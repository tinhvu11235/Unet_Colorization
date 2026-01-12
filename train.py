import os
import re
import tempfile
import urllib.parse
import urllib.request

import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
from skimage.color import lab2rgb
from datetime import datetime
from diffusers.schedulers import DDPMScheduler
from tqdm.auto import tqdm

from config import DEVICE
from model import build_model
from data_loader import create_dataloaders

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

config = {}


def lab_to_rgb(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)
    return lab2rgb(Lab)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_ckpt_path(save_dir, epoch, prefix="checkpoint"):
    return os.path.join(save_dir, f"{prefix}_epoch_{epoch}.pth")


def _is_url(s: str) -> bool:
    try:
        u = urllib.parse.urlparse(s)
        return u.scheme in ("http", "https")
    except Exception:
        return False


def _extract_gdrive_file_id(url: str) -> str | None:
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "id" in qs and len(qs["id"]) > 0:
        return qs["id"][0]
    return None


def _download_gdrive(url: str, out_path: str) -> str:
    file_id = _extract_gdrive_file_id(url)
    if file_id is None:
        raise ValueError("Không trích được file id từ link Google Drive.")

    try:
        import gdown  # type: ignore
        gdown.download(id=file_id, output=out_path, quiet=False, fuzzy=True)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("gdown download failed (file rỗng hoặc không tồn tại).")
        return out_path
    except Exception:
        direct = f"https://drive.google.com/uc?export=download&id={file_id}"
        urllib.request.urlretrieve(direct, out_path)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(
                "Tải bằng direct link thất bại. Cài gdown (pip install gdown) rồi thử lại."
            )
        return out_path


def resolve_checkpoint_path(checkpoint_path: str | None, cache_dir: str) -> str | None:
    if not checkpoint_path:
        return None

    if os.path.exists(checkpoint_path):
        return checkpoint_path

    if _is_url(checkpoint_path):
        ensure_dir(cache_dir)
        basename = os.path.basename(urllib.parse.urlparse(checkpoint_path).path).strip()
        if not basename or "." not in basename:
            basename = "checkpoint_download.pth"
        local_path = os.path.join(cache_dir, basename)

        if "drive.google.com" in checkpoint_path:
            return _download_gdrive(checkpoint_path, local_path)

        urllib.request.urlretrieve(checkpoint_path, local_path)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise RuntimeError("Download checkpoint từ URL thất bại.")
        return local_path

    raise FileNotFoundError(f"Không tìm thấy checkpoint_path: {checkpoint_path}")


def save_checkpoint_local(epoch_next, model, optimizer, scheduler, run_id, save_dir, best_val=None, is_best=False):
    ensure_dir(save_dir)
    payload = {
        "epoch": int(epoch_next),
        "epoch_saved_as_next": True,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "run_id": run_id,
        "best_val": best_val,
        "timestamp": datetime.now().isoformat(),
    }
    ckpt_path = get_ckpt_path(save_dir, epoch_next)
    torch.save(payload, ckpt_path)
    if is_best:
        torch.save(payload, os.path.join(save_dir, "best.pth"))
    return ckpt_path


def log_image_wandb(L, ab, num=5, captions=None):
    L = L.detach().cpu().numpy()
    ab = ab.detach().cpu().numpy()
    B = min(L.shape[0], num)
    images = []
    for i in range(B):
        rgb = lab_to_rgb(L[i], ab[i])
        cap = captions[i] if captions is not None else f"img_{i}"
        images.append(wandb.Image(rgb, caption=cap))
    return images


@torch.no_grad()
def sample_colorization(model, L, scheduler, num_steps, show_tqdm=False):
    model.eval()
    B, _, H, W = L.shape
    ab = torch.randn(B, 2, H, W, device=L.device)

    scheduler.set_timesteps(num_steps)
    iterator = scheduler.timesteps
    if show_tqdm:
        iterator = tqdm(iterator, desc="Sampling", leave=False)

    for t in iterator:
        t_batch = torch.full((B,), int(t), device=L.device, dtype=torch.long)
        pred_noise = model(ab, L, t_batch)
        ab = scheduler.step(pred_noise, int(t), ab).prev_sample

    return ab


def train_model(
    net_G,
    train_dl,
    val_dl,
    epochs,
    lr,
    checkpoint_path=None,
    save_dir="/kaggle/working/checkpoints",
    inference_steps=50,
    show_sampling_tqdm=False,
):
    ensure_dir(save_dir)
    ckpt_cache_dir = os.path.join(save_dir, "_ckpt_cache")
    checkpoint_path = resolve_checkpoint_path(checkpoint_path, ckpt_cache_dir)

    optimizer = optim.Adam(net_G.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.95, patience=5
    )

    train_noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
    )

    infer_noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
    )

    start_epoch = 0
    run_id = None
    best_val = float("inf")

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        net_G.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        saved_epoch = int(ckpt.get("epoch", 0))
        saved_as_next = bool(ckpt.get("epoch_saved_as_next", False))
        start_epoch = saved_epoch if saved_as_next else (saved_epoch + 1)
        run_id = ckpt.get("run_id", None)
        best_val = ckpt.get("best_val", best_val)

    if run_id:
        wandb.init(
            project=config["WANDB_PROJECT"],
            name=config["WANDB_RUN_NAME"],
            id=run_id,
            resume="must",
        )
    else:
        wandb.init(
            project=config["WANDB_PROJECT"],
            name=config["WANDB_RUN_NAME"],
            config={"lr": lr, "epochs": epochs},
        )
        run_id = wandb.run.id

    fixed_batch = next(iter(val_dl))
    L_fix_all = fixed_batch["L"].to(DEVICE)
    ab_fix_all = fixed_batch["ab"].to(DEVICE)
    val_iter = iter(val_dl)
    check = True
    for epoch in range(start_epoch, epochs):
        net_G.train()
        train_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Train {epoch+1}/{epochs}", leave=False)
        for data in pbar:
            L = data["L"].to(DEVICE, non_blocking=True)
            ab = data["ab"].to(DEVICE, non_blocking=True)

            B = ab.size(0)
            t = torch.randint(
                0, train_noise_scheduler.num_train_timesteps, (B,), device=DEVICE
            ).long()

            noise = torch.randn_like(ab)
            ab_t = train_noise_scheduler.add_noise(ab, noise, t)

            pred_noise = net_G(ab_t, L, t)
            loss = F.mse_loss(pred_noise, noise)
            if epoch == 0 and check == True: 
                baseline = F.mse_loss(torch.zeros_like(noise), noise).item()
                print("noise mean/std:", noise.mean().item(), noise.std().item())
                print("baseline mse (pred=0):", baseline)
                print("current loss:", loss.item())
                check = False
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss /= max(1, len(train_dl))

        net_G.eval()
        val_loss = 0.0
        with torch.no_grad():
            vbar = tqdm(val_dl, desc="Val", leave=False)
            for val in vbar:
                L_v = val["L"].to(DEVICE, non_blocking=True)
                ab_v = val["ab"].to(DEVICE, non_blocking=True)

                B = ab_v.size(0)
                t = torch.randint(
                    0, train_noise_scheduler.num_train_timesteps, (B,), device=DEVICE
                ).long()

                noise = torch.randn_like(ab_v)
                ab_t = train_noise_scheduler.add_noise(ab_v, noise, t)

                pred_noise = net_G(ab_t, L_v, t)
                batch_v = F.mse_loss(pred_noise, noise).item()
                val_loss += batch_v
                vbar.set_postfix(val_mse=f"{batch_v:.4f}")

        val_loss /= max(1, len(val_dl))
        lr_scheduler.step(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        save_checkpoint_local(
            epoch + 1,
            net_G,
            optimizer,
            lr_scheduler,
            run_id,
            save_dir,
            best_val=best_val,
            is_best=is_best,
        )

        with torch.no_grad():
            n_vis = min(5, L_fix_all.size(0))
            L_fix = L_fix_all[:n_vis]
            ab_fix = ab_fix_all[:n_vis]

            fake_fix = sample_colorization(
                net_G, L_fix, infer_noise_scheduler, inference_steps, show_tqdm=show_sampling_tqdm
            )

            try:
                data_rand = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dl)
                data_rand = next(val_iter)

            L_r_all = data_rand["L"].to(DEVICE)
            ab_r_all = data_rand["ab"].to(DEVICE)
            n_vis_r = min(5, L_r_all.size(0))
            L_r = L_r_all[:n_vis_r]
            ab_r = ab_r_all[:n_vis_r]

            fake_rand = sample_colorization(
                net_G, L_r, infer_noise_scheduler, inference_steps, show_tqdm=show_sampling_tqdm
            )

        net_G.train()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "images/fake_fix": log_image_wandb(L_fix, fake_fix),
            "images/real_fix": log_image_wandb(L_fix, ab_fix),
            "images/fake_rand": log_image_wandb(L_r, fake_rand),
            "images/real_rand": log_image_wandb(L_r, ab_r),
        })

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}"
        )

    wandb.finish()


def train_from_scratch(cfg):
    global config
    config = cfg

    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"],
        cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"],
        cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"],
        cfg["VAL_SIZE"],
    )

    net_G = build_model().to(DEVICE)

    train_model(
        net_G,
        train_dl,
        val_dl,
        epochs=cfg["EPOCHS"],
        lr=cfg["LR"],
        save_dir=cfg["CHECKPOINT_DIR"],
        inference_steps=cfg.get("INFERENCE_STEPS", 50),
        show_sampling_tqdm=cfg.get("SHOW_SAMPLING_TQDM", False),
    )


def continue_training(cfg, checkpoint_path):
    global config
    config = cfg

    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"],
        cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"],
        cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"],
        cfg["VAL_SIZE"],
    )

    net_G = build_model().to(DEVICE)

    train_model(
        net_G,
        train_dl,
        val_dl,
        epochs=cfg["EPOCHS"],
        lr=cfg["LR"],
        checkpoint_path=checkpoint_path,
        save_dir=cfg["CHECKPOINT_DIR"],
        inference_steps=cfg.get("INFERENCE_STEPS", 50),
        show_sampling_tqdm=cfg.get("SHOW_SAMPLING_TQDM", False),
    )
