import os
import re
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

# Log sampling số step giống inference thực tế
LOG_STEPS_FIX = 1000
LOG_STEPS_RAND = 1000

# Min-SNR
MIN_SNR_GAMMA = 5.0

# X0 reconstruction loss (dùng ab groundtruth tham gia loss)
X0_LOSS_WEIGHT = 0.3
X0_LOSS_TYPE = "smooth_l1"  # "l1" | "smooth_l1" | "mse"

# Dynamic thresholding (áp trong sampling/log)
DT_PERCENTILE = 0.995   # gợi ý: 0.99 ~ 0.995; 0.8 thường quá gắt => nhạt màu
DT_CLAMP_MIN = 1.0      # s >= 1 để không phóng đại (scale-up)
DT_APPLY_EVERY_STEP = True
DT_LAST_K_STEPS = 200   # dùng nếu DT_APPLY_EVERY_STEP=False


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


def stat_ab(name: str, ab: torch.Tensor):
    m = ab.mean(dim=[0, 2, 3]).detach().cpu().numpy()
    s = ab.std(dim=[0, 2, 3]).detach().cpu().numpy()
    mx = ab.abs().max().item()
    print(f"{name}: mean={m}, std={s}, |max|={mx:.3f}")


def min_snr_weight(scheduler: DDPMScheduler, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(t.device)
    a = alphas_cumprod[t]
    snr = a / (1 - a)
    w = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    return w


def x0_loss_fn(x0_hat: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    if X0_LOSS_TYPE == "l1":
        return (x0_hat - x0).abs().mean()
    if X0_LOSS_TYPE == "mse":
        return (x0_hat - x0).pow(2).mean()
    return F.smooth_l1_loss(x0_hat, x0, reduction="mean")


def _right_pad_dims_to(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # t: (B,) -> (B,1,1,1,...) broadcast theo x
    while t.ndim < x.ndim:
        t = t.view(t.shape[0], *([1] * (x.ndim - 1)))
    return t


@torch.no_grad()
def dynamic_threshold(img: torch.Tensor, percentile: float = 0.995, clamp_min: float = 1.0) -> torch.Tensor:
    """
    img: (B,C,H,W)
    Lưu ý: đây KHÔNG phải clamp cứng. Nó clamp theo +/-s rồi chia s -> về [-1,1] theo rescale mềm.
    """
    B = img.shape[0]
    flat = img.abs().reshape(B, -1)
    s = torch.quantile(flat, percentile, dim=1)           # (B,)
    s = torch.clamp(s, min=clamp_min)                     # s >= 1
    s = _right_pad_dims_to(img, s)                        # (B,1,1,1)
    img = img.clamp(-s, s) / s
    return img


@torch.no_grad()
def sample_colorization(
    model,
    L,
    scheduler: DDPMScheduler,
    num_steps: int,
    show_tqdm: bool = False,
    init_ab=None,
    generator=None,
    dt_percentile: float | None = DT_PERCENTILE,
    dt_apply_every_step: bool = DT_APPLY_EVERY_STEP,
    dt_last_k_steps: int = DT_LAST_K_STEPS,
):
    """
    Trả về x0_hat đã dynamic-threshold ở bước cuối (dùng để render/log).
    Không clamp x_t (ab) mỗi step.
    """
    model.eval()
    B, _, H, W = L.shape

    scheduler.set_timesteps(num_steps)
    alphas_cumprod = scheduler.alphas_cumprod.to(L.device)

    if init_ab is None:
        ab = torch.randn(B, 2, H, W, device=L.device, generator=generator)
        ab = ab * scheduler.init_noise_sigma
    else:
        ab = init_ab.clone()

    iterator = scheduler.timesteps
    if show_tqdm:
        iterator = tqdm(iterator, desc="Sampling", leave=False)

    timesteps_list = list(iterator)
    nT = len(timesteps_list)

    x0_hat = None

    for i, t in enumerate(timesteps_list):
        t_int = int(t)
        t_batch = torch.full((B,), t_int, device=L.device, dtype=torch.long)

        model_in = scheduler.scale_model_input(ab, t_int)
        eps_hat = model(model_in, L, t_batch)

        a = alphas_cumprod[t_int].view(1, 1, 1, 1)
        x0_hat = (ab - (1.0 - a).sqrt() * eps_hat) / (a.sqrt() + 1e-8)

        if dt_percentile is not None:
            do_dt = dt_apply_every_step or (i >= (nT - int(dt_last_k_steps)))
            if do_dt:
                x0_hat = dynamic_threshold(x0_hat, percentile=float(dt_percentile), clamp_min=float(DT_CLAMP_MIN))
                eps_hat = (ab - a.sqrt() * x0_hat) / ((1.0 - a).sqrt() + 1e-8)

        ab = scheduler.step(eps_hat, t_int, ab).prev_sample

    return x0_hat


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

    optimizer = optim.AdamW(net_G.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.95, patience=5)

    train_noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )

    # Inference scheduler: để clip_sample=False (tránh double clip),
    # dynamic threshold đã chịu trách nhiệm kìm x0_hat.
    infer_noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
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
        wandb.init(project=config["WANDB_PROJECT"], name=config["WANDB_RUN_NAME"], id=run_id, resume="must")
    else:
        wandb.init(project=config["WANDB_PROJECT"], name=config["WANDB_RUN_NAME"], config={"lr": lr, "epochs": epochs})
        run_id = wandb.run.id

    fixed_batch = next(iter(val_dl))
    L_fix_all = fixed_batch["L"].to(DEVICE)
    ab_fix_all = fixed_batch["ab"].to(DEVICE)

    n_vis_fix = min(5, L_fix_all.size(0))
    L_fix = L_fix_all[:n_vis_fix]
    ab_fix = ab_fix_all[:n_vis_fix]

    gen_fix = torch.Generator(device=DEVICE).manual_seed(1234)
    fixed_init_ab = torch.randn(ab_fix.shape, device=ab_fix.device, dtype=ab_fix.dtype, generator=gen_fix)
    fixed_init_ab = fixed_init_ab * infer_noise_scheduler.init_noise_sigma

    val_iter = iter(val_dl)
    check = True

    alphas_cumprod_train = train_noise_scheduler.alphas_cumprod.to(DEVICE)

    for epoch in range(start_epoch, epochs):
        net_G.train()
        train_loss = 0.0
        train_eps = 0.0
        train_x0 = 0.0

        pbar = tqdm(train_dl, desc=f"Train {epoch+1}/{epochs}", leave=False)
        for data in pbar:
            L = data["L"].to(DEVICE, non_blocking=True)
            ab = data["ab"].to(DEVICE, non_blocking=True)

            B = ab.size(0)
            t = torch.randint(0, train_noise_scheduler.num_train_timesteps, (B,), device=DEVICE).long()

            noise = torch.randn_like(ab)
            ab_t = train_noise_scheduler.add_noise(ab, noise, t)

            pred_noise = net_G(ab_t, L, t)

            mse = (pred_noise - noise).pow(2).mean(dim=[1, 2, 3])
            w = min_snr_weight(train_noise_scheduler, t, gamma=MIN_SNR_GAMMA)
            loss_eps = (w * mse).mean()

            # x0_hat từ epsilon (KHÔNG clamp)
            a = alphas_cumprod_train[t].view(-1, 1, 1, 1)
            x0_hat = (ab_t - (1.0 - a).sqrt() * pred_noise) / (a.sqrt() + 1e-8)
            loss_x0 = x0_loss_fn(x0_hat, ab)

            loss = loss_eps + float(X0_LOSS_WEIGHT) * loss_x0

            if epoch == 0 and check:
                baseline = F.mse_loss(torch.zeros_like(noise), noise).item()
                print("noise mean/std:", noise.mean().item(), noise.std().item())
                print("baseline mse (pred=0):", baseline)
                print("loss_eps:", loss_eps.item(), "loss_x0:", loss_x0.item(), "loss_total:", loss.item())
                check = False

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_eps += loss_eps.item()
            train_x0 += loss_x0.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                eps=f"{loss_eps.item():.4f}",
                x0=f"{loss_x0.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        n_train = max(1, len(train_dl))
        train_loss /= n_train
        train_eps /= n_train
        train_x0 /= n_train

        net_G.eval()
        val_loss = 0.0
        val_eps = 0.0
        val_x0 = 0.0

        with torch.no_grad():
            vbar = tqdm(val_dl, desc="Val", leave=False)
            for val in vbar:
                L_v = val["L"].to(DEVICE, non_blocking=True)
                ab_v = val["ab"].to(DEVICE, non_blocking=True)

                B = ab_v.size(0)
                t = torch.randint(0, train_noise_scheduler.num_train_timesteps, (B,), device=DEVICE).long()

                noise = torch.randn_like(ab_v)
                ab_t = train_noise_scheduler.add_noise(ab_v, noise, t)

                pred_noise = net_G(ab_t, L_v, t)

                mse = (pred_noise - noise).pow(2).mean(dim=[1, 2, 3])
                w = min_snr_weight(train_noise_scheduler, t, gamma=MIN_SNR_GAMMA)
                loss_eps_b = (w * mse).mean()

                a = alphas_cumprod_train[t].view(-1, 1, 1, 1)
                x0_hat = (ab_t - (1.0 - a).sqrt() * pred_noise) / (a.sqrt() + 1e-8)
                loss_x0_b = x0_loss_fn(x0_hat, ab_v)

                loss_b = loss_eps_b + float(X0_LOSS_WEIGHT) * loss_x0_b

                val_loss += loss_b.item()
                val_eps += loss_eps_b.item()
                val_x0 += loss_x0_b.item()
                vbar.set_postfix(val=f"{loss_b.item():.4f}", eps=f"{loss_eps_b.item():.4f}", x0=f"{loss_x0_b.item():.4f}")

        n_val = max(1, len(val_dl))
        val_loss /= n_val
        val_eps /= n_val
        val_x0 /= n_val

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

        # --- Sampling/log đúng kiểu inference 1000 step, dùng dynamic threshold ---
        with torch.no_grad():
            fake_fix = sample_colorization(
                net_G,
                L_fix,
                infer_noise_scheduler,
                num_steps=LOG_STEPS_FIX,
                show_tqdm=show_sampling_tqdm,
                init_ab=fixed_init_ab,
                dt_percentile=DT_PERCENTILE,
                dt_apply_every_step=DT_APPLY_EVERY_STEP,
                dt_last_k_steps=DT_LAST_K_STEPS,
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

            gen_rand = torch.Generator(device=DEVICE).manual_seed(999 + epoch)
            fake_rand = sample_colorization(
                net_G,
                L_r,
                infer_noise_scheduler,
                num_steps=LOG_STEPS_RAND,
                show_tqdm=show_sampling_tqdm,
                init_ab=None,
                generator=gen_rand,
                dt_percentile=DT_PERCENTILE,
                dt_apply_every_step=DT_APPLY_EVERY_STEP,
                dt_last_k_steps=DT_LAST_K_STEPS,
            )

        stat_ab("real_fix", ab_fix)
        stat_ab("fake_fix_x0_dt", fake_fix)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_eps": train_eps,
            "train_x0": train_x0,
            "val_loss": val_loss,
            "val_eps": val_eps,
            "val_x0": val_x0,
            "lr": optimizer.param_groups[0]['lr'],

            "images/fake_fix": log_image_wandb(L_fix, fake_fix),
            "images/real_fix": log_image_wandb(L_fix, ab_fix),
            "images/fake_rand": log_image_wandb(L_r, fake_rand),
            "images/real_rand": log_image_wandb(L_r, ab_r),

            "log_steps_fix": LOG_STEPS_FIX,
            "log_steps_rand": LOG_STEPS_RAND,

            "min_snr_gamma": MIN_SNR_GAMMA,
            "x0_loss_weight": float(X0_LOSS_WEIGHT),
            "x0_loss_type": X0_LOSS_TYPE,

            "dt_percentile": float(DT_PERCENTILE),
            "dt_clamp_min": float(DT_CLAMP_MIN),
            "dt_apply_every_step": bool(DT_APPLY_EVERY_STEP),
            "dt_last_k_steps": int(DT_LAST_K_STEPS),
        })

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | "
            f"train_eps: {train_eps:.6f} | train_x0: {train_x0:.6f} | "
            f"val_eps: {val_eps:.6f} | val_x0: {val_x0:.6f}"
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
