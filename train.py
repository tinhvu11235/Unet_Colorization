import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
from skimage.color import lab2rgb
from datetime import datetime
from diffusers.schedulers import DDPMScheduler

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

def save_checkpoint_local(epoch, model, optimizer, scheduler, run_id, save_dir, best_val=None, is_best=False):
    ensure_dir(save_dir)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "run_id": run_id,
        "best_val": best_val,
        "timestamp": datetime.now().isoformat(),
    }
    ckpt_path = get_ckpt_path(save_dir, epoch)
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
def sample_colorization(model, L, scheduler, num_steps):
    model.eval()
    B, _, H, W = L.shape
    ab = torch.randn(B, 2, H, W, device=L.device)
    scheduler.set_timesteps(num_steps)
    for t in scheduler.timesteps:
        t_batch = t.expand(B)
        pred_noise = model(ab, L, t_batch)
        ab = scheduler.step(pred_noise, t, ab).prev_sample
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
):

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

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        net_G.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
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

    for epoch in range(start_epoch, epochs):
        net_G.train()
        train_loss = 0.0

        for data in train_dl:
            L = data["L"].to(DEVICE)
            ab = data["ab"].to(DEVICE)

            B = ab.size(0)
            t = torch.randint(
                0, train_noise_scheduler.num_train_timesteps, (B,), device=DEVICE
            ).long()

            noise = torch.randn_like(ab)
            ab_t = train_noise_scheduler.add_noise(ab, noise, t)

            pred_noise = net_G(ab_t, L, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)

        net_G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val in val_dl:
                L_v = val["L"].to(DEVICE)
                ab_v = val["ab"].to(DEVICE)

                B = ab_v.size(0)
                t = torch.randint(
                    0, train_noise_scheduler.num_train_timesteps, (B,), device=DEVICE
                ).long()

                noise = torch.randn_like(ab_v)
                ab_t = train_noise_scheduler.add_noise(ab_v, noise, t)

                pred_noise = net_G(ab_t, L_v, t)
                val_loss += F.mse_loss(pred_noise, noise).item()

        val_loss /= len(val_dl)
        lr_scheduler.step(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        save_checkpoint_local(
            epoch,
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
                net_G, L_fix, infer_noise_scheduler, inference_steps
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
                net_G, L_r, infer_noise_scheduler, inference_steps
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
    )
