import os
import gdown
import requests
import urllib.parse as _urlparse
import torch
import numpy as np
import random
from tqdm.auto import tqdm
from skimage.color import lab2rgb
from torch.utils.data import DataLoader
import wandb
from data_loader import create_dataloaders
from config import Config as cfg
from model import GAN, load_trained_model, pretrain_discriminator, get_encoder_weights, tv_loss

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)
    return lab2rgb(Lab)

def log_image_wandb(L, ab, num=5, captions=None):
    L = L.cpu().detach().numpy()
    ab = ab.cpu().detach().numpy()
    wandb_image = []
    n = min(num, L.shape[0])
    for i in range(n):
        image = lab_to_rgb(L[i], ab[i])
        cap = captions[i] if captions is not None and i < len(captions) else f"Image {i}"
        wandb_image.append(wandb.Image(image, caption=cap))
    return wandb_image

def evaluate_L1_on_val(GAN_model, val_dl):
    GAN_model.net_G.eval()
    total_L1 = 0.0
    with torch.no_grad():
        for data in val_dl:
            GAN_model.setup_input(data)
            GAN_model.forward()
            l1 = GAN_model.L1criterion(GAN_model.fake_color, GAN_model.ab) * GAN_model.lambda_L1
            total_L1 += l1.item()
    return total_L1 / max(1, len(val_dl))

def save_checkpoint_local(path, model, epoch, metric_name=None, metric_value=None):
    ckpt = {
        'epoch': epoch + 1,
        'Unet_state_dict': model.net_G.state_dict(),
        'Gray_state_dict': model.net_F.state_dict() if hasattr(model, 'net_F') else None,
        'Disc_state_dict': model.net_D.state_dict(),
        'optimizer_Unet_state_dict': model.opt_G.state_dict(),
        'optimizer_Gray_state_dict': model.opt_F.state_dict() if hasattr(model, 'opt_F') else None,
        'optimizer_Disc_state_dict': model.opt_D.state_dict(),
        'scheduler_state_dict': model.scheduler_G.state_dict(),
        'schedulerF_state_dict': model.scheduler_F.state_dict() if hasattr(model, 'scheduler_F') else None,
        'metric_name': metric_name,
        'metric_value': metric_value,
    }
    torch.save(ckpt, path)

def is_better(curr, best, mode='min'):
    if best is None:
        return True
    return (curr < best) if mode == 'min' else (curr > best)

def download_model(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

def download_checkpoint_from_url(url: str, out_dir="/kaggle/working") -> str:
    os.makedirs(out_dir, exist_ok=True)
    if "drive.google.com" in url or "id=" in url:
        parsed = _urlparse.urlparse(url)
        name = os.path.basename(parsed.path) or "checkpoint.pth"
        if not name.endswith(".pth"):
            name = "checkpoint.pth"
        out_path = os.path.join(out_dir, name)
        gdown.download(url=url, output=out_path, quiet=False, fuzzy=True)
        if not os.path.exists(out_path):
            raise RuntimeError(f"gdown failed to download: {url}")
        return out_path
    filename = os.path.basename(_urlparse.urlparse(url).path) or "checkpoint.pth"
    if not filename.endswith(".pth"):
        filename = "checkpoint.pth"
    out_path = os.path.join(out_dir, filename)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return out_path

def download_pretrain_generator():
    model_url = 'https://drive.google.com/uc?id=1dD7PQt1RB-IqNVJFHlnsG9MdkmdDuRxH'
    model_path = 'model.pth'
    download_model(model_url, model_path)
    return load_trained_model(model_path)

def pretrain_encoder_weights():
    model_url = 'https://drive.google.com/uc?id=1dD7PQt1RB-IqNVJFHlnsG9MdkmdDuRxH'
    model_path = 'model.pth'
    download_model(model_url, model_path)
    return get_encoder_weights(model_path)

def train_GAN(GAN_model, train_dl, val_dl, log_interval, checkpoint_path=None, warmup_epochs=2, out_dir="/kaggle/working", best_metric_name="val_L1", best_mode="min"):
    epochs = cfg["EPOCHS"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_last = os.path.join(out_dir, "checkpoint_last.pth")
    ckpt_best = os.path.join(out_dir, "checkpoint_best.pth")
    start_epoch = 0
    best_metric = None
    best_epoch = None
    run_id = None
    if checkpoint_path:
        if checkpoint_path.startswith("http"):
            checkpoint_file = download_checkpoint_from_url(checkpoint_path, out_dir=out_dir)
        else:
            checkpoint_file = checkpoint_path
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        start_epoch = checkpoint.get('epoch', 0)
        GAN_model.net_G.load_state_dict(checkpoint['Unet_state_dict'])
        if checkpoint.get('Gray_state_dict') is not None and hasattr(GAN_model, 'net_F'):
            GAN_model.net_F.load_state_dict(checkpoint['Gray_state_dict'])
        GAN_model.net_D.load_state_dict(checkpoint['Disc_state_dict'])
        GAN_model.opt_G.load_state_dict(checkpoint['optimizer_Unet_state_dict'])
        if checkpoint.get('optimizer_Gray_state_dict') is not None and hasattr(GAN_model, 'opt_F'):
            GAN_model.opt_F.load_state_dict(checkpoint['optimizer_Gray_state_dict'])
        GAN_model.opt_D.load_state_dict(checkpoint['optimizer_Disc_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            GAN_model.scheduler_G.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('schedulerF_state_dict') is not None and hasattr(GAN_model, 'scheduler_F'):
            GAN_model.scheduler_F.load_state_dict(checkpoint['schedulerF_state_dict'])
        for state in GAN_model.opt_G.state.values():
            if 'momentum_buffer' in state: state['momentum_buffer'].zero_()
        for state in GAN_model.opt_D.state.values():
            if 'momentum_buffer' in state: state['momentum_buffer'].zero_()
        if hasattr(GAN_model, 'opt_F'):
            for state in GAN_model.opt_F.state.values():
                if 'momentum_buffer' in state: state['momentum_buffer'].zero_()
        best_metric = checkpoint.get('metric_value', None)
    wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"], config=cfg)
    for epoch in range(start_epoch, epochs):
        if epoch == 0:
            for warmup_epoch in range(warmup_epochs):
                step_warmup = 0
                for data in tqdm(train_dl, desc=f"Warmup Epoch {warmup_epoch+1}"):
                    GAN_model.setup_input(data)
                    GAN_model.warmup_optimize()
                    step_warmup += 1
                    if step_warmup % log_interval == 0:
                        with torch.no_grad():
                            bs = cfg["BATCH_SIZE"]
                            caps = [f"warmup{warmup_epoch+1}_step{step_warmup}_img{i+1}" for i in range(bs)]
                            data_fix = next(iter(val_dl))
                            GAN_model.setup_input(data_fix)
                            GAN_model.forward()
                            fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps)
                            real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab, captions=caps)
                            batch_rand = next(iter(val_dl))
                            GAN_model.setup_input(batch_rand)
                            GAN_model.forward()
                            val_fake = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5, captions=caps)
                            val_real = log_image_wandb(GAN_model.L, GAN_model.ab, num=5, captions=caps)
                        wandb.log({
                            "fix_fake_images": fake_imgs,
                            "fix_real_images": real_imgs,
                            "random_fake_images": val_fake,
                            "random_real_images": val_real,
                        })
        running_loss_G = running_loss_D = 0.0
        running_loss_G_GAN = running_loss_G_L1 = 0.0
        running_loss_D_fake = running_loss_D_real = 0.0
        step = 0
        for data in tqdm(train_dl, desc=f"Training Epoch {epoch+1}"):
            GAN_model.setup_input(data)
            GAN_model.optimize()
            running_loss_G += GAN_model.loss_G.item()
            running_loss_D += GAN_model.loss_D.item()
            running_loss_G_GAN += GAN_model.loss_G_GAN.item()
            running_loss_G_L1 += GAN_model.loss_G_L1.item()
            running_loss_D_fake += GAN_model.loss_D_fake.item()
            running_loss_D_real += GAN_model.loss_D_real.item()
            step += 1
            if step % log_interval == 0:
                with torch.no_grad():
                    bs = cfg["BATCH_SIZE"]
                    caps = [f"{epoch+1}_step{step}_img{i+1}" for i in range(bs)]
                    data_fix = next(iter(val_dl))
                    GAN_model.setup_input(data_fix)
                    GAN_model.forward()
                    fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps)
                    real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab, captions=caps)
                    batch_rand = next(iter(val_dl))
                    GAN_model.setup_input(batch_rand)
                    GAN_model.forward()
                    val_fake = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5, captions=caps)
                    val_real = log_image_wandb(GAN_model.L, GAN_model.ab, num=5, captions=caps)
                wandb.log({
                    "fix_fake_images": fake_imgs,
                    "fix_real_images": real_imgs,
                    "random_fake_images": val_fake,
                    "random_real_images": val_real,
                }, commit=False)
        num_batches = max(1, len(train_dl))
        average_loss_G = running_loss_G / num_batches
        average_loss_D = running_loss_D / num_batches
        average_loss_G_GAN = running_loss_G_GAN / num_batches
        average_loss_G_L1 = running_loss_G_L1 / num_batches
        average_loss_D_fake = running_loss_D_fake / num_batches
        average_loss_D_real = running_loss_D_real / num_batches
        val_L1 = evaluate_L1_on_val(GAN_model, val_dl)
        GAN_model.scheduler_G.step(val_L1)
        if hasattr(GAN_model, 'scheduler_F'):
            GAN_model.scheduler_F.step(val_L1)
        with torch.no_grad():
            data_fix = next(iter(val_dl))
            GAN_model.setup_input(data_fix)
            GAN_model.forward()
            fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color)
            real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab)
            batch_rand = next(iter(val_dl))
            GAN_model.setup_input(batch_rand)
            GAN_model.forward()
            val_fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5)
            val_real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab, num=5)
            ssim_val = GAN_model.SSIMcriterion(GAN_model.rec_L, GAN_model.L).item()
            tv_val = tv_loss(GAN_model.fake_color).item()
        wandb.log({
            "epoch_train_loss_G": average_loss_G,
            "epoch_train_loss_D": average_loss_D,
            "epoch_train_loss_G_GAN": average_loss_G_GAN,
            "epoch_train_loss_G_L1": average_loss_G_L1,
            "epoch_val_loss_G_L1": val_L1,
            "epoch_train_loss_D_fake": average_loss_D_fake,
            "epoch_train_loss_D_real": average_loss_D_real,
            "lr_G": GAN_model.opt_G.param_groups[0]['lr'],
            "lr_F": GAN_model.opt_F.param_groups[0]['lr'],
            "cycle_ssim_sample": ssim_val,
            "tv_ab_sample": tv_val,
            "end_fake_images": fake_imgs,
            "end_real_images": real_imgs,
            "end_val_fake_images": val_fake_imgs,
            "end_val_real_images": val_real_imgs,
        })
        print(f"Epoch {epoch+1}/{epochs} — train L1: {average_loss_G_L1:.4f}, val L1: {val_L1:.4f}")
        save_checkpoint_local(ckpt_last, GAN_model, epoch, metric_name=best_metric_name, metric_value=val_L1)
        current_metric = val_L1
        if is_better(current_metric, best_metric, mode=best_mode):
            best_metric = current_metric
            best_epoch = epoch + 1
            save_checkpoint_local(ckpt_best, GAN_model, epoch, metric_name=best_metric_name, metric_value=best_metric)
            print(f"[BEST] epoch {best_epoch}: {best_metric_name} = {best_metric:.4f} -> saved to {ckpt_best}")

def train_from_scratch():
    train_dl, val_dl = create_dataloaders(cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"], cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"])
    net_GAN = GAN(
        lr_G=cfg["LR_G"], lr_D=cfg["LR_D"],
        lambda_L1=cfg.get("LAMBDA_L1", 100.0),
        lambda_cycle=cfg.get("LAMBDA_CYCLE", 10.0),
        tv_weight=cfg.get("TV_WEIGHT", 0.1),
        blur_k=cfg.get("BLUR_K", 3),
        blur_sigma=cfg.get("BLUR_SIGMA", 1.0),
        blur_p=cfg.get("BLUR_P", 0.5),
        noise_std=cfg.get("NOISE_STD", 0.0),
    )
    train_GAN(net_GAN, train_dl, val_dl, log_interval=cfg["LOG_INTERVAL"], warmup_epochs=cfg.get("WARMUP_EPOCHS", 2), out_dir="/kaggle/working", best_metric_name="val_L1", best_mode="min")

def train_from_checkpoint(url):
    checkpoint_file = download_checkpoint_from_url(url, out_dir="/kaggle/working")
    train_dl, val_dl = create_dataloaders(cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"], cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"])
    net_GAN = GAN(
        lr_G=cfg["LR_G"], lr_D=cfg["LR_D"],
        lambda_L1=cfg.get("LAMBDA_L1", 100.0),
        lambda_cycle=cfg.get("LAMBDA_CYCLE", 10.0),
        tv_weight=cfg.get("TV_WEIGHT", 0.1),
        blur_k=cfg.get("BLUR_K", 3),
        blur_sigma=cfg.get("BLUR_SIGMA", 1.0),
        blur_p=cfg.get("BLUR_P", 0.5),
        noise_std=cfg.get("NOISE_STD", 0.0),
    )
    train_GAN(net_GAN, train_dl, val_dl, log_interval=cfg["LOG_INTERVAL"], checkpoint_path=checkpoint_file, warmup_epochs=cfg.get("WARMUP_EPOCHS", 2), out_dir="/kaggle/working", best_metric_name="val_L1", best_mode="min")
