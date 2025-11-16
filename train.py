import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import numpy as np
from skimage.color import lab2rgb
import gdown
from datetime import datetime
from config import DEVICE
from model import build_model
from data_loader import create_dataloaders

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

config = {}

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)
    return lab2rgb(Lab)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_ckpt_path(save_dir, epoch, prefix="checkpoint"):
    fname = f"{prefix}_epoch_{epoch}.pth"
    return os.path.join(save_dir, fname)
def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
def save_checkpoint_local(epoch, model, optimizer, scheduler, run_id, save_dir, best_val=None, is_best=False):
    ensure_dir(save_dir)
    payload = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'run_id': run_id,
        'best_val': best_val,
        'timestamp': datetime.now().isoformat()
    }
    ckpt_path = get_ckpt_path(save_dir, epoch)
    torch.save(payload, ckpt_path)
    if is_best:
        best_path = os.path.join(save_dir, "best.pth")
        torch.save(payload, best_path)
    return ckpt_path

def download_ckpt_from_gdrive(gdrive_id_or_url, dst_dir):
    ensure_dir(dst_dir)
    outfile = os.path.join(dst_dir, "resume_from_drive.pth")
    gdown.download(url=gdrive_id_or_url, output=outfile, quiet=False, fuzzy=True)
    if not os.path.exists(outfile):
        raise ValueError("Cannot download checkpoint from Google Drive.")
    return outfile

def train_model(net_G, train_dl, val_dl, epochs, lr,
                    beta_kl=1e-3,
                    checkpoint_path=None, save_dir="/kaggle/working/checkpoints",
                    save_every=1, save_best=True):

    optimizer = optim.Adam(net_G.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)
    criterion = nn.L1Loss()

    start_epoch = 0
    run_id = None
    best_val = float("inf")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        net_G.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        run_id = ckpt.get('run_id', None)
        best_val = ckpt.get('best_val', best_val)

    if run_id:
        wandb.init(project=config["WANDB_PROJECT"], name=config["WANDB_RUN_NAME"],
                   id=run_id, resume="must")
    else:
        wandb.init(project=config["WANDB_PROJECT"], name=config["WANDB_RUN_NAME"],
                   config={'lr': lr, 'epochs': epochs, 'beta_kl': beta_kl})
        run_id = wandb.run.id

    for epoch in range(start_epoch, epochs):
        net_G.train()
        total = 0; total_rec = 0; total_kl = 0

        for data in train_dl:
            L = data['L'].to(DEVICE)
            ab = data['ab'].to(DEVICE)

            fake_ab, mu, logvar = net_G(L)

            loss_rec = criterion(fake_ab, ab)
            loss_kl = kl_loss(mu, logvar)
            loss = loss_rec + beta_kl * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_rec += loss_rec.item()
            total_kl += loss_kl.item()
            total += loss.item()

        avg_rec = total_rec / len(train_dl)
        avg_kl = total_kl / len(train_dl)
        avg_total = total / len(train_dl)

        net_G.eval()
        v_total = 0; v_rec = 0; v_kl = 0
        with torch.no_grad():
            for val in val_dl:
                L_v = val['L'].to(DEVICE)
                ab_v = val['ab'].to(DEVICE)

                fake_ab_v, mu_v, logvar_v = net_G(L_v)

                r = criterion(fake_ab_v, ab_v)
                k = kl_loss(mu_v, logvar_v)
                t = r + beta_kl * k

                v_rec += r.item()
                v_kl += k.item()
                v_total += t.item()

        v_rec /= len(val_dl)
        v_kl /= len(val_dl)
        v_total /= len(val_dl)

        scheduler.step(v_total)

        is_best = v_total < best_val if save_best else False
        if is_best:
            best_val = v_total

        if ((epoch + 1) % save_every == 0) or is_best:
            save_checkpoint_local(epoch, net_G, optimizer, scheduler, run_id,
                                  save_dir, best_val=best_val, is_best=is_best)

        wandb.log({
            "epoch": epoch + 1,
            "train_rec": avg_rec,
            "train_kl": avg_kl,
            "train_total": avg_total,
            "val_rec": v_rec,
            "val_kl": v_kl,
            "val_total": v_total,
            "lr": optimizer.param_groups[0]['lr'],
        })

        print(epoch+1, avg_rec, avg_kl, avg_total, v_rec, v_kl, v_total)

    wandb.finish()

def train_from_scratch(cfg):
    global config
    config = cfg
    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"], cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"], cfg["VAL_SIZE"]
    )
    net_G = build_model().to(DEVICE)
    train_model(
        net_G, train_dl, val_dl,
        epochs=cfg["EPOCHS"], log_interval=1, lr=cfg["LR"],
        checkpoint_path=None,
        save_dir=cfg["CHECKPOINT_DIR"],
        save_every=cfg.get("SAVE_EVERY", 1),
        save_best=cfg.get("SAVE_BEST", True)
    )

def continue_training(cfg, gdrive_id_or_url):
    global config
    config = cfg
    local_ckpt = download_ckpt_from_gdrive(gdrive_id_or_url, cfg["CHECKPOINT_DIR"])
    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"], cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"], cfg["VAL_SIZE"]
    )
    net_G = build_model().to(DEVICE)
    train_model(
        net_G, train_dl, val_dl,
        epochs=cfg["EPOCHS"], log_interval=1, lr=cfg["LR"],
        checkpoint_path=local_ckpt,
        save_dir=cfg["CHECKPOINT_DIR"],
        save_every=cfg.get("SAVE_EVERY", 1),
        save_best=cfg.get("SAVE_BEST", True)
    )
