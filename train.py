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
from model import UNetGenerator, init_weights
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

def train_model(net_G, train_dl, val_dl, epochs, log_interval, lr,
                checkpoint_path=None, save_dir="/kaggle/working/checkpoints",
                save_every=1, save_best=True):
    optimizer = optim.Adam(net_G.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)
    criterion = nn.L1Loss()
    start_epoch = 0
    run_id = None
    best_val = float("inf")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        net_G.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        run_id = checkpoint.get('run_id', None)
        best_val = checkpoint.get('best_val', best_val)
    if run_id:
        wandb.init(project=config["WANDB_PROJECT"], name=config["WANDB_RUN_NAME"], id=run_id, resume="must")
    else:
        wandb.init(project=config["WANDB_PROJECT"], name=config["WANDB_RUN_NAME"], config={
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': getattr(train_dl, 'batch_size', None),
        })
        run_id = wandb.run.id
    for epoch in range(start_epoch, epochs):
        net_G.train()
        running_loss = 0.0
        for data in tqdm(train_dl):
            L = data['L'].to(DEVICE)
            ab = data['ab'].to(DEVICE)
            fake_ab = net_G(L)
            loss = criterion(fake_ab, ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_dl)
        net_G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in val_dl:
                L_val = val_data['L'].to(DEVICE)
                ab_val = val_data['ab'].to(DEVICE)
                fake_ab_val = net_G(L_val)
                loss = criterion(fake_ab_val, ab_val)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dl)
        scheduler.step(avg_val_loss)
        is_best = avg_val_loss < best_val if save_best else False
        if is_best:
            best_val = avg_val_loss
        if ((epoch + 1) % save_every == 0) or is_best:
            save_checkpoint_local(
                epoch, net_G, optimizer, scheduler, run_id, save_dir, best_val=best_val, is_best=is_best
            )
        with torch.no_grad():
            sample_data = next(iter(train_dl))
            L_sample = sample_data['L'].to(DEVICE)
            ab_sample = sample_data['ab'].to(DEVICE)
            fake_ab_sample = net_G(L_sample)
            n_train = min(5, L_sample.size(0))
            real_images_train = [wandb.Image(lab_to_rgb(L_sample[i].cpu().numpy(), ab_sample[i].cpu().numpy()),
                                             caption=f"GT Train {i}") for i in range(n_train)]
            fake_images_train = [wandb.Image(lab_to_rgb(L_sample[i].cpu().numpy(), fake_ab_sample[i].cpu().numpy()),
                                             caption=f"Predicted Train {i}") for i in range(n_train)]
            val_sample = next(iter(val_dl))
            L_val_s = val_sample['L'].to(DEVICE)
            ab_val_s = val_sample['ab'].to(DEVICE)
            fake_ab_val_s = net_G(L_val_s)
            n_val = min(5, L_val_s.size(0))
            real_images_val = [wandb.Image(lab_to_rgb(L_val_s[i].cpu().numpy(), ab_val_s[i].cpu().numpy()),
                                           caption=f"GT Val {i}") for i in range(n_val)]
            fake_images_val = [wandb.Image(lab_to_rgb(L_val_s[i].cpu().numpy(), fake_ab_val_s[i].cpu().numpy()),
                                           caption=f"Predicted Val {i}") for i in range(n_val)]
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'Ground Truth Train': real_images_train,
                'Predicted Train': fake_images_train,
                'Ground Truth Val': real_images_val,
                'Predicted Val': fake_images_val
            })
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    wandb.finish()

def train_from_scratch(cfg):
    global config
    config = cfg
    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"], cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"], cfg["VAL_SIZE"]
    )
    net_G = UNetGenerator().to(DEVICE)
    net_G.apply(init_weights)
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
    net_G = UNetGenerator().to(DEVICE)
    train_model(
        net_G, train_dl, val_dl,
        epochs=cfg["EPOCHS"], log_interval=1, lr=cfg["LR"],
        checkpoint_path=local_ckpt,
        save_dir=cfg["CHECKPOINT_DIR"],
        save_every=cfg.get("SAVE_EVERY", 1),
        save_best=cfg.get("SAVE_BEST", True)
    )
