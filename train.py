# train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import numpy as np
from skimage.color import lab2rgb
import gdown

from config import CHECKPOINT_PATH_TEMPLATE, DEVICE
from model import UNetGenerator, init_weights
from data_loader import create_dataloaders

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)
    return lab2rgb(Lab)

def train_model(net_G, train_dl, val_dl, epochs, log_interval, lr, checkpoint_path=None, checkpoint_template=CHECKPOINT_PATH_TEMPLATE):
    optimizer = optim.Adam(net_G.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)
    criterion = nn.L1Loss()

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        net_G.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch + 1}")
    
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
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss}")

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
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

        scheduler.step(avg_val_loss)

        # Lưu checkpoint
        checkpoint_file = checkpoint_template.format(epoch=epoch+1)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net_G.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)

        # Log các số liệu lên wandb
        wandb.log({'epoch': epoch+1, 'train_loss': avg_loss, 'val_loss': avg_val_loss, 'lr': optimizer.param_groups[0]['lr']})
        
        # Log ảnh mẫu lên wandb mỗi log_interval epoch
        if (epoch + 1) % log_interval == 0:
            with torch.no_grad():
                sample_data = next(iter(train_dl))
                L_sample = sample_data['L'].to(DEVICE)
                ab_sample = sample_data['ab'].to(DEVICE)
                fake_ab_sample = net_G(L_sample)
                
                # Lấy 5 ảnh mẫu từ tập train
                real_images = [wandb.Image(lab_to_rgb(L_sample[i].cpu().numpy(), ab_sample[i].cpu().numpy()), caption=f"Real {i}") for i in range(5)]
                fake_images = [wandb.Image(lab_to_rgb(L_sample[i].cpu().numpy(), fake_ab_sample[i].cpu().numpy()), caption=f"Fake {i}") for i in range(5)]
                
                wandb.log({'Generated Images': fake_images, 'Ground Truth Images': real_images})
    
    wandb.finish()

def train_from_scratch(cfg):
    train_dl, val_dl = create_dataloaders(cfg["DATASET_PATH"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"], cfg["BATCH_SIZE"], cfg["NUM_WORKERS"])
    net_G = UNetGenerator().to(DEVICE)
    net_G.apply(init_weights)
    
    wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"],
               config={'learning_rate': cfg["LR"], 'epochs': cfg["EPOCHS"], 'batch_size': cfg["BATCH_SIZE"]})
    train_model(net_G, train_dl, val_dl, epochs=cfg["EPOCHS"], log_interval=1, lr=cfg["LR"])

def download_checkpoint(checkpoint_url):
    checkpoint_path = "checkpoint.pth"
    gdown.download(checkpoint_url, checkpoint_path, quiet=False)
    print(f"Checkpoint downloaded to {checkpoint_path}")
    return checkpoint_path

def continue_training(cfg):
    # Tải checkpoint từ URL
    checkpoint_path = download_checkpoint(cfg["CHECKPOINT_URL"])
    train_dl, val_dl = create_dataloaders(cfg["DATASET_PATH"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"], cfg["BATCH_SIZE"], cfg["NUM_WORKERS"])
    net_G = UNetGenerator().to(DEVICE)
    wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"],
               config={'learning_rate': cfg["LR"], 'epochs': cfg["EPOCHS"], 'batch_size': cfg["BATCH_SIZE"]})
    train_model(net_G, train_dl, val_dl, epochs=cfg["EPOCHS"], log_interval=1, lr=cfg["LR"], checkpoint_path=checkpoint_path)
