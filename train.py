import gdown
import requests
import torch
import numpy as np
from tqdm.auto import tqdm
from data_loader import create_dataloaders
import wandb
import os
from skimage.color import lab2rgb
from config import Config as cfg
from model import GAN, load_trained_model, get_encoder_weights
from torch.utils.data import DataLoader, Subset
import random
import itertools

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)
    return lab2rgb(Lab)

def save_checkpoint_as_artifact(epoch, model, run_id, artifact_base_name="checkpoint"):
    checkpoint_file = f"{artifact_base_name}_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch + 1,
        'Unet_state_dict': model.net_G.state_dict(),
        'Disc_state_dict': model.net_D.state_dict(),
        'optimizer_Unet_state_dict': model.opt_G.state_dict(),
        'optimizer_Disc_state_dict': model.opt_D.state_dict(),
        'scheduler_state_dict': model.scheduler_G.state_dict(),
        'run_id': run_id,
    }, checkpoint_file)
    artifact_name = f"{artifact_base_name}_epoch_{epoch}"
    artifact = wandb.Artifact(name=artifact_name, type='model')
    artifact.add_file(checkpoint_file)
    wandb.log_artifact(artifact)
    os.remove(checkpoint_file)

def log_image_wandb(L, ab, num=5, captions=None):
    L = L.cpu().detach().numpy()
    ab = ab.cpu().detach().numpy()
    wandb_image = []
    for i in range(num):
        image = lab_to_rgb(L[i], ab[i])
        caption = captions[i] if captions is not None else f"Image {i}"
        wandb_image.append(wandb.Image(image, caption=caption))
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
    avg_L1 = total_L1 / len(val_dl)
    return avg_L1

def train_GAN(GAN_model, train_dl, val_dl, log_interval, checkpoint_path=None, warmup_epochs=0):
    epochs   = cfg["EPOCHS"]
    start_ep = 0
    run_id   = None

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        start_ep = ckpt['epoch']
        GAN_model.net_G.load_state_dict(ckpt['Unet_state_dict'])
        GAN_model.net_D.load_state_dict(ckpt['Disc_state_dict'])
        GAN_model.opt_G.load_state_dict(ckpt['optimizer_Unet_state_dict'])
        GAN_model.opt_D.load_state_dict(ckpt['optimizer_Disc_state_dict'])
        GAN_model.scheduler_G.load_state_dict(ckpt['scheduler_state_dict'])
        run_id = ckpt.get('run_id')

    if run_id:
        wandb.init(project=cfg["WANDB_PROJECT"],
                   name=cfg["WANDB_RUN_NAME"],
                   id=run_id,
                   resume="must")
    else:
        wandb.init(project=cfg["WANDB_PROJECT"],
                   name=cfg["WANDB_RUN_NAME"],
                   config=cfg)

    for epoch in range(start_ep, epochs):
        running_G        = running_D        = 0.0
        running_G_GAN    = running_G_L1     = running_G_P     = 0.0
        running_D_fake   = running_D_real   = 0.0
        step = 0

        # Warmup phase only for G-L1
        if epoch == 0:
            for w_epoch in range(warmup_epochs):
                for data in tqdm(train_dl, desc=f"Warmup Epoch {w_epoch+1}"):
                    GAN_model.setup_input(data)
                    GAN_model.warmup_optimize()
        
        # Main training loop
        for data in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            GAN_model.setup_input(data)
            GAN_model.optimize()

            running_G      += GAN_model.loss_G.item()
            running_D      += GAN_model.loss_D.item()
            running_G_GAN  += GAN_model.loss_G_GAN.item()
            running_G_L1   += GAN_model.loss_G_L1.item()
            running_G_P    += GAN_model.loss_G_perc.item()
            running_D_fake += GAN_model.loss_D_fake.item() if hasattr(GAN_model, 'loss_D_fake') else 0.0
            running_D_real += GAN_model.loss_D_real.item() if hasattr(GAN_model, 'loss_D_real') else 0.0

            step += 1
            if step % log_interval == 0:
                # log a fixed batch and a random validation batch
                with torch.no_grad():
                    bs = cfg["BATCH_SIZE"]
                    caps_train = [f"ep{epoch+1}_st{step}_img{i+1}" for i in range(bs)]
                    # fixed train batch
                    data_train = next(iter(train_dl))
                    GAN_model.setup_input(data_train)
                    GAN_model.forward()
                    fake_train = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps_train)
                    real_train = log_image_wandb(GAN_model.L, GAN_model.ab, captions=caps_train)
                    wandb.log({
                        "train/fake_images": fake_train,
                        "train/real_images": real_train
                    }, commit=False)
                    # random val batch
                    idx = random.randrange(len(val_dl))
                    data_val = list(val_dl)[idx]
                    GAN_model.setup_input(data_val)
                    GAN_model.forward()
                    fake_val = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps_train)
                    real_val = log_image_wandb(GAN_model.L, GAN_model.ab, captions=caps_train)
                    wandb.log({
                        "val/fake_images": fake_val,
                        "val/real_images": real_val
                    }, commit=False)

        # compute averages
        n = len(train_dl)
        avg_G      = running_G / n
        avg_D      = running_D / n
        avg_G_GAN  = running_G_GAN / n
        avg_G_L1   = running_G_L1 / n
        avg_G_P    = running_G_P / n
        avg_D_fake = running_D_fake / n
        avg_D_real = running_D_real / n

        # scheduler step
        GAN_model.scheduler_G.step(avg_G)

        # validation L1
        val_L1 = evaluate_L1_on_val(GAN_model, val_dl)

        # log metrics
        wandb.log({
            "epoch": epoch+1,
            "train/total_G_loss": avg_G,
            "train/G_GAN_loss":   avg_G_GAN,
            "train/G_L1_loss":    avg_G_L1,
            "train/G_Perc_loss":  avg_G_P,
            "train/D_loss":       avg_D,
            "train/D_fake_loss":  avg_D_fake,
            "train/D_real_loss":  avg_D_real,
            "val/L1_loss":        val_L1,
            "lr":                 GAN_model.opt_G.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{epochs} — "
              f"G_L1: {avg_G_L1:.4f}, G_P: {avg_G_P:.4f}, D: {avg_D:.4f}, val_L1: {val_L1:.4f}")

        # save checkpoint
        save_checkpoint_as_artifact(epoch, GAN_model, wandb.run.id, artifact_base_name="checkpoint")

def download_model(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists, skipping download.")

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

def train_from_scratch():
    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"],
        cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"],
        cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"],
        cfg["VAL_SIZE"]
    )
    net_GAN = GAN(lr_G=cfg["LR_G"], lr_D=cfg["LR_D"])
    train_GAN(net_GAN, train_dl, val_dl, log_interval=cfg["LOG_INTERVAL"])

def train_from_checkpoint(path):
    if not path.startswith("http"):
        raise ValueError(f"Invalid URL: {path}")
    response = requests.get(path)
    if response.status_code == 200:
        ckpt_file = os.path.basename(path)
        with open(ckpt_file, "wb") as f:
            f.write(response.content)
    else:
        raise ValueError(f"Failed to download checkpoint from {path}")
    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"],
        cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"],
        cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"],
        cfg["VAL_SIZE"]
    )
    net_GAN = GAN(lr_G=cfg["LR_G"], lr_D=cfg["LR_D"])
    train_GAN(net_GAN, train_dl, val_dl, log_interval=cfg["LOG_INTERVAL"], checkpoint_path=ckpt_file)
