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
from model import GAN, load_trained_model, pretrain_discriminator, get_encoder_weights
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

def train_GAN(GAN_model, train_dl, val_dl, log_interval, checkpoint_path=None, warmup_epochs=2):
    epochs = cfg["EPOCHS"]
    start_epoch = 0
    run_id = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        GAN_model.net_G.load_state_dict(checkpoint['Unet_state_dict'])
        GAN_model.net_D.load_state_dict(checkpoint['Disc_state_dict'])
        GAN_model.opt_G.load_state_dict(checkpoint['optimizer_Unet_state_dict'])
        GAN_model.opt_D.load_state_dict(checkpoint['optimizer_Disc_state_dict'])
        GAN_model.scheduler_G.load_state_dict(checkpoint['scheduler_state_dict'])
        for state in GAN_model.opt_G.state.values():
            if 'momentum_buffer' in state:
                state['momentum_buffer'].zero_()
        for state in GAN_model.opt_D.state.values():
            if 'momentum_buffer' in state:
                state['momentum_buffer'].zero_()
        run_id = checkpoint['run_id']
    if run_id:
        wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"], id=run_id, resume="must")
    else:
        wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"], config=cfg)
    for epoch in range(start_epoch, epochs):
        running_loss_G = running_loss_D = 0.0
        running_loss_G_GAN = running_loss_G_L1 = 0.0
        running_loss_D_fake = running_loss_D_real = 0.0
        step = 0
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
                            caps_train = [f"warmup{warmup_epoch+1}_step{step_warmup}_img{i+1}" for i in range(bs)]
                            caps_val = caps_train.copy()
                            data_fix = next(iter(val_dl))
                            GAN_model.setup_input(data_fix)
                            GAN_model.forward()
                            fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps_train)
                            real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab, captions=caps_train)
                            rand_idx = random.randrange(len(val_dl))
                            batch_rand = list(val_dl)[rand_idx]
                            GAN_model.setup_input(batch_rand)
                            GAN_model.forward()
                            val_fake = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5, captions=caps_val)
                            val_real = log_image_wandb(GAN_model.L, GAN_model.ab, num=5, captions=caps_val)
                        wandb.log({
                            "fix_fake_images": fake_imgs,
                            "fix_real_images": real_imgs,
                            "random_fake_images": val_fake,
                            "random_real_images": val_real,
                        })
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
                    caps_train = [f"{epoch+1}_step{step}_img{i+1}" for i in range(bs)]
                    caps_val = caps_train.copy()
                    data_fix = next(iter(val_dl))
                    GAN_model.setup_input(data_fix)
                    GAN_model.forward()
                    fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps_train)
                    real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab, captions=caps_train)
                    rand_idx = random.randrange(len(val_dl))
                    batch_rand = list(val_dl)[rand_idx]
                    GAN_model.setup_input(batch_rand)
                    GAN_model.forward()
                    val_fake = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5, captions=caps_val)
                    val_real = log_image_wandb(GAN_model.L, GAN_model.ab, num=5, captions=caps_val)
                wandb.log({
                    "fix_fake_images": fake_imgs,
                    "fix_real_images": real_imgs,
                    "random_fake_images": val_fake,
                    "random_real_images": val_real,
                }, commit=False)
        num_batches = len(train_dl)
        average_loss_G = running_loss_G / num_batches
        average_loss_D = running_loss_D / num_batches
        average_loss_G_GAN = running_loss_G_GAN / num_batches
        average_loss_G_L1 = running_loss_G_L1 / num_batches
        average_loss_D_fake = running_loss_D_fake / num_batches
        average_loss_D_real = running_loss_D_real / num_batches
        GAN_model.scheduler_G.step(average_loss_G)
        val_L1 = evaluate_L1_on_val(GAN_model, val_dl)
        with torch.no_grad():
            data_fix = next(iter(val_dl))
            GAN_model.setup_input(data_fix)
            GAN_model.forward()
            fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color)
            real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab)
            rand_idx = random.randrange(len(val_dl))
            batch_rand = list(val_dl)[rand_idx]
            GAN_model.setup_input(batch_rand)
            GAN_model.forward()
            val_fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5)
            val_real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab, num=5)
        wandb.log({
            "epoch_train_loss_G": average_loss_G,
            "epoch_train_loss_D": average_loss_D,
            "epoch_train_loss_G_GAN": average_loss_G_GAN,
            "epoch_train_loss_G_L1": average_loss_G_L1,
            "epoch_val_loss_G_L1": val_L1,
            "epoch_train_loss_D_fake": average_loss_D_fake,
            "epoch_train_loss_D_real": average_loss_D_real,
            "lr": GAN_model.opt_G.param_groups[0]['lr'],
            "end_fake_images": fake_imgs,
            "end_real_images": real_imgs,
            "end_val_fake_images": val_fake_imgs,
            "end_val_real_images": val_real_imgs,
        })
        print(f"Epoch {epoch+1}/{epochs} — train L1: {average_loss_G_L1:.4f}, val L1: {val_L1:.4f}")
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
    Unet_Generator = load_trained_model(model_path)
    return Unet_Generator

def pretrain_encoder_weights():
    model_url = 'https://drive.google.com/uc?id=1dD7PQt1RB-IqNVJFHlnsG9MdkmdDuRxH'
    model_path = 'model.pth'
    download_model(model_url, model_path)
    return get_encoder_weights(model_path)

def train_from_scratch():
    train_dl, val_dl = create_dataloaders(cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"], cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"])
    net_GAN = GAN(lr_G=cfg["LR_G"], lr_D=cfg["LR_D"])
    train_GAN(net_GAN, train_dl, val_dl, log_interval=cfg["LOG_INTERVAL"])

def train_from_checkpoint(path):
    if not path.startswith("http"):
        raise ValueError(f"Invalid URL: {path}")
    checkpoint_url = path
    checkpoint_file = os.path.join(".", os.path.basename(checkpoint_url))
    response = requests.get(checkpoint_url)
    if response.status_code == 200:
        with open(checkpoint_file, "wb") as f:
            f.write(response.content)
    else:
        raise ValueError(f"Failed to download checkpoint from {checkpoint_url}")
    train_dl, val_dl = create_dataloaders(cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"], cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"])
    net_GAN = GAN(lr_G=cfg["LR_G"], lr_D=cfg["LR_D"])
    train_GAN(net_GAN, train_dl, val_dl, log_interval=cfg["LOG_INTERVAL"], checkpoint_path=checkpoint_file)
