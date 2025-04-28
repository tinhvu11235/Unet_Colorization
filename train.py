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

def log_image_wandb(L, ab,num = 5,captions = None):
    L = L.cpu().detach().numpy()
    ab = ab.cpu().detach().numpy()
    wandb_image = []
    for i in range(num):
        image = lab_to_rgb(L[i], ab[i])
        if captions is not None:
            wandb_image.append(wandb.Image(image, caption=captions[i]))
        else:
            wandb_image.append(wandb.Image(image, caption=f"Image {i}"))
    return wandb_image


def train_GAN(GAN_model, train_dl, val_dl, log_interval, checkpoint_path = None, warmup_epochs = 0):
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
        run_id = checkpoint['run_id']

    if run_id :
        wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"], id = run_id, resume = "must")
    if run_id is None:
        wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"], config=cfg)
    for epoch in range(start_epoch,epochs):   
        #start training
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_G_GAN = 0.0
        running_loss_G_L1 = 0.0
        running_loss_D_fake = 0.0
        running_loss_D_real = 0.0
        step = 0
        if epoch == 0 :
           for warmup_epoch in range(warmup_epochs):
                step_warmup = 0
                for data in tqdm(train_dl, desc=f"Warmup Epoch {warmup_epoch+1}"):
                    GAN_model.setup_input(data)
                    GAN_model.warmup_optimize()
                    step_warmup += 1
                    if step_warmup % log_interval == 0:
                        with torch.no_grad():
                            bs_train = cfg["BATCH_SIZE"]
                            caps_train = [f"warmup{warmup_epoch+1}_step{step_warmup}_img{i+1}" for i in range(bs_train)]
                            bs_val =  cfg["BATCH_SIZE"]
                            caps_val   = [f"warmup{warmup_epoch+1}_step{step_warmup}_img{i+1}" for i in range(bs_val)]
                            data_fix = next(iter(val_dl))
                            GAN_model.setup_input(data_fix)
                            GAN_model.forward()
                            fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps_train)
                            real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab,         captions=caps_train)
                            num_batches = len(val_dl)
                            rand_idx    = random.randrange(1, num_batches)
                            it = iter(val_dl)
                            batch_random = next(itertools.islice(it, rand_idx, rand_idx+1))
                            GAN_model.setup_input(batch_random)
                            GAN_model.forward()
                            val_fake  = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5, captions=caps_val)
                            val_real  = log_image_wandb(GAN_model.L, GAN_model.ab,         num=5, captions=caps_val)
                        wandb.log({
                            "fix_fake_images": fake_imgs,
                            "fix_real_images": real_imgs,
                            "random_fake_images":   val_fake,
                            "random_real_images":   val_real,
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
                    bs_train =  cfg["BATCH_SIZE"]
                    caps_train = [f"{epoch+1}_step{step}_img{i+1}" for i in range(bs_train)]
                    bs_val =  cfg["BATCH_SIZE"]
                    caps_val   = [f"{epoch+1}_step{step}_img{i+1}" for i in range(bs_val)]
                    data = next(iter(val_dl))
                    GAN_model.setup_input(data)
                    GAN_model.forward()
                    fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, captions=caps_train)
                    real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab,         captions=caps_train)

                    num_batches = len(val_dl)                         
                    rand_idx    = random.randrange(1,num_batches)      
                    it = iter(val_dl)
                    batch_random = next(itertools.islice(it, rand_idx, rand_idx+1))
                    GAN_model.setup_input(batch_random)
                    GAN_model.forward()
                    val_fake  = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5, captions=caps_val)
                    val_real  = log_image_wandb(GAN_model.L, GAN_model.ab,         num=5, captions=caps_val)
                wandb.log({
                            "fix_fake_images": fake_imgs,
                            "fix_real_images": real_imgs,
                            "random_fake_images":   val_fake,
                            "random_real_images":   val_real,
                        })   
        num_batches = len(train_dl)
        average_loss_G = running_loss_G / num_batches
        average_loss_D = running_loss_D / num_batches
        average_loss_G_GAN = running_loss_G_GAN / num_batches
        average_loss_G_L1 = running_loss_G_L1 / num_batches
        average_loss_D_fake = running_loss_D_fake / num_batches
        average_loss_D_real = running_loss_D_real / num_batches
        GAN_model.scheduler_G.step(average_loss_G)

        with torch.no_grad():
            data = next(iter(val_dl))
            GAN_model.setup_input(data)
            GAN_model.forward()
            fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color)
            real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab)

            num_batches = len(val_dl)                         
            rand_idx    = random.randrange(1,num_batches)      
            it = iter(val_dl)
            batch_random = next(itertools.islice(it, rand_idx, rand_idx+1))
            GAN_model.setup_input(batch_random)
            GAN_model.forward()
            val_fake_imgs = log_image_wandb(GAN_model.L, GAN_model.fake_color, num=5)
            val_real_imgs = log_image_wandb(GAN_model.L, GAN_model.ab, num=5)
        wandb.log({
            "train_loss": average_loss_G,
            'average_loss_D': average_loss_D,
            'average_loss_G_GAN': average_loss_G_GAN,
            'average_loss_G_L1': average_loss_G_L1,
            'average_loss_D_fake': average_loss_D_fake,
            'average_loss_D_real': average_loss_D_real,
            "end_fake_images": fake_imgs,
            "end_real_images": real_imgs,
            "end_val_fake_images": val_fake_imgs,
            "end_val_real_images": val_real_imgs,
            'lr': GAN_model.opt_G.param_groups[0]['lr']
        },step = epoch )
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss_G}")
        
        save_checkpoint_as_artifact(epoch, GAN_model,  wandb.run.id, artifact_base_name="checkpoint") 
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
    train_dl, val_dl = create_dataloaders(cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"],cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"])
    net_GAN = GAN(lr_G=cfg["LR_G"], lr_D=cfg["LR_D"])
    # net_GAN.net_G.load_state_dict(download_pretrain_generator().state_dict())
    # # net_GAN.net_G.load_state_dict(pretrain_encoder_weights(),strict=False)
    # pretrain_discriminator(train_dl,net_GAN)
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
    train_dl, val_dl = create_dataloaders(cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"],cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"])
    net_GAN = GAN(lr_G=cfg["LR_G"], lr_D=cfg["LR_D"])
    train_GAN(net_GAN, train_dl, val_dl, log_interval=cfg["LOG_INTERVAL"], checkpoint_path=checkpoint_file)
