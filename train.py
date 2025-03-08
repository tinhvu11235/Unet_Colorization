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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = np.concatenate([L, ab], axis=0).transpose(1, 2, 0)
    return lab2rgb(Lab)

def save_checkpoint_as_artifact(epoch, model, optimizer, scheduler, run_id, artifact_base_name="checkpoint"):
    checkpoint_file = f"{artifact_base_name}_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'run_id': run_id,
    }, checkpoint_file)

    artifact_name = f"{artifact_base_name}_epoch_{epoch}"
    artifact = wandb.Artifact(name=artifact_name, type='model')
    artifact.add_file(checkpoint_file)
    wandb.log_artifact(artifact)
    os.remove(checkpoint_file)

def train_model(net_G, train_dl, val_dl, epochs, log_interval, lr, checkpoint_path=None):
    optimizer = optim.Adam(net_G.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)
    criterion = nn.L1Loss()

    start_epoch = 0
    run_id = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        net_G.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        run_id = checkpoint['run_id']

    if run_id:
        wandb.init(project="image-colorization", name="Unet", id=run_id, resume="must")
    else:
        wandb.init(project="image-colorization", name="Unet", resume=False)

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
        save_checkpoint_as_artifact(epoch, net_G, optimizer, scheduler, wandb.run.id)
        with torch.no_grad():
            sample_data = next(iter(train_dl))
            L_sample = sample_data['L'].to(DEVICE)
            ab_sample = sample_data['ab'].to(DEVICE)
            fake_ab_sample = net_G(L_sample)
            real_images_train = [wandb.Image(lab_to_rgb(L_sample[i].cpu().numpy(), ab_sample[i].cpu().numpy()),
                                                    caption=f"GT Train {i}") for i in range(5)]
            fake_images_train = [wandb.Image(lab_to_rgb(L_sample[i].cpu().numpy(), fake_ab_sample[i].cpu().numpy()),
                                                    caption=f"Predicted Train {i}") for i in range(5)]
                    
            val_sample = next(iter(val_dl))
            L_val = val_sample['L'].to(DEVICE)
            ab_val = val_sample['ab'].to(DEVICE)
            fake_ab_val = net_G(L_val)       
            real_images_val = [wandb.Image(lab_to_rgb(L_val[i].cpu().numpy(), ab_val[i].cpu().numpy()),
                                                caption=f"GT Val {i}") for i in range(5)]
            fake_images_val = [wandb.Image(lab_to_rgb(L_val[i].cpu().numpy(), fake_ab_val[i].cpu().numpy()),
                                                caption=f"Predicted Val {i}") for i in range(5)]
                    
            wandb.log({
                'epoch': epoch+1,
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
    train_dl, val_dl = create_dataloaders(cfg["TRAIN_DATASET_PATH"], cfg["VAL_DATASET_PATH"],cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], cfg["TRAIN_SIZE"], cfg["VAL_SIZE"])
    net_G = UNetGenerator().to(DEVICE)
    net_G.apply(init_weights)

    wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"], config={
        'learning_rate': cfg["LR"],
        'epochs': cfg["EPOCHS"],
        'batch_size': cfg["BATCH_SIZE"],
    })
    train_model(net_G, train_dl, val_dl, epochs=cfg["EPOCHS"], log_interval=1, lr=cfg["LR"])

def continue_training(cfg, path):
    run = wandb.init(project=cfg["WANDB_PROJECT"], name=cfg["WANDB_RUN_NAME"], resume=True)
    
    artifact = run.use_artifact(path, type="model")
    artifact_dir = artifact.download(root=".")
    
    checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        raise ValueError("No checkpoint file found in artifact directory.")

    checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
    net_G = UNetGenerator().to(DEVICE)
    
    train_dl, val_dl = create_dataloaders(
        cfg["TRAIN_DATASET_PATH"],
        cfg["VAL_DATASET_PATH"],
        cfg["BATCH_SIZE"],
        cfg["NUM_WORKERS"],
        cfg["TRAIN_SIZE"],
        cfg["VAL_SIZE"]
    )
    
    train_model(net_G, train_dl, val_dl, epochs=cfg["EPOCHS"], log_interval=1, lr=cfg["LR"], checkpoint_path=checkpoint_path)
