import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Config = {
    "TRAIN_DATASET_PATH": "/kaggle/input/coco-stuff-image-only/train2017/train2017",
    "VAL_DATASET_PATH": "/kaggle/input/coco-stuff-image-only/val2017/val2017",
    "BATCH_SIZE": 32,
    "EPOCHS": 200,
    "LR_G": 4e-4,
    "LR_D": 2e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": 1000,
    "VAL_SIZE": 200,
    "WANDB_PROJECT": "image-colorization-123k-img-version2",
    "WANDB_RUN_NAME": "Unet-GAN",
    "LOG_INTERVAL": 300,
}



CHECKPOINT_PATH_TEMPLATE = "checkpoint_epoch_{epoch}.pth"
