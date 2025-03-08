import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_FROM_SCRATCH = {
    "TRAIN_DATASET_PATH": "/kaggle/input/coco-stuff-image-only/train2017/train2017",
    "VAL_DATASET_PATH": "/kaggle/input/coco-stuff-image-only/val2017/val2017",
    "BATCH_SIZE": 32,
    "EPOCHS": 70,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": 118000,
    "VAL_SIZE": 5000,
    "WANDB_PROJECT": "image-colorization-123k-img",
    "WANDB_RUN_NAME": "Unet",
}

CONTINUE_TRAINING = {
    "TRAIN_DATASET_PATH": "/kaggle/input/coco-stuff-image-only/train2017/train2017",
    "VAL_DATASET_PATH": "/kaggle/input/coco-stuff-image-only/val2017/val2017",
    "BATCH_SIZE": 32,
    "EPOCHS": 70,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": 118000,
    "VAL_SIZE": 5000,
    "WANDB_PROJECT": "image-colorization-123k-img",
    "WANDB_RUN_NAME": "Unet",
}

CHECKPOINT_PATH_TEMPLATE = "checkpoint_epoch_{epoch}.pth"
