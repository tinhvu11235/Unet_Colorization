import torch

# Thiết lập thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



# Cấu hình cho huấn luyện từ đầu
TRAIN_FROM_SCRATCH = {
    "TRAIN_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/train",
    "VAL_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/valid",
    "BATCH_SIZE": 32,
    "EPOCHS": 70,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": 100,
    "VAL_SIZE": 10,
    "WANDB_PROJECT": "image-colorization-123k-img",
    "WANDB_RUN_NAME": "Unet",
}

# Cấu hình cho tiếp tục huấn luyện (sử dụng checkpoint từ wandb/drive)
CONTINUE_TRAINING = {
    "TRAIN_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/train",
    "VAL_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/valid",
    "BATCH_SIZE": 32,
    "EPOCHS": 70,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": 118000,
    "VAL_SIZE": 5000,
    "WANDB_PROJECT": "image-colorization-123k-img",
    "WANDB_RUN_NAME": "Unet",
}


# Mẫu tên file checkpoint
CHECKPOINT_PATH_TEMPLATE = "checkpoint_epoch_{epoch}.pth"
