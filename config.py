# config.py
import torch

# Thiết lập thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Cấu hình cho huấn luyện từ đầu
TRAIN_FROM_SCRATCH = {
    "TRAIN_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/train",
    "VAL_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/valid",
    "BATCH_SIZE": 64,
    "EPOCHS": 70,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": None,
    "VAL_SIZE": None,
    "WANDB_PROJECT": "image-colorization",
    "WANDB_RUN_NAME": "Unet",
}

# Cấu hình cho tiếp tục huấn luyện (sử dụng checkpoint từ wandb/drive)
CONTINUE_TRAINING = {
    "TRAIN_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/train",
    "VAL_DATASET_PATH": "/kaggle/input/coco-stuff-256x256-image-only-split/CoCo_Stuff_256x256/valid",
    "BATCH_SIZE": 64,
    "EPOCHS": 70,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": None,
    "VAL_SIZE": None,
    "WANDB_PROJECT": "image-colorization",
    "WANDB_RUN_NAME": "Unet",
    "MODEL_ARTIFACT": "matbinhtinh20-f/vae-training-VAE-phase/vae-model-epoch-69:v0", 
}


# Mẫu tên file checkpoint
CHECKPOINT_PATH_TEMPLATE = "checkpoint_epoch_{epoch}.pth"
