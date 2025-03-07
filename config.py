# config.py
import torch

# Thiết lập thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Cấu hình cho huấn luyện từ đầu
TRAIN_FROM_SCRATCH = {
    "DATASET_PATH": "/kaggle/input/cocostuff-10k-withoutgray/coco10k/images/train256",
    "BATCH_SIZE": 8,
    "EPOCHS": 20,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": 8000,
    "VAL_SIZE": 1000,
    "WANDB_PROJECT": "image-colorization",
    "WANDB_RUN_NAME": "Unet",
}

# Cấu hình cho tiếp tục huấn luyện (sử dụng checkpoint từ wandb/drive)
CONTINUE_TRAINING = {
    "DATASET_PATH": "/kaggle/input/cocostuff-10k-withoutgray/coco10k/images/train256",
    "BATCH_SIZE": 32,
    "EPOCHS": 36,
    "LR": 4e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": 8000,
    "VAL_SIZE": 1000,
    "CHECKPOINT_URL": "https://drive.google.com/uc?export=download&id=1dD7PQt1RB-IqNVJFHlnsG9MdkmdDuRxH",
    "WANDB_PROJECT": "image-colorization-123k",
    "WANDB_RUN_NAME": "Unet",
}

# Mẫu tên file checkpoint
CHECKPOINT_PATH_TEMPLATE = "checkpoint_epoch_{epoch}.pth"
