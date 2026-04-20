
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Config = {
    "TRAIN_DATASET_PATH": "/kaggle/input/datasets/duynguynp9pou/coco-stuff-image-only/train2017/train2017",
    "VAL_DATASET_PATH": "/kaggle/input/datasets/duynguynp9pou/coco-stuff-image-only/val2017/val2017",
    "TRAIN_SEG_PATH": "/kaggle/input/datasets/wxli408/cocostuff/stuffthingmaps_trainval2017/train2017",
    "VAL_SEG_PATH": "/kaggle/input/datasets/wxli408/cocostuff/stuffthingmaps_trainval2017/val2017",
    "IMAGE_SIZE": 256,
    "USE_SEGMENTATION": True,
    "NUM_SEG_CLASSES": 182,
    "SEG_IGNORE_INDEX": 255,
    "LAMBDA_SEG": 1.0,
    "BATCH_SIZE": 16,
    "EPOCHS": 200,
    "LR_G": 4e-4,
    "LR_D": 2e-4,
    "NUM_WORKERS": 4,
    "TRAIN_SIZE": None,
    "VAL_SIZE": None,
    "WANDB_PROJECT": "image-colorization-123k-img-version2",
    "WANDB_RUN_NAME": "Unet-GAN-SwinDeep",
    "LOG_INTERVAL": 300,
    "LAMBDA_OBJ": 0.01,
    "OBJECT_MIN_PIXELS": 16,
    "OBJECT_USE_CONNECTED_COMPONENTS": True,
    "USE_SWIN_DEEP": True,
    "SWIN_WINDOW_SIZE": 8,
    "SWIN_ENC3_DEPTH": 1,
    "SWIN_ENC4_DEPTH": 1,
    "SWIN_NUM_HEADS_ENC3": 4,
    "SWIN_NUM_HEADS_ENC4": 8,
    "SWIN_ATTN_DIM_ENC3": 256,
    "SWIN_ATTN_DIM_ENC4": 256,
    "SWIN_MLP_RATIO": 4.0,
    "SWIN_DROPOUT": 0.0,
    "SWIN_ATTN_DROPOUT": 0.0,
}

CHECKPOINT_PATH_TEMPLATE = "checkpoint_epoch_{epoch}.pth"
