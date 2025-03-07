import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab

class ColorizationDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.transform = transform
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Chuyển đổi sang không gian màu Lab
        img_lab = rgb2lab(np.array(img)).astype("float32")
        img_lab = torch.tensor(img_lab).permute(2, 0, 1)
        # Chuẩn hóa các kênh
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)

def create_dataloaders(train_dataset_path, val_dataset_path, batch_size, num_workers, train_size=None, val_size=None):
    # Lấy danh sách ảnh trong thư mục train và validation
    train_paths = glob.glob(train_dataset_path + "/*.jpg")
    val_paths = glob.glob(val_dataset_path + "/*.jpg")
    
    # Nếu train_size hoặc val_size là None, lấy toàn bộ ảnh
    if train_size is None:
        train_size = len(train_paths)
    if val_size is None:
        val_size = len(val_paths)
    
    # Kiểm tra nếu số lượng train_size và val_size hợp lệ
    if train_size > len(train_paths):
        raise ValueError(f"train_size ({train_size}) cannot be greater than the number of available training images ({len(train_paths)})")
    if val_size > len(val_paths):
        raise ValueError(f"val_size ({val_size}) cannot be greater than the number of available validation images ({len(val_paths)})")

    # Tách danh sách ảnh thành tập train và validation (nếu cần)
    np.random.seed(123)
    train_paths = np.random.choice(train_paths, train_size, replace=False)
    val_paths = np.random.choice(val_paths, val_size, replace=False)
    
    # Các phép biến đổi cho tập train và validation
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor()
    ])

    train_dl = DataLoader(ColorizationDataset(train_paths, transform=train_transforms),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(ColorizationDataset(val_paths, transform=val_transforms),
                          batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl
