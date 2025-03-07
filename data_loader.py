# data_loader.py
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
        img_lab = rgb2lab(img.permute(1, 2, 0).numpy()).astype("float32")
        img_lab = torch.tensor(img_lab).permute(2, 0, 1)
        # Chuẩn hóa các kênh
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)

def create_dataloaders(dataset_path, train_size, val_size, batch_size, num_workers):
    paths = glob.glob(dataset_path + "/*.jpg")
    np.random.seed(123)
    paths_subset = np.random.choice(paths, train_size + val_size, replace=False)
    rand_idxs = np.random.permutation(train_size + val_size)
    train_paths = paths_subset[rand_idxs[:train_size]]
    val_paths = paths_subset[rand_idxs[train_size:]]
    
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

    train_dl = DataLoader(ColorizationDataset(train_paths, transform=train_transforms), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(ColorizationDataset(val_paths, transform=val_transforms), batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl
