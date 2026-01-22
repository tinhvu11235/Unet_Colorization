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
        img = np.array(img)

        if self.transform:
            img = self.transform(Image.fromarray(img))

        img_lab = rgb2lab(img.permute(1, 2, 0).numpy()).astype("float32")
        img_lab = torch.tensor(img_lab).permute(2, 0, 1)

        L = img_lab[[0], ...] / 50.0 - 1.0
        ab = img_lab[[1, 2], ...] / 110.0

        return {"L": L, "ab": ab}

    def __len__(self):
        return len(self.paths)


def create_dataloaders(
    train_dataset_path,
    val_dataset_path,
    batch_size,
    num_workers,
    train_size=None,
    val_size=None,
):
    train_paths = glob.glob(train_dataset_path + "/*.jpg")
    val_paths = glob.glob(val_dataset_path + "/*.jpg")

    if train_size is None:
        train_size = len(train_paths)
    if val_size is None:
        val_size = len(val_paths)

    if train_size > len(train_paths):
        raise ValueError(
            f"train_size ({train_size}) cannot be greater than the number of available training images ({len(train_paths)})"
        )
    if val_size > len(val_paths):
        raise ValueError(
            f"val_size ({val_size}) cannot be greater than the number of available validation images ({len(val_paths)})"
        )

    np.random.seed(123)
    train_paths = np.random.choice(train_paths, train_size, replace=False)
    val_paths = np.random.choice(val_paths, val_size, replace=False)

    # Aug nhẹ: giữ hình học đơn giản để L vẫn khớp cấu trúc với ab
    train_transforms = transforms.Compose(
        [
            transforms.Resize((286, 286), Image.BICUBIC),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )

    train_dl = DataLoader(
        ColorizationDataset(train_paths, transform=train_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        ColorizationDataset(val_paths, transform=val_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dl, val_dl
