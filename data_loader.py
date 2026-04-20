
import glob
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
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

        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


class SegmentationPairTransform:
    def __init__(self, image_size=256, train=True):
        self.image_size = image_size
        self.train = train
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def __call__(self, img, seg):
        img = TF.resize(img, (self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC)
        seg = TF.resize(seg, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST)

        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                seg = TF.hflip(seg)

            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BICUBIC, fill=0)
            seg = TF.rotate(seg, angle, interpolation=InterpolationMode.NEAREST, fill=255)

            translate_x = int(round(random.uniform(-0.1, 0.1) * self.image_size))
            translate_y = int(round(random.uniform(-0.1, 0.1) * self.image_size))
            scale = random.uniform(0.9, 1.0)
            shear = random.uniform(-10, 10)
            img = TF.affine(
                img,
                angle=0.0,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=[shear, 0.0],
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
            )
            seg = TF.affine(
                seg,
                angle=0.0,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=[shear, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=255,
            )
            img = self.color_jitter(img)

        img_tensor = TF.to_tensor(img)
        seg_tensor = torch.from_numpy(np.array(seg, dtype=np.int64))
        return img_tensor, seg_tensor


class ColorizationSegmentationDataset(Dataset):
    def __init__(self, image_paths, seg_root, transform=None):
        self.transform = transform
        self.image_paths = list(image_paths)
        self.seg_root = seg_root

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        seg_path = os.path.join(self.seg_root, base_name + ".png")

        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation map not found for {img_path}: {seg_path}")

        img = Image.open(img_path).convert("RGB")
        seg = Image.open(seg_path)

        if self.transform:
            img_tensor, seg_tensor = self.transform(img, seg)
        else:
            img_tensor = TF.to_tensor(img)
            seg_tensor = torch.from_numpy(np.array(seg, dtype=np.int64))

        img_lab = rgb2lab(img_tensor.permute(1, 2, 0).numpy()).astype("float32")
        img_lab = torch.tensor(img_lab).permute(2, 0, 1)

        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.

        return {'L': L, 'ab': ab, 'seg': seg_tensor}

    def __len__(self):
        return len(self.image_paths)


def create_dataloaders(
    train_dataset_path,
    val_dataset_path,
    batch_size,
    num_workers,
    train_size=None,
    val_size=None,
    train_seg_path=None,
    val_seg_path=None,
    image_size=256,
):
    train_paths = glob.glob(train_dataset_path + "/*.jpg")
    val_paths = glob.glob(val_dataset_path + "/*.jpg")

    if train_size is None:
        train_size = len(train_paths)
    if val_size is None:
        val_size = len(val_paths)

    if train_size > len(train_paths):
        raise ValueError(f"train_size ({train_size}) cannot be greater than the number of available training images ({len(train_paths)})")
    if val_size > len(val_paths):
        raise ValueError(f"val_size ({val_size}) cannot be greater than the number of available validation images ({len(val_paths)})")

    np.random.seed(123)
    train_paths = np.random.choice(train_paths, train_size, replace=False)
    val_paths = np.random.choice(val_paths, val_size, replace=False)

    if train_seg_path is not None and val_seg_path is not None:
        train_transform = SegmentationPairTransform(image_size=image_size, train=True)
        val_transform = SegmentationPairTransform(image_size=image_size, train=False)

        train_dataset = ColorizationSegmentationDataset(train_paths, train_seg_path, transform=train_transform)
        val_dataset = ColorizationSegmentationDataset(val_paths, val_seg_path, transform=val_transform)
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
            transforms.ToTensor()
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.BICUBIC),
            transforms.ToTensor()
        ])

        train_dataset = ColorizationDataset(train_paths, transform=train_transforms)
        val_dataset = ColorizationDataset(val_paths, transform=val_transforms)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_dl, val_dl
