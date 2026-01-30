import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_MD5 = None  # md5 not always stable across mirrors


def _ensure_tiny_imagenet(root: str, download: bool):
    root = os.path.expanduser(root)
    base = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(base):
        return base
    if not download:
        raise FileNotFoundError(
            f"Tiny-ImageNet not found at {base}. Set data.download=true to auto-download."
        )
    os.makedirs(root, exist_ok=True)
    download_and_extract_archive(TINY_IMAGENET_URL, download_root=root, filename="tiny-imagenet-200.zip", md5=TINY_IMAGENET_MD5)
    if not os.path.isdir(base):
        raise RuntimeError(f"Download finished but folder {base} not found. Please check the archive structure.")
    return base


def get_tiny_imagenet_loaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Tiny-ImageNet (200 classes, 64x64). Uses CIFAR-style augmentation and ImageNet normalization."""
    root = cfg.get("root", "./data")
    download = bool(cfg.get("download", True))
    base = _ensure_tiny_imagenet(root, download=download)

    # train: tiny-imagenet-200/train/<wnid>/images/*.JPEG
    train_dir = os.path.join(base, "train")
    # val: tiny-imagenet-200/val/images/*.JPEG + val_annotations.txt; we convert to ImageFolder layout on the fly.
    val_dir = os.path.join(base, "val")

    # CIFAR-style aug (random crop + flip) but keep 64x64
    transform_train = T.Compose([
        T.RandomCrop(64, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = ImageFolder(train_dir, transform=transform_train)
    # For val, ImageFolder doesn't work directly because images aren't in class folders by default.
    # We create a small cached re-organization under root/tiny-imagenet-200/val_imagefolder.
    val_img_root = os.path.join(base, "val_imagefolder")
    if not os.path.isdir(val_img_root):
        _prepare_val_imagefolder(val_dir, val_img_root)

    val_set = ImageFolder(val_img_root, transform=transform_test)

    bs = int(cfg.get("batch_size", 128))
    workers = int(cfg.get("workers", 4))
    pin = bool(cfg.get("pin_memory", True))

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader = val_loader  # Tiny-ImageNet has val only
    num_classes = 200
    return train_loader, val_loader, test_loader, num_classes


def _prepare_val_imagefolder(val_dir: str, out_root: str):
    """Convert Tiny-ImageNet val split to ImageFolder-compatible structure."""
    import shutil

    os.makedirs(out_root, exist_ok=True)
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    img_dir = os.path.join(val_dir, "images")
    if not os.path.isfile(ann_file) or not os.path.isdir(img_dir):
        raise FileNotFoundError("Tiny-ImageNet val structure not found.")
    mapping = {}
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    for img_name, wnid in mapping.items():
        src = os.path.join(img_dir, img_name)
        if not os.path.isfile(src):
            continue
        dst_dir = os.path.join(out_root, wnid)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, img_name)
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
