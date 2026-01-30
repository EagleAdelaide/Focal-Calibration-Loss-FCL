from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .simple_transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

# Official CIFAR python dataset tarballs
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

CIFAR10_FOLDER = "cifar-10-batches-py"
CIFAR100_FOLDER = "cifar-100-python"

# Standard CIFAR normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _download(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        return
    print(f"Downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def _extract_tar_gz(path: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dst_dir)


def _ensure_cifar(root: str, name: str, download: bool = True):
    root = os.path.expanduser(root)
    if name == "cifar10":
        folder = os.path.join(root, CIFAR10_FOLDER)
        if os.path.isdir(folder):
            return folder
        if not download:
            raise FileNotFoundError(
                f"{folder} not found. Either set dataset.download=true to auto-download, "
                f"or manually download/extract CIFAR-10 to {folder}."
            )
        archive = os.path.join(root, "cifar-10-python.tar.gz")
        _download(CIFAR10_URL, archive)
        _extract_tar_gz(archive, root)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected extracted folder {folder} not found")
        return folder

    if name == "cifar100":
        folder = os.path.join(root, CIFAR100_FOLDER)
        if os.path.isdir(folder):
            return folder
        if not download:
            raise FileNotFoundError(
                f"{folder} not found. Either set dataset.download=true to auto-download, "
                f"or manually download/extract CIFAR-100 to {folder}."
            )
        archive = os.path.join(root, "cifar-100-python.tar.gz")
        _download(CIFAR100_URL, archive)
        _extract_tar_gz(archive, root)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected extracted folder {folder} not found")
        return folder

    raise ValueError(f"Unknown CIFAR dataset: {name}")


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _load_cifar10(folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(1, 6):
        d = _load_pickle(os.path.join(folder, f"data_batch_{i}"))
        xs.append(d["data"])
        ys.append(np.array(d["labels"], dtype=np.int64))
    xtr = np.concatenate(xs, axis=0)
    ytr = np.concatenate(ys, axis=0)

    dt = _load_pickle(os.path.join(folder, "test_batch"))
    xte = dt["data"]
    yte = np.array(dt["labels"], dtype=np.int64)

    # reshape to N, C, H, W
    xtr = xtr.reshape(-1, 3, 32, 32)
    xte = xte.reshape(-1, 3, 32, 32)
    return xtr, ytr, xte, yte


def _load_cifar100(folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = _load_pickle(os.path.join(folder, "train"))
    te = _load_pickle(os.path.join(folder, "test"))
    xtr = tr["data"].reshape(-1, 3, 32, 32)
    ytr = np.array(tr["fine_labels"], dtype=np.int64)
    xte = te["data"].reshape(-1, 3, 32, 32)
    yte = np.array(te["fine_labels"], dtype=np.int64)
    return xtr, ytr, xte, yte


class CIFAR(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert images.ndim == 4 and images.shape[1] == 3
        self.images = images  # uint8, NCHW
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = self.images[idx]
        y = int(self.labels[idx])
        # NCHW uint8 -> HWC uint8 for ToTensor
        x = np.transpose(x, (1, 2, 0))
        x = ToTensor()(x)  # CHW float
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def get_cifar(name: str, root: str = "./data", download: bool = True):
    """Return (train_set, test_set, num_classes)."""
    name = name.lower()
    folder = _ensure_cifar(root, name, download=download)
    if name == "cifar10":
        xtr, ytr, xte, yte = _load_cifar10(folder)
        num_classes = 10
    elif name == "cifar100":
        xtr, ytr, xte, yte = _load_cifar100(folder)
        num_classes = 100
    else:
        raise ValueError(name)

    # Train augmentations (standard)
    train_tf = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(0.5),
            Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_tf = Compose([Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_set = CIFAR(xtr, ytr, transform=train_tf)
    test_set = CIFAR(xte, yte, transform=test_tf)
    return train_set, test_set, num_classes


def build_loaders(
    name: str,
    root: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int = 4,
    download: bool = True,
):
    train_set, test_set, num_classes = get_cifar(name=name, root=root, download=download)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader, num_classes
