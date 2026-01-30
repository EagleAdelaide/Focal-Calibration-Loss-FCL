from __future__ import annotations

import os
import urllib.request
from typing import Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .simple_transforms import Compose, Normalize, ToTensor

SVHN_URLS = {
    "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
    "extra": "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
}

# Use CIFAR normalization by default to match common OoD protocols when ID is CIFAR.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _download(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        return
    print(f"Downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


class SVHNDataset(Dataset):
    def __init__(self, mat_path: str, transform=None):
        d = loadmat(mat_path)
        # X: (32,32,3,N) uint8, y: (N,1)
        x = d["X"]
        y = d["y"].squeeze(1)
        y = y.astype(np.int64)
        y[y == 10] = 0

        x = np.transpose(x, (3, 0, 1, 2))  # N,H,W,C
        self.images = x
        self.labels = y
        self.transform = transform

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        img = self.images[idx]  # HWC uint8
        y = int(self.labels[idx])
        x = ToTensor()(img)
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def get_svhn(root: str = "./data", split: str = "test"):
    """SVHN loader (cropped digits).

    Downloads the official .mat file and returns a torch Dataset.
    """
    split = split.lower()
    if split not in ["train", "test", "extra"]:
        raise ValueError("split must be one of: train, test, extra")

    svhn_dir = os.path.join(os.path.expanduser(root), "SVHN")
    os.makedirs(svhn_dir, exist_ok=True)
    fname = f"{split}_32x32.mat"
    path = os.path.join(svhn_dir, fname)
    _download(SVHN_URLS[split], path)

    transform = Compose([
        Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return SVHNDataset(path, transform=transform)
