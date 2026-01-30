import os
import tarfile
import urllib.request
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .simple_transforms import ToTensor

# CIFAR-10-C distributed as numpy arrays (.npy).
# Canonical Zenodo mirror. If the link changes, set `root` to a manually downloaded directory.
CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10C_FILENAME = "CIFAR-10-C.tar"
CIFAR10C_FOLDER = "CIFAR-10-C"

CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def _download(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        return
    print(f"Downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def _extract_tar(tar_path: str, root: str):
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=root)


class CIFAR10C(Dataset):
    """CIFAR-10-C corruption benchmark (Hendrycks & Dietterich).

    Expected directory structure after extraction:
      root/CIFAR-10-C/{corruption}.npy
      root/CIFAR-10-C/labels.npy
    """

    def __init__(
        self,
        root: str = "./data",
        corruption: str = "gaussian_noise",
        severity: int = 5,
        transform=None,
        download: bool = False,
    ):
        super().__init__()
        assert 1 <= int(severity) <= 5
        self.root = os.path.expanduser(root)
        self.corruption = corruption
        self.severity = int(severity)
        self.transform = transform

        self.base_dir = os.path.join(self.root, CIFAR10C_FOLDER)
        if not os.path.isdir(self.base_dir):
            if download:
                self._download()
            else:
                raise FileNotFoundError(
                    f"CIFAR-10-C not found at {self.base_dir}. Set download=True or place files there."
                )

        images_path = os.path.join(self.base_dir, f"{corruption}.npy")
        labels_path = os.path.join(self.base_dir, "labels.npy")
        if not os.path.isfile(images_path) or not os.path.isfile(labels_path):
            raise FileNotFoundError(
                f"Missing {images_path} or {labels_path}. Available corruptions are the .npy files in {self.base_dir}."
            )

        images = np.load(images_path)  # [50000,32,32,3]
        labels = np.load(labels_path)  # [50000]

        start = (self.severity - 1) * 10000
        end = self.severity * 10000
        self.images = images[start:end]
        self.labels = labels[start:end].astype(np.int64)

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        tar_path = os.path.join(self.root, CIFAR10C_FILENAME)
        _download(CIFAR10C_URL, tar_path)
        _extract_tar(tar_path, self.root)

        # Some extractors may create nested folder; try to locate
        if not os.path.isdir(self.base_dir):
            for cand in [
                os.path.join(self.root, "CIFAR-10-C"),
                os.path.join(self.root, "CIFAR-10-C", "CIFAR-10-C"),
            ]:
                if os.path.isdir(cand):
                    self.base_dir = cand
                    break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # HWC uint8
        y = int(self.labels[idx])
        x = ToTensor()(img)
        if self.transform is not None:
            x = self.transform(x)
        return x, y
