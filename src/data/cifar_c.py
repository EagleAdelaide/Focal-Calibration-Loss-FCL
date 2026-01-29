import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


# CIFAR-10-C is distributed as numpy arrays (.npy) in the official release.
# We use the canonical Zenodo mirror. If the link changes, set `root` to a manually downloaded directory.
CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10C_FILENAME = "CIFAR-10-C.tar"
CIFAR10C_FOLDER = "CIFAR-10-C"


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

        # severity slices: each severity is 10k samples in order
        start = (self.severity - 1) * 10000
        end = self.severity * 10000
        self.images = images[start:end]
        self.labels = labels[start:end].astype(np.int64)

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        download_and_extract_archive(
            url=CIFAR10C_URL,
            download_root=self.root,
            filename=CIFAR10C_FILENAME,
            md5=None,
        )
        # The extracted folder name should be CIFAR-10-C
        if not os.path.isdir(self.base_dir):
            # some extractors may create a nested folder; try to locate
            for cand in [os.path.join(self.root, "CIFAR-10-C"), os.path.join(self.root, "CIFAR-10-C", "CIFAR-10-C")]:
                if os.path.isdir(cand):
                    self.base_dir = cand
                    break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        y = int(self.labels[idx])
        # to PIL
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, y
