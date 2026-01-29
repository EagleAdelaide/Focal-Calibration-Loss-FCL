from __future__ import annotations
from typing import Optional, Tuple

import torchvision
import torchvision.transforms as T


def get_svhn(root: str = "./data", split: str = "test"):
    """
    SVHN loader (cropped digits). We use CIFAR normalization by default to match
    common OoD protocol when the ID model is trained on CIFAR.
    """
    split = split.lower()
    assert split in ["train", "test", "extra"]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = torchvision.datasets.SVHN(root=root, split=split, download=True, transform=transform)
    return ds
