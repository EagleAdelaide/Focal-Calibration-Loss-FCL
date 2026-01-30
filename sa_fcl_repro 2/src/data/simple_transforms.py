"""A tiny subset of torchvision-style transforms (torch-only).

This repo originally used torchvision transforms, but torchvision may be
unavailable or incompatible in some environments. These transforms cover
what we need for CIFAR/SVHN experiments: random crop, horizontal flip,
tensor conversion, and normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


@dataclass
class ToTensor:
    """Convert a HWC uint8 numpy array or PIL image to CHW float tensor in [0,1]."""

    def __call__(self, x):
        if hasattr(x, "mode") and hasattr(x, "size"):
            # PIL image
            x = np.array(x)
        if isinstance(x, np.ndarray):
            if x.ndim != 3:
                raise ValueError(f"Expected HWC array, got shape={x.shape}")
            x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        if not torch.is_tensor(x):
            raise TypeError(f"Unsupported type for ToTensor: {type(x)}")
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = x.float()
        return x


@dataclass
class Normalize:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Normalize expects CHW tensor")
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)[:, None, None]
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)[:, None, None]
        return (x - mean) / std


@dataclass
class RandomHorizontalFlip:
    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.p:
            return torch.flip(x, dims=[2])  # flip width
        return x


@dataclass
class RandomCrop:
    size: int
    padding: int = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        _, h, w = x.shape
        th = tw = int(self.size)
        if h < th or w < tw:
            raise ValueError(f"Crop size {th} larger than input {(h,w)}")
        i = int(torch.randint(0, h - th + 1, (1,)).item())
        j = int(torch.randint(0, w - tw + 1, (1,)).item())
        return x[:, i : i + th, j : j + tw]
