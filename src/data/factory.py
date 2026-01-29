from __future__ import annotations

from .cifar import build_loaders as build_cifar
from .tiny_imagenet import get_tiny_imagenet_loaders


def build_dataloaders(cfg):
    name = cfg["dataset"]["name"].lower()

    if name in ["cifar10", "cifar100"]:
        root = cfg["dataset"].get("root", "./data")
        num_workers = int(cfg["dataset"].get("num_workers", 4))
        batch_size = int(cfg["train"]["batch_size"])
        test_batch_size = int(cfg["train"].get("test_batch_size", 512))
        return build_cifar(name, root, batch_size, test_batch_size, num_workers)

    if name in ["tiny_imagenet", "tinyimagenet", "tiny-imagenet"]:
        # expects cfg.dataset.tiny_imagenet dict
        ti_cfg = dict(cfg["dataset"].get("tiny_imagenet", {}))
        ti_cfg.setdefault("root", cfg["dataset"].get("root", "./data"))
        ti_cfg.setdefault("download", cfg["dataset"].get("download", True))
        ti_cfg.setdefault("batch_size", int(cfg["train"]["batch_size"]))
        ti_cfg.setdefault("workers", int(cfg["dataset"].get("num_workers", 4)))
        ti_cfg.setdefault("pin_memory", True)
        return get_tiny_imagenet_loaders(ti_cfg)

    raise ValueError(f"Dataset not implemented in this scaffold: {name}")
