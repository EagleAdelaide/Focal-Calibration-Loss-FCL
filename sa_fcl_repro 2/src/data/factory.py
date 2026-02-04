from __future__ import annotations

from .cifar import build_loaders as build_cifar
from .cifar import get_cifar as _get_cifar


def get_cifar(name: str, root: str = "./data"):
    """Small convenience wrapper used by some scripts (e.g., ood_eval.py)."""
    return _get_cifar(name=name, root=root)


def build_dataloaders(cfg):
    name = cfg["dataset"]["name"].lower()

    if name in ["cifar10", "cifar100"]:
        root = cfg["dataset"].get("root", "./data")
        download = bool(cfg["dataset"].get("download", True))
        num_workers = int(cfg["dataset"].get("num_workers", 4))
        batch_size = int(cfg["train"]["batch_size"])
        test_batch_size = int(cfg["train"].get("test_batch_size", 512))
        return build_cifar(name, root, batch_size, test_batch_size, num_workers, download=download)

    if name in ["tiny_imagenet", "tinyimagenet", "tiny-imagenet"]:
        # Lazy import to avoid hard dependency on torchvision unless needed.
        from .tiny_imagenet import get_tiny_imagenet_loaders
        # expects cfg.dataset.tiny_imagenet dict
        ti_cfg = dict(cfg["dataset"].get("tiny_imagenet", {}))
        ti_cfg.setdefault("root", cfg["dataset"].get("root", "./data"))
        ti_cfg.setdefault("download", cfg["dataset"].get("download", True))
        ti_cfg.setdefault("batch_size", int(cfg["train"]["batch_size"]))
        ti_cfg.setdefault("workers", int(cfg["dataset"].get("num_workers", 4)))
        ti_cfg.setdefault("pin_memory", True)
        train_loader, val_loader, test_loader, num_classes = get_tiny_imagenet_loaders(ti_cfg)
        # Keep the public contract consistent: return (train_loader, test_loader, num_classes).
        # Tiny-ImageNet uses val as its canonical evaluation split; the helper returns
        # both val and test (alias). Use val_loader here to avoid mismatched unpacking.
        return train_loader, val_loader, num_classes

    raise ValueError(f"Dataset not implemented in this scaffold: {name}")
