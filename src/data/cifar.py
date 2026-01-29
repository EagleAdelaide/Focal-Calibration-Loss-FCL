from __future__ import annotations
from typing import Tuple
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_cifar(name: str, root: str = "./data"):
    name = name.lower()
    if name not in ["cifar10", "cifar100"]:
        raise ValueError(f"Unsupported dataset: {name}")
    num_classes = 10 if name == "cifar10" else 100

    # Standard CIFAR augmentation
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    if name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    else:
        train_set = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform_test)

    return train_set, test_set, num_classes

def build_loaders(name: str, root: str, batch_size: int, test_batch_size: int, num_workers: int):
    train_set, test_set, num_classes = get_cifar(name=name, root=root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    return train_loader, test_loader, num_classes


from torch.utils.data import Subset
import numpy as np

def split_train_val(train_set, val_ratio: float = 0.1, seed: int = 0):
    n = len(train_set)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val].tolist()
    tr_idx = idx[n_val:].tolist()
    return Subset(train_set, tr_idx), Subset(train_set, val_idx)
