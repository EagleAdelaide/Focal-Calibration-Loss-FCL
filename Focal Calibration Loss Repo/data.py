from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import torch

def get_cifar_loaders(dataset: str, data_root: str, batch_size=128, num_workers=4, val_size=5000, seed=42):
    dataset = dataset.lower()
    assert dataset in ("cifar10", "cifar100")
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    test_tf  = T.Compose([T.ToTensor(), normalize])

    if dataset == "cifar10":
        TrainDS = torchvision.datasets.CIFAR10
        TestDS  = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        TrainDS = torchvision.datasets.CIFAR100
        TestDS  = torchvision.datasets.CIFAR100
        num_classes = 100

    full_train = TrainDS(root=data_root, train=True,  download=True, transform=train_tf)
    test_set   = TestDS (root=data_root, train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(seed)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes
