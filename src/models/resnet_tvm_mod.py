import torch.nn as nn
import torchvision.models as tvm


def _apply_cifar_stem(resnet: nn.Module):
    # Replace 7x7 stride2 + maxpool with 3x3 stride1 (CIFAR-style stem).
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    return resnet


def resnet18(num_classes: int, cifar_stem: bool = False, weights=None):
    m = tvm.resnet18(weights=weights)
    if cifar_stem:
        _apply_cifar_stem(m)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def resnet50(num_classes: int, cifar_stem: bool = False, weights=None):
    m = tvm.resnet50(weights=weights)
    if cifar_stem:
        _apply_cifar_stem(m)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
