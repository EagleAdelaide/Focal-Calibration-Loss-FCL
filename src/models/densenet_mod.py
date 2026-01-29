import torch.nn as nn
import torchvision.models as tvm


def _apply_cifar_stem(dn: nn.Module):
    # DenseNet stem: conv0 7x7 stride2 + pool0. Replace with 3x3 stride1, remove pool.
    dn.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    dn.features.pool0 = nn.Identity()
    return dn


def densenet121(num_classes: int, cifar_stem: bool = False, weights=None):
    m = tvm.densenet121(weights=weights)
    if cifar_stem:
        _apply_cifar_stem(m)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m
