"""DenseNet wrapper.

We keep the option to use torchvision's DenseNet-121 if torchvision is
available. Import is done lazily to avoid hard failures in environments
where torchvision cannot be imported.
"""

from __future__ import annotations

import torch.nn as nn


def _apply_cifar_stem(dn: nn.Module):
    # DenseNet stem: conv0 7x7 stride2 + pool0. Replace with 3x3 stride1, remove pool.
    dn.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    dn.features.pool0 = nn.Identity()
    return dn


def densenet121(num_classes: int, cifar_stem: bool = False, weights=None):
    try:
        import torchvision.models as tvm
    except Exception as e:
        raise RuntimeError(
            "DenseNet-121 requires torchvision, but torchvision could not be imported. "
            "Either install a compatible torchvision build or use a different model (e.g., resnet18)."
        ) from e

    m = tvm.densenet121(weights=weights)
    if cifar_stem:
        _apply_cifar_stem(m)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m
