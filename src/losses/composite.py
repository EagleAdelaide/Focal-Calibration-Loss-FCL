from __future__ import annotations
import torch
import torch.nn as nn

from .ce import CrossEntropyLoss
from .brier import BrierScoreLoss
from .mmce import MMCE

class CEBrier(nn.Module):
    def __init__(self, num_classes: int, lam: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.brier = BrierScoreLoss(num_classes=num_classes)
        self.lam = float(lam)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.lam * self.brier(logits, target)

class CEMMCE(nn.Module):
    def __init__(self, lam: float = 1.0, sigma: float = 0.4, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mmce = MMCE(sigma=sigma)
        self.lam = float(lam)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, target) + self.lam * self.mmce(logits, target)
