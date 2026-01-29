from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BrierScoreLoss(nn.Module):
    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1)
        y = F.one_hot(target, num_classes=self.num_classes).float()
        loss = (p - y).pow(2).sum(dim=1)
        return loss.mean() if self.reduction == "mean" else loss.sum()
