from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return F.cross_entropy(logits, target, reduction="mean")
        # Custom label smoothing (works in older torch versions too)
        n_classes = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            y = torch.zeros_like(logp).fill_(self.label_smoothing / (n_classes - 1))
            y.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        return -(y * logp).sum(dim=1).mean()
