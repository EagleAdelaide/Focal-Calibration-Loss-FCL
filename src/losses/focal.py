from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t).clamp_min(0).pow(self.gamma)) * logp_t
        return loss.mean() if self.reduction == "mean" else loss.sum()
