from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLUngated(nn.Module):
    """Ungated FCL: focal + λ * Brier."""
    def __init__(self, num_classes: int, gamma_focal: float = 2.0, lam: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma_focal = float(gamma_focal)
        self.lam = float(lam)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = -((1 - p_t).pow(self.gamma_focal)) * logp_t

        y = F.one_hot(target, num_classes=self.num_classes).float()
        brier = (p - y).pow(2).sum(dim=1)

        loss = focal + self.lam * brier
        return loss.mean() if self.reduction == "mean" else loss.sum()

class SAFCL(nn.Module):
    """
    SA-FCL (Ours): focal + λ * w(x) * Brier
      w(x) = detach(1 - p_t)^{gamma_cal}
    """
    def __init__(self, num_classes: int, gamma_focal: float = 2.0, gamma_cal: float = 2.0, lam: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma_focal = float(gamma_focal)
        self.gamma_cal = float(gamma_cal)
        self.lam = float(lam)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()

        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = -((1 - p_t).pow(self.gamma_focal)) * logp_t

        y = F.one_hot(target, num_classes=self.num_classes).float()
        brier = (p - y).pow(2).sum(dim=1)

        w = (1 - p_t).detach().pow(self.gamma_cal)
        loss = focal + self.lam * w * brier
        return loss.mean() if self.reduction == "mean" else loss.sum()

class SAFCLNoDetach(nn.Module):
    """Gate without stop-gradient (ablation)."""
    def __init__(self, num_classes: int, gamma_focal: float = 2.0, gamma_cal: float = 2.0, lam: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma_focal = float(gamma_focal)
        self.gamma_cal = float(gamma_cal)
        self.lam = float(lam)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()

        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = -((1 - p_t).pow(self.gamma_focal)) * logp_t

        y = F.one_hot(target, num_classes=self.num_classes).float()
        brier = (p - y).pow(2).sum(dim=1)

        w = (1 - p_t).pow(self.gamma_cal)  # NOT detached
        loss = focal + self.lam * w * brier
        return loss.mean() if self.reduction == "mean" else loss.sum()
