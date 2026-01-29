from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float):
    # x: [N], y: [M]
    # returns [N, M]
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(- (x - y).pow(2) / (2 * sigma * sigma))

class MMCE(nn.Module):
    """
    Differentiable MMCE (Kumar et al., 2018). This is a lightweight implementation for classification.

    We compute:
      MMCE = sqrt( E_{i,j}[ (c_i - a_i)(c_j - a_j) k(c_i, c_j) ]_+ )

    where:
      c_i = max prob (confidence), a_i = 1[pred_i == y_i].
    """
    def __init__(self, sigma: float = 0.4, eps: float = 1e-12):
        super().__init__()
        self.sigma = float(sigma)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        acc = (pred == target).float()
        diff = (conf - acc)  # [N]

        K = _rbf_kernel(conf, conf, self.sigma)  # [N,N]
        mmce2 = (diff.unsqueeze(1) * diff.unsqueeze(0) * K).mean()
        mmce = torch.sqrt(torch.clamp(mmce2, min=0.0) + self.eps)
        return mmce
