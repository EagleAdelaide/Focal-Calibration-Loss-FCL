from __future__ import annotations
from typing import Tuple

import torch


@torch.no_grad()
def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute AUROC (area under ROC) from scores and binary labels.

    Args:
        scores: [N] higher means more likely positive
        labels: [N] in {0,1}, 1 is positive
    """
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()

    # Sort by score descending
    order = torch.argsort(scores, descending=True)
    scores = scores[order]
    labels = labels[order]

    # True/false positives as we move threshold
    P = labels.sum().item()
    N = (labels.numel() - labels.sum()).item()
    if P == 0 or N == 0:
        return float("nan")

    tps = torch.cumsum(labels, dim=0).float()
    fps = torch.cumsum(1 - labels, dim=0).float()

    tpr = tps / P
    fpr = fps / N

    # Add (0,0) at start and (1,1) at end for integration
    tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
    fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])

    # Trapezoidal integration
    auc = torch.trapz(tpr, fpr).item()
    return float(auc)


@torch.no_grad()
def msp_scores(probs: torch.Tensor) -> torch.Tensor:
    """Max softmax probability (MSP) score: higher = more in-distribution."""
    return probs.max(dim=1).values


@torch.no_grad()
def ood_auroc_from_probs(id_probs: torch.Tensor, ood_probs: torch.Tensor) -> float:
    """
    AUROC for OOD detection using MSP as a score, with OOD as positive class.
    We use score = -MSP so that higher => more OOD.
    """
    id_msp = msp_scores(id_probs)
    ood_msp = msp_scores(ood_probs)
    scores = torch.cat([-id_msp, -ood_msp], dim=0)
    labels = torch.cat([torch.zeros_like(id_msp, dtype=torch.long), torch.ones_like(ood_msp, dtype=torch.long)], dim=0)
    return auroc(scores, labels)
