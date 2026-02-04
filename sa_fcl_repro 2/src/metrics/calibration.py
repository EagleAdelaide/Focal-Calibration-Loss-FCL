import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class EvalStats:
    """Common evaluation stats for classification."""
    acc: float
    ece: float
    ada_ece: float
    class_ece: float
    sm_ece: float
    nll: float
    brier: float


@torch.no_grad()
def _ece_from_conf_correct(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    # conf, correct: [N]
    device = conf.device
    bins = torch.linspace(0, 1, n_bins + 1, device=device)
    ece = torch.zeros((), device=device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # last bin includes right edge
        mask = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi + 1e-12)
        if mask.any():
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            ece += mask.float().mean() * (acc_bin - conf_bin).abs()
    return float(ece.item())


@torch.no_grad()
def ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    Standard (top-label) ECE with uniform bins over max confidence.
    probs: [N, K], targets: [N]
    """
    conf, pred = probs.max(dim=1)
    correct = (pred == targets).float()
    return _ece_from_conf_correct(conf, correct, n_bins=n_bins)


# Backwards-compat name used by calibrate_ts.py
def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    return ece(probs, targets, n_bins=n_bins)


@torch.no_grad()
def adaptive_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    Adaptive ECE (adaECE): equal-mass bins (quantile bins) over max confidence.
    This reduces sensitivity to empty bins when confidences cluster.
    """
    conf, pred = probs.max(dim=1)
    correct = (pred == targets).float()

    # sort by confidence and split into equal-sized bins
    conf_sorted, idx = torch.sort(conf)
    correct_sorted = correct[idx]
    n = conf.numel()
    if n == 0:
        return 0.0

    # bin boundaries by index (avoid repeated quantile values issues)
    edges = [0]
    for i in range(1, n_bins):
        edges.append((i * n) // n_bins)
    edges.append(n)

    ada = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        conf_bin = conf_sorted[lo:hi].mean()
        acc_bin = correct_sorted[lo:hi].mean()
        w = (hi - lo) / n
        ada += w * float((acc_bin - conf_bin).abs().item())
    return float(ada)


@torch.no_grad()
def classwise_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    Classwise ECE (classECE): average over classes of binary ECE computed on p_k vs 1{y=k}.
    probs: [N, K], targets: [N]
    """
    n, k = probs.shape
    if n == 0:
        return 0.0
    # For each class k, treat probs[:,k] as confidence of class k.
    # correct_k = 1{y==k}. We compute ECE_k over uniform bins and average over classes.
    device = probs.device
    bins = torch.linspace(0, 1, n_bins + 1, device=device)
    class_ece = 0.0

    for c in range(k):
        conf = probs[:, c]
        correct = (targets == c).float()
        ece_c = torch.zeros((), device=device)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi + 1e-12)
            if mask.any():
                acc_bin = correct[mask].mean()
                conf_bin = conf[mask].mean()
                ece_c += mask.float().mean() * (acc_bin - conf_bin).abs()
        class_ece += float(ece_c.item())
    return float(class_ece / max(k, 1))


@torch.no_grad()
def smooth_ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_grid: int = 101,
    bandwidth: float = 0.1,
    eps: float = 1e-12,
) -> float:
    """
    Smooth ECE (smECE): kernel-smoothed calibration curve approximation.

    Implementation note:
    - Uses top-label confidence (max prob).
    - Estimates acc(c) via kernel regression and integrates |acc(c) - c| weighted by estimated density.
    - This is an approximation suitable for benchmarking; it is not an exact reproduction of any single paper's code.

    Args:
        n_grid: number of grid points over [0,1]
        bandwidth: Gaussian kernel bandwidth h
    """
    conf, pred = probs.max(dim=1)
    correct = (pred == targets).float()
    n = conf.numel()
    if n == 0:
        return 0.0

    device = conf.device
    grid = torch.linspace(0.0, 1.0, n_grid, device=device)  # [G]
    h = float(bandwidth)

    # compute gaussian weights in chunks to control memory: [N, G] can be big.
    # We'll do [chunk, G] blocks.
    G = n_grid
    chunk = 4096 if n > 4096 else n
    num = torch.zeros((G,), device=device)
    den = torch.zeros((G,), device=device)

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        c = conf[start:end].unsqueeze(1)          # [B,1]
        y = correct[start:end].unsqueeze(1)       # [B,1]
        # Gaussian kernel
        w = torch.exp(-0.5 * ((c - grid.unsqueeze(0)) / (h + eps)) ** 2)  # [B,G]
        num += (w * y).sum(dim=0)
        den += w.sum(dim=0)

    acc_hat = num / (den + eps)                  # [G]
    # density proxy (normalized)
    dens = den / (den.sum() + eps)               # [G]
    # Riemann sum (grid step)
    delta = 1.0 / (G - 1)
    sm = (dens * (acc_hat - grid).abs()).sum() * delta * G  # scale to approx integral
    return float(sm.item())


@torch.no_grad()
def evaluate(model, loader, device: str, num_classes: int, n_bins: int = 15,
             sm_grid: int = 101, sm_bandwidth: float = 0.1) -> EvalStats:
    model.eval()
    total = 0
    correct = 0
    nll_sum = 0.0
    brier_sum = 0.0

    all_probs = []
    all_targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        nll_sum += float(F.nll_loss(logp, y, reduction="sum").item())

        onehot = F.one_hot(y, num_classes=num_classes).float()
        brier_sum += float((p - onehot).pow(2).sum(dim=1).sum().item())

        all_probs.append(p)
        all_targets.append(y)

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    ece_v = ece(probs, targets, n_bins=n_bins)
    ada_v = adaptive_ece(probs, targets, n_bins=n_bins)
    cls_v = classwise_ece(probs, targets, n_bins=n_bins)
    sm_v = smooth_ece(probs, targets, n_grid=sm_grid, bandwidth=sm_bandwidth)

    return EvalStats(
        acc=correct / max(total, 1),
        ece=ece_v,
        ada_ece=ada_v,
        class_ece=cls_v,
        sm_ece=sm_v,
        nll=nll_sum / max(total, 1),
        brier=brier_sum / max(total, 1),
    )
