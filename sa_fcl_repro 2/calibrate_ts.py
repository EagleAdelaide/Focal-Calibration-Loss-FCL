import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.checkpoint import load_ckpt
from src.data.cifar import get_cifar, split_train_val
from src.models.factory import build_model
from src.metrics.calibration import expected_calibration_error, EvalStats

class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32)))

    @property
    def T(self):
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.clamp_min(1e-6)

@torch.no_grad()
def eval_with_temperature(model, loader, device: str, num_classes: int, T: float, n_bins: int = 15) -> EvalStats:
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
        logits = model(x) / T
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
    ece = expected_calibration_error(probs, targets, n_bins=n_bins)

    return EvalStats(
        acc=correct / max(total, 1),
        ece=ece,
        nll=nll_sum / max(total, 1),
        brier=brier_sum / max(total, 1),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", "--cfg", dest="config", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_iter", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)

    # Build model
    ds_name = cfg["dataset"]["name"]
    root = cfg["dataset"].get("root", "./data")
    train_set, test_set, num_classes = get_cifar(ds_name, root=root)
    tr_set, val_set = split_train_val(train_set, val_ratio=args.val_ratio, seed=int(cfg.get("seed", 0)))

    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=int(cfg["dataset"].get("num_workers", 4)), pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=int(cfg["dataset"].get("num_workers", 4)), pin_memory=True)

    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    ckpt = load_ckpt(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    # Collect logits/targets on val (no grad through model)
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits_list.append(model(x).detach())
            targets_list.append(y.detach())
    logits_val = torch.cat(logits_list, dim=0)
    targets_val = torch.cat(targets_list, dim=0)

    scaler = TemperatureScaler(init_T=1.0).to(device)
    opt = torch.optim.LBFGS([scaler.log_T], lr=0.5, max_iter=args.max_iter)

    def closure():
        opt.zero_grad(set_to_none=True)
        scaled = scaler(logits_val)
        loss = F.cross_entropy(scaled, targets_val)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(scaler.T.item())

    # Pre vs Post on test
    pre = eval_with_temperature(model, test_loader, device, num_classes, T=1.0, n_bins=int(cfg.get("metrics", {}).get("ece_bins", 15)))
    post = eval_with_temperature(model, test_loader, device, num_classes, T=T, n_bins=int(cfg.get("metrics", {}).get("ece_bins", 15)))

    print(f"[TS] optimal T={T:.4f}")
    print(f"[Pre]  acc={pre.acc*100:.2f}%  ece={pre.ece:.4f}  nll={pre.nll:.4f}  brier={pre.brier:.4f}")
    print(f"[Post] acc={post.acc*100:.2f}%  ece={post.ece:.4f}  nll={post.nll:.4f}  brier={post.brier:.4f}")

if __name__ == "__main__":
    main()
