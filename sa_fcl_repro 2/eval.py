import argparse
import torch

from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.checkpoint import load_ckpt
from src.data.factory import build_dataloaders
from src.models.factory import build_model
from src.metrics.calibration import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", "--cfg", dest="config", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)

    train_loader, test_loader, num_classes = build_dataloaders(cfg)
    model = build_model(cfg["model"], num_classes=num_classes).to(device)

    ckpt = load_ckpt(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    stats = evaluate(model, test_loader, device=device, num_classes=num_classes, n_bins=int(cfg.get("metrics", {}).get("ece_bins", 15)))
    print(f"acc={stats.acc*100:.2f}%  ece={stats.ece:.4f}  ada={stats.ada_ece:.4f}  class={stats.class_ece:.4f}  sm={stats.sm_ece:.4f}  nll={stats.nll:.4f}  brier={stats.brier:.4f}")

if __name__ == "__main__":
    main()
