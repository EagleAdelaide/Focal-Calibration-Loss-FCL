import argparse
import json
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.device import resolve_device
from src.data.cifar import get_cifar
from src.data.svhn import get_svhn
from src.data.cifar_c import CIFAR10C, CIFAR10C_CORRUPTIONS
from src.models.factory import build_model
from src.metrics.ood import ood_auroc_from_probs


@torch.no_grad()
def collect_probs(model, loader, device: str) -> torch.Tensor:
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).detach())
    return torch.cat(probs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)

    # Either specify an explicit config (recommended) or use the standalone flags.
    ap.add_argument("--config", "--cfg", dest="config", type=str, default=None, help="Optional YAML config")

    ap.add_argument("--id_dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--model", type=str, default="resnet18")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")

    # OOD options
    ap.add_argument("--svhn", action="store_true", help="Evaluate CIFAR -> SVHN shift AUROC (MSP).")
    ap.add_argument("--cifar10c", action="store_true", help="Evaluate CIFAR -> CIFAR-10-C corruptions AUROC (MSP).")
    ap.add_argument(
        "--cifar10c_root",
        type=str,
        default="./data",
        help="Root dir to store/find CIFAR-10-C (will create <root>/CIFAR-10-C)",
    )
    ap.add_argument("--cifar10c_severity", type=int, default=5)
    ap.add_argument(
        "--cifar10c_corruptions",
        type=str,
        nargs="*",
        default=None,
        help="Subset of corruptions; default: all 15 corruptions.",
    )
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    # If a YAML config is provided, use it to set dataset/model defaults.
    cfg = None
    root = "./data"
    model_cfg = {"name": args.model}
    if args.config:
        cfg = load_config(args.config)
        args.id_dataset = str(cfg["dataset"]["name"]).lower()
        root = str(cfg["dataset"].get("root", root))
        model_cfg = cfg.get("model", model_cfg)

    device = resolve_device(args.device)

    train_set, test_set, num_classes = get_cifar(args.id_dataset, root=root)
    id_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    model = build_model(model_cfg, num_classes=num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    id_probs = collect_probs(model, id_loader, device=device)

    results: Dict[str, float] = {}

    if args.svhn:
        svhn_ds = get_svhn(root=root, split="test")
        svhn_loader = DataLoader(
            svhn_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(args.workers > 0),
        )
        ood_probs = collect_probs(model, svhn_loader, device=device)
        results["auroc_cifar_to_svhn"] = ood_auroc_from_probs(id_probs, ood_probs) * 100.0

    if args.cifar10c:
        if args.id_dataset != "cifar10":
            raise ValueError("CIFAR-10-C evaluation is only defined for id_dataset=cifar10.")

        corrs = args.cifar10c_corruptions or CIFAR10C_CORRUPTIONS
        aurocs = []

        # Match ID normalization for corrupted samples.
        transform = getattr(test_set, "transform", None)

        for c in corrs:
            ds = CIFAR10C(
                root=args.cifar10c_root,
                corruption=c,
                severity=args.cifar10c_severity,
                transform=transform,
                download=True,
            )
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=(args.workers > 0),
            )
            probs_c = collect_probs(model, loader, device=device)
            aurocs.append(ood_auroc_from_probs(id_probs, probs_c) * 100.0)

        results[f"auroc_cifar_to_cifar10c_s{args.cifar10c_severity}_mean"] = float(sum(aurocs) / max(len(aurocs), 1))
        for c, a in zip(corrs, aurocs):
            results[f"auroc_cifar_to_cifar10c_s{args.cifar10c_severity}_{c}"] = float(a)

    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
