import argparse
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config import load_config, merge_overrides
from src.utils.seed import set_seed
from src.utils.lr import cosine_lr, piecewise_lr
from src.utils.device import resolve_device
from src.utils.jsonl import append_jsonl
from src.utils.checkpoint import save_ckpt

from src.data.factory import build_dataloaders
from src.models.factory import build_model
from src.losses import build_loss
from src.metrics.calibration import evaluate


def _maybe_safcl_decompose(logits, y, num_classes, loss_cfg: dict, loss_name: str):
    """Return per-sample focal/brier/p_t/w for SA-FCL style losses (or None...)."""
    loss_name = str(loss_name).lower()
    if loss_name not in ["safcl", "safcl_nodetach", "fcl_ungated"]:
        return None, None, None, None

    gamma_focal = float(loss_cfg.get("gamma_focal", 2.0))
    gamma_cal = float(loss_cfg.get("gamma_cal", 2.0))

    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    logp_t = logp.gather(1, y.unsqueeze(1)).squeeze(1)
    p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)

    focal_vec = -((1 - p_t).pow(gamma_focal)) * logp_t
    onehot = F.one_hot(y, num_classes=num_classes).float()
    brier_vec = (p - onehot).pow(2).sum(dim=1)

    if loss_name == "fcl_ungated":
        w_vec = torch.ones_like(p_t)
    elif loss_name == "safcl":
        w_vec = (1 - p_t).detach().pow(gamma_cal)
    else:
        w_vec = (1 - p_t).pow(gamma_cal)

    return focal_vec, brier_vec, p_t, w_vec


def _make_log_path(base: Optional[str], run_dir: str, seed: int, multi_seed: bool) -> str:
    """Resolve the JSONL log path.

    If the user provides --out and uses multiple seeds, we make paths unique.
    - If --out contains '{seed}', we format it.
    - Else we append '_seed{seed}' before the suffix.
    """
    if base is None:
        return os.path.join(run_dir, "metrics.jsonl")

    base = os.path.expanduser(base)
    if multi_seed:
        if "{seed}" in base:
            return base.format(seed=seed)
        p = Path(base)
        stem = p.stem
        suffix = "".join(p.suffixes) or ".jsonl"
        return str(p.with_name(f"{stem}_seed{seed}{suffix}"))
    return base


def _current_lr(tr_cfg: Dict, epoch: int) -> float:
    # Priority: explicit piecewise schedule -> cosine -> constant
    if tr_cfg.get("lr_schedule", None) is not None:
        return float(piecewise_lr(tr_cfg["lr_schedule"], epoch))

    sched = tr_cfg.get("scheduler", None)
    if isinstance(sched, str) and sched.lower() in ["cosine", "cosine_lr", "cosineannealing"]:
        return float(cosine_lr(float(tr_cfg["lr"]), epoch, int(tr_cfg["epochs"])))
    if isinstance(sched, dict) and str(sched.get("type", "")).lower() in ["cosine", "cosine_lr"]:
        return float(cosine_lr(float(tr_cfg["lr"]), epoch, int(tr_cfg["epochs"])))

    return float(tr_cfg["lr"])


def train_one_seed(cfg: dict, seed: int, out_jsonl: Optional[str] = None) -> str:
    cfg = deepcopy(cfg)
    cfg["seed"] = int(seed)

    device = resolve_device(cfg.get("device", "auto"))
    set_seed(seed)

    train_loader, test_loader, num_classes = build_dataloaders(cfg)
    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    criterion = build_loss(cfg, num_classes=num_classes)

    tr_cfg = cfg["train"]
    opt_name = str(tr_cfg.get("optimizer", "sgd")).lower()
    if opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(tr_cfg["lr"]),
            momentum=float(tr_cfg.get("momentum", 0.9)),
            weight_decay=float(tr_cfg.get("weight_decay", 5e-4)),
            nesterov=bool(tr_cfg.get("nesterov", True)),
        )
    elif opt_name == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=float(tr_cfg["lr"]),
            betas=tuple(tr_cfg.get("betas", (0.9, 0.999))),
            weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Optional ReduceLROnPlateau scheduler (in addition to per-epoch LR assignment above).
    plateau = None
    sched_monitor = "val_ece"
    sched_cfg = tr_cfg.get("scheduler", {})
    if isinstance(sched_cfg, dict) and str(sched_cfg.get("type", "")).lower() == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(sched_cfg.get("mode", "min")),
            factor=float(sched_cfg.get("factor", 0.1)),
            patience=int(sched_cfg.get("patience", 5)),
            threshold=float(sched_cfg.get("threshold", 1e-4)),
            min_lr=float(sched_cfg.get("min_lr", 0.0)),
            verbose=False,
        )
        sched_monitor = str(sched_cfg.get("monitor", "val_ece"))

    exp_name = cfg.get("exp_name", "exp")
    run_dir = os.path.join(cfg.get("out_dir", "runs"), f"{exp_name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = _make_log_path(out_jsonl, run_dir, seed, multi_seed=False)

    best = {"ece": 1e9, "epoch": -1}
    t0 = time.time()

    for ep in range(int(tr_cfg["epochs"])):
        model.train()

        # LR schedule
        cur_lr = _current_lr(tr_cfg, ep)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        # trackers
        total = 0
        loss_sum = 0.0
        focal_sum = 0.0
        brier_sum = 0.0
        w_sum = 0.0
        wbrier_sum = 0.0
        pt_sum = 0.0
        gate_grad_norm = float("nan")

        pbar = tqdm(train_loader, desc=f"train ep {ep+1:03d}", leave=False)
        for it, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            bs = y.size(0)
            total += bs
            loss_sum += float(loss.item()) * bs

            focal_vec, brier_vec, p_t, w_vec = _maybe_safcl_decompose(
                logits, y, num_classes, cfg["loss"], cfg["loss"]["name"]
            )
            if focal_vec is not None:
                with torch.no_grad():
                    focal_sum += float(focal_vec.mean().item()) * bs
                    brier_sum += float(brier_vec.mean().item()) * bs
                    w_sum += float(w_vec.mean().item()) * bs
                    wbrier_sum += float((w_vec * brier_vec).mean().item()) * bs
                    pt_sum += float(p_t.mean().item()) * bs

                # first batch: measure ||d w / d logits|| for nodetach
                if it == 0 and str(cfg["loss"]["name"]).lower() == "safcl_nodetach":
                    logp_g = F.log_softmax(logits, dim=1)
                    p_g = logp_g.exp()
                    p_t_g = p_g.gather(1, y.unsqueeze(1)).squeeze(1)
                    gamma_cal = float(cfg["loss"].get("gamma_cal", 2.0))
                    w_g = (1 - p_t_g).pow(gamma_cal)
                    gw = torch.autograd.grad(w_g.mean(), logits, retain_graph=True, create_graph=False)[0]
                    gate_grad_norm = float(gw.norm().item())
                elif it == 0 and str(cfg["loss"]["name"]).lower() == "safcl":
                    gate_grad_norm = 0.0

            pbar.set_postfix(loss=loss_sum / max(total, 1))

        # epoch averages
        tr_loss = loss_sum / max(total, 1)
        tr_focal = focal_sum / max(total, 1) if total > 0 else float("nan")
        tr_brier = brier_sum / max(total, 1) if total > 0 else float("nan")
        tr_w = w_sum / max(total, 1) if total > 0 else float("nan")
        tr_wbrier = wbrier_sum / max(total, 1) if total > 0 else float("nan")
        tr_pt = pt_sum / max(total, 1) if total > 0 else float("nan")

        stats = evaluate(
            model,
            test_loader,
            device=device,
            num_classes=num_classes,
            n_bins=int(cfg.get("metrics", {}).get("ece_bins", 15)),
        )

        if plateau is not None:
            if sched_monitor == "val_ece":
                plateau.step(stats.ece)
            elif sched_monitor == "val_nll":
                plateau.step(stats.nll)
            elif sched_monitor == "val_brier":
                plateau.step(stats.brier)
            else:
                plateau.step(stats.ece)

        if stats.ece < best["ece"]:
            best.update({"ece": float(stats.ece), "epoch": int(ep)})
            save_ckpt(
                os.path.join(run_dir, "ckpt_best.pt"),
                {"model": model.state_dict(), "cfg": cfg, "epoch": ep, "best": best},
            )

        save_ckpt(
            os.path.join(run_dir, "ckpt_last.pt"),
            {"model": model.state_dict(), "cfg": cfg, "epoch": ep, "best": best},
        )

        rec = {
            "exp_name": exp_name,
            "seed": seed,
            "epoch": ep,
            "lr": cur_lr,
            "train_loss": tr_loss,
            "train_focal": tr_focal,
            "train_brier": tr_brier,
            "train_w_mean": tr_w,
            "train_wbrier": tr_wbrier,
            "train_pt_mean": tr_pt,
            "train_gate_grad_norm": gate_grad_norm,
            "test_acc": stats.acc,
            "test_ece": stats.ece,
            "test_ada_ece": getattr(stats, "ada_ece", float("nan")),
            "test_class_ece": getattr(stats, "class_ece", float("nan")),
            "test_sm_ece": getattr(stats, "sm_ece", float("nan")),
            "test_nll": stats.nll,
            "test_brier": stats.brier,
            "best_ece_sofar": best["ece"],
            "best_epoch_sofar": best["epoch"],
            "wall_time_sec": time.time() - t0,
        }
        append_jsonl(log_path, rec)

        print(
            f"[EP {ep+1:03d}/{int(tr_cfg['epochs'])}] {exp_name} seed={seed} "
            f"lr={cur_lr:.3e} train_loss={tr_loss:.4f} "
            f"test(acc={stats.acc*100:.2f}%, ece={stats.ece:.4f}, "
            f"ada={getattr(stats,'ada_ece',float('nan')):.4f}, "
            f"class={getattr(stats,'class_ece',float('nan')):.4f}, "
            f"sm={getattr(stats,'sm_ece',float('nan')):.4f}, "
            f"nll={stats.nll:.4f}, brier={stats.brier:.4f}) "
            f"best_ece={best['ece']:.4f}@{best['epoch']+1:03d}"
        )

    print(f"[DONE] run_dir={run_dir}  log={log_path}")
    return run_dir


def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "--cfg", dest="config", type=str, required=True, help="YAML config path")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional list of seeds")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", dest="out_jsonl", type=str, default=None, help="Optional metrics JSONL path")
    ap.add_argument(
        "--opts",
        nargs="*",
        default=None,
        help="Optional overrides like train.epochs=1 loss.lam=0.5",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.opts:
        cfg = merge_overrides(cfg, args.opts)
    if args.device is not None:
        cfg["device"] = args.device

    seeds = args.seeds if args.seeds is not None and len(args.seeds) > 0 else [int(cfg.get("seed", 0))]
    multi = len(seeds) > 1

    for s in seeds:
        # If user passes --out for multi-seed, create unique paths.
        out_path = _make_log_path(args.out_jsonl, run_dir=os.path.join(cfg.get("out_dir", "runs"), f"{cfg.get('exp_name','exp')}_seed{s}"), seed=s, multi_seed=multi)
        train_one_seed(cfg, s, out_jsonl=out_path)


if __name__ == "__main__":
    main()
