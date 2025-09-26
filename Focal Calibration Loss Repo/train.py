import os, time, csv, argparse, json, traceback
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils import set_seed, fmt_eta, acquire_lock, release_lock
from data import get_cifar_loaders
from models import make_backbone
from losses import BrierScore, BrierCombo, FocalLoss, LabelSmoothingLoss
from calibration import eval_logits, ModelWithTemperature, metrics_from_logits
from reliability import reliability_diagram

def make_scheduler(opt, total_epochs=350):
    return optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 250], gamma=0.1)

def init_csv(path, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    if new_file:
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header)

def append_csv(path, row):
    with open(path, "a", newline="") as f:
        w = csv.writer(f); w.writerow(row)

def get_criterion(name: str, num_classes: int, device: str):
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "brier":
        return BrierScore()
    if name == "ce+brier0.1":
        return BrierCombo(gamma=0.1)
    if name == "focal":
        return FocalLoss(gamma=5.0, reduction="mean")
    if name == "labelsmooth":
        return LabelSmoothingLoss(classes=num_classes, smoothing=0.1, device=device)
    raise ValueError(f"Unknown loss: {name}")

def train_one(cfg):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.log_csv = os.path.join(cfg.out_dir, cfg.log_csv)

    run_key = f"{cfg.dataset}|{cfg.backbone}|{cfg.loss}"
    lock_file = acquire_lock(cfg.out_dir, run_key, stale_hours=48)
    if lock_file is None:
        print(f"[SKIP-LOCK] {run_key} — another worker is training (lock present).")
        return 0

    done_path = os.path.join(cfg.out_dir, f"done_{cfg.dataset}_{cfg.backbone}_{cfg.loss}.json")
    if cfg.skip_if_done and os.path.exists(done_path):
        print(f"[SKIP-DONE] {run_key} — done marker found at {done_path}")
        return 0

    try:
        train_loader, val_loader, test_loader, num_classes = get_cifar_loaders(
            cfg.dataset, cfg.data, batch_size=cfg.batch_size, num_workers=cfg.num_workers, val_size=5000, seed=cfg.seed
        )

        model = make_backbone(cfg.backbone, num_classes=num_classes).to(device)
        criterion = get_criterion(cfg.loss, num_classes, device)
        opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
        sched = make_scheduler(opt, total_epochs=cfg.epochs)

        init_csv(cfg.log_csv, header=[
            "dataset","num_classes","backbone","method","epoch",
            "test_acc","test_ece","test_adaece","test_classwise_ece","test_nll",
            "test_acc_ts","test_ece_ts","test_adaece_ts","test_classwise_ece_ts","test_nll_ts",
            "temperature_T"
        ])

        scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda" and cfg.amp))

        start_all = time.time()
        best = {"acc": 0.0, "acc_t": 0.0}
        for ep in range(1, cfg.epochs+1):
            ep_t0 = time.time()
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {ep}/{cfg.epochs}", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device=='cuda' and cfg.amp)):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            sched.step()

            logits_test, labels_test = eval_logits(model, test_loader, device)
            acc, ece, adae, cwe, nll, probs_before = metrics_from_logits(
                logits_test, labels_test, n_bins=cfg.ece_bins, num_classes=num_classes, temperature=None
            )

            if cfg.ts_every > 0 and (ep % cfg.ts_every == 0):
                mwT = ModelWithTemperature(model, device=device)
                Tval = mwT.set_temperature(val_loader, max_iter=50, lr=0.01)
                acc_t, ece_t, adae_t, cwe_t, nll_t, probs_after = metrics_from_logits(
                    logits_test, labels_test, n_bins=cfg.ece_bins, num_classes=num_classes, temperature=mwT.temperature.detach()
                )
            else:
                Tval = 1.0
                acc_t, ece_t, adae_t, cwe_t, nll_t = acc, ece, adae, cwe, nll
                probs_after = probs_before

            append_csv(cfg.log_csv, [
                cfg.dataset, num_classes, cfg.backbone, cfg.loss, ep,
                acc, ece, adae, cwe, nll,
                acc_t, ece_t, adae_t, cwe_t, nll_t, Tval
            ])

            if (cfg.rd_every > 0 and ep % cfg.rd_every == 0) or (ep == cfg.epochs):
                out_rel = os.path.join(cfg.out_dir, f"reliability_{cfg.dataset}_{cfg.backbone}_{cfg.loss}_ep{ep}.png")
                reliability_diagram(probs_before, probs_after, labels_test, out_rel, n_bins=cfg.ece_bins,
                                    title=f"{cfg.dataset}|{cfg.backbone}|{cfg.loss} (T={Tval:.3f})")

            if cfg.save_dir:
                os.makedirs(cfg.save_dir, exist_ok=True)
                if acc_t >= best["acc_t"]:
                    best.update({"acc_t": acc_t, "acc": acc})
                    ckpt = {
                        "model_state": model.state_dict(),
                        "backbone": cfg.backbone,
                        "num_classes": num_classes,
                        "epoch": ep
                    }
                    torch.save(ckpt, os.path.join(cfg.save_dir, f"best_{cfg.dataset}_{cfg.backbone}_{cfg.loss}.pt"))

            ep_t = time.time() - ep_t0
            elapsed = time.time() - start_all
            remaining = (cfg.epochs - ep) * (elapsed / ep)
            print(f"[{cfg.dataset}|{cfg.backbone}|{cfg.loss}] ep {ep:03d}/{cfg.epochs} "
                  f"acc={acc*100:.2f}%  ECE={ece:.4f}  NLL={nll:.3f} "
                  f"| TS acc={acc_t*100:.2f}% ECE_T={ece_t:.4f}  "
                  f"[time/epoch={ep_t:.1f}s | ETA={fmt_eta(remaining)}]")

        with open(done_path, "w") as f:
            json.dump({"finished": True}, f)
        return 0

    except Exception as e:
        print("[ERROR] Training crashed:\n" + traceback.format_exc())
        return 1
    finally:
        release_lock(lock_file)

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--backbone', type=str, default='resnet50')
    ap.add_argument('--loss', type=str, default='ce', help='ce | brier | ce+brier0.1 | focal | labelsmooth')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--amp', action='store_true', default=False)
    ap.add_argument('--ece-bins', type=int, default=15)
    ap.add_argument('--ts-every', type=int, default=1)
    ap.add_argument('--rd-every', type=int, default=10, help='reliability diagram frequency (epochs)')
    ap.add_argument('--out-dir', type=str, default='runs')
    ap.add_argument('--log-csv', type=str, default='metrics.csv')
    ap.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save best checkpoint; "" to disable')
    ap.add_argument('--skip-if-done', action='store_true', default=True)
    return ap

if __name__ == '__main__':
    args = build_parser().parse_args()
    raise SystemExit(train_one(args))
