import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def find_logs(run_dir: str) -> List[str]:
    run_dir = str(run_dir)
    if os.path.isfile(os.path.join(run_dir, "metrics.jsonl")):
        return [os.path.join(run_dir, "metrics.jsonl")]
    return sorted(glob.glob(os.path.join(run_dir, "*", "metrics.jsonl")))

def stack_curves(runs: List[List[Dict]], key: str):
    # assumes epochs start at 0 and contiguous
    max_len = max(len(r) for r in runs)
    arr = np.full((len(runs), max_len), np.nan, dtype=np.float64)
    for i, r in enumerate(runs):
        for rec in r:
            ep = int(rec["epoch"])
            arr[i, ep] = float(rec.get(key, np.nan))
    return arr

def plot_mean_std(x, arr, title: str, ylabel: str, out_path: str):
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    plt.figure()
    plt.plot(x, mean)
    plt.fill_between(x, mean-std, mean+std, alpha=0.2)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="runs/<exp_name>_seed* or a single seed dir")
    args = ap.parse_args()

    log_paths = find_logs(args.run_dir)
    if len(log_paths) == 0:
        raise FileNotFoundError(f"No metrics.jsonl found under: {args.run_dir}")

    runs = [read_jsonl(p) for p in log_paths]
    out_dir = Path(args.run_dir)
    if not (out_dir / "metrics.jsonl").exists():
        out_dir = out_dir  # parent can hold plots too

    # epochs
    max_len = max(len(r) for r in runs)
    x = np.arange(max_len)

    # Evidence: train decomposition
    keys1 = [
        ("train_loss", "train_loss"),
        ("train_focal", "train_focal"),
        ("train_brier", "train_brier"),
        ("train_w_mean", "w_mean"),
        ("train_wbrier", "w*brier"),
        ("train_pt_mean", "p_t"),
    ]
    for key, ylabel in keys1:
        arr = stack_curves(runs, key)
        plot_mean_std(x, arr, title=f"{ylabel}", ylabel=ylabel, out_path=str(out_dir / f"evidence_{key}.png"))

    # Test metrics
    keys2 = [
        ("test_acc", "acc"),
        ("test_ece", "ece"),
        ("test_nll", "nll"),
        ("test_brier", "brier"),
    ]
    for key, ylabel in keys2:
        arr = stack_curves(runs, key)
        plot_mean_std(x, arr, title=f"test_{ylabel}", ylabel=ylabel, out_path=str(out_dir / f"metrics_{key}.png"))

    print(f"[OK] wrote plots to: {out_dir}")

if __name__ == "__main__":
    main()
