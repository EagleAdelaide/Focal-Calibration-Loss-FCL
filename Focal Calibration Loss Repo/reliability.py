import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def reliability_diagram(probs_before, probs_after, labels, out_path, n_bins=15, title="Reliability Diagram (Before vs After TS)"):
    def compute_bin_stats(probs):
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        centers, accs, confs, counts = [], [], [], []
        for i in range(n_bins):
            if i == 0:
                m = (conf <= bins[i+1])
            elif i == n_bins - 1:
                m = (conf > bins[i]) & (conf <= bins[i+1] + 1e-8)
            else:
                m = (conf > bins[i]) & (conf <= bins[i+1])
            num = int(m.sum().item())
            centers.append(((bins[i] + bins[i+1]) / 2).item())
            if num > 0:
                accs.append(correct[m].mean().item())
                confs.append(conf[m].mean().item())
            else:
                accs.append(float('nan')); confs.append(float('nan'))
            counts.append(num)
        return centers, accs, confs, counts

    cb_x, cb_acc, cb_conf, cb_cnt = compute_bin_stats(probs_before)
    ca_x, ca_acc, ca_conf, ca_cnt = compute_bin_stats(probs_after)

    def nan_to_val(a, val=0.0):
        a = np.array(a, dtype=float); a[np.isnan(a)] = val; return a
    cb_acc_n = nan_to_val(cb_acc); cb_conf_n = nan_to_val(cb_conf)
    ca_acc_n = nan_to_val(ca_acc); ca_conf_n = nan_to_val(ca_conf)
    xs = np.array(cb_x, dtype=float)

    group_width = 0.24
    w = group_width / 4.0
    offsets = np.array([-1.5*w, -0.5*w, 0.5*w, 1.5*w])

    def alpha_for_counts(cnt, base=0.85, empty=0.15):
        cnt = np.array(cnt, dtype=float)
        return np.where(cnt > 0, base, empty)
    a_cb = alpha_for_counts(cb_cnt); a_ca = alpha_for_counts(ca_cnt)

    plt.figure(figsize=(7.2, 5.2))
    bars_cb_acc  = plt.bar(xs + offsets[0], cb_acc_n,  width=w, alpha=0.85, label="Acc (Before)")
    bars_cb_conf = plt.bar(xs + offsets[1], cb_conf_n, width=w, alpha=0.60, label="Conf (Before)")
    bars_ca_acc  = plt.bar(xs + offsets[2], ca_acc_n,  width=w, alpha=0.85, label="Acc (After TS)")
    bars_ca_conf = plt.bar(xs + offsets[3], ca_conf_n, width=w, alpha=0.60, label="Conf (After TS)")

    for i, r in enumerate(bars_cb_acc.patches):  r.set_alpha(float(a_cb[i]))
    for i, r in enumerate(bars_cb_conf.patches): r.set_alpha(float(a_cb[i] * 0.8))
    for i, r in enumerate(bars_ca_acc.patches):  r.set_alpha(float(a_ca[i]))
    for i, r in enumerate(bars_ca_conf.patches): r.set_alpha(float(a_ca[i] * 0.8))

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlim(0.0 - group_width, 1.0 + group_width)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Confidence (bin centers)")
    plt.ylabel("Accuracy / Confidence")
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
