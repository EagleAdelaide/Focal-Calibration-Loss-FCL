import argparse, os, torch
from data import get_cifar_loaders
from models import make_backbone
from calibration import eval_logits, metrics_from_logits, ModelWithTemperature
from reliability import reliability_diagram

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--backbone', type=str, default='resnet50')
    ap.add_argument('--checkpoint', type=str, required=False, help='optional: path to a trained checkpoint')
    ap.add_argument('--ece-bins', type=int, default=15)
    ap.add_argument('--out-dir', type=str, default='runs_eval')
    ap.add_argument('--no-ts', action='store_true', default=False, help='disable temperature scaling')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, num_classes = get_cifar_loaders(args.dataset, args.data, seed=42)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model = make_backbone(args.backbone, num_classes=num_classes)
        model.load_state_dict(ckpt['model_state']); model = model.to(device)
    else:
        model = make_backbone(args.backbone, num_classes=num_classes).to(device)

    logits_test, labels_test = eval_logits(model, test_loader, device)
    acc, ece, adae, cwe, nll, probs_before = metrics_from_logits(logits_test, labels_test, n_bins=args.ece_bins, num_classes=num_classes)

    if args.no_ts:
        acc_t, ece_t, adae_t, cwe_t, nll_t = acc, ece, adae, cwe, nll
        probs_after = probs_before; T=1.0
    else:
        mwT = ModelWithTemperature(model, device=device)
        T = mwT.set_temperature(val_loader, max_iter=50, lr=0.01)
        acc_t, ece_t, adae_t, cwe_t, nll_t, probs_after = metrics_from_logits(logits_test, labels_test, n_bins=args.ece_bins, num_classes=num_classes, temperature=mwT.temperature.detach())

    print(f"Test acc={acc:.4f} ece={ece:.4f} nll={nll:.3f} | TS acc={acc_t:.4f} ece_t={ece_t:.4f} T={T:.3f}")

    out_rel = os.path.join(args.out_dir, f"reliability_{args.dataset}_{args.backbone}.png")
    reliability_diagram(probs_before, probs_after, labels_test, out_rel, n_bins=args.ece_bins,
                        title=f"{args.dataset}|{args.backbone} (T={T:.3f})")
    print(f"Saved reliability diagram to {out_rel}")

if __name__ == '__main__':
    main()
