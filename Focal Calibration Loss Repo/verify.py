import argparse, os, torch
from data import get_cifar_loaders
from models import make_backbone
from calibration import eval_logits, metrics_from_logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--backbone', type=str, default='resnet50')
    ap.add_argument('--checkpoint', type=str, required=True, help='path to a trained checkpoint (.pt)')
    ap.add_argument('--ece-bins', type=int, default=15)
    args = ap.parse_args()

    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader, num_classes = get_cifar_loaders(args.dataset, args.data, seed=42)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = make_backbone(args.backbone, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state']); model = model.to(device)

    logits_test, labels_test = eval_logits(model, test_loader, device)
    acc, ece, adae, cwe, nll, _ = metrics_from_logits(logits_test, labels_test, n_bins=args.ece_bins, num_classes=num_classes)
    print(f"[VERIFY] Test acc={acc:.4f} ece={ece:.4f} nll={nll:.3f}")

if __name__ == '__main__':
    main()
