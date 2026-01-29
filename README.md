# Self-Adaptive Focal Calibration Loss for Trustworthy Deep Learning

<img width="3817" height="2300" alt="image" src="https://github.com/user-attachments/assets/b754eaf7-6835-4994-b5b9-cd70e8d9a733" />

## Reproducible CIFAR-10/100 Training: Multi-loss & Multi-backbone

Modular, reproducible code for training classification backbones on CIFAR with multiple loss functions, post-hoc **temperature scaling**, and **reliability diagrams**. The code supports **multi-machine mutual exclusion** via lock files so the same run won't start twice across servers.

## Evolution of Reliabilty Diagram （Restnet-18 on CIFAR-10）
Definition of OverConfidence/UnderConfidence:

<p align="center">
  <img src="https://github.com/user-attachments/assets/0fbb02f2-114c-4848-95d1-08143f824ad0" width="804" alt="Definition of OverConfidence/UnderConfidence" />
</p>


| CE | Focal |
|---|---|
| <img src="https://github.com/user-attachments/assets/cd87b7dd-cdc1-4082-8e17-7f794057e9dc" width="420" alt="CE reliability evolution" /> | <img src="https://github.com/user-attachments/assets/d14792e9-a7f3-4065-be7f-b1b5d74dabe1" width="420" alt="Focal reliability evolution" /> |

| Brier | Self-Adaptive Focal Calibration |
|---|---|
| <img src="https://github.com/user-attachments/assets/06cd5425-ddb6-4928-acef-2a77395092b4" width="420" alt="Brier reliability evolution" /> | <img src="https://github.com/user-attachments/assets/5ce7cd8b-d5d8-4aac-81e9-b480854afbdf" width="420" alt="SA-FCL reliability evolution" /> |


### Loss baselines
- **CE + weight decay** (`wd_ce`)
- **Label smoothing** (`label_smoothing`, default 0.05)
- **Brier** (`brier`)
- **MMCE** (`mmce`)
- **FLSD-53** (`flsd53`)
- **Dual Focal** (`dual_focal`)
- **Ungated FCL** = focal + λ·Brier (`fcl_ungated`)
- **SA-FCL (ours)** (`safcl`) and **SA-FCL w/o stop-grad** (`safcl_nodetach`)

### Metrics
- Accuracy
- ECE / **adaECE** / **classECE**
- **smECE**
- NLL, Brier
- Evidence components (mean focal / brier / w / w·brier / p_t)

### Robustness / OoD table
- CIFAR → **SVHN** AUROC (MSP)
- CIFAR → **CIFAR-10-C** AUROC (MSP), per-corruption and mean

## Install

```bash
pip install -r requirements.txt
```

## Quick start (CIFAR-10, ResNet-50)

Each method has its own YAML under `configs/<dataset>/<model>/<method>.yaml`, and inherits a dataset/model base config from `configs/_base_/*.yaml`.

Train **SA-FCL**:
```bash
python train.py --cfg configs/cifar10/resnet50/safcl.yaml --out runs/cifar10_resnet50_safcl.jsonl
```

Train **all baselines** for the same dataset/model (example):
```bash
for m in wd_ce label_smooth brier mmce flsd53 dual_focal fcl_ungated safcl safcl_nostop; do
  python train.py --cfg configs/cifar10/resnet50/${m}.yaml --out runs/cifar10_resnet50_${m}.jsonl
done
```

> Note: `safcl_nostop.yaml` uses the **no-stop-gradient** variant (`safcl_nodetach`) to demonstrate the “confidence collapse” behavior.

## Training schedules 

- **CIFAR-10/100**: 350 epochs, LR = 0.1 (0–149), 0.01 (150–249), 0.001 (250–349), SGD (momentum=0.9, wd=5e-4), batch=128.
- **Tiny-ImageNet**: 100 epochs, LR = 0.1 (0–39), 0.01 (40–59), 0.001 (60–99), SGD, batch=128.
- **Tiny-ImageNet uses a CIFAR-style stem** for ResNet/DenseNet (3×3 stride-1, no maxpool).

These are encoded as `train.lr_schedule` inside `configs/_base_/*.yaml`.

## Evaluate (metrics)

You can compute metrics from a saved checkpoint:

```bash
python eval.py --ckpt path/to/checkpoint.pt --cfg configs/cifar10/resnet50/safcl.yaml
```

## OoD robustness (SVHN, CIFAR-10-C)

```bash
python ood_eval.py --ckpt path/to/checkpoint.pt --cfg configs/cifar10/resnet50/safcl.yaml --svhn --cifar10c
```

- **SVHN** is downloaded via torchvision.
- **CIFAR-10-C** is auto-downloaded from a canonical mirror into `./data/CIFAR-10-C/` (can be overridden by `--cifar10c_root`).

## Evidence plot (per-epoch trends)

Given a JSONL log produced by `train.py`, plot evidence components:

```bash
python calibrate_stats.py --jsonl runs/cifar10_resnet50_safcl.jsonl --out runs/plots/cifar10_resnet50_safcl
```

This produces:
- `evidence_components.png`
- `ece_vs_acc.png`
- optional gradient-ratio plots if logged


See `src/data/` and `src/models/` for extension points.

## Cite
For Citing.

---

