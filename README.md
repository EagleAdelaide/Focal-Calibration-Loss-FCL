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


## Structure
```
.
├── calibration.py    # metrics, temperature scaling, ECE/AdaECE/Classwise-ECE helpers
├── data.py           # CIFAR loaders
├── environment.yml   # conda environment
├── evaluate.py       # evaluate (optionally with TS) + reliability diagram
├── losses.py         # CE, Brier, CE+Brier, Focal, Label Smoothing
├── models.py         # CIFAR backbones: ResNet-50 (CIFAR-style), ResNet-110, WRN-28-10, DenseNet-121
├── reliability.py    # plotting utilities (reliability diagram)
├── train.py          # training loop with CSV logging, TS, checkpoint saving, lock
├── utils.py          # seeding, ETA formatting, cross-host lock helpers
└── verify.py         # load a saved checkpoint and verify metrics on test set
```

## Environment
Create the conda environment (GPU optional):
```bash
conda env create -f environment.yml
conda activate euclid-calib
# If you don't have a compatible NVIDIA driver, comment the `pytorch-cuda` line in environment.yml and reinstall CPU-only PyTorch:
#   conda install pytorch torchvision cpuonly -c pytorch -c conda-forge
```

## Quick Start (Train)
Example: ResNet-50 on CIFAR-10 with cross-entropy, 50 epochs, AMP on GPU, save the best checkpoint and reliability plots every 10 epochs.
```bash
python train.py   --dataset cifar10 --data ./data   --backbone resnet50 --loss ce   --epochs 50 --batch-size 128 --lr 0.1 --weight-decay 5e-4   --amp   --ece-bins 15 --ts-every 1 --rd-every 10   --out-dir runs/c10_resnet50_ce   --log-csv metrics.csv   --save-dir checkpoints   --skip-if-done
```
Notes:
- The code downloads CIFAR automatically to `--data`.
- A lock file named like `lock_cifar10_resnet50_ce.lock` is created under `--out-dir`. If another server starts the same run, it will see the lock and skip.

## Evaluate (with or without TS)
```bash
python evaluate.py --dataset cifar10 --data ./data --backbone resnet50 --out-dir runs_eval
python evaluate.py --dataset cifar10 --data ./data --backbone resnet50 --out-dir runs_eval --no-ts
python evaluate.py --dataset cifar10 --data ./data --backbone resnet50   --checkpoint checkpoints/best_cifar10_resnet50_ce.pt   --out-dir runs_eval
```

This produces a reliability diagram image under `--out-dir` and prints metrics.

## Verify a Saved Model
```bash
python verify.py --dataset cifar10 --data ./data --backbone resnet50   --checkpoint checkpoints/best_cifar10_resnet50_ce.pt
```

## Multi-Machine Mutual Exclusion
We use a simple lock-file protocol. If a lock is older than 48h, it is treated as stale and automatically recovered. The lock guards per-run key `{dataset}|{backbone}|{loss}`.

## Reproducibility Tips
- We set seeds (`--seed`, default 42). Due to cuDNN and data order, small variations can still occur.
- Set `--ts-every 1` to perform temperature scaling each epoch for tracking TS metrics; or run `evaluate.py` at the end.
- CSV logs are stored at `--out-dir/--log-csv` (default `metrics.csv`).
- 
## Cite
For Citing.

---

If only want a "single command to reproduce", point them to:

```bash
python many_losses_multibackbone_training.py --dataset cifar100 --out-dir runs/c100   --backbones resnet50,resnet110,wideresnet28x10,densenet121
```

This will download CIFAR-100 automatically and produce logs & figures under `runs/c100`.
