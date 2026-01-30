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

# SA-FCL Reproduction Guideline

This repository is a minimal, **runnable** scaffold for training and evaluating **Self-Adaptive Focal Calibration Loss (SA-FCL)** on CIFAR.

Key points:
- **No hard dependency on torchvision** for CIFAR/SVHN (some environments have incompatible torchvision builds).  
- CIFAR-10/100 and SVHN are **downloaded on first run** (internet required). You can also **manually place** the extracted datasets under `./data/`.
- Training outputs **checkpoints** and a **metrics JSONL** file under `./runs/`.

---

## 1) Setup

```bash
# (recommended) create a fresh env
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2) Train

Example: CIFAR-10 + ResNet-18 + SA-FCL

```bash
python train.py --cfg configs/cifar10_resnet18_safcl.yaml
```

By default this creates:
- `runs/<exp_name>_seed<seed>/best.pt`
- `runs/<exp_name>_seed<seed>/last.pt`
- `runs/<exp_name>_seed<seed>/metrics.jsonl`

### Useful CLI options

- Multiple seeds:

```bash
python train.py --cfg configs/cifar10_resnet18_safcl.yaml --seeds 0 1 2
```

- Change output folder / experiment name:

```bash
python train.py --cfg configs/cifar10_resnet18_safcl.yaml \
  --out_dir runs \
  --exp_name my_safcl_exp
```

- Override config values from the command line:

```bash
python train.py --cfg configs/cifar10_resnet18_safcl.yaml \
  --opts train.epochs=1 train.batch_size=64
```

(That `epochs=1` override is handy for a quick smoke test.)

---

## 3) Evaluate a checkpoint

```bash
python eval.py --cfg configs/cifar10_resnet18_safcl.yaml \
  --ckpt runs/cifar10_resnet18_safcl_seed0/best.pt
```

---

## 4) Temperature scaling

Fits temperature on the validation split (here: CIFAR test split in this scaffold) and writes a JSON file.

```bash
python calibrate_ts.py --cfg configs/cifar10_resnet18_safcl.yaml \
  --ckpt runs/cifar10_resnet18_safcl_seed0/best.pt \
  --out runs/cifar10_resnet18_safcl_seed0/ts.json
```

---

## 5) OOD / Corruption evaluation

### SVHN OOD

```bash
python ood_eval.py --config configs/cifar10_resnet18_safcl.yaml \
  --ckpt runs/cifar10_resnet18_safcl_seed0/best.pt \
  --ood svhn
```

### CIFAR-10-C corruptions

```bash
python ood_eval.py --config configs/cifar10_resnet18_safcl.yaml \
  --ckpt runs/cifar10_resnet18_safcl_seed0/best.pt \
  --ood cifar10c \
  --corruption gaussian_noise \
  --severity 5 \
  --download_cifar10c
```

---

## 6) Datasets (manual placement)

If you cannot auto-download, place the extracted folders like this:

- CIFAR-10:
  - `data/cifar-10-batches-py/...`
- CIFAR-100:
  - `data/cifar-100-python/...`

For SVHN, place the `.mat` files under:
- `data/SVHN/train_32x32.mat`
- `data/SVHN/test_32x32.mat`
- `data/SVHN/extra_32x32.mat`

---

## Notes

- Tiny-ImageNet support in `src/data/tiny_imagenet.py` still relies on torchvision.
- If you run on CPU and it feels very slow, you can limit threads:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```
## Cite
For Citing.

---

