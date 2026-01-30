# SA-FCL Reproduction Scaffold

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
