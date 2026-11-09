# ContinualSteg

ContinualSteg is a runnable research baseline for continual robust image
information hiding. It combines UDH/HiNet hiding, RwF isolated domain paths,
AIM-style parameter-efficient adapters, and prompt/prototype routing.

The first implementation is self-contained and trains a UDH-style encoder and
reveal network. It accepts the image folders already used on the server, while
the original HiNet/UDH repositories remain unchanged.

## Task stream

The default stream is:

```
0 clean -> 1 JPEG-like quantization -> 2 Gaussian noise -> 3 resize -> 4 crop
```

Stage 0 trains the base hiding network. From stage 1 onward the base and old
adapters are frozen and only the current adapter/prompt is updated. A replay
memory and output distillation loss preserve previous tasks.

## Dataset layout

```
data/
├── train/       # DIV2K_train_HR or another clean image directory
└── val/         # DIV2K_valid_HR or another clean image directory
```

Images can be PNG or JPG and do not require class labels. A cover and a secret
are sampled independently from the same split.

## Installation

```bash
python -m pip install torch torchvision pillow pyyaml scikit-image tqdm
```

## Smoke test

```bash
python train.py \
  --train_dir /data/zhangle/HiNet/DIV2K_train_HR \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --image_size 64 \
  --batch_size 2 \
  --epochs_per_task 1 \
  --tasks clean,gaussian \
  --max_train_batches 2 \
  --max_val_batches 1 \
  --checkpoint_dir runs/smoke
```

## Full training

```bash
python train.py \
  --train_dir /data/zhangle/HiNet/DIV2K_train_HR \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --image_size 256 \
  --batch_size 8 \
  --epochs_per_task 50 \
  --tasks clean,jpeg,gaussian,resize,crop \
  --checkpoint_dir runs/div2k_continual
```

The trainer writes `latest.pt` after every epoch and `stage_XX.pt` after each
completed task. Resume an interrupted stage with:

```bash
python train.py --resume runs/div2k_continual/latest.pt \
  --train_dir /data/zhangle/HiNet/DIV2K_train_HR \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --checkpoint_dir runs/div2k_continual
```

## Evaluation

```bash
python train.py --mode eval \
  --train_dir /data/zhangle/HiNet/DIV2K_train_HR \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --checkpoint runs/div2k_continual/stage_04.pt \
  --checkpoint_dir runs/div2k_continual \
  --tasks clean,jpeg,gaussian,resize,crop
```

The CSV contains cover PSNR/SSIM, recovered-secret PSNR/SSIM, APD and
task-level results. The forgetting calculation can be added directly from this
matrix.

## Connecting the exact original baselines

This repository is a runnable adapter baseline. To use the uploaded projects,
replace the model calls in `continual_steg/models.py`:

- UDH: wrap `models/HidingUNet.py` and `models/RevealNet.py`, injecting an
  adapter at the U-Net bottleneck and RevealNet middle feature.
- HiNet: pass the same prompt through `hinet.py` and `invblock.py`; prompt
  biases must be used in both forward and reverse coupling equations.

Do not load a Restormer or AIM checkpoint into HiNet/UDH. Only the adapter and
routing mechanisms are shared.
