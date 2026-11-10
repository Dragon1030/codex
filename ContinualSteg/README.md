# ContinualSteg

ContinualSteg is a runnable research baseline for continual robust image
information hiding. It combines a compact UDH-style hiding/reveal network,
RwF-style isolated task paths, AIM-style parameter-efficient adapters, replay,
distillation, and prompt/prototype routing.

The implementation is self-contained and accepts the DIV2K folders already
used on the server. It is an adapter baseline for research experiments; it
does not silently replace the original HiNet or UDH repositories.

## What is trained

The default continual stream is:

```text
clean -> JPEG-like quantization -> Gaussian noise -> resize -> crop
```

The clean stage trains the base hiding/reveal network. Later stages freeze the
base and old task paths, train a new task adapter and prompt, and use replay
plus output distillation to reduce forgetting.

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

Use an existing environment that already has a working CUDA-enabled PyTorch,
or install the dependencies in a new environment:

```bash
python -m pip install torch torchvision
python -m pip install pillow pyyaml scikit-image tqdm
```

The repository's `requirements.txt` lists PyTorch as a dependency, but on a
server with a preconfigured CUDA image it is safer to keep the existing
PyTorch installation and install only the remaining packages.

The current trainer reads its parameters from command-line arguments. The
file `configs/default.yaml` documents defaults but is not loaded automatically
by `train.py`.

## Docker on the server

If the host does not provide the expected Conda path, use a PyTorch container
and mount the host directory:

```bash
docker run -d \
  --name continualsteg_div2k \
  --gpus '"device=0"' \
  --shm-size=8g \
  -v /data/zhangle/HiNet:/data/zhangle/HiNet \
  -w /data/zhangle/HiNet/ContinualSteg \
  pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel sleep infinity

docker exec -it continualsteg_div2k bash
```

Inside the container, `/data/zhangle/HiNet/ContinualSteg` is the same project
directory as on the host, so checkpoints and logs remain available after the
container exits.

## Smoke test

```bash
python train.py \
  --data_root /data/zhangle/HiNet \
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

`--data_root` is currently required by the argument parser even when explicit
`--train_dir` and `--val_dir` are provided, so keep it in every command.

## Full training

```bash
python train.py \
  --data_root /data/zhangle/HiNet \
  --train_dir /data/zhangle/HiNet/DIV2K_train_HR \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --image_size 256 \
  --batch_size 8 \
  --epochs_per_task 50 \
  --tasks clean,jpeg,gaussian,resize,crop \
  --checkpoint_dir runs/div2k_continual
```

For the five default tasks, `epochs_per_task=50` means 50 epochs per task,
not 50 epochs total. Reduce `--batch_size` or `--image_size` if the GPU runs
out of memory.

The trainer writes `latest.pt` after every epoch and `stage_XX.pt` after each
completed task. Resume an interrupted stage with:

```bash
python train.py --resume runs/div2k_continual/latest.pt \
  --data_root /data/zhangle/HiNet \
  --train_dir /data/zhangle/HiNet/DIV2K_train_HR \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --checkpoint_dir runs/div2k_continual
```

## Evaluation

```bash
python train.py --mode eval \
  --data_root /data/zhangle/HiNet \
  --train_dir /data/zhangle/HiNet/DIV2K_train_HR \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --checkpoint runs/div2k_continual/stage_04.pt \
  --checkpoint_dir runs/div2k_continual \
  --tasks clean,jpeg,gaussian,resize,crop
```

The CSV contains cover PSNR/SSIM, recovered-secret PSNR/SSIM, APD and
task-level results. Summarize secret-PSNR forgetting with:

```bash
python tools/forgetting.py runs/div2k_continual/metrics.csv
```

The trainer does not currently write PNG visualizations. Its outputs are the
checkpoints, `metrics.csv`, and the console metrics; image visualization can
be added as a separate evaluation script without changing the training loop.

## Checkpoints and interruption recovery

`latest.pt` is written after every epoch and `stage_XX.pt` after each completed
task. Run long jobs inside `tmux`, or start them with `nohup`/`docker exec -d`,
so an SSH disconnect does not stop the process. Resume with the same model
shape arguments and `--resume runs/div2k_continual/latest.pt`.

The model uses a prompt-aware reveal wrapper. The training and evaluation
paths call `model.reveal_container(...)`, which supplies the task prompt to
`RevealNet`; calling `model.reveal(container, task_id)` directly is incomplete.

## Connecting the exact original baselines

This repository is a runnable adapter baseline. To use the uploaded projects,
replace the model calls in `continual_steg/models.py`:

- UDH: wrap `models/HidingUNet.py` and `models/RevealNet.py`, injecting an
  adapter at the U-Net bottleneck and RevealNet middle feature.
- HiNet: pass the same prompt through `hinet.py` and `invblock.py`; prompt
  biases must be used in both forward and reverse coupling equations.

Do not load a Restormer or AIM checkpoint into HiNet/UDH. Only the adapter and
routing mechanisms are shared.
