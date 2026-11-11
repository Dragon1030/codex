# ContinualSteg

ContinualSteg is a runnable research baseline for continual robust image
information hiding. It combines a compact UDH-style hiding/reveal network,
RwF-style isolated task paths, AIM-style parameter-efficient adapters, replay,
distillation, and prompt/prototype routing.

The implementation is self-contained and accepts the DIV2K folders already
used on the server. It is an adapter baseline for research experiments; it
does not silently replace the original HiNet or UDH repositories.

## Structure diagram

The code-aligned data flow and tensor shapes are documented in
[`docs/continual_steg_structure.md`](docs/continual_steg_structure.md). A
static overview is available as
[`docs/continual_steg_structure.svg`](docs/continual_steg_structure.svg).

## What is trained

The default continual stream is:

```text
clean -> JPEG-like quantization -> Gaussian noise -> resize -> crop
```

The clean stage trains the base hiding/reveal network. Later stages freeze the
base and old task paths, train a new task adapter and prompt, and use replay
plus output distillation to reduce forgetting.

## Where continual learning appears

This is a sequential task protocol rather than ordinary joint training:

1. Stage 0 learns the clean task.
2. Stage 1 learns JPEG-like degradation while the clean adapter is frozen.
3. Later stages add one attack at a time and evaluate every task seen so far.
4. A replay buffer revisits examples from old tasks.
5. Output distillation keeps the new model close to the frozen previous model
   on replayed examples.
6. Each task has its own adapter and prompt; old paths are not overwritten.

The `metrics.csv` file is therefore a continual-learning matrix: each stage
contains results for all tasks seen up to that stage. Comparing a task's best
secret PSNR with its final secret PSNR gives its forgetting. The model also
stores task prototypes and exposes `route()`/`recover_unknown()` for future
unknown-attack routing experiments; the default benchmark uses known task IDs
so that the task-adaptation and forgetting effects can be measured separately.

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
python -m pip install pillow pyyaml scikit-image matplotlib tqdm
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

## Reproducible ablations

Run the proposed method and three controls with the same data split, task
order, seed, and evaluation code:

```bash
EPOCHS_PER_TASK=10 MAX_TRAIN_BATCHES=20 MAX_VAL_BATCHES=5 \
  ./scripts/run_ablation.sh
```

Use `EPOCHS_PER_TASK=50` and omit the batch limits for the reported run. The
script creates:

```text
runs/ablation/
├── full/                 # adapters + replay + distillation
├── no_distill/           # adapters + replay
├── adapter_only/         # isolated adapters only
├── sequential_finetune/  # naive full-model sequential fine-tuning
└── comparison.csv
```

For each control, compare final secret recovery quality and forgetting. The
sequential fine-tuning control updates the full network at every task and is
the most direct test for catastrophic forgetting. The other two controls
isolate the contributions of distillation and replay.

The ablation script is intentionally explicit and can be run in a Docker
container after installing the non-PyTorch dependencies. Use a separate
`--checkpoint_dir` for each method; never overwrite the proposed run.
Checkpoints record the strategy used to create them; resuming with a different
strategy is rejected to prevent mixing experiment conditions.

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
be generated with the included evaluation script below.

## Qualitative visualization

Use a completed checkpoint and the validation images to save one six-column
panel per sample and task. The columns are:

```text
cover | container | amplified cover difference |
secret | recovered secret | amplified secret difference
```

The difference columns are amplified for display only; the CSV keeps the
unamplified PSNR, SSIM, and APD measurements.

```bash
python tools/visualize.py \
  --checkpoint runs/div2k_continual/stage_04.pt \
  --val_dir /data/zhangle/HiNet/DIV2K_valid_HR \
  --image_size 128 \
  --tasks clean,jpeg,gaussian,resize,crop \
  --max_samples 8 \
  --out_dir runs/div2k_continual/visuals
```

This writes `sample_XXXX_<task>.png` and
`visualization_metrics.csv`. The script infers the adapter and prompt sizes
from the checkpoint, so they do not need to be repeated.

Plot stage-wise metric curves after training:

```bash
python tools/plot_metrics.py \
  --metrics runs/div2k_continual/metrics.csv \
  --out_dir runs/div2k_continual/plots
```

Compare multiple ablation runs:

```bash
python tools/compare_runs.py \
  --run full=runs/ablation/full/metrics.csv \
  --run no_distill=runs/ablation/no_distill/metrics.csv \
  --run adapter_only=runs/ablation/adapter_only/metrics.csv \
  --run sequential_finetune=runs/ablation/sequential_finetune/metrics.csv \
  --out runs/ablation/comparison.csv
```

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
