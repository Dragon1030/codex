#!/usr/bin/env bash
set -euo pipefail

# Run four comparable streams. Override these values for a short pilot:
#   EPOCHS_PER_TASK=5 MAX_TRAIN_BATCHES=20 ./scripts/run_ablation.sh
DATA_ROOT="${DATA_ROOT:-/data/zhangle/HiNet}"
TRAIN_DIR="${TRAIN_DIR:-/data/zhangle/HiNet/DIV2K_train_HR}"
VAL_DIR="${VAL_DIR:-/data/zhangle/HiNet/DIV2K_valid_HR}"
OUT_ROOT="${OUT_ROOT:-runs/ablation}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
TASKS="clean,jpeg,gaussian,resize,crop"

run_one() {
  local name="$1"
  local strategy="$2"
  local replay_capacity="$3"
  local replay_per_batch="$4"
  local lambda_distill="$5"

  echo "===== ${name} ====="
  python -u train.py \
    --data_root "$DATA_ROOT" \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs_per_task "$EPOCHS_PER_TASK" \
    --tasks "$TASKS" \
    --strategy "$strategy" \
    --replay_capacity "$replay_capacity" \
    --replay_per_batch "$replay_per_batch" \
    --lambda_distill "$lambda_distill" \
    --max_train_batches "$MAX_TRAIN_BATCHES" \
    --max_val_batches "$MAX_VAL_BATCHES" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --checkpoint_dir "$OUT_ROOT/$name" \
    2>&1 | tee "$OUT_ROOT/${name}.log"
}

mkdir -p "$OUT_ROOT"

# Proposed continual method: isolated adapters + replay + distillation.
run_one full continual 256 2 0.2

# Adapter isolation without output distillation.
run_one no_distill continual 256 2 0.0

# Adapter isolation without replay or distillation.
run_one adapter_only continual 0 0 0.0

# Naive sequential fine-tuning baseline.
run_one sequential_finetune sequential_finetune 0 0 0.0

python tools/compare_runs.py \
  --run full="$OUT_ROOT/full/metrics.csv" \
  --run no_distill="$OUT_ROOT/no_distill/metrics.csv" \
  --run adapter_only="$OUT_ROOT/adapter_only/metrics.csv" \
  --run sequential_finetune="$OUT_ROOT/sequential_finetune/metrics.csv" \
  --out "$OUT_ROOT/comparison.csv"
