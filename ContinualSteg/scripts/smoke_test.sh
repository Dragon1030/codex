#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?usage: smoke_test.sh /path/to/data}"
python train.py --data_root "$DATA_ROOT" --image_size 64 --batch_size 2 \
  --epochs_per_task 1 --tasks clean,gaussian --max_train_batches 2 \
  --max_val_batches 1 --checkpoint_dir runs/smoke
