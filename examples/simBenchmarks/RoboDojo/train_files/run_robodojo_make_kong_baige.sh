#!/usr/bin/env bash
# Run the converted make_kong v2 dataset in a BaiGe job with the generic PI0.5 launcher.
# The job must mount the training CFS at /mnt/cfs and provide a staged repo with
# an accelerate environment (or set ACCEL_BIN to an existing environment).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
CFS_ROOT="${CFS_ROOT:-/mnt/cfs}"
DATA_ROOT="$CFS_ROOT/data"
DATASET="$DATA_ROOT/robodojo/RoboDojo_make_kong_v30"
CONFIG="${TRAIN_CONFIG:-examples/simBenchmarks/RoboDojo/train_files/robodojo_pi05_make_kong_v2.yaml}"

[[ -f "$REPO_ROOT/$CONFIG" ]] || { echo "Missing config: $REPO_ROOT/$CONFIG" >&2; exit 1; }
for path in \
  "$DATASET/meta/info.json" \
  "$DATASET/meta/tasks.parquet" \
  "$DATASET/meta/episodes/chunk-000/file-000.parquet" \
  "$DATASET/data/chunk-000/file-000.parquet" \
  "$DATA_ROOT/pretrain_models/paligemma-3b-pt-224/tokenizer.model" \
  "$DATA_ROOT/pretrain_models/pi05_base_starvla/pi05_base_starvla/fp32/model.safetensors"; do
  [[ -s "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for camera in cam_high cam_left_wrist cam_right_wrist; do
  [[ -d "$DATASET/videos/observation.images.$camera" ]] || {
    echo "Missing video camera: $camera" >&2; exit 1;
  }
done

# Existing PI0.5 configs and the generic launcher resolve model, data and output
# paths through /root/data. Ensure that link points to this job's mounted CFS.
if [[ -L /root/data ]]; then
  [[ "$(readlink -f /root/data)" == "$(readlink -f "$DATA_ROOT")" ]] || {
    echo "/root/data points outside $DATA_ROOT" >&2; exit 1;
  }
elif [[ -e /root/data ]]; then
  echo "/root/data exists and is not a CFS symlink" >&2
  exit 1
else
  ln -s "$DATA_ROOT" /root/data
fi

export REPO_ROOT
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_PROJECT="${WANDB_PROJECT:-robodojo_pi05_make_kong}"
export BATCH="${BATCH:-4}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export SAVE_EVERY="${SAVE_EVERY:-2500}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH="${PREFETCH:-2}"

if [[ -z "${ACCEL_BIN:-}" ]]; then
  ACCEL_BIN="$REPO_ROOT/.venv/bin/accelerate"
fi
[[ -x "$ACCEL_BIN" ]] || { echo "Missing accelerate: $ACCEL_BIN (set ACCEL_BIN)" >&2; exit 1; }
export ACCEL_BIN

cd "$REPO_ROOT"
exec bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh "$CONFIG"
