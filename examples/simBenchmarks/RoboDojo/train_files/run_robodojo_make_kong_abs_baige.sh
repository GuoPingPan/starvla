#!/usr/bin/env bash
# Use the absolute EE training config while retaining the existing BaiGe setup.
set -euo pipefail

export TRAIN_CONFIG=examples/simBenchmarks/RoboDojo/train_files/robodojo_pi05_make_kong_v2_abs.yaml
export BATCH="${BATCH:-8}"
export MAX_STEPS="${MAX_STEPS:-40000}"
export SAVE_EVERY="${SAVE_EVERY:-5000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/run_robodojo_make_kong_baige.sh"
