#!/usr/bin/env bash
# ============================================================================
# scripts/dev/overfit.sh — overfit smoke test (1 episode, 几百步, loss 能否→0)
# ----------------------------------------------------------------------------
# 用法:
#   bash scripts/dev/overfit.sh <config_yaml> [MAX_STEPS]
#
#   例:
#     bash scripts/dev/overfit.sh examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml 300
#
# 前提: config_yaml 里已通过 overfit_episode_id 显式开启单 episode 过拟合
#       (语义见 agent.md 附录 A.3)。本脚本只负责单 GPU、短步数、本地日志地跑起来。
# 行为: 单 GPU 跑训练入口, 默认 300 步、SwanLab 本地、复用 worktree 的 .venv。
#       结果只能作为 pipeline/fitting evidence, 不能作为泛化证据。
# 详见 docs/zerith_docs/workflow/testing_layers.md
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "用法: $0 <config_yaml> [MAX_STEPS]" >&2
  echo "  例: $0 examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml 300" >&2
  exit 1
fi

CONFIG_YAML="$1"
MAX_STEPS="${2:-300}"
[ -f "$CONFIG_YAML" ] || { echo "ERROR: 配置不存在: $CONFIG_YAML" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 优先用 worktree 的 .venv 里的 accelerate
ACCEL_BIN="${ACCEL_BIN:-$REPO_ROOT/.venv/bin/accelerate}"
if [ ! -x "$ACCEL_BIN" ]; then
  ACCEL_BIN="$(command -v accelerate || true)"
fi
[ -n "$ACCEL_BIN" ] || { echo "ERROR: 找不到 accelerate (先 uv sync 或激活 venv)" >&2; exit 1; }

echo ">> overfit smoke: $CONFIG_YAML  steps=$MAX_STEPS  GPU=$CUDA_VISIBLE_DEVICES"
"$ACCEL_BIN" launch --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG_YAML" \
  --trainer.max_train_steps "$MAX_STEPS"
