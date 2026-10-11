#!/usr/bin/env bash
# ============================================================================
# scripts/dev/smoke.sh — 模块级 smoke test (单框架/dataloader 独立运行)
# ----------------------------------------------------------------------------
# 用法:
#   bash scripts/dev/smoke.sh <框架名或 .py 路径> <config.yaml>
#
#   例:
#     bash scripts/dev/smoke.sh QwenOFT examples/.../xxx.yaml
#     bash scripts/dev/smoke.sh starVLA/model/framework/VLM4A/PI05.py examples/.../xxx.yaml
#     bash scripts/dev/smoke.sh starVLA/dataloader/lerobot_datasets.py examples/.../xxx.yaml
#
# 行为: 单进程跑指定框架/dataloader 文件的 __main__ smoke (验证 forward/数据形状),
#       默认单 GPU、SwanLab 本地模式。耗时秒~分。
# 详见 docs/zerith_docs/workflow/testing_layers.md
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "用法: $0 <框架名或 .py 路径> <config.yaml>" >&2
  echo "  例: $0 QwenOFT examples/simBenchmarks/LIBERO/train_files/xxx.yaml" >&2
  exit 1
fi

TARGET="$1"
CONFIG_YAML="$2"
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 框架名 → 文件路径自动解析
if [[ "$TARGET" != *.py ]]; then
  PY="starVLA/model/framework/VLM4A/${TARGET}.py"
  if [ ! -f "$PY" ]; then
    echo "ERROR: 找不到框架文件 $PY (可用名见 starVLA/model/framework/VLM4A/)" >&2
    exit 1
  fi
else
  PY="$TARGET"
fi
[ -f "$PY" ] || { echo "ERROR: 文件不存在: $PY" >&2; exit 1; }
[ -f "$CONFIG_YAML" ] || { echo "ERROR: 配置不存在: $CONFIG_YAML" >&2; exit 1; }

echo ">> smoke: $PY  --config_yaml $CONFIG_YAML  (GPU=$CUDA_VISIBLE_DEVICES, swanlab=$SWANLAB_MODE)"
python "$PY" --config_yaml "$CONFIG_YAML"
