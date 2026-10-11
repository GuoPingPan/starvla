#!/usr/bin/env bash
# RoboDojo · PI05 开环评测 launcher（适配自 z_umi）。
#
# 用法:
#   bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh [run-dir] [extra args...]
#
# 示例:
#   # smoke（2 episode, 每 episode 5 anchor）
#   bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh <run-dir> --episodes 2400 2401 --max-anchors 5
#   # 全量评测
#   bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh <run-dir>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}

RUN_DIR="${1:-}"
shift || true
[[ -d "${RUN_DIR}" ]] || { echo "用法: $0 <run-dir> [extra args]. run-dir 不存在: ${RUN_DIR}" >&2; exit 1; }

EVAL_YAML="examples/simBenchmarks/RoboDojo/eval_files/eval_robodojo_pi05_overlay.yaml"
DATA_ROOT_OVERRIDE=${DATA_ROOT_OVERRIDE:-/nix/data/robodojo}

PYTHON=/root/workspace/vla/starvla/.venv/bin/python

"${PYTHON}" -m starVLA.training.eval_starvla \
  --run-dir "${RUN_DIR}" \
  --eval-yaml "${EVAL_YAML}" \
  --data-root-override "${DATA_ROOT_OVERRIDE}" \
  "$@"
