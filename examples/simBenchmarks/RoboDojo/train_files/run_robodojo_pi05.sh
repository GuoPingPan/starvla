#!/usr/bin/env bash
# RoboDojo · PI05 训练启动脚本（适配自 zerith_pi05，单卡 A100）。
#
# 用法:
#   bash examples/simBenchmarks/RoboDojo/train_files/run_robodojo_pi05.sh [config.yaml] [run_tag]
#
# 可覆盖的环境变量:
#   TASK_INDEX    单任务训练的任务 id（默认 24=Pour Liquid Into Cup；设为 "all" 多任务）
#   NUM_GPUS      GPU 数（默认 1）
#   BATCH         per_device_batch_size（默认 4）
#   MAX_STEPS     训练步数（默认 10000）
#   SAVE_EVERY    保存间隔（默认 2500）
#   WANDB_PROJECT PORT
set -euo pipefail

if [[ $# -lt 1 ]]; then
  CONFIG_YAML="examples/simBenchmarks/RoboDojo/train_files/robodojo_pi05.yaml"
else
  CONFIG_YAML="$1"
fi
RUN_TAG="${2:-}"
CONFIG_YAML="$(readlink -f "${CONFIG_YAML}")"
[[ -f "${CONFIG_YAML}" ]] || { echo "错误: 找不到配置 $CONFIG_YAML" >&2; exit 1; }

SELF="$(readlink -f "$0")"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
# editable 安装指向主仓库；显式把 worktree 放到 PYTHONPATH 最前，使 starVLA + examples
# 解析到本 worktree（registry 才能发现 RoboDojo 的 data_config）。
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

NUM_GPUS=${NUM_GPUS:-1}
BATCH=${BATCH:-4}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_EVERY=${SAVE_EVERY:-2500}
LOG_EVERY=${LOG_EVERY:-10}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH=${PREFETCH:-2}
PORT=${PORT:-29501}
WANDB_PROJECT=${WANDB_PROJECT:-robodojo_pi05}
TASK_INDEX=${TASK_INDEX:-24}

yaml_stem="$(basename "${CONFIG_YAML}" .yaml)"
run_id="${RUN_ID:-${yaml_stem}${RUN_TAG:+_${RUN_TAG}}_task${TASK_INDEX}_${MAX_STEPS}step}"

run_root_dir=/nix/data/model_outputs/
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "${SELF}" "${output_dir}/" 2>/dev/null || true
cp "${CONFIG_YAML}" "${output_dir}/"

# 禁用 WandB（快速测试）；正式训练删此行并 wandb login
export WANDB_MODE=${WANDB_MODE:-disabled}

# 任务过滤覆盖：TASK_INDEX=all 时多任务（不传 task_index）
TASK_OVERRIDE=()
if [[ "${TASK_INDEX}" != "all" ]]; then
  TASK_OVERRIDE=(--datasets.vla_data.task_index "${TASK_INDEX}")
fi

PYTHON=/root/workspace/vla/starvla/.venv/bin/python
ACCEL=/root/workspace/vla/starvla/.venv/bin/accelerate

"${ACCEL}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.prefetch_factor "${PREFETCH}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${WANDB_PROJECT}" \
  "${TASK_OVERRIDE[@]}"
