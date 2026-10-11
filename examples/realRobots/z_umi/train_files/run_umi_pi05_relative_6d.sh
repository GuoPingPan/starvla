#!/usr/bin/env bash
# UMI · PI05 通用训练启动脚本（一个 bash + 任意 yaml 配置）
#
# 想换训练配置，只需换 yaml，不再需要为每个配置复制一份 bash。
#
# 用法:
#   bash run_umi_pi05_relative_6d.sh <config.yaml> [run_tag]
#
# 示例:
#   # 1) 训练某一配置（从 repo root 或任意目录均可）
#   bash examples/realRobots/z_umi/train_files/run_umi_pi05_relative_6d.sh \
#        examples/realRobots/z_umi/train_files/run_umi_shenzhen_relative_6d.yaml
#
#   # 2) 同一配置跑多组实验：传 run_tag 区分输出目录
#   bash run_umi_pi05_relative_6d.sh run_umi_shenzhen_relative_6d.yaml exp2
#
#   # 3) 用环境变量临时改训练参数
#   MAX_STEPS=30000 BATCH=4 bash run_umi_pi05_relative_6d.sh run_umi_shenzhen_relative_6d.yaml
#
# run_id 默认由 yaml 文件名推导（不同配置互不覆盖），可用 RUN_ID 整体覆盖。
# 可覆盖的环境变量: NUM_GPUS BATCH MAX_STEPS SAVE_EVERY LOG_EVERY WANDB_PROJECT PORT RUN_ID
#
# 无需离线预处理：norm 统计在训练首次启动时在线计算并缓存到
#   meta/stats_gr00t.json（按 norm_stride 失效重算）；缺失的 meta/modality.json
#   也会由 UMILeRobotDataset 自动从 umi_modality_template.json 生成。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <config.yaml> [run_tag]" >&2
  echo "示例: $0 examples/realRobots/z_umi/train_files/run_umi_shenzhen_relative_6d.yaml" >&2
  exit 1
fi

# 脚本自身路径（cd 到 repo root 后仍可拷贝到输出目录）
SELF="$(readlink -f "$0")"

# ---- 解析参数 ----
CONFIG_YAML="$1"
RUN_TAG="${2:-}"

# yaml 解析为绝对路径，之后 cd 到 repo root 仍可访问
CONFIG_YAML="$(readlink -f "${CONFIG_YAML}")"
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "错误: 找不到配置文件 '$1'（解析为 ${CONFIG_YAML}）" >&2
  exit 1
fi

# 切到 repo root，使 starVLA/... 相对路径生效（可在任意目录调用本脚本）
# train.sh 等外部可预设 REPO_ROOT (如 zip 解压目录无 .git 时); 否则取 git 根。
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${REPO_ROOT}"

# DeepSpeed 需要真实的 nvcc。登录节点的存根 $HOME/.local/bin/nvcc 不够用
# — 将 CUDA_HOME 指向系统的 CUDA toolkit 并将其 bin 目录添加到 PATH。
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
export NCCL_DEBUG=WARN
# expandable_segments 显著降低显存碎片,避免大 batch 下 NCCL 缓冲区 OOM
# (实测 batch 10 不加会在 profile 窗口后 NCCL OOM,加上后稳定)。可用环境变量覆盖。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---- 训练参数 (可用环境变量覆盖) ----
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
# BATCH 默认 10: 实测吞吐甜点 (~62 samples/s, 比 batch 8 快 ~12%, peak 67.5GB)。
# 想要最大余量/零配置可改回 8 (56.8GB)。
BATCH=${BATCH:-10}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_EVERY=${SAVE_EVERY:-5000}
LOG_EVERY=${LOG_EVERY:-10}
WANDB_PROJECT=${WANDB_PROJECT:-pi05_z_umi_relative_6d}
PORT=${PORT:-29501}
# 数据加载: 开 compile 后计算变快, num_workers=4 的视频解码喂不饱 GPU (data_time 飙到
# ~1s)。本机 128 核, 默认 8 worker × 8 卡 = 64 进程, 余量足。仍饿可加到 12/16。
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH=${PREFETCH:-4}

# run_id 默认 = <yaml_stem>[_<run_tag>]_<MAX_STEPS>step；可用 RUN_ID 整体覆盖
yaml_stem="$(basename "${CONFIG_YAML}" .yaml)"
run_id="${RUN_ID:-${yaml_stem}${RUN_TAG:+_${RUN_TAG}}_${MAX_STEPS}step}"

run_root_dir=/root/data/model_outputs/   # 经 /root/data 软链落到 /nix
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "${SELF}" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

# 禁用 WandB 进行快速测试；正式训练时删除此行并执行 `wandb login`。
# export WANDB_MODE=${WANDB_MODE:-disabled}

# accelerate 二进制: 默认开发机 worktree venv; train.sh (用 uv 建的 env) 可经 ACCEL_BIN 覆盖
ACCEL_BIN="${ACCEL_BIN:-/root/workspace/vla/starvla.worktrees/dev-main/.venv/bin/accelerate}"

# ---- 多机开关 ----
# 默认单机: 仅 --num_processes NUM_GPUS。
# 多机: 设 NUM_MACHINES(>1) 或 MAIN_PROCESS_IP / MASTER_ADDR, 自动追加
#       --num_machines / --machine_rank / --main_process_ip / --num_processes(总数)。
# 兼容 SLURM: MACHINE_RANK 默认取 $SLURM_NODEID; 主进程 IP 默认取 $MASTER_ADDR。
# 预览不启动: DRY_RUN=1 只打印 accelerate 命令。
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-${SLURM_NODEID:-0}}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-}}"

ACCEL_FLAGS=(--config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --main_process_port "${PORT}")
if [ "${NUM_MACHINES}" -gt 1 ] || [ -n "${MAIN_PROCESS_IP}" ]; then
  TOTAL_GPUS="${TOTAL_GPUS:-$((NUM_MACHINES * NUM_GPUS))}"
  ACCEL_FLAGS+=(--num_machines "${NUM_MACHINES}" --machine_rank "${MACHINE_RANK}" --num_processes "${TOTAL_GPUS}")
  [ -n "${MAIN_PROCESS_IP}" ] && ACCEL_FLAGS+=(--main_process_ip "${MAIN_PROCESS_IP}")
  echo ">> 多机模式: num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} total_gpus=${TOTAL_GPUS} main_process_ip=${MAIN_PROCESS_IP:-<未设>}"
else
  ACCEL_FLAGS+=(--num_processes "${NUM_GPUS}")
  echo ">> 单机模式: num_processes=${NUM_GPUS}"
fi

set -- "${ACCEL_BIN}" launch "${ACCEL_FLAGS[@]}" \
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
  --wandb_project "${WANDB_PROJECT}"

echo ">> CMD: $*"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "(DRY_RUN=1, 不实际启动)"; exit 0
fi
"$@"
