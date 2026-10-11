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
#   bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh \
#        examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d.yaml
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
  echo "示例: $0 examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d.yaml" >&2
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
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}   # 调试时可 export NCCL_DEBUG=INFO (train.sh 透传)
# expandable_segments 显著降低显存碎片,避免大 batch 下 NCCL 缓冲区 OOM
# (实测 batch 10 不加会在 profile 窗口后 NCCL OOM,加上后稳定)。可用环境变量覆盖。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---- 训练参数 (可用环境变量覆盖) ----
# 优先 NPROC_PER_NODE (训练平台多机注入), 否则探测本地卡数。
# 探测用本仓库 .venv 的 python (系统 python 常无 torch), 无 venv 再退回 python。
_PY_BIN="${REPO_ROOT}/.venv/bin/python"
[ -x "${_PY_BIN}" ] || _PY_BIN="python"
NUM_GPUS=${NUM_GPUS:-${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}
# 守卫: NUM_GPUS 超过本机实际卡数时快速失败。否则多个 rank 挤同一张卡,
# 要到 NCCL init 才炸 "Duplicate GPU detected", 报错难懂还可能挂死机器。
LOCAL_GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${LOCAL_GPU_COUNT}" -gt 0 ] && [ "${NUM_GPUS}" -gt "${LOCAL_GPU_COUNT}" ]; then
  echo "错误: NUM_GPUS=${NUM_GPUS} 超过本机实际 GPU 数 (${LOCAL_GPU_COUNT})。" >&2
  echo "      本机卡数: $(nvidia-smi -L 2>/dev/null | tr '\n' ' ')" >&2
  echo "      若要在多机跑, 请把命令放到有 ${NUM_GPUS} 卡的机器上执行 (NUM_GPUS 为每节点卡数)。" >&2
  exit 1
fi
# BATCH 默认 10: 实测吞吐甜点 (~62 samples/s, 比 batch 8 快 ~12%, peak 67.5GB)。
# 想要最大余量/零配置可改回 8 (56.8GB)。
BATCH=${BATCH:-10}
# 梯度累计: effective_batch = BATCH × NUM_GPUS × GRAD_ACCUM。增大时需同步上调
# MAX_STEPS (保 epoch 数) 和 yaml learning_rate (linear scaling ×~1.5-2), 否则 epoch 变少
# + LR 偏小, 收敛更慢。本变量只调梯度累计; LR 仍需改 yaml (train.sh 未暴露 LR 变量)。
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_EVERY=${SAVE_EVERY:-5000}
LOG_EVERY=${LOG_EVERY:-10}
WANDB_PROJECT=${WANDB_PROJECT:-pi05_z_umi_relative_6d}
PORT=${PORT:-${MASTER_PORT:-29501}}
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
# 多机共享同一 CFS 输出目录，只由节点 rank 0 写配置副本。
if [ "${RANK:-0}" = "0" ]; then
  cp "${SELF}" "${output_dir}/"
  cp "${CONFIG_YAML}" "${output_dir}/"
fi

# 禁用 WandB 进行快速测试；正式训练时删除此行并执行 `wandb login`。
# export WANDB_MODE=${WANDB_MODE:-disabled}

# accelerate 二进制解析顺序 (只认本仓库路径 / 当前环境, 不写死任何外部机器路径):
#   1. ACCEL_BIN 环境变量显式指定
#   2. repo 根目录 .venv/bin/accelerate (本仓库标准 venv 布局)
#   3. 当前 PATH 里的 accelerate (如已 source .venv/bin/activate)
# 都找不到时报错退出, 避免跑到一半才挂。
if [ -z "${ACCEL_BIN:-}" ]; then
  for _cand in "${REPO_ROOT}/.venv/bin/accelerate" "$(command -v accelerate 2>/dev/null || true)"; do
    if [ -n "${_cand}" ] && [ -x "${_cand}" ]; then ACCEL_BIN="${_cand}"; break; fi
  done
fi
if [ -z "${ACCEL_BIN:-}" ] || [ ! -x "${ACCEL_BIN}" ]; then
  echo "错误: 找不到可执行的 accelerate (试过 ${REPO_ROOT}/.venv/bin/accelerate 和 PATH)。" >&2
  echo "      请先 source .venv/bin/activate, 或显式指定: ACCEL_BIN=/path/to/accelerate bash $0 ..." >&2
  exit 1
fi

# ---- 多机开关 ----
# 默认单机: 仅 --num_processes NUM_GPUS。
# 多机: 设 NUM_MACHINES(>1) 或 MAIN_PROCESS_IP / MASTER_ADDR, 自动追加
#       --same_network / --num_machines / --machine_rank / --main_process_ip / --num_processes(总数)。
# 兼容 SLURM: MACHINE_RANK 默认取 $SLURM_NODEID; 主进程 IP 默认取 $MASTER_ADDR。
# 兼容训练平台 (多机自动注入): WORLD_SIZE→节点数, RANK→本机 rank,
#   NPROC_PER_NODE→每节点卡数, MASTER_ADDR→主进程 IP, MASTER_PORT→端口。
# 预览不启动: DRY_RUN=1 只打印 accelerate 命令。
# 注: 不加 --multi_gpu (会把 config 里的 distributed_type:DEEPSPEED 顶成 MULTI_GPU,
#     导致 ZeRO-2 失效); DeepSpeed 走 standard multinode launcher + --same_network。
NUM_MACHINES="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
MACHINE_RANK="${MACHINE_RANK:-${RANK:-${SLURM_NODEID:-0}}}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-}}"

# --gradient_accumulation_steps 走 accelerate launch 通道: 它会同时设置
# accelerator.gradient_accumulation_steps 并 (因 zerith_ds_config 里该字段为 "auto")
# 透传给 DeepSpeed engine。注意必须走 launch flag, 不能用 --trainer.xxx cfg
# 参数 —— train_starvla.py 不读 cfg.trainer.gradient_accumulation_steps。
# ACCEL_CONFIG 可覆盖 accelerate 入口配置 (如 zerith_deepspeed_zero2_fast.yaml:
# bf16 梯度通信 + 小 bucket, 修 8 卡下梯度同步与 backward 零重叠的问题)。
ACCEL_CONFIG="${ACCEL_CONFIG:-starVLA/config/deepseeds/zerith_deepspeed_zero2.yaml}"
ACCEL_FLAGS=(--config_file "${ACCEL_CONFIG}" --main_process_port "${PORT}" --gradient_accumulation_steps "${GRAD_ACCUM}")
if [ "${NUM_MACHINES}" -gt 1 ] || [ -n "${MAIN_PROCESS_IP}" ]; then
  TOTAL_GPUS="${TOTAL_GPUS:-$((NUM_MACHINES * NUM_GPUS))}"
  ACCEL_FLAGS+=(--same_network --num_machines "${NUM_MACHINES}" --machine_rank "${MACHINE_RANK}" --num_processes "${TOTAL_GPUS}")
  [ -n "${MAIN_PROCESS_IP}" ] && ACCEL_FLAGS+=(--main_process_ip "${MAIN_PROCESS_IP}")
  echo ">> 多机模式: num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} total_gpus=${TOTAL_GPUS} main_process_ip=${MAIN_PROCESS_IP:-<未设>}"
else
  ACCEL_FLAGS+=(--num_processes "${NUM_GPUS}")
  echo ">> 单机模式: num_processes=${NUM_GPUS}"
fi

# ── STAGE_SHM: 数据集预拷到 /dev/shm (内存盘) ────────────────────────────
# CFS 上的视频解码实测 ~60-80ms/样本 (供给赤字 → 周期性 20-30s 停顿);
# 拷入 RAM 后解码读 I/O 变内存速度, worker 供给能力翻数倍。适用于数据集
# 总量 << 节点 RAM 的情况 (umi 741552b7 1.28GB; 大数据集勿用, 看 STAGE_SHM_MAX_GB)。
STAGE_SHM="${STAGE_SHM:-0}"
STAGE_SHM_MAX_GB="${STAGE_SHM_MAX_GB:-20}"
DATA_ROOT_OVERRIDE=""
if [ "${STAGE_SHM}" = "1" ]; then
  PYBIN="${ACCEL_BIN%/*}/python"   # venv python (系统 python3 无 omegaconf)
  SRC_ROOT=$("${PYBIN}" -c "from omegaconf import OmegaConf; print(OmegaConf.load('${CONFIG_YAML}').datasets.vla_data.data_root_dir)")
  SRC_MIX=$("${PYBIN}" -c "from omegaconf import OmegaConf; print(OmegaConf.load('${CONFIG_YAML}').datasets.vla_data.data_mix)")
  SIZE_GB=$(du -sm "${SRC_ROOT}" 2>/dev/null | cut -f1); SIZE_GB=$((SIZE_GB / 1024))
  if [ "${SIZE_GB}" -le "${STAGE_SHM_MAX_GB}" ] && [ -n "${SRC_ROOT}" ] && [ -d "${SRC_ROOT}" ]; then
    DST="/dev/shm$(echo ${SRC_ROOT} | sed 's|/root||')"
    mkdir -p "${DST}" && rsync -a "${SRC_ROOT}/" "${DST}/" && echo ">> STAGE_SHM: ${SRC_ROOT} (${SIZE_GB}GB) → ${DST}"
    DATA_ROOT_OVERRIDE="--datasets.vla_data.data_root_dir ${DST}"
  else
    echo ">> STAGE_SHM: 跳过 (${SIZE_GB}GB > 上限 ${STAGE_SHM_MAX_GB}GB 或路径无效)"
  fi
fi

set -- "${ACCEL_BIN}" launch "${ACCEL_FLAGS[@]}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.prefetch_factor "${PREFETCH}" \
  ${DATA_ROOT_OVERRIDE} \
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
