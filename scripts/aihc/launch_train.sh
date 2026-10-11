#!/usr/bin/env bash
# ============================================================================
# launch_train.sh — 百舸 (AIHC) H800 训练任务一键提交
#
# 以正式跑 (qwenvla-test-h100-copy18) 的任务规格为模板, 参数化生成 job JSON
# 后经 aihc CLI 提交。免控制台手点。
#
# 用法 (在装有 aihc CLI 且已 aihc config 配好凭证的机器上):
#   bash scripts/aihc/launch_train.sh --run-id my_run --yaml <config.yaml> [选项]
#
# 常用选项:
#   --run-id ID          RUN_ID (必填, 模型输出目录名)
#   --yaml PATH          训练 yaml 相对 repo_30a45b2b 的路径 (必填, 与 --command 二选一)
#   --command STR        完整自定义命令 (与 --yaml 二选一, 如 canary 探针)
#   --name NAME          任务名 (默认 <run_id>-<MMdd_HHmmss>)
#   --replicas N         机器数 (默认 1)
#   --gpus N             每机 GPU 数 (默认 8)
#   --batch N            每卡 batch (默认 32)
#   --workers N          dataloader workers (默认 8)
#   --prefetch N         prefetch_factor (默认 8)
#   --max-steps N        默认 200
#   --save-every N       默认 100000 (=不中途存 ckpt)
#   --grad-accum N       默认 1
#   --commit HASH        train.sh 的 COMMIT 参数 (默认 3c768d9)
#   --local              train.sh 追加 --local
#   --swanlab-key KEY    注入 SWANLAB_API_KEY (缺省读环境变量, 再缺省不注入)
#   --priority P         low/normal/high (默认 normal)
#   --cpu N / --memory N 单机 CPU 核数 / 内存 GB (默认按 GPU 数缩放)
#   --env K=V            追加 export 行 (可多次, 如 --env STARVLA_PROFILE=1)
#   --dry-run            只渲染 JSON 到 stdout, 不提交
#
# 示例:
#   # 正式 50k 步 4 机:
#   bash scripts/aihc/launch_train.sh --run-id h1_d18480a6_qwenpi_v3_first \
#     --yaml examples/data-analyze/train_files/run_h1_d18480a6_qwenpi_v3.yaml \
#     --replicas 4 --max-steps 50000 --save-every 5000
#   # canary 探针 (1 机 1 卡跑 30s 验证链路):
#   bash scripts/aihc/launch_train.sh --run-id canary --command "hostname; nvidia-smi -L; sleep 30" \
#     --replicas 1 --gpus 1 --name orion-launcher-canary
# ============================================================================
set -euo pipefail

POOL="aihc-serverless"
QUEUE="aihcq-gwzuewk6egoh"
IMAGE="ccr-2owfeef4-vpc.cnc.bj.baidubce.com/orion/base_trian:0819new"
GPU_TYPE="baidu.com/h800_80g_cgpu"

RUN_ID="" YAML="" COMMAND=""
NAME="" REPLICAS=1 GPUS=8 BATCH=32 WORKERS=8 PREFETCH=8
MAX_STEPS=200 SAVE_EVERY=100000 GRAD_ACCUM=1 COMMIT="3c768d9"
PRIORITY="normal" CPU="" MEMORY="" LOCAL=0 DRY_RUN=0
EXTRA_ENVS=()
SWANLAB_KEY="${SWANLAB_API_KEY:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2;;
    --yaml) YAML="$2"; shift 2;;
    --command) COMMAND="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --replicas) REPLICAS="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --gpu-type) GPU_TYPE="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --prefetch) PREFETCH="$2"; shift 2;;
    --max-steps) MAX_STEPS="$2"; shift 2;;
    --save-every) SAVE_EVERY="$2"; shift 2;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2;;
    --commit) COMMIT="$2"; shift 2;;
    --swanlab-key) SWANLAB_KEY="$2"; shift 2;;
    --priority) PRIORITY="$2"; shift 2;;
    --cpu) CPU="$2"; shift 2;;
    --memory) MEMORY="$2"; shift 2;;
    --env) EXTRA_ENVS+=("$2"); shift 2;;
    --local) LOCAL=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    *) echo "未知参数: $1 (看脚本头注释)" >&2; exit 1;;
  esac
done

if [ -z "$COMMAND" ]; then
  [ -n "$RUN_ID" ] && [ -n "$YAML" ] || { echo "--run-id 和 --yaml 必填 (或用 --command)" >&2; exit 1; }
  LOCAL_FLAG=""; [ "$LOCAL" = 1 ] && LOCAL_FLAG=" --local"
  ENV_LINES=""
  for kv in "${EXTRA_ENVS[@]:-}"; do [ -n "$kv" ] && ENV_LINES="${ENV_LINES}export ${kv}
"; done
  # ── train.sh 来源: CFS (与镜像解耦) ──────────────────────────────────────
  # 提交时把仓库当前 scripts/train.sh + cloud_io.py 原子同步到 /mnt/cfs/code/,
  # 任务容器直接执行 CFS 上的版本, 不再依赖镜像内置的 bootstrap (旧镜像也能跑新流水线)。
  # 前提: 提交机挂载了 CFS (/mnt/cfs); 可用 TRAIN_SH_DIR 覆盖目标目录。
  REPO_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  TRAIN_SH_DIR="${TRAIN_SH_DIR:-/mnt/cfs/code}"
  if [ ! -d "$TRAIN_SH_DIR" ] || [ ! -w "$TRAIN_SH_DIR" ]; then
    echo "ERROR: $TRAIN_SH_DIR 不存在或不可写 (提交机需挂载 CFS) — 队列启动依赖 CFS 上的 train.sh" >&2
    exit 1
  fi
  cp "$REPO_SCRIPTS/train.sh" "$TRAIN_SH_DIR/.train.sh.tmp" && mv -f "$TRAIN_SH_DIR/.train.sh.tmp" "$TRAIN_SH_DIR/train.sh"
  cp "$REPO_SCRIPTS/cloud_io.py" "$TRAIN_SH_DIR/.cloud_io.py.tmp" && mv -f "$TRAIN_SH_DIR/.cloud_io.py.tmp" "$TRAIN_SH_DIR/cloud_io.py"
  chmod +x "$TRAIN_SH_DIR/train.sh"
  COMMAND=$(cat <<EOF
${ENV_LINES}${SWANLAB_KEY:+export SWANLAB_API_KEY=$SWANLAB_KEY
}export MAX_STEPS=$MAX_STEPS
export GRAD_ACCUM=$GRAD_ACCUM
CLOUD=bos NUM_GPUS=$GPUS BATCH=$BATCH NUM_WORKERS=$WORKERS PREFETCH=$PREFETCH SAVE_EVERY=$SAVE_EVERY \\
  RUN_ID=$RUN_ID \\
  OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  bash /mnt/cfs/code/train.sh \\
  $YAML $COMMIT$LOCAL_FLAG
EOF
)
fi
[ -n "$RUN_ID" ] || RUN_ID="custom"

[ -n "$NAME" ] || NAME="$(echo "$RUN_ID" | tr '[:upper:]' '[:lower:]' | cut -c1-40)-$(date +%m%d_%H%M%S)"
# cpu/memory 固定 175/2767: 与 copy18 (正式跑) 完全一致, 按节点粒度申请;
# 自行缩放会产生 176/2768 之类非标准值, 调度器无法匹配节点。
CPU="${CPU:-175}"
MEMORY="${MEMORY:-2767}"

# hang-detection 的"无日志超时": 冷 torch.compile 期间可静默 10+ 分钟, 默认 8 会把
# 正在编译的进程误判为挂起并杀掉。HANG_LOG_TIMEOUT_MIN 可放宽 (调试/冷缓存首跑用)。
HANG_LOG_TIMEOUT_MIN="${HANG_LOG_TIMEOUT_MIN:-8}"
HANG_STARTUP_TOLERATION_MIN="${HANG_STARTUP_TOLERATION_MIN:-10}"

JOB_JSON="${TMPDIR:-/tmp}/aihc_job_${NAME}.json"
export POOL QUEUE IMAGE GPU_TYPE NAME REPLICAS GPUS BATCH CPU MEMORY PRIORITY COMMAND
export HANG_LOG_TIMEOUT_MIN HANG_STARTUP_TOLERATION_MIN
python3 - <<'PYEOF' > "$JOB_JSON"
import json, os
env = os.environ
job = {
    "name": env["NAME"],
    "queue": env["QUEUE"],
    "jobFramework": "pytorch",
    "jobSpec": {
        "command": env["COMMAND"],
        "image": env["IMAGE"],
        "imageConfig": {"username": "", "password": ""},
        "replicas": int(env["REPLICAS"]),
        "resources": [
            {"name": env["GPU_TYPE"], "quantity": int(env["GPUS"])},
            {"name": "cpu", "quantity": int(env["CPU"])},
            {"name": "memory", "quantity": int(env["MEMORY"])},
            {"name": "rdma/ehca", "quantity": 1},
            {"name": "rdma/hca", "quantity": 1},
            {"name": "sharedMemory", "quantity": 0},
        ],
        "envs": [
            {"name": "AIHC_JOB_NAME", "value": env["NAME"]},
            {"name": "LOG_COLLECTION", "value": "true"},
            {"name": "NCCL_IB_DISABLE", "value": "0"},
        ],
        "enableRDMA": True,
    },
    "faultTolerance": True,
    "labels": [
        {"key": "aihc.baidubce.com/userid", "value": "248b9fd4247d42c4b4cf3a564c83ebb0"},
        {"key": "aijob.cce.baidubce.com/ai-user-id", "value": "248b9fd4247d42c4b4cf3a564c83ebb0"},
        {"key": "aijob.cce.baidubce.com/ai-user-name", "value": "Orion"},
        {"key": "aijob.cce.baidubce.com/create-from-aihcp", "value": "true"},
        {"key": "aijob.cce.baidubce.com/serverless-job", "value": "true"},
        {"key": "aihc.baidubce.com/username", "value": "Orion"},
        {"key": "aihc.baidubce.com/workload-type", "value": "train"},
        {"key": "aijob.cce.baidubce.com/log-collection", "value": "v3"},
    ],
    "priority": env["PRIORITY"],
    "datasources": [
        {"id": "", "type": "cfs", "sourcePath": "/", "mountPath": "/mnt/cfs", "name": "",
         "options": {"sizeLimit": 0, "medium": "", "readOnly": False,
                     "cfsInstanceId": "cfs-pVlRwhavQO",
                     "cfsMountPoint": "cfs-pVlRwhavQO.lb-2etad1gk.cfs.bj.baidubce.com"}},
    ],
    "faultToleranceArgs": (
        "--enable-hang-detection=true "
        f"--hang-detection-log-timeout-minutes={env['HANG_LOG_TIMEOUT_MIN']} "
        f"--hang-detection-startup-toleration-minutes={env['HANG_STARTUP_TOLERATION_MIN']} "
        "--hang-detection-stack-timeout-minutes=5 --max-num-of-unconditional-retry=0"
    ),
}
json.dump(job, open(1, "w"), ensure_ascii=False, indent=2)
PYEOF

if [ "$DRY_RUN" = 1 ]; then
  echo "=== DRY-RUN: $JOB_JSON ===" >&2
  cat "$JOB_JSON"
  exit 0
fi

echo ">> 提交任务: $NAME (replicas=$REPLICAS gpus=$GPUS)" >&2
aihc job create -f "$JOB_JSON" -p "$POOL" -q "$QUEUE"
