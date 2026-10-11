#!/usr/bin/env bash
# ============================================================================
# scripts/train.sh — 任务队列直接拉起的训练流水线 (UMI · PI05) · boto3 版
# ----------------------------------------------------------------------------
#   命令行参数 (均可选): bash train.sh [config.yaml] [commit] [--local]
#     不传则回退环境变量/默认值; CONFIG_YAML 与 COMMIT 已改为位置参数解析。
#   6 段: 配置环境 → 拉代码 → 拉数据 → 拉预训练 → 训练 → 上传全部产物
#   BOS 读写走 scripts/bos_io.py (boto3 / S3 兼容, 凭证从环境变量读, 不入库)。
#   所有产物落 /root 下的软链 (-> /mnt/cfs 共享盘), 方便查看。
# ----------------------------------------------------------------------------
#   前置 (主机/队列机):
#     · export BOS_ACCESS_KEY_ID / BOS_SECRET_ACCESS_KEY   (或写进 /root/.bosenv, 本脚本会 source)
#     · BOS_ENDPOINT 默认 s3.bj.bcebos.com ; BOS_BUCKET 默认 umi-gripper ; BOS_NAMESPACE 默认 Orion
#     · uv 在 PATH (本机 /usr/local/bin/uv)
#     · NFS 客户端工具 (仅首次, 提供 /sbin/mount.nfs4 helper):
#         sudo apt update && sudo apt install -y nfs-common
#   注: 默认 CONFIG_YAML=shenzhen (目前唯一配了 BOS 数据路径的); 跑 hefei 前需在
#       data_config.py 给 hefei_relative_6d 补第4个元素 bos:// URI。
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOS_IO="$SCRIPT_DIR/bos_io.py"
DL_DATA="$SCRIPT_DIR/download_data.py"

# ── /mnt/cfs 挂载 (NFS 共享盘, 百度云 CFS) ────────────────────────────────────
# 数据/代码/预训练全部落这块共享盘 (替代原 /nix 本地盘, /nix 有问题)。
# 前置 (仅首次, 装 NFS 客户端 helper /sbin/mount.nfs4):
#     sudo apt update && sudo apt install -y nfs-common
# 已挂载则跳过; 重复运行安全。
# if mountpoint -q /mnt/cfs; then
#   echo ">> /mnt/cfs 已挂载, 跳过 mount"
# else
#   mkdir -p /mnt/cfs
#   mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport \
#     cfs-pVlRwhavQO.lb-2etad1gk.cfs.bj.baidubce.com:/ /mnt/cfs
# fi

# ── 命令行参数 (可选) ──────────────────────────────────────────────────────────
#   用法: bash train.sh [config.yaml] [commit] [--local]
#     $1       CONFIG_YAML  训练配置 (相对 repo root); 默认 shenzhen 配置
#     $2       COMMIT       源码产物 commit sha (任意位数前缀均可); 默认 latest (取 starvla_res 最新 zip)
#     --local  SwanLab 切本地模式 (默认云端 online, 需 swanlab login); 等价 SWANLAB_MODE=local
#   位置参数优先; 留空回退环境变量, 再回退默认值 (向后兼容旧调用)。--local 可出现在任意位置。
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "用法: $0 [config.yaml] [commit] [--local]" >&2
  echo "  config.yaml  训练配置 (相对 repo root), 默认 shenzhen 配置" >&2
  echo "  commit       源码产物 commit sha (任意位数前缀) 或 latest, 默认 latest" >&2
  echo "  --local      SwanLab 用本地模式 (默认云端 online, 需先 swanlab login)" >&2
  exit 0
fi
# 提取 --local flag (可出现在任意位置); 剩余参数作为 config/commit 位置参数
SWANLAB_LOCAL=0
_pos=()
for _a in "$@"; do
  case "$_a" in
    --local) SWANLAB_LOCAL=1 ;;
    *) _pos+=("$_a") ;;
  esac
done
set -- "${_pos[@]}"
[ "$SWANLAB_LOCAL" = 1 ] && export SWANLAB_MODE=local
# ── 顶部可调项 ────────────────────────────────────────────────────────────────
CONFIG_YAML="${1:-${CONFIG_YAML:-examples/realRobots/z_umi/train_files/run_umi_shenzhen_relative_6d.yaml}}"
COMMIT="${2:-${COMMIT:-latest}}"          # commit sha (任意位数前缀); latest=取 starvla_res 最新 zip
# 训练超参 (与 run_umi_pi05_relative_6d.sh 默认一致)
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"   # 默认全部可见 GPU; 可 env 覆盖
BATCH="${BATCH:-10}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
PORT="${PORT:-29501}"
WANDB_PROJECT="${WANDB_PROJECT:-starvla}"
# 所有产物走 /root 下的软链 (-> /mnt/cfs 共享盘), 方便查看
DATA_DIR="/root/data"           # -> /mnt/cfs/data (含 z-umi/, model_outputs/)
CODE_DIR="/root/code"           # -> /mnt/cfs/code
RUN_ROOT_DIR="${DATA_DIR}/model_outputs"
PRETRAINED_LOCAL="${DATA_DIR}/pretrain_models"

# ============================================================================
# 1. 配置环境
# ============================================================================
# BOS 凭证: 优先环境里已有; 否则 source 一个本地(不入库)文件
if [ -z "${BOS_ACCESS_KEY_ID:-}" ] && [ -f "${BOS_ENV_FILE:-/root/.bosenv}" ]; then
  set -a; . "${BOS_ENV_FILE:-/root/.bosenv}"; set +a
fi
# SwanLab 凭证: 优先环境里已有; 否则 source 一个本地(不入库)文件 (与 BOS 同模式)。
# SWANLAB_API_KEY 通常只写在 ~/.zshrc, 而队列/后台用非交互 bash 拉起 train.sh 时
# 不会 source ~/.zshrc → 训练子进程拿不到 key → logger.py (b8a8bba) 的无 TTY 非交互
# login 失效, 退化成 "no TTY for interactive login" → continuing without tracker。
# 凭据文件路径可被 SWANLAB_ENV_FILE 覆盖 (例如指向 /mnt/cfs 共享盘, 让所有队列节点共用)。
if [ -z "${SWANLAB_API_KEY:-}" ] && [ -f "${SWANLAB_ENV_FILE:-/root/.swanlab_env}" ]; then
  set -a; . "${SWANLAB_ENV_FILE:-/root/.swanlab_env}"; set +a
fi
export BOS_ENDPOINT="${BOS_ENDPOINT:-s3.bj.bcebos.com}"
export BOS_BUCKET="${BOS_BUCKET:-umi-gripper}"
export BOS_NAMESPACE="${BOS_NAMESPACE:-Orion}"

# 路径归一: /root/data -> /mnt/cfs/data, /root/code -> /mnt/cfs/code (软链目标必须是真实 /mnt/cfs 路径)
mkdir -p /mnt/cfs/data/z-umi /mnt/cfs/code
for _l in /root/data:/mnt/cfs/data /root/code:/mnt/cfs/code; do
  link="${_l%%:*}"; target="${_l#*:}"
  if [ -L "$link" ]; then :
  elif [ -e "$link" ]; then echo "ERROR: $link 已存在且非软链, 拒绝覆盖" >&2; exit 1
  else ln -s "$target" "$link"; fi
done

# CUDA / NCCL / HF / WandB
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=10000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-disabled}"   # 调试用; 正式训练删此行 + wandb login

# ============================================================================
# 2. 拉取代码 (按 commit 从 BOS 取 CI 产物 zip -> 解压 -> uv 建运行环境)
#    并发安全 (多任务共享 /mnt/cfs/code):
#      · zip 下到唯一临时目录, 不互相覆盖下载中的产物
#      · 解压目录按真实 commit sha 命名 repo_<sha>, 不同 commit 任务互不覆盖
#      · flock 串行化同 commit 的解压+uv sync; repo_<sha>/.venv 已存在则直接复用
#    注: 每次都会下载 zip (latest 需下载后才能解析真实 sha); 复用靠 repo_<sha>/.venv。
# ============================================================================
mkdir -p "$CODE_DIR"
DL_TMP="$(mktemp -d "$CODE_DIR/.srczip.XXXXXX")"
trap 'rm -rf "$DL_TMP" 2>/dev/null || true' EXIT INT TERM
# 此刻还没建好环境, 用 uv run --with boto3 临时带上 boto3 来跑 bos_io
uv run --no-project --with boto3 python "$BOS_IO" src-zip "$COMMIT" --dest-dir "$DL_TMP/"
ZIP="$(ls -t "$DL_TMP"/starvla_*.zip 2>/dev/null | head -1)"
[ -n "$ZIP" ] || { echo "ERROR: 没拉到源码 zip (先跑 GitLab CI 上传产物到 starvla_res/)" >&2; exit 1; }
# zip 命名 starvla_<ref>_<ts>_<sha>.zip -> 末段为真实 commit sha (与 bos_io.fetch_src_zip 解析一致)
REAL_SHA="$(basename "$ZIP" .zip | awk -F_ '{print $NF}' | tr 'A-Z' 'a-z')"
REPO_DIR="$CODE_DIR/repo_$REAL_SHA"
# flock: 同 commit 并发任务排队解压+建 venv, 避免竞态; 双重检查 .venv 复用已建环境
(
  flock 9
  if [ -d "$REPO_DIR/.venv" ]; then
    echo ">> [段2] 复用已有环境 $REPO_DIR/.venv (跳过解压 + uv sync)"
  else
    rm -rf "$REPO_DIR" && mkdir -p "$REPO_DIR"
    unzip -q "$ZIP" -d "$REPO_DIR"
    ( cd "$REPO_DIR" && uv sync --python 3.10 )
  fi
) 9>"$CODE_DIR/.repo_$REAL_SHA.lock"
rm -rf "$DL_TMP"; trap - EXIT INT TERM
cd "$REPO_DIR"
. .venv/bin/activate
# bos_io.py 需要 boto3; pyproject 里它在 sagemaker extra (uv sync 默认不装), 这里补装
uv pip install --quiet boto3
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 段3+ 改用「刚解压的 commit」的 scripts/bos_io.py + download_data.py（即 $PWD/scripts/），
# 不再用调用方 train.sh 旁边的版本。这样下载/训练代码同源（都是 <commit> 的），
# 避免：调用方 checkout 是旧分支（如 dev/train_anylze）→ 旧 download_data.py 没有
# data_config 自动发现 → 误读 z_umi data_config → "data_mix 不在 DATASET_NAMED_MIXTURES"。
# 段2 拉 zip 时 REPO_DIR 还不存在，已用原 $SCRIPT_DIR/bos_io.py，不受影响。
BOS_IO="$PWD/scripts/bos_io.py"
DL_DATA="$PWD/scripts/download_data.py"

# ============================================================================
# 3. 拉取数据 (按 yaml + data_config 从 BOS 拉 -> $DATA_DIR/z-umi/data)
# ============================================================================
python "$DL_DATA" "$CONFIG_YAML"

# ============================================================================
# 4. 拉取预训练 (Orion/pretrained/ -> $PRETRAINED_LOCAL)
# ============================================================================
mkdir -p "$PRETRAINED_LOCAL"
python "$BOS_IO" get-dir "bos://${BOS_BUCKET}/${BOS_NAMESPACE}/pretrained/" "$PRETRAINED_LOCAL" \
  --exclude '.hfd/*'

# ============================================================================
# 5. 训练 (复用 launcher: bash run_umi_pi05_relative_6d.sh <yaml>, 不重写其逻辑)
# ============================================================================
yaml_stem="$(basename "$CONFIG_YAML" .yaml)"
RUN_ID="${RUN_ID:-${yaml_stem}_${MAX_STEPS}step}"    # 与 launcher 默认一致; 段6 上传用它
# 导出 launcher 读取的训练参数 + 指向 uv 建的 env / 解压代码目录
# (launcher 已支持 REPO_ROOT / ACCEL_BIN 覆盖, 默认值不变)
export RUN_ID MAX_STEPS BATCH SAVE_EVERY LOG_EVERY NUM_WORKERS PREFETCH PORT WANDB_PROJECT NUM_GPUS
export REPO_ROOT="$PWD"                              # 已 cd 到 $CODE_DIR/repo (zip 解压, 无 .git)
export ACCEL_BIN="$PWD/.venv/bin/accelerate"         # uv sync 建的 venv
echo ">> [训练] RUN_ID=${RUN_ID}  (复用 launcher: run_umi_pi05_relative_6d.sh)"
bash examples/realRobots/z_umi/train_files/run_umi_pi05_relative_6d.sh "$CONFIG_YAML"
output_dir="${RUN_ROOT_DIR}/${RUN_ID}"               # launcher 把产物写这里, 段6 上传

# ============================================================================
# 6. 上传全部产物 (整个 run 目录 -> Orion/ckpts/<RUN_ID>/, 防训练中断后无法恢复)
# ============================================================================
echo ">> [上传] ${output_dir} -> bos://${BOS_BUCKET}/${BOS_NAMESPACE}/ckpts/${RUN_ID}/"
python "$BOS_IO" put-dir "$output_dir" "bos://${BOS_BUCKET}/${BOS_NAMESPACE}/ckpts/${RUN_ID}/"
echo "✅ 完成: ${RUN_ID}  ->  bos://${BOS_BUCKET}/${BOS_NAMESPACE}/ckpts/${RUN_ID}/"
