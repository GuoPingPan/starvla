#!/usr/bin/env bash
# ============================================================================
# scripts/train.sh — 任务队列直接拉起的训练流水线 (UMI · PI05) · 双云版 (阿里云 OSS / 百度云 BOS)
# ----------------------------------------------------------------------------
#   命令行参数 (均可选): bash train.sh [config.yaml] [commit] [--local] [--local-code]
#     不传则回退环境变量/默认值; CONFIG_YAML 与 COMMIT 已改为位置参数解析。
#     --local-code  跳过段2云端 src-zip, 直接用当前 checkout + 其 .venv (本机测试用,
#                   未提交/未上传 CI 的配置也能跑; 等价 LOCAL_CODE=1)
#   6 段: 配置环境 → 拉代码 → 拉数据 → 拉预训练 → 训练 → 上传全部产物
#   云读写走 scripts/cloud_io.py (OSS=oss2 / BOS=boto3, 凭证从环境变量读, 不入库)。
#   云自动选择: OSS 凭证可得 (OSS_ACCESS_KEY_ID 或 PAI DSW 凭证服务 localhost:7002)
#   优先 OSS, 否则 BOS; 可用 CLOUD=oss|bos 强制。
#   共享盘: 阿里云 PAI 用 /mnt/data (CPFS), 百度云用 /mnt/cfs (CFS NFS); SHARED_DIR 可覆盖。
#   所有产物落 /root 下的软链 (-> 共享盘), 方便查看。
# ----------------------------------------------------------------------------
#   前置 (主机/队列机):
#     · 阿里云: export OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET  (或写进 /root/.ossenv, 本脚本会 source)
#     · 百度云: export BOS_ACCESS_KEY_ID / BOS_SECRET_ACCESS_KEY  (或写进 /root/.bosenv, 本脚本会 source)
#     · OSS 默认桶 algo-public0 / 命名空间 orion ; BOS 默认 umi-gripper / Orion
#     · uv 在 PATH (本机 /usr/local/bin/uv)
#   注: 默认 CONFIG_YAML=shenzhen (目前唯一配了云端数据路径的); 跑 hefei 前需在
#       data_config.py 给 hefei_relative_6d 补第4个元素 oss:// 或 bos:// URI。
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_IO="$SCRIPT_DIR/cloud_io.py"
DL_DATA="$SCRIPT_DIR/download_data.py"

# ── 共享盘挂载说明 ────────────────────────────────────────────────────────────
# 阿里云 PAI (DSW/EAS): CPFS 已由平台挂到 /mnt/data (fuse.aliyun-alinas-efc), 无需手动 mount。
# 百度云: CFS NFS (历史上手动 mount 到 /mnt/cfs, 命令留档):
#   sudo apt update && sudo apt install -y nfs-common
#   mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport \
#     cfs-pVlRwhavQO.lb-2etad1gk.cfs.bj.baidubce.com:/ /mnt/cfs

# ── 命令行参数 (可选) ──────────────────────────────────────────────────────────
#   用法: bash train.sh [config.yaml] [commit] [--local] [--local-code]
#     $1       CONFIG_YAML  训练配置 (相对 repo root); 默认 shenzhen 配置
#     $2       COMMIT       源码产物 commit sha (任意位数前缀均可); 默认 latest (取 starvla_res 最新 zip)
#     --local  SwanLab 切本地模式 (默认云端 online, 需 swanlab login); 等价 SWANLAB_MODE=local
#     --local-code  段2跳过云端 src-zip, 用当前 checkout + 其 .venv (本机测试用:
#                   未提交/未上传 CI 产物的配置也能跑; 等价 LOCAL_CODE=1)
#   位置参数优先; 留空回退环境变量, 再回退默认值 (向后兼容旧调用)。--local/--local-code 可出现在任意位置。
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "用法: $0 [config.yaml] [commit] [--local] [--local-code]" >&2
  echo "  config.yaml   训练配置 (相对 repo root), 默认 shenzhen 配置" >&2
  echo "  commit        源码产物 commit sha (任意位数前缀) 或 latest, 默认 latest" >&2
  echo "  --local       SwanLab 用本地模式 (默认云端 online, 需先 swanlab login)" >&2
  echo "  --local-code  跳过云端 src-zip, 用当前 checkout + .venv (本机测试)" >&2
  exit 0
fi
# 提取 --local / --local-code flag (可出现在任意位置); 剩余参数作为 config/commit 位置参数
SWANLAB_LOCAL=0
LOCAL_CODE="${LOCAL_CODE:-0}"
_pos=()
for _a in "$@"; do
  case "$_a" in
    --local) SWANLAB_LOCAL=1 ;;
    --local-code) LOCAL_CODE=1 ;;
    *) _pos+=("$_a") ;;
  esac
done
set -- "${_pos[@]}"
[ "$SWANLAB_LOCAL" = 1 ] && export SWANLAB_MODE=local
# ── 顶部可调项 ────────────────────────────────────────────────────────────────
CLOUD="${CLOUD:-bos}"                        # 云后端: oss=阿里云 / bos=百度云 (手动指定, 不自动探测; 默认百度云)
CONFIG_YAML="${1:-${CONFIG_YAML:-examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d.yaml}}"
COMMIT="${2:-${COMMIT:-latest}}"          # commit sha (任意位数前缀); latest=取 starvla_res 最新 zip
# 训练超参 (与 run_umi_pi05_relative_6d.sh 默认一致)
# 默认用全部可见 GPU (nvidia-smi 计数); 平台多机注入 NPROC_PER_NODE 时优先用它。
NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}"
BATCH="${BATCH:-10}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
PORT="${PORT:-${MASTER_PORT:-29501}}"
WANDB_PROJECT="${WANDB_PROJECT:-starvla}"
# ---- 多机 (默认单机; 兼容训练平台注入的 WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT) ----
# 这些在段3/4 下载守卫 和 段5 launcher 透传都要用, 故在顶部解析。
NUM_MACHINES="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
MACHINE_RANK="${MACHINE_RANK:-${RANK:-${SLURM_NODEID:-0}}}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-}}"
# 所有产物走 /root 下的软链 (-> 共享盘), 方便查看
# 共享盘选择: SHARED_DIR 可覆盖; 默认 /mnt/data (阿里云 CPFS, 存在即用), 回退 /mnt/cfs (百度云 CFS)
if [ -z "${SHARED_DIR:-}" ]; then
  if [ -d /mnt/data ]; then SHARED_DIR="/mnt/data"; else SHARED_DIR="/mnt/cfs"; fi
fi
DATA_DIR="/root/data"           # -> $SHARED_DIR/data (含 z-umi/, model_outputs/)
CODE_DIR="/root/code"           # -> $SHARED_DIR/code
RUN_ROOT_DIR="${DATA_DIR}/model_outputs"
PRETRAINED_LOCAL="${DATA_DIR}/pretrain_models"

# ============================================================================
# 1. 配置环境
# ============================================================================
# 云凭证: 优先环境里已有; 否则 source 本地(不入库)文件 (阿里云 .ossenv / 百度云 .bosenv, 两者可并存)
for _envf in "${OSS_ENV_FILE:-/root/.ossenv}" "${BOS_ENV_FILE:-/root/.bosenv}"; do
  if [ -f "$_envf" ]; then set -a; . "$_envf"; set +a; fi
done
# 云选择: CLOUD 强制 > OSS 凭证可得(含 PAI DSW 凭证服务 localhost:7002) > BOS
# 注意: 本机 .ossenv/.bosenv 可能并存, 自动探测会优先 OSS; 为避免误传, 建议显式 export CLOUD=oss|bos。
if [ -z "${CLOUD:-}" ]; then
  if [ -n "${OSS_ACCESS_KEY_ID:-}" ] && [ -n "${OSS_ACCESS_KEY_SECRET:-}" ] \
     || curl -sf -m 1 -o /dev/null "${OSS_CREDENTIALS_URI:-http://localhost:7002/api/v1/credentials/0}" 2>/dev/null; then
    CLOUD=oss
  elif [ -n "${BOS_ACCESS_KEY_ID:-}" ] && [ -n "${BOS_SECRET_ACCESS_KEY:-}" ]; then
    CLOUD=bos
  else
    echo "ERROR: 无任何云端凭证 (OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET 或 BOS_ACCESS_KEY_ID/BOS_SECRET_ACCESS_KEY), 也不可写 /root/.ossenv|/root/.bosenv" >&2
    exit 1
  fi
fi
case "$CLOUD" in
  oss)
    export OSS_ENDPOINT="${OSS_ENDPOINT:-https://oss-cn-beijing.aliyuncs.com}"   # 华北2
    : "${OSS_BUCKET:?ERROR: OSS_BUCKET 未设置 (请 export 或写入 /root/.ossenv)}"
    : "${OSS_NAMESPACE:?ERROR: OSS_NAMESPACE 未设置 (请 export 或写入 /root/.ossenv)}"
    export OSS_BUCKET OSS_NAMESPACE
    SCHEME=oss; BUCKET="$OSS_BUCKET"; NAMESPACE="$OSS_NAMESPACE"
    ;;
  bos)
    export BOS_ENDPOINT="${BOS_ENDPOINT:-s3.bj.bcebos.com}"
    # 桶/命名空间默认团队标准值 (容器内 .bosenv 通常只有密钥; 显式 export 可覆盖)
    export BOS_BUCKET="${BOS_BUCKET:-umi-gripper}"
    export BOS_NAMESPACE="${BOS_NAMESPACE:-Orion}"
    SCHEME=bos; BUCKET="$BOS_BUCKET"; NAMESPACE="$BOS_NAMESPACE"
    ;;
  *) echo "ERROR: CLOUD 必须是 oss 或 bos" >&2; exit 1 ;;
esac
echo ">> [云] CLOUD=$CLOUD  ${SCHEME}://${BUCKET}/${NAMESPACE}/"
# 跨云数据集镜像: OSS 环境下把 data_config 里的 bos:// 数据 URI 重映射到 OSS 同名前缀
# (数据同步到 OSS 后无需改 data_config; 可用 DATA_URI_REMAP 覆盖, 格式 旧前缀=新前缀[,..])
if [ "$CLOUD" = oss ] && [ -z "${DATA_URI_REMAP:-}" ]; then
  export DATA_URI_REMAP="bos://robot-data-from-disk/=${SCHEME}://${BUCKET}/robot-data-from-disk/,bos://umi-gripper/=${SCHEME}://${BUCKET}/umi-gripper/"
fi

# SwanLab 凭证: 优先环境里已有; 否则 source 一个本地(不入库)文件 (与云凭证同模式)。
# SWANLAB_API_KEY 通常只写在 ~/.zshrc, 而队列/后台用非交互 bash 拉起 train.sh 时
# 不会 source ~/.zshrc → 训练子进程拿不到 key → logger.py (b8a8bba) 的无 TTY 非交互
# login 失效, 退化成 "no TTY for interactive login" → continuing without tracker。
# 凭据文件路径可被 SWANLAB_ENV_FILE 覆盖 (例如指向共享盘, 让所有队列节点共用)。
if [ -z "${SWANLAB_API_KEY:-}" ] && [ -f "${SWANLAB_ENV_FILE:-/root/.swanlab_env}" ]; then
  set -a; . "${SWANLAB_ENV_FILE:-/root/.swanlab_env}"; set +a
fi

# 路径归一: /root/data -> $SHARED_DIR/data, /root/code -> $SHARED_DIR/code
# (悬空的旧软链 —— 目标不存在, 如百度云机器迁阿里云后 /mnt/cfs 不再挂载 —— 自动重指向新共享盘)
mkdir -p "$SHARED_DIR/data/z-umi" "$SHARED_DIR/code"
for _l in /root/data:"$SHARED_DIR/data" /root/code:"$SHARED_DIR/code"; do
  link="${_l%%:*}"; target="${_l#*:}"
  if [ -L "$link" ] && { [ ! -e "$link" ] || [ "$(readlink -f "$link" 2>/dev/null)" = "$(readlink -f "$target")" ]; }; then
    ln -sfn "$target" "$link"   # 悬空或已指向本目标: (重)指向, 幂等
  elif [ -L "$link" ]; then
    echo "ERROR: $link 已指向其它存在目标 ($(readlink "$link")), 拒绝覆盖 (可用 SHARED_DIR 显式指定)" >&2; exit 1
  elif [ -e "$link" ]; then
    echo "ERROR: $link 已存在且非软链, 拒绝覆盖" >&2; exit 1
  else
    ln -s "$target" "$link"
  fi
done

# CUDA / NCCL / HF / WandB
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_BLOCKING_WAIT=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=10000
# ── RDMA 用户态库自愈 ─────────────────────────────────────────────────────────
# 平台 (DLC/eRDMA/kangaroo) 注入 /dev/infiniband + mlx5 设备, 但镜像常缺 libibverbs
# 用户态库 → NCCL dlopen 失败后静默退回 NET/Socket (TCP), 多机 allreduce 慢 ~2x。
# 检测到 RDMA 设备且无 libibverbs 时 apt 安装; 换镜像重试兼容百度云/阿里云源。
if [ -d /sys/class/infiniband ] && ! ldconfig -p 2>/dev/null | grep -q libibverbs; then
  echo ">> [RDMA] 检测到 RDMA 设备 ($(ls /sys/class/infiniband | tr '\n' ' ')) 但缺 libibverbs, 安装..."
  _rdma_ok=0
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq libibverbs1 libmlx5-1 >/dev/null 2>&1 && _rdma_ok=1
  if [ "${_rdma_ok}" = 0 ]; then
    echo ">> [RDMA] 默认源失败, 切阿里云镜像重试..."
    sed -i.bak 's|mirrors.baidubce.com|mirrors.aliyun.com|g' \
      /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null || true
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq libibverbs1 libmlx5-1 >/dev/null 2>&1 && _rdma_ok=1
  fi
  if [ "${_rdma_ok}" = 1 ] && ldconfig -p 2>/dev/null | grep -q libibverbs; then
    echo ">> [RDMA] libibverbs 就绪 — NCCL 将走 NET/IB (RoCE)"
  else
    echo "WARN: [RDMA] rdma-core 安装失败, NCCL 将退回 TCP (多机通信慢)" >&2
  fi
fi
# RoCE v2 GID: gid[0]=RoCE v1 (默认, 不通), gid[3]=RoCE v2 + 业务 IP; 固定指 3。
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# 梯度爆炸看门狗 (train_starvla.py::_grad_watch): grad_norm 非有限或超阈值时, 在
# optimizer step 前把案发现场 batch dump 到 {output_dir}/grad_watch/。默认开;
# 复现完整雪崩用, 取证见 scripts/replay_grad_explosion.py。关闭: STARVLA_GRAD_WATCH=0。
export STARVLA_GRAD_WATCH="${STARVLA_GRAD_WATCH:-1}"
export STARVLA_GRAD_WATCH_THRESH="${STARVLA_GRAD_WATCH_THRESH:-100}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# torch.compile 缓存共享盘化: FX graph + triton kernel 落共享盘, 跨容器/跨任务/跨 rank 复用。
# 解决 max-autotune 8 卡死锁: 无缓存时各 rank 独立调优 (benchmark 期间不进 NCCL 集合通信)
# → 先到的 rank 卡在 allreduce 等, 全体互等死锁 (实测 1.5h 零进展)。
# 有共享缓存后: 冷启动只有 rank0 生成缓存, 其余 rank 秒级命中; 二次任务全员命中。
# 首跑安全姿势: COMPILE_WARMUP=1 (段5 前自动单卡预热 3 步产出缓存) 或手动先跑一次 NUM_GPUS=1。
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SHARED_DIR}/inductor-cache}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SHARED_DIR}/triton-cache}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# uv 缓存持久化到共享盘 (CFS/CPFS): 跨容器/任务复用, 不依赖 docker 镜像层缓存;
# 首次 uv sync 后 wheel 全部命中缓存, 换新容器建新 venv 也不再重复下载。
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SHARED_DIR}/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"   # NFS 上 hardlink 不可靠, 显式用 copy
mkdir -p "$UV_CACHE_DIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"   # 调试用; 正式训练删此行 + wandb login

# ============================================================================
# 2. 拉取代码 (按 commit 从云端取 CI 产物 zip -> 解压 -> uv 建运行环境)
#    并发安全 (多任务共享共享盘):
#      · zip 下到唯一临时目录, 不互相覆盖下载中的产物
#      · 解压目录按真实 commit sha 命名 repo_<sha>, 不同 commit 任务互不覆盖
#      · flock 串行化同 commit 的解压+uv sync; repo_<sha>/.venv 已存在则直接复用
#    注: 每次都会下载 zip (latest 需下载后才能解析真实 sha); 复用靠 repo_<sha>/.venv。
# ============================================================================
mkdir -p "$CODE_DIR"
DL_TMP="$(mktemp -d "$CODE_DIR/.srczip.XXXXXX")"
trap 'rm -rf "$DL_TMP" 2>/dev/null || true' EXIT INT TERM
# 此刻还没建好环境, 用 uv run --with 临时带上 oss2/boto3 来跑 cloud_io
uv run --no-project --with oss2 --with boto3 python "$CLOUD_IO" src-zip "$COMMIT" \
  --cloud "$CLOUD" --dest-dir "$DL_TMP/"
ZIP="$(ls -t "$DL_TMP"/starvla_*.zip 2>/dev/null | head -1)"
[ -n "$ZIP" ] || { echo "ERROR: 没拉到源码 zip (先跑 GitLab CI 上传产物到 starvla_res/)" >&2; exit 1; }
# zip 命名 starvla_<ref>_<ts>_<sha>.zip -> 末段为真实 commit sha (与 cloud_io.fetch_src_zip 解析一致)
REAL_SHA="$(basename "$ZIP" .zip | awk -F_ '{print $NF}' | tr 'A-Z' 'a-z')"
REPO_DIR="$CODE_DIR/repo_$REAL_SHA"
# flock: 同 commit 并发任务排队解压+建 venv, 避免竞态; 双重检查 .venv 复用已建环境
(
  flock 9
  if [ -x "$REPO_DIR/.venv/bin/python" ]; then   # 只看 -d .venv 不够: 中断的 uv sync 会留下无 python 的残缺 venv
    echo ">> [段2] 复用已有环境 $REPO_DIR/.venv (跳过解压 + uv sync)"
  else
    echo ">> [段2] 环境不完整/不存在, 重建 $REPO_DIR (共享盘上删旧 venv 可能要几分钟, 无输出属正常, 请勿 Ctrl-C)"
    rm -rf "$REPO_DIR" || { sleep 3; rm -rf "$REPO_DIR"; }   # 偶发"目录非空"(fuse/共享盘目录缓存过期), 重试一次
    mkdir -p "$REPO_DIR"
    unzip -q "$ZIP" -d "$REPO_DIR"
    ( cd "$REPO_DIR" && uv sync --python 3.10 )
  fi
  # ── 高性能 attention 三件套自愈 (在 flock 内, 同 commit 并发任务串行安装) ──
  # flash_attn (视觉 cu_seqlens 变长内核) + flash-linear-attention + causal-conv1d
  # (Qwen3.5 DeltaNet 线性注意力 fast path)。pyproject 的 attention extra 默认不装,
  # flash-attn 源码编译要 1h+; 曾靠手工装进 repo_3c768d9* 一个 venv 撑着, 换 commit
  # 即失效 (QWen3_5 静默回退 sdpa/慢路径, 吞吐 -30%+)。这里从 CFS 共享 wheel 秒级补齐
  # (/mnt/cfs/code/wheels/, --no-deps 离线装), fla 走 PyPI 镜像; 版本坑见
  # scripts/dev/setup_qwen35_fastpath.sh。已装即跳过 (幂等); 无 CFS wheel 的机器
  # 跳过对应件, 训练走回退路径不报错。
  VENV_PY="$REPO_DIR/.venv/bin/python"
  WHEELS_DIR="/mnt/cfs/code/wheels"
  _has_mod() { "$VENV_PY" -c "import $1" >/dev/null 2>&1; }
  # transformers 5.x: QwenPI_v3 需要 Qwen3_5ForConditionalGeneration (>=5.2), 而
  # pyproject 全局 pin 4.57.0 (其它框架用)。yaml 用 QwenPI_v3 时给本 venv 升 5.3.0
  # (调优定版版本, 与历史手工 venv 一致); 其它 yaml 保持 lockfile 版本不动。
  if grep -q "QwenPI_v3" "$CONFIG_YAML" 2>/dev/null \
     && ! "$VENV_PY" -c "from transformers import Qwen3_5ForConditionalGeneration" >/dev/null 2>&1; then
    echo ">> [段2] QwenPI_v3 需要 transformers>=5.2 (当前 venv 不满足), 升级到 5.3.0"
    uv pip install --python "$VENV_PY" --quiet "transformers==5.3.0" \
      || echo "WARN: transformers 5.3.0 升级失败, Qwen3.5 路径将 ImportError" >&2
  fi
  if ! _has_mod flash_attn; then
    _fa_wheel="$(ls $WHEELS_DIR/flash_attn-*.whl 2>/dev/null | head -1)"
    if [ -n "$_fa_wheel" ]; then
      echo ">> [段2] 补装 flash_attn ($_fa_wheel)"
      uv pip install --python "$VENV_PY" --quiet --no-deps "$_fa_wheel" \
        || echo "WARN: flash_attn 安装失败, 视觉注意力回退 sdpa" >&2
    else
      echo "WARN: $WHEELS_DIR 无 flash_attn wheel, 视觉注意力回退 sdpa" >&2
    fi
  fi
  if ! _has_mod causal_conv1d || ! _has_mod fla; then
    _cc1_wheel="$(ls $WHEELS_DIR/causal_conv1d-*.whl /mnt/cfs/Pretrained_models/wheels/causal_conv1d-*.whl 2>/dev/null | head -1)"
    echo ">> [段2] 补装 Qwen3.5 线性注意力 fast path (fla + causal-conv1d)"
    _has_mod fla || uv pip install --python "$VENV_PY" --quiet flash-linear-attention \
      || echo "WARN: flash-linear-attention 安装失败 (需外网/镜像)" >&2
    if ! _has_mod causal_conv1d && [ -n "$_cc1_wheel" ]; then
      uv pip install --python "$VENV_PY" --quiet --no-deps --reinstall-package causal-conv1d "$_cc1_wheel" \
        || echo "WARN: causal-conv1d 安装失败, DeltaNet 走 torch 慢速实现" >&2
    elif ! _has_mod causal_conv1d; then
      echo "WARN: 无 causal_conv1d wheel 缓存, DeltaNet 走 torch 慢速实现" >&2
    fi
  fi
  # ── 底座模型软链: yaml 的 base_vlm 指 ./playground/Pretrained_models/<模型> ──
  # repo zip 不含 playground; 底座统一放 $PRETRAINED_LOCAL (段4 下载/共享盘常驻)。
  # 已有真实目录 (老 repo 手工拷过模型) 不动, 只给缺失的建软链。
  if [ ! -e "$REPO_DIR/playground/Pretrained_models" ]; then
    mkdir -p "$REPO_DIR/playground"
    ln -s "$PRETRAINED_LOCAL" "$REPO_DIR/playground/Pretrained_models"
    echo ">> [段2] playground/Pretrained_models -> $PRETRAINED_LOCAL"
  fi
) 9>"$CODE_DIR/.repo_$REAL_SHA.lock"
rm -rf "$DL_TMP"; trap - EXIT INT TERM
cd "$REPO_DIR"
. .venv/bin/activate
# cloud_io.py 需要 oss2+boto3; pyproject 里 boto3 在 sagemaker extra (uv sync 默认不装), 这里补装
uv pip install --quiet oss2 boto3
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 段3+ 改用「刚解压的 commit」的 scripts/cloud_io.py + download_data.py（即 $PWD/scripts/），
# 不再用调用方 train.sh 旁边的版本。这样下载/训练代码同源（都是 <commit> 的），
# 避免：调用方 checkout 是旧分支（如 dev/train_anylze）→ 旧 download_data.py 没有
# data_config 自动发现 → 误读 zerith_pi05 data_config → "data_mix 不在 DATASET_NAMED_MIXTURES"。
# 段2 拉 zip 时 REPO_DIR 还不存在，已用原 $SCRIPT_DIR/cloud_io.py，不受影响。
CLOUD_IO="$PWD/scripts/cloud_io.py"
DL_DATA="$PWD/scripts/download_data.py"

# ============================================================================
# 3/4. 拉取数据 + 预训练
#     多机时只在 rank0 下载 (产物落共享盘, 全节点可见); 其余节点轮询 sentinel 等其完成,
#     避免多节点同时首次下载竞争写同一份共享数据。单机: 直接下载, 不写 sentinel。
#     sentinel 按 CONFIG_YAML 命名 (同配置多次跑幂等); rank0 先 rm 旧 sentinel 再下载再 touch。
# ============================================================================
DL_SENTINEL="${DATA_DIR}/.dl_ready_$(basename "${CONFIG_YAML}")"
if [ "${STARVLA_SYNTHETIC:-0}" = "1" ] || [ "${STARVLA_SYNTHETIC:-}" = "true" ] || [ "${STARVLA_SYNTHETIC:-}" = "yes" ]; then
  # STARVLA_SYNTHETIC=1: 完全跳过数据下载 (训练侧用 stdlib 生成的假 batch), 只拉预训练权重。
  echo ">> [段3] STARVLA_SYNTHETIC=1 — 跳过数据下载 (假数据压测)"
  if [ "${NUM_MACHINES}" -gt 1 ] && [ "${MACHINE_RANK}" -ne 0 ]; then
    DL_WAIT_MIN="${DL_WAIT_MIN:-60}"
    for _ in $(seq 1 $((DL_WAIT_MIN * 6))); do [ -f "$DL_SENTINEL" ] && break; sleep 10; done
  else
    mkdir -p "$PRETRAINED_LOCAL"
    python "$CLOUD_IO" get-dir "${SCHEME}://${BUCKET}/${NAMESPACE}/pretrained/" "$PRETRAINED_LOCAL" \
      --exclude '.hfd/*'
    [ "${NUM_MACHINES}" -gt 1 ] && touch "$DL_SENTINEL"
  fi
elif [ "${NUM_MACHINES}" -gt 1 ] && [ "${MACHINE_RANK}" -ne 0 ]; then
  echo ">> [段3/4] 多机 worker(rank=${MACHINE_RANK}): 跳过下载, 等待 rank0 完成..."
  DL_WAIT_MIN="${DL_WAIT_MIN:-60}"
  for _ in $(seq 1 $((DL_WAIT_MIN * 6))); do   # 每 10s 轮询, 最多 DL_WAIT_MIN 分钟
    [ -f "$DL_SENTINEL" ] && break
    sleep 10
  done
  [ -f "$DL_SENTINEL" ] || { echo "ERROR: 等待 rank0 下载超时 (${DL_WAIT_MIN}min, $DL_SENTINEL 未出现)" >&2; exit 1; }
  echo ">> [段3/4] rank0 下载完成, 继续"
else
  [ "${NUM_MACHINES}" -gt 1 ] && rm -f "$DL_SENTINEL"   # 多机 rank0: 清旧 sentinel
  # 3. 数据 (按 yaml + data_config 从云端拉 -> $DATA_DIR/z-umi/data; DATA_URI_REMAP 自动跨云镜像)
  python "$DL_DATA" "$CONFIG_YAML"
  # 4. 预训练 (${NAMESPACE}/pretrained/ -> $PRETRAINED_LOCAL)
  mkdir -p "$PRETRAINED_LOCAL"
  python "$CLOUD_IO" get-dir "${SCHEME}://${BUCKET}/${NAMESPACE}/pretrained/" "$PRETRAINED_LOCAL" \
    --exclude '.hfd/*'
  [ "${NUM_MACHINES}" -gt 1 ] && touch "$DL_SENTINEL"   # 通知 worker
fi

# ============================================================================
# 4.5 torch.compile 单卡预热 (COMPILE_WARMUP=1 时; 多卡正式训练前先产出编译缓存)
#     背景: max-autotune 等模式各 rank 独立调优会与 NCCL 互等死锁 (见段1 注释)。
#     做法: 用 1 张卡跑 3 步真实训练 (足够触发全部 hot-path 编译), 缓存落共享盘;
#           随后的多卡训练里所有 rank 命中缓存, 编译秒级完成, 无死锁窗口。
#     只在 rank0 节点 + 多卡 + STARVLA_COMPILE 开启时执行; 缓存已存在时会快速跳过
#     (FX graph cache 命中, 预热步本身 ~1min)。COMPILE_WARMUP=0 可禁用。
# ============================================================================
if [ "${COMPILE_WARMUP:-0}" = "1" ] && [ "${NUM_MACHINES}" -eq 1 ] && [ "${MACHINE_RANK}" -eq 0 ] \
   && [ "${NUM_GPUS}" -gt 1 ] && [ -n "${STARVLA_COMPILE:-}" ]; then
  echo ">> [段4.5] compile 单卡预热 (STARVLA_COMPILE=${STARVLA_COMPILE}, 3 步产缓存 → ${TORCHINDUCTOR_CACHE_DIR})"
  ( export NUM_GPUS=1 RUN_ID="${RUN_ID}_compile_warmup" MAX_STEPS=3 SAVE_EVERY=1000000 PORT=$((PORT+100)) SWANLAB_MODE=local
    bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh "$CONFIG_YAML" >/dev/null 2>&1 \
      || echo "WARN: 预热子进程异常退出 (缓存可能部分生成, 多卡训练会自行补编)" ) >&2
  echo ">> [段4.5] 预热完成, 继续多卡训练"
fi

# ============================================================================
# 5. 训练 (复用 launcher: bash run_umi_pi05_relative_6d.sh <yaml>, 不重写其逻辑)
# ============================================================================
yaml_stem="$(basename "$CONFIG_YAML" .yaml)"
RUN_ID="${RUN_ID:-${yaml_stem}_${MAX_STEPS}step}"    # 与 launcher 默认一致; 段6 上传用它
# 导出 launcher 读取的训练参数 + 指向 uv 建的 env / 解压代码目录
# (launcher 已支持 REPO_ROOT / ACCEL_BIN 覆盖, 默认值不变)
export RUN_ID MAX_STEPS BATCH GRAD_ACCUM SAVE_EVERY LOG_EVERY NUM_WORKERS PREFETCH PORT WANDB_PROJECT NUM_GPUS
export NUM_MACHINES MACHINE_RANK MAIN_PROCESS_IP     # 多机变量透传给 launcher (段5)
export REPO_ROOT="$PWD"                              # 已 cd 到 $CODE_DIR/repo (zip 解压, 无 .git)
export ACCEL_BIN="$PWD/.venv/bin/accelerate"         # uv sync 建的 venv
if [ "${NUM_MACHINES}" -gt 1 ]; then
  echo ">> [训练] RUN_ID=${RUN_ID}  多机: num_machines=${NUM_MACHINES} rank=${MACHINE_RANK} total_gpus=$((NUM_MACHINES*NUM_GPUS)) main_ip=${MAIN_PROCESS_IP:-<未设>}"
else
  echo ">> [训练] RUN_ID=${RUN_ID}  单机: num_processes=${NUM_GPUS}  (复用 launcher: run_umi_pi05_relative_6d.sh)"
fi
bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh "$CONFIG_YAML"
output_dir="${RUN_ROOT_DIR}/${RUN_ID}"               # launcher 把产物写这里, 段6 上传

# ============================================================================
# 6. 上传全部产物 (整个 run 目录 -> <UPLOAD_DIR>, 防训练中断后无法恢复)
#    UPLOAD_DIR 可覆盖上传目标 (默认 ${SCHEME}://${BUCKET}/${NAMESPACE}/ckpts/<RUN_ID>/), 例如:
#      UPLOAD_DIR="oss://other-bucket/Alex/pi_checkpoints/" 时权重传该目录
# ============================================================================
UPLOAD_DIR="${UPLOAD_DIR:-${SCHEME}://${BUCKET}/${NAMESPACE}/ckpts/${RUN_ID}/}"
echo ">> [上传] ${output_dir} -> ${UPLOAD_DIR}"
python "$CLOUD_IO" put-dir "$output_dir" "$UPLOAD_DIR"
echo "✅ 完成: ${RUN_ID}  ->  ${UPLOAD_DIR}"
