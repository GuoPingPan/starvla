#!/usr/bin/env bash
# =============================================================================
# StarVLA 环境一键安装脚本
# -----------------------------------------------------------------------------
# 用法:
#   ./install.sh                      # 仅核心依赖
#   ./install.sh --dev                # 额外装 dev 工具 (black/ruff/pre-commit)
#   ./install.sh --attention          # 额外装高性能 attention (triton/flash-attn, 编译慢)
#   ./install.sh --dev --attention
#
# 详细说明 / 已知问题排错见同目录 install.md。
# =============================================================================
set -euo pipefail

WITH_DEV=0
WITH_ATTENTION=0
for arg in "$@"; do
    case "$arg" in
        --dev)        WITH_DEV=1 ;;
        --attention)  WITH_ATTENTION=1 ;;
        -h|--help)    sed -n '2,13s/^# \{0,1\}//p' "$0"; exit 0 ;;
        *) echo "未知参数: $arg (用 --help 查看用法)" >&2; exit 1 ;;
    esac
done

# 切到仓库根目录 (脚本位于 docs/z-docs/, 根在往上两级)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n\033[1;34m[install]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# -----------------------------------------------------------------------------
# 1. 系统依赖 (apt: FFmpeg 开发头文件)
#    av==12.3.0 在 Python 3.13 上没有预编译 wheel, 需要从源码编译;
#    编译依赖 FFmpeg 的 libav* 开发库 (通过 pkg-config 查找)。
#    缺失时 `uv sync` 会报: pkg-config was not found libraries
#    ['avformat','avcodec','avdevice','avutil','avfilter','swscale','swresample']
# -----------------------------------------------------------------------------
install_apt_deps() {
    log "检查系统 FFmpeg 开发库 (av 源码编译依赖)..."
    if pkg-config --exists libavformat 2>/dev/null; then
        log "libavformat 已存在, 跳过 apt 安装。"
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        err "未检测到 apt-get (非 Debian/Ubuntu?)。请手动安装 FFmpeg 开发库:"
        err "  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev"
        err "  libavfilter-dev libswscale-dev libswresample-dev pkg-config"
        err "详见 install.md 「备选方案」一节。"
        exit 1
    fi
    log "通过 apt 安装 FFmpeg 开发库 (需要 sudo)..."
    sudo apt-get update
    sudo apt-get install -y \
        ffmpeg \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libavfilter-dev libswscale-dev libswresample-dev \
        pkg-config
}
install_apt_deps

# -----------------------------------------------------------------------------
# 2. 检查 uv
# -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    err "未检测到 uv, 请先安装:"
    err "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
log "uv: $(uv --version)"

# -----------------------------------------------------------------------------
# 3. uv sync
#    固定 python 3.10: .python-version 若缺失 (常见于 git worktree,
#    该文件历史上被 .gitignore 忽略), uv 会回退到系统默认 python,
#    wheel ABI 不匹配导致无法复用 uv cache 的硬链接, 全量重新下载。
#    pyproject.toml 已配置 aliyun 镜像 (默认 index) + PyTorch cu124 显式索引,
#    torch/torchvision 会走 cu124 wheel (CUDA 13 driver 向下兼容 cu124)。
# -----------------------------------------------------------------------------
log "固定 python 版本为 3.10..."
uv python pin 3.10

log "uv sync (核心依赖)..."
uv sync

if [[ "${WITH_DEV}" == "1" ]]; then
    log "uv sync --extra dev (black/ruff/pre-commit)..."
    uv sync --extra dev
fi

if [[ "${WITH_ATTENTION}" == "1" ]]; then
    log "uv sync --extra attention (triton/flash-attn, 编译耗时较长, 请耐心)..."
    uv sync --extra attention
fi

# -----------------------------------------------------------------------------
# 4. 验证关键依赖可 import
# -----------------------------------------------------------------------------
log "验证关键依赖..."
uv run --no-sync python - <<'PY'
import importlib
for mod in ["torch", "av", "decord", "transformers", "accelerate"]:
    try:
        m = importlib.import_module(mod)
        print(f"  ✓ {mod:<12} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  ✗ {mod:<12} FAILED: {e}")
PY

log "完成。运行训练示例:"
echo "  uv run python train.py --config_yaml <your_config.yaml>"
