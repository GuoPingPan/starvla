#!/usr/bin/env bash
# ============================================================================
# scripts/dev/setup_qwen35_fastpath.sh — Qwen3.5 DeltaNet 线性注意力 fast path 一键安装
# ----------------------------------------------------------------------------
# 背景: Qwen3.5 的 18 层 Gated DeltaNet 线性注意力在缺 fla + causal-conv1d 时退化为
#       纯 torch 慢速实现 (训练日志会出现 "The fast path is not available ...")。
#       transformers 要求 4 个符号全就绪才启用: chunk_gated_delta_rule /
#       fused_recurrent_gated_delta_rule (来自 flash-linear-attention) +
#       causal_conv1d_fn / causal_conv1d_update (来自 causal-conv1d)。
#
# 版本坑 (踩过的, 勿踩第二次):
#   - causal-conv1d 1.7.0 官方 "cxx11abiFALSE" wheel 与自编译 wheel 均会报
#     undefined symbol: c10::detail::torchCheckFail(...std::__cxx11::basic_string...)
#     (Dao-AILab/causal-conv1d#77, state-spaces/mamba#808)。
#   - 社区验证可用: 1.5.0.post8 abiFALSE wheel @ torch 2.6.0+cu124 + py3.10。
#   - 本仓库 torch 为内部构建 (旧 ABI), 只匹配该 wheel。
#
# 用法 (在带 .venv 的 repo 根目录):
#   bash scripts/dev/setup_qwen35_fastpath.sh            # 默认 .venv
#   VENV=/path/to/.venv bash scripts/dev/setup_qwen35_fastpath.sh
#
# 产物缓存: wheel 存 CFS 共享目录 (全机器共享, 他人免重下): 首选
#           /mnt/cfs/code/wheels (与 flash_attn wheel 同处, train.sh 段2 自愈也读这里),
#           旧位置 /mnt/cfs/Pretrained_models/wheels 作回退; 两者皆无时从 GitHub 拉
#           (需代理: 自动读 setproxy/https_proxy)。causal-conv1d 的 GitHub wheel 在
#           百度云内网拉不动时, 可由 train.sh 段2 自愈从 donor venv 重建的 wheel 安装。
# 验证: 安装后打印 FAST_PATH: True + kernel 冒烟; 训练日志不再出现 "fast path" 警告。
# ============================================================================
set -euo pipefail

VENV="${VENV:-$PWD/.venv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: 找不到 $PY (在 repo 根目录跑, 或 VENV=... 指定)" >&2; exit 1; }

WHEEL_NAME="causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# 同名 donor 重建 wheel (train.sh 自愈产物, 版本 1.5.0.post8, 内容一致)
WHEEL_NAME_LOCAL="causal_conv1d-1.5.0.post8+local-cp310-cp310-linux_x86_64.whl"
WHEEL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/$WHEEL_NAME"
UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/cfs/uv-cache}"
export UV_CACHE_DIR

# ── 1. causal-conv1d: CFS 缓存优先 (两个目录两种命名), 缺则下载 ──────────────
WHEEL=""
for _dir in /mnt/cfs/code/wheels /mnt/cfs/Pretrained_models/wheels; do
  for _name in "$WHEEL_NAME" "$WHEEL_NAME_LOCAL"; do
    [ -f "$_dir/$_name" ] && WHEEL="$_dir/$_name" && break
  done
  [ -n "$WHEEL" ] && break
done
if [ -z "$WHEEL" ]; then
  mkdir -p /mnt/cfs/code/wheels
  WHEEL="/mnt/cfs/code/wheels/$WHEEL_NAME"
  echo ">> CFS 无缓存 wheel, 从 GitHub releases 下载 (走代理)..."
  [ -n "${https_proxy:-}" ] || { echo "   (提示: 未检测到 https_proxy, 可先 source setproxy)" >&2; }
  curl -sL --retry 3 -o "$WHEEL" "$WHEEL_URL"
fi
echo ">> wheel: $WHEEL"

echo ">> 安装 fla (PyPI) + causal-conv1d (CFS wheel) -> $VENV"
uv pip install --python "$PY" -q flash-linear-attention
uv pip install --python "$PY" -q --reinstall-package causal-conv1d "$WHEEL"

# ── 2. 验证: fast path + GPU kernel 冒烟 ──────────────────────────────────
echo ">> 验证 fast path..."
"$PY" - <<'EOF' 2>&1 | grep -vE "Triton version|Python version"
import torch
import causal_conv1d
from causal_conv1d import causal_conv1d_fn
import transformers
import transformers.models.qwen3_5.modeling_qwen3_5 as m

assert m.is_fast_path_available, "fast path 仍未启用!"

if torch.cuda.is_available():
    x = torch.randn(2, 64, 32, device="cuda"); w = torch.randn(64, 4, device="cuda"); b = torch.randn(64, device="cuda")
    y = causal_conv1d_fn(x=x, weight=w, bias=b)
    assert y.isfinite().all(), "kernel 输出含 NaN"
    kernel = f"kernel OK {tuple(y.shape)}"
else:
    kernel = "kernel 跳过 (无 CUDA)"

print(f"FAST_PATH: True | transformers {transformers.__version__} | torch {torch.__version__} | {kernel}")
EOF

echo "✅ Qwen3.5 fast path 就绪 (fla + causal-conv1d 1.5.0.post8)"
echo "   下次训练日志中不应再出现 'The fast path is not available' 警告。"
