#!/usr/bin/env bash
set -euo pipefail

# UMI · PI05 部署启动脚本 (Zenoh serve)。
#
# launcher 只管路径 / 环境 / preflight; 部署参数 (tokenizer / device / zenoh addr / ...)
# 全在 deploy-yaml 的 deploy: 段, 不在 CLI (跟 eval / train launcher 同原则)。
#
# 用法:
#   bash examples/realRobots/z_umi/deploy_files/run_umi_pi05_serve.sh
#
# 改部署参数 (device / dtype / listen_addr / ...): 编辑 DEPLOY_YAML 的 deploy: 段, 不走 CLI。
#
# 环境变量覆盖 (有默认值):
#   CHECKPOINT     ckpt run 目录或 .pt 文件 (默认本机实例 ckpt; 扁平/标准布局都行)
#   DEPLOY_YAML    部署 overlay (默认同目录 deploy_umi_pi05_relative_6d.yaml)
#   SERVE_PYTHON   serve 用的 python (默认 starVLA conda env —— 有 zenoh + sentencepiece)
#
# 注: serve 需要 zenoh, 故默认走 starVLA conda env 的 python。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# ── 默认值 ─────────────────────────────────────────────────────────────
# 实例 ckpt (扁平导出: .pt 直接平铺在 run_dir); resolve_checkpoint 支持扁平/标准两种布局。
CHECKPOINT=${CHECKPOINT:-/home/zerith/.models/run_umi_shenzhen_relative_6d_20000step_0727_100151}
DEPLOY_YAML=${DEPLOY_YAML:-"${SCRIPT_DIR}/deploy_umi_pi05_relative_6d.yaml"}

# ── preflight ──────────────────────────────────────────────────────────
[[ -e "${CHECKPOINT}" ]] || { echo "CHECKPOINT not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${DEPLOY_YAML}" ]] || { echo "DEPLOY_YAML not found: ${DEPLOY_YAML}" >&2; exit 1; }

# serve 需要 zenoh → 默认 starVLA conda env (本机装了 zenoh + sentencepiece)。
PY_CMD=(${SERVE_PYTHON:-/home/zerith/miniforge3/envs/starVLA/bin/python})
if ! PYTHONPATH="${REPO_ROOT}" "${PY_CMD[@]}" -c "import zenoh" 2>/dev/null; then
  echo "zenoh 未安装 → serve 需要。用装有 zenoh 的 env (SERVE_PYTHON=<python>)。" >&2
  exit 1
fi

echo "[run_umi_pi05_serve] checkpoint:  ${CHECKPOINT}"
echo "  deploy_yaml:  ${DEPLOY_YAML}"
echo "  python:       ${PY_CMD[*]}"

PYTHONPATH="${REPO_ROOT}" exec "${PY_CMD[@]}" -m deployment.model_server.server_umi_pi05_rio \
  serve \
  --checkpoint "${CHECKPOINT}" \
  --deploy-yaml "${DEPLOY_YAML}"
