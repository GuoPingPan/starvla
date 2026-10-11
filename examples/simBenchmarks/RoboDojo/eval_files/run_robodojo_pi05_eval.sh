#!/usr/bin/env bash
# RoboDojo · PI05 开环评测 launcher（适配自 zerith_pi05）。
#
# 两段式: Stage 1 跑 starVLA.training.eval_starvla (开环预测写 npz/json),
# Stage 2 (默认开) 用 rerun 把 predictions.npz + metrics.json + obs_images.npz
# 写成离线 .rrd (双臂 16 维, 3 相机)。
#
# 用法:
#   bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh <run-dir> [extra args...]
#
# 示例:
#   # smoke（2 episode, 每 episode 5 anchor）
#   bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh <run-dir> \
#       --episodes 0 1 --max-anchors 5 --checkpoint-step 10000
#   # 全量评测 (默认含 .rrd)
#   bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh <run-dir>
#   # 跳过 .rrd 生成
#   bash .../run_robodojo_pi05_eval.sh <run-dir> --no-visualize
#   # 仅重跑 Stage 2 (Stage 1 已跑过):
#   bash .../run_robodojo_pi05_eval.sh <run-dir> --stage2-only
#
# 数据根: 默认 /root/data/robodojo (本机路径, 其下有 RoboDojo_ee_lerobot_v30_video);
#         其它镜像机可显式 DATA_ROOT_OVERRIDE=... 覆盖。
#
# 环境变量: DATA_ROOT_OVERRIDE / STAGE2_SCRIPT / STAGE2_INPUT_DIR(--stage2-only) /
#           CKPT_STEP / SEED (后两者仅用于 --stage2-only 默认目录派生)。
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
STAGE2_SCRIPT=${STAGE2_SCRIPT:-examples/simBenchmarks/RoboDojo/eval_files/stage2_visualize.py}
DATA_ROOT_OVERRIDE=${DATA_ROOT_OVERRIDE:-/root/data/robodojo}
# --stage2-only 默认目录派生 (匹配 eval_starvla output_dir 命名 {name}_step{N}_seed{S})。
CKPT_STEP=${CKPT_STEP:-10000}
SEED=${SEED:-42}

PYTHON=/root/workspace/vla/starvla/.venv/bin/python

[[ -f "${EVAL_YAML}" ]] || { echo "EVAL_YAML not found: ${EVAL_YAML}" >&2; exit 1; }
[[ -f "${STAGE2_SCRIPT}" ]] || { echo "STAGE2_SCRIPT not found: ${STAGE2_SCRIPT}" >&2; exit 1; }

# ── 扫 launcher-only flag, 其余透传给 Stage 1 ─────────────────────────
# 默认 Stage 1 后自动生成 .rrd; --no-visualize 关闭, --stage2-only 仅跑 Stage 2。
DO_VISUALIZE=1
STAGE2_ONLY=0
STAGE1_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --visualize)    DO_VISUALIZE=1 ;;
    --no-visualize) DO_VISUALIZE=0 ;;
    --stage2-only)  STAGE2_ONLY=1 ;;
    *)              STAGE1_ARGS+=("${arg}") ;;
  esac
done

# ── Stage 2 单独跑 (Stage 1 已经跑过) ──────────────────────────────────
if [[ "${STAGE2_ONLY}" == "1" ]]; then
  STAGE2_INPUT_DIR=${STAGE2_INPUT_DIR:-playground/Evals/$(basename "${RUN_DIR}")_step${CKPT_STEP}_seed${SEED}}
  [[ -d "${STAGE2_INPUT_DIR}" ]] || { echo "STAGE2_INPUT_DIR not found: ${STAGE2_INPUT_DIR}" >&2; exit 1; }
  echo "[run_robodojo_pi05_eval] Stage 2 only on ${STAGE2_INPUT_DIR}"
  EPISODE_DIRS=("${STAGE2_INPUT_DIR}"/episode_*/)
  if [[ ${#EPISODE_DIRS[@]} -eq 0 ]]; then
    echo "[run_robodojo_pi05_eval] no episode_*/ subdirs in ${STAGE2_INPUT_DIR}" >&2; exit 1
  fi
  PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${STAGE2_SCRIPT}" "${EPISODE_DIRS[@]}"
  exit 0
fi

echo "Launching RoboDojo PI05 open-loop eval (Stage 1)"
echo "  run_dir:          ${RUN_DIR}"
echo "  eval_yaml:        ${EVAL_YAML}"
echo "  data_root:        ${DATA_ROOT_OVERRIDE}"
echo "  stage1_args:      ${STAGE1_ARGS[*]:-}"
echo "  visualize:        ${DO_VISUALIZE}"

# ── Stage 1 ───────────────────────────────────────────────────────────
# 输出 tee 到临时日志: Stage 2 据其中的 __EVAL_OUTPUT_DIR__ 标记精确定位 output_dir
# (旧版按 mtime 猜, 多 run 共存会串目录)。
STAGE1_LOG="$(mktemp)"
trap 'rm -f "${STAGE1_LOG}"' EXIT

"${PYTHON}" -m starVLA.training.eval_starvla \
  --run-dir "${RUN_DIR}" \
  --eval-yaml "${EVAL_YAML}" \
  --data-root-override "${DATA_ROOT_OVERRIDE}" \
  "${STAGE1_ARGS[@]}" 2>&1 | tee "${STAGE1_LOG}"

# ── Stage 2 (可选) ────────────────────────────────────────────────────
# 定位 Stage 1 刚写出来的 output_dir: 优先 __EVAL_OUTPUT_DIR__ 标记 (确定性),
# 缺失 (Stage 1 崩了 / 没打标记) 才回退 mtime。grep 带 || true 防 pipefail 中断。
if [[ "${DO_VISUALIZE}" == "1" ]]; then
  LAST_OUTPUT="$( { grep '__EVAL_OUTPUT_DIR__' "${STAGE1_LOG}" 2>/dev/null || true; } | tail -1 | sed 's/.*__EVAL_OUTPUT_DIR__[[:space:]]*//')"
  if [[ -n "${LAST_OUTPUT}" ]]; then
    echo "[run_robodojo_pi05_eval] Stage 2 output dir (from Stage 1 marker): ${LAST_OUTPUT}"
  else
    LAST_OUTPUT="$(ls -td playground/Evals/*/ 2>/dev/null | head -1)"
    LAST_OUTPUT="${LAST_OUTPUT%/}"
    [[ -n "${LAST_OUTPUT}" ]] && \
      echo "[run_robodojo_pi05_eval] Stage 2 output dir (mtime fallback): ${LAST_OUTPUT}"
  fi
  if [[ -z "${LAST_OUTPUT}" ]]; then
    echo "[run_robodojo_pi05_eval] WARNING: no output dir found to visualize" >&2
  else
    EPISODE_DIRS=("${LAST_OUTPUT}"/episode_*/)
    if [[ ${#EPISODE_DIRS[@]} -eq 0 ]]; then
      echo "[run_robodojo_pi05_eval] WARNING: no episode_*/ in ${LAST_OUTPUT}" >&2
    else
      echo "[run_robodojo_pi05_eval] Stage 2 → ${#EPISODE_DIRS[@]} episodes under ${LAST_OUTPUT}"
      PYTHONPATH="${REPO_ROOT}" "${PYTHON}" "${STAGE2_SCRIPT}" "${EPISODE_DIRS[@]}"
    fi
  fi
fi
