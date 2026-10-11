#!/usr/bin/env bash
set -euo pipefail

# UMI · PI05 开环评测 (open-eval) 启动脚本。
#
# Stage 1: 运行 starVLA.training.eval_starvla (多卡入口, 复刻 train_starvla.py 结构)。
# Stage 2 (默认在 Stage 1 后自动跑; --no-visualize 关闭 / --stage2-only 仅跑 Stage 2):
#   运行同目录 stage2_visualize.py 写 .rrd。
#
# 跟 examples/Z-umi/eval_files/run_umi_open_eval.sh 的关键区别 (本脚本修掉的两处):
#   1. base_vlm 处理框架无关 —— PI05/paligemma 没有 framework.qwenvl.base_vlm 键,
#      旧 launcher 解析该键会抛异常。本脚本 try/except 取, 取不到就跳过 symlink,
#      所以 PI05 / Qwen 两类 run 都能用。
#   2. data_root 默认覆盖到 /data/z-umi/data —— ckpt config 里存的是 /nix/data/z-umi
#      (本机已失效), 真实数据在 /data/z-umi/data/<dataset>。默认直接覆盖; 想换镜像再
#      显式设 DATA_ROOT_OVERRIDE。
#
# 默认值都指向实例 ckpt + 本目录 overlay/stage2。launcher 只管 uv 环境 / PYTHONPATH /
# preflight, 不通过 CLI hardcode 训练参数 (跟 train launcher 同原则)。
#
# 用法:
#   bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh
#   bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --max-anchors 5   # smoke (默认含 .rrd)
#   bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --no-visualize     # 跳过 .rrd 生成
#   bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --episode-id 3 --checkpoint-step 20000
#   bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --episodes 0 1 2
#
# 多卡: 设 NUM_PROCESSES=N, launcher 自动切到 accelerate launch
#   NUM_PROCESSES=4 bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh
#
# 换数据集评测 (DATA_MIX_OVERRIDE 覆盖 ckpt 的 data_mix, 透传 --data-mix):
#   # 同 embodiment 的另一份 shenzhen 数据 (泛化测试):
#   DATA_MIX_OVERRIDE=shenzhen_relative_6d_2 \
#     bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --max-anchors 5
#   # 跨分布数据集:
#   DATA_MIX_OVERRIDE=hefei_relative_6d \
#     bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --max-anchors 2
#   # 产物自动落到 ..._seed42_mix-<name>/, 不覆盖原数据集结果。
#   # 注意: norm 始终用模型固化的 _per_key (训练 shenzhen 统计); 跨分布数据集的绝对数字
#   #       不可定量比较, 仅用于 pipeline/可视化/同 embodiment 泛化烟测。
#
# Stage 2 单独跑 (Stage 1 已经跑过):
#   STAGE2_INPUT_DIR=playground/Evals/<run>_step20000_seed42 \
#     bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --stage2-only
#
# 环境变量覆盖 (有默认值):
#   RUN_DIR / EVAL_YAML / STAGE2_SCRIPT / DATA_ROOT_OVERRIDE / DATA_MIX_OVERRIDE / NUM_PROCESSES
# 环境: 不再用 mamba/conda, 统一走 `uv run` (项目 .venv)。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# ── 环境: uv (不再用 mamba/conda) ─────────────────────────────────────
# 所有 python / accelerate 调用走 `uv run`, 使用项目 .venv (由 `uv sync` 创建)。
# 首次运行前装依赖: uv sync  (Stage 2 可视化另需 rerun: uv pip install rerun-sdk)
command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

# ── 默认值 (eval 参数走 EVAL_YAML, 这里只有路径 / 环境) ──────────────
# 本实例 ckpt: 扁平导出 (.pt 直接平铺在 run_dir, 无 checkpoints/ 子目录)。
# eval/checkpoint 代码 (checkpoint_utils.py + share_tools._resolve_run_dir) 已支持扁平布局,
# 故这里直接指 run_dir 即可, 无需手动 mv 进 checkpoints/。
RUN_DIR=${RUN_DIR:-/root/.models/run_umi_shenzhen_relative_6d_20000step_0727_100151}
EVAL_YAML=${EVAL_YAML:-examples/realRobots/z_umi/eval_files/eval_umi_pi05_relative_6d_overlay.yaml}
STAGE2_SCRIPT=${STAGE2_SCRIPT:-examples/realRobots/z_umi/eval_files/stage2_visualize.py}
# 数据根: ckpt config 里存的是 /nix/data/z-umi (本机已失效), 真实数据在 /data/z-umi/data。
# 默认直接覆盖到正确位置; 想换镜像再显式设 DATA_ROOT_OVERRIDE。
DATA_ROOT_OVERRIDE=${DATA_ROOT_OVERRIDE:-/data/z-umi/data}
# DATA_MIX_OVERRIDE 留空 = 沿用 ckpt 的 data_mix (shenzhen_relative_6d)。
# 换数据集评测: DATA_MIX_OVERRIDE=shenzhen_relative_6d_2 (或 hefei_relative_6d 等, 见 data_config.py)。
# 注意: norm 始终用模型固化的 _per_key (训练时 shenzhen df899c7d 统计), 跨数据集为泛化/可视化用,
#       output_dir 自动追加 _mix-<name> 隔离产物, 不覆盖原数据集结果。
DATA_MIX_OVERRIDE=${DATA_MIX_OVERRIDE:-}

# ── preflight ─────────────────────────────────────────────────────────
[[ -d "${RUN_DIR}" ]] || { echo "RUN_DIR not found: ${RUN_DIR}" >&2; exit 1; }
[[ -f "${EVAL_YAML}" ]] || { echo "EVAL_YAML not found: ${EVAL_YAML}" >&2; exit 1; }
[[ -f "${STAGE2_SCRIPT}" ]] || { echo "STAGE2_SCRIPT not found: ${STAGE2_SCRIPT}" >&2; exit 1; }

if [[ -f "${RUN_DIR}/config.full.yaml" ]]; then
  SAVED_CONFIG="${RUN_DIR}/config.full.yaml"
elif [[ -f "${RUN_DIR}/config.yaml" ]]; then
  SAVED_CONFIG="${RUN_DIR}/config.yaml"
else
  echo "Missing config.full.yaml / config.yaml in ${RUN_DIR}" >&2
  exit 1
fi

# ── base_vlm symlink (框架无关): 仅当 saved config 真有 framework.qwenvl.base_vlm
#    且指向本地相对/绝对路径时才 symlink。PI05/paligemma 没有该键 → try/except 返回空 → 跳过。──
SAVED_BASE_VLM=$(uv run python -c 'from omegaconf import OmegaConf; import sys
cfg = OmegaConf.load(sys.argv[1])
try:
    print(cfg.framework.qwenvl.base_vlm)
except Exception:
    print("")' "${SAVED_CONFIG}")

case "${SAVED_BASE_VLM}" in
  ./*) SAVED_BASE_VLM_ABS="${REPO_ROOT}/${SAVED_BASE_VLM#./}" ;;
  /*)  SAVED_BASE_VLM_ABS="${SAVED_BASE_VLM}" ;;
  *)   SAVED_BASE_VLM_ABS="" ;;  # 空 (PI05) 或 HF repo id, 都不需要 symlink
esac

if [[ -n "${SAVED_BASE_VLM_ABS}" && ! -e "${SAVED_BASE_VLM_ABS}" && -n "${BASE_VLM:-}" ]]; then
  mkdir -p "$(dirname "${SAVED_BASE_VLM_ABS}")"
  ln -sfn "${BASE_VLM}" "${SAVED_BASE_VLM_ABS}"
  echo "[run_umi_pi05_open_eval] symlinked ${SAVED_BASE_VLM_ABS} -> ${BASE_VLM}"
fi

# ── 扫 launcher-only flag, 其余透传给 Stage 1 ─────────────────────────
# 默认 Stage 1 后自动生成 .rrd; --no-visualize 关闭, --visualize 显式开启 (等价默认)
DO_VISUALIZE=1
STAGE2_ONLY=0
STAGE1_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --visualize) DO_VISUALIZE=1 ;;
    --no-visualize) DO_VISUALIZE=0 ;;
    --stage2-only) STAGE2_ONLY=1 ;;
    *) STAGE1_ARGS+=("${arg}") ;;
  esac
done

if [[ "${STAGE2_ONLY}" == "1" ]]; then
  MIX_SUFFIX=""
  [[ -n "${DATA_MIX_OVERRIDE}" ]] && MIX_SUFFIX="_mix-${DATA_MIX_OVERRIDE}"
  STAGE2_INPUT_DIR=${STAGE2_INPUT_DIR:-playground/Evals/$(basename "${RUN_DIR}")_step20000_seed42${MIX_SUFFIX}}
  if [[ ! -d "${STAGE2_INPUT_DIR}" ]]; then
    echo "STAGE2_INPUT_DIR not found: ${STAGE2_INPUT_DIR}" >&2
    exit 1
  fi
  echo "[run_umi_pi05_open_eval] Stage 2 only on ${STAGE2_INPUT_DIR}"
  # 多 episode 输出布局: 每个 episode_*/ 子目录独立 visualize
  EPISODE_DIRS=("${STAGE2_INPUT_DIR}"/episode_*/)
  if [[ ${#EPISODE_DIRS[@]} -eq 0 ]]; then
    echo "[run_umi_pi05_open_eval] no episode_*/ subdirs in ${STAGE2_INPUT_DIR}" >&2
    exit 1
  fi
  PYTHONPATH="${REPO_ROOT}" uv run python "${STAGE2_SCRIPT}" "${EPISODE_DIRS[@]}"
  exit 0
fi

echo "Launching UMI PI05 open-loop eval (Stage 1)"
echo "  run_dir:        ${RUN_DIR}"
echo "  eval_yaml:      ${EVAL_YAML}"
echo "  data_root:      ${DATA_ROOT_OVERRIDE:-(from ckpt config)}"
echo "  data_mix:       ${DATA_MIX_OVERRIDE:-(from ckpt: shenzhen_relative_6d)}"
echo "  base_vlm:       ${BASE_VLM:-n/a} (config saved: ${SAVED_BASE_VLM:-none})"
echo "  stage1_args:    ${STAGE1_ARGS[*]:-}"
echo "  visualize:      ${DO_VISUALIZE}"
echo "  num_processes:  ${NUM_PROCESSES:-1}"

# ── Stage 1 ───────────────────────────────────────────────────────────
# 输出 tee 到临时日志: Stage 2 据其中的 __EVAL_OUTPUT_DIR__ 标记精确定位 output_dir
# (旧版 `ls -td | head -1` 按 mtime 猜, 多 run 共存会串目录)。
STAGE1_LOG="$(mktemp)"
trap 'rm -f "${STAGE1_LOG}"' EXIT

# NUM_PROCESSES>1 切到 accelerate launch (单 episode 内 anchor 跨卡切分, gather 回 rank 0)
if [[ -n "${NUM_PROCESSES:-}" && "${NUM_PROCESSES}" -gt 1 ]]; then
  PYTHONPATH="${REPO_ROOT}" uv run accelerate launch --multi-gpu --num_processes "${NUM_PROCESSES}" \
    -m starVLA.training.eval_starvla \
    --run-dir "${RUN_DIR}" \
    --eval-yaml "${EVAL_YAML}" \
    $( [[ -n "${DATA_ROOT_OVERRIDE}" ]] && echo --data-root-override "${DATA_ROOT_OVERRIDE}" ) \
    $( [[ -n "${DATA_MIX_OVERRIDE}" ]] && echo --data-mix "${DATA_MIX_OVERRIDE}" ) \
    "${STAGE1_ARGS[@]}" 2>&1 | tee "${STAGE1_LOG}"
else
  PYTHONPATH="${REPO_ROOT}" uv run python -m starVLA.training.eval_starvla \
    --run-dir "${RUN_DIR}" \
    --eval-yaml "${EVAL_YAML}" \
    $( [[ -n "${DATA_ROOT_OVERRIDE}" ]] && echo --data-root-override "${DATA_ROOT_OVERRIDE}" ) \
    $( [[ -n "${DATA_MIX_OVERRIDE}" ]] && echo --data-mix "${DATA_MIX_OVERRIDE}" ) \
    "${STAGE1_ARGS[@]}" 2>&1 | tee "${STAGE1_LOG}"
fi

# ── Stage 2 (可选) ────────────────────────────────────────────────────
# 默认可视化 Stage 1 刚写出来的 output_dir 的所有 episode_*。
# 目录定位优先用 Stage 1 打的 __EVAL_OUTPUT_DIR__ 标记 (确定性); 缺失 (Stage 1 崩了 /
# 旧版没打标记) 才回退到 `ls -td | head -1` 按 mtime 取最新。
if [[ "${DO_VISUALIZE}" == "1" ]]; then
  LAST_OUTPUT="$( { grep '__EVAL_OUTPUT_DIR__' "${STAGE1_LOG}" 2>/dev/null || true; } | tail -1 | sed 's/.*__EVAL_OUTPUT_DIR__[[:space:]]*//')"
  if [[ -n "${LAST_OUTPUT}" ]]; then
    echo "[run_umi_pi05_open_eval] Stage 2 output dir (from Stage 1 marker): ${LAST_OUTPUT}"
  else
    LAST_OUTPUT="$(ls -td playground/Evals/*/ 2>/dev/null | head -1)"
    LAST_OUTPUT="${LAST_OUTPUT%/}"
    [[ -n "${LAST_OUTPUT}" ]] && \
      echo "[run_umi_pi05_open_eval] Stage 2 output dir (mtime fallback): ${LAST_OUTPUT}"
  fi
  if [[ -z "${LAST_OUTPUT}" ]]; then
    echo "[run_umi_pi05_open_eval] WARNING: no output dir found to visualize" >&2
  else
    EPISODE_DIRS=("${LAST_OUTPUT}"/episode_*/)
    if [[ ${#EPISODE_DIRS[@]} -eq 0 ]]; then
      echo "[run_umi_pi05_open_eval] WARNING: no episode_*/ in ${LAST_OUTPUT}" >&2
    else
      echo "[run_umi_pi05_open_eval] Stage 2 → ${#EPISODE_DIRS[@]} episodes under ${LAST_OUTPUT}"
      PYTHONPATH="${REPO_ROOT}" uv run python "${STAGE2_SCRIPT}" "${EPISODE_DIRS[@]}"
    fi
  fi
fi
