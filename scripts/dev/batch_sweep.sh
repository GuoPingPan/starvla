#!/usr/bin/env bash
# ============================================================================
# scripts/dev/batch_sweep.sh — 单卡 batch size 扫描 (找最大可用 / 吞吐甜点)
# ----------------------------------------------------------------------------
# 背景: A100 8卡时代 (f9ac6dc) 是手动 BATCH=8/9/10/11 逐个跑 + STARVLA_PROFILE=1
#       得出 batch=10 甜点的; 本脚本把该流程固化: 逐个 BATCH 起短训练,
#       记录 吞吐(samples/s) + 峰值显存, OOM 自动停止并汇总表格。
#
# 用法 (在已有 .venv 的 repo 根目录跑, 数据/预训练已缓存则秒进训练):
#   bash scripts/dev/batch_sweep.sh <config.yaml> [batch 列表]
#
#   例: 8 卡 H200, 从 10 扫到 32 (默认 10 12 14 16 20 24 28 32):
#     NUM_GPUS=8 bash scripts/dev/batch_sweep.sh \
#       examples/realRobots/zerith_pi05/train_files/umi_all_nohjerk_noproblem_smooth_oss.yaml
#   自定义列表:
#     NUM_GPUS=8 bash scripts/dev/batch_sweep.sh <yaml> 8 10 12 16 24
#
# 可调环境变量:
#   NUM_GPUS   卡数 (默认 nvidia-smi 计数; launcher 也读 NPROC_PER_NODE)
#   SWEEP_STEPS 每个 batch 跑的 train steps (默认 20; 太小数据侧未热身, 太大浪费)
#   WARMUP_SKIP 计吞吐时跳过前 N step (默认 5, 排除建图/首个 batch 抖动)
#   SWEEP_PROFILE=1  开 STARVLA_PROFILE=1 (perf_analyzer, SM util/通信占比,
#                     产物在各 run 目录 profile_summary.json)
#   SWEEP_KEEP=0     默认每个 run 结束即删 checkpoint 目录 (大权重很占盘); =1 保留
#   其余 BATCH 无效(被扫), MAX_STEPS/SAVE_EVERY 被脚本接管; NUM_WORKERS/PREFETCH 等透传
#
# 输出:
#   <SWEEP_OUT_ROOT>/batch_sweep_<ts>/summary.md — 每行: batch | 状态 | samples/s | 峰值显存 | run 目录
#   SWEEP_OUT_ROOT 默认 /root/data/batch_sweeps (共享盘, 可追溯, 不随 /tmp 清理丢失);
#   每轮目录含 summary.md + 各 batch 的完整训练日志(.log)/显存采样(.mem)。
# 判据: 状态 = OK / OOM / FAIL(其它错误); 首个 OOM 后停止扫描 (列表按升序)
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_YAML="${1:?用法: bash scripts/dev/batch_sweep.sh <config.yaml> [batch 列表, 默认 10 12 14 16 20 24 28 32]}"
shift || true
BATCHES=("$@")
[ ${#BATCHES[@]} -gt 0 ] || BATCHES=(10 12 14 16 20 24 28 32)

NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}}"
SWEEP_STEPS="${SWEEP_STEPS:-20}"
WARMUP_SKIP="${WARMUP_SKIP:-5}"
SWEEP_PROFILE="${SWEEP_PROFILE:-0}"
SWEEP_KEEP="${SWEEP_KEEP:-0}"
LAUNCHER="examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh"

# 环境检查: .venv + launcher + yaml
[ -f ".venv/bin/activate" ] || { echo "ERROR: 请在带 .venv 的 repo 根目录跑 (train.sh 段2 建好的, 如 /root/code/repo_<sha>)" >&2; exit 1; }
[ -f "$LAUNCHER" ] || { echo "ERROR: 未找到 launcher: $LAUNCHER" >&2; exit 1; }
[ -f "$CONFIG_YAML" ] || { echo "ERROR: yaml 不存在: $CONFIG_YAML" >&2; exit 1; }
. .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export REPO_ROOT="$PWD"
export ACCEL_BIN="$PWD/.venv/bin/accelerate"
export NUM_GPUS
export WANDB_MODE=disabled SWANLAB_MODE=local
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ "$SWEEP_PROFILE" = "1" ] && export STARVLA_PROFILE=1

TS="$(date +%m%d_%H%M%S)"
# 结果落共享盘 (/root/data 软链 -> CFS/CPFS), 不用 /tmp (清理即丢, 无法追溯)
SWEEP_OUT_ROOT="${SWEEP_OUT_ROOT:-/root/data/batch_sweeps}"
OUT_DIR="${SWEEP_OUT_ROOT}/batch_sweep_${TS}"
SUMMARY="$OUT_DIR/summary.md"
mkdir -p "$OUT_DIR"
echo "# batch sweep @ $(hostname) · ${NUM_GPUS}x$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) · $(basename "$CONFIG_YAML")" > "$SUMMARY"
echo "| batch | 状态 | samples/s | 峰值显存(卡0) | run |" >> "$SUMMARY"
echo "|---|---|---|---|---|" >> "$SUMMARY"

info() { printf '\033[1;34m>>\033[0m %s\n' "$*" >&2; }

peak_mem_gb() { awk -v m="$1" 'BEGIN{printf "%.1f", m/1024}'; }  # nvidia-smi nounits = MiB

# 从 tqdm 进度行提稳态吞吐:
#   非交互 tqdm 在 log 里留一系列 "N%|...|[MM:SS<..." 快照; 取 (首个>=skip阈值的快照)
#   到 (首个 100% 快照) 的时间差算 it/s (排除热身 + 结束后的 save/teardown),
#   samples/s = it/s × batch × NUM_GPUS
parse_tput() {
  local log="$1" bs="$2" steps="$3" skip="$4"
  python3 - "$log" "$bs" "$NUM_GPUS" "$steps" "$skip" <<'PYEOF'
import re, sys

log, bs, gpus, steps, skip = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
snaps = []  # (pct, seconds)
for m in re.finditer(rf"(\d+)%\|[^[]*\|\s*\d+/{steps} \[([\d:]+)<", open(log, errors="ignore").read()):
    pct = int(m.group(1))
    t = sum(int(x) * 60**i for i, x in enumerate(reversed(m.group(2).split(":"))))
    snaps.append((pct, t))
if not snaps:
    sys.exit(1)
skip_pct = skip * 100 // steps if steps else 0
start = next((t for pct, t in snaps if pct >= skip_pct), None)
end = next((t for pct, t in snaps if pct >= 100), None)  # 首个 100% = 循环真正结束, 不含 save
if start is None or end is None or end <= start or steps - skip <= 0:
    sys.exit(1)
rate = (steps - skip) / (end - start)
print(f"{rate * bs * gpus:.1f}")
PYEOF
}

for bs in "${BATCHES[@]}"; do
  run_id="bs_sweep_b${bs}_${TS}"
  log="$OUT_DIR/b${bs}.log"
  info "===== batch=$bs (${SWEEP_STEPS} steps) → $log ====="

  # 显存采样器: 每 2s 记录卡0 used (MiB), run 结束取 max
  mem_log="$OUT_DIR/b${bs}.mem"
  : > "$mem_log"
  while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$mem_log" 2>/dev/null; sleep 2; done &
  MEM_PID=$!

  RUN_ID="$run_id" BATCH="$bs" MAX_STEPS="$SWEEP_STEPS" SAVE_EVERY=1000000 LOG_EVERY=1 \
    bash "$LAUNCHER" "$CONFIG_YAML" >"$log" 2>&1
  rc=$?
  kill "$MEM_PID" 2>/dev/null; wait "$MEM_PID" 2>/dev/null

  peak=$(sort -n "$mem_log" 2>/dev/null | tail -1); peak=${peak:-0}
  peak_gb="$(peak_mem_gb "${peak:-0}")"
  run_dir="$(ls -td /root/data/model_outputs/"$run_id"/*/ 2>/dev/null | head -1)"

  if grep -qE "CUDA out of memory|OutOfMemoryError|Tried to allocate" "$log"; then
    status=OOM
  elif [ $rc -ne 0 ]; then
    status=FAIL
  else
    status=OK
  fi
  tput=$(parse_tput "$log" "$bs" "$SWEEP_STEPS" "$WARMUP_SKIP" 2>/dev/null) || tput="-"

  echo "| $bs | $status | $tput | ${peak_gb} GB | ${run_dir:-<无>} |" >> "$SUMMARY"
  info "batch=$bs → $status, 吞吐=${tput} samples/s, 峰值=${peak_gb} GB"

  # 默认删除本轮 ckpt (final+steps), 只留日志/汇总; swanlog 很小保留
  if [ "$SWEEP_KEEP" = "0" ] && [ -n "$run_dir" ]; then
    rm -rf "${run_dir}checkpoints" "${run_dir}final_model" 2>/dev/null
  fi

  [ "$status" = "OOM" ] && { info "首个 OOM @ batch=$bs, 停止扫描 (最大可用 = 上一个 OK 的 batch)"; break; }
  [ "$status" = "FAIL" ] && { info "非 OOM 失败 @ batch=$bs (查 $log), 停止扫描"; break; }
done

info "汇总 → $SUMMARY"
cat "$SUMMARY"
