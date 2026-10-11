#!/usr/bin/env bash
# ============================================================================
# check_job.sh — 百舸训练任务巡检 (状态/步进/健康/异常一屏)
# 用法: bash scripts/aihc/check_job.sh <job_id> [master-pod 后缀默认 master-0]
# 需: aihc CLI 已配置 (aihc config)
# ============================================================================
set -euo pipefail
J="${1:?用法: check_job.sh <job_id>}"
POD="${2:-$J-master-0}"
P="aihc-serverless"; Q="aihcq-gwzuewk6egoh"

echo "═══ $J @ $(date +%H:%M:%S) ═══"
timeout 20 aihc job get "$J" -p $P -q $Q </dev/null 2>&1 | grep -aE "status|runningat" | head -2

TS=$(date -d '-12 minutes' +%s)
LOGS=$(timeout 60 aihc job logs "$J" --podname "$POD" -s "$TS" -m 1000 -p $P -q $Q </dev/null 2>&1 | tr "\r" "\n")

echo "--- 最近 Step (loss/grad_norm/步时):"
echo "$LOGS" | grep -aoE "Step [0-9]+ \| action_dit_loss=[0-9.]+ grad_norm=[0-9.]+ timing/data=[0-9.e-]+ timing/model=[0-9.]+" | tail -4

echo "--- tok memo:"
echo "$LOGS" | grep -aoE "tok_cache/hit_rate=[0-9.]+" | tail -2

echo "--- 异常扫描:"
echo "$LOGS" | grep -aiE "traceback|out of memory|nan |inf " | grep -av "info" | head -3 || true
echo "$LOGS" | grep -acE "\[audit\]" | sed 's/^/audit 触发: /'

echo "--- 大停顿 (data>5s 或 model>3s):"
echo "$LOGS" | grep -aoE "timing/data=[0-9]{1,3}\.|timing/model=[0-9]{2,3}\." | wc -l | sed 's/^/计数: /'
