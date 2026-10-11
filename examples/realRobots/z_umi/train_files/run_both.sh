#!/usr/bin/env bash
# 顺序训练 Hefei + Shenzhen 两个 relative_6d 配置。
#
# 用法:
#   bash run_both.sh            # 全量训练（默认）
#   bash run_both.sh smoke      # 冒烟测试（各 100 步，跑通即可）
#
# 两个任务串行执行；第一个失败也会继续跑第二个，最后统一汇报 exit code。
# 训练参数可用环境变量覆盖，例如:
#   MAX_STEPS=30000 bash run_both.sh
set -uo pipefail

MODE="${1:-full}"

case "${MODE}" in
  smoke)
    export MAX_STEPS=${MAX_STEPS:-100}
    export SAVE_EVERY=${SAVE_EVERY:-100}
    export LOG_EVERY=${LOG_EVERY:-10}
    TAG="smoke"
    ;;
  full|"")
    # 全量训练：20000 步，每 5000 步存一次，每 50 步记一次 log
    export MAX_STEPS=${MAX_STEPS:-20000}
    export SAVE_EVERY=${SAVE_EVERY:-5000}
    export LOG_EVERY=${LOG_EVERY:-50}
    TAG="full"
    ;;
  *)
    echo "用法: $0 [smoke|full]" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

LAUNCHER="examples/realRobots/z_umi/train_files/run_umi_pi05_relative_6d.sh"
HEFEI_YAML="examples/realRobots/z_umi/train_files/run_umi_hefei_relative_6d.yaml"
SHENZHEN_YAML="examples/realRobots/z_umi/train_files/run_umi_shenzhen_relative_6d.yaml"

echo "########## MODE=${MODE} · MAX_STEPS=${MAX_STEPS} · SAVE_EVERY=${SAVE_EVERY} ##########"

echo "==================== [1/2] Hefei · relative_6d · ${MAX_STEPS} steps ===================="
bash "${LAUNCHER}" "${HEFEI_YAML}" "${TAG}"
hefei_status=$?

echo "==================== [2/2] Shenzhen · relative_6d · ${MAX_STEPS} steps ===================="
bash "${LAUNCHER}" "${SHENZHEN_YAML}" "${TAG}"
shenzhen_status=$?

echo "==================== DONE ===================="
echo "Hefei    exit: ${hefei_status}"
echo "Shenzhen exit: ${shenzhen_status}"
# 两个都成功才返回 0
[ "${hefei_status}" -eq 0 ] && [ "${shenzhen_status}" -eq 0 ]
