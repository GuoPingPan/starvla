#!/usr/bin/env bash
# ============================================================================
# scripts/dev/new-worktree.sh — 为 feature 分支创建 git worktree + uv 环境
# ----------------------------------------------------------------------------
# 用法:
#   bash scripts/dev/new-worktree.sh <分支名>
#   bash scripts/dev/new-worktree.sh feat/my-topic        # 斜杠自动转连字符
#
# 行为:
#   1. 从 dev/main(最新集成分支) 创建/复用指定分支
#   2. 在 ../starvla.worktrees/<分支名-连字符> 建 worktree (与主仓库同级)
#   3. 在 worktree 内 uv sync --python 3.10 建独立 .venv (uv 全局缓存复用, 秒级)
#
> 设计: 主仓库 /root/workspace/vla/starvla 只留给 dev/main; 所有活跃 feature 工作放 worktree。
# 详见 docs/zerith_docs/workflow/worktree_dev.md
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE_ROOT="$(cd "$REPO_ROOT/.." && pwd)/starvla.worktrees"
BASE_BRANCH="${BASE_BRANCH:-dev/main}"
PYTHON_VER="${PYTHON_VER:-3.10}"

if [ "$#" -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "用法: $0 <分支名>   (从 $BASE_BRANCH 创建)" >&2
  echo "  例: $0 feat/my-topic" >&2
  exit 1
fi

BRANCH="$1"
# 分支名里的 / 转连字符作为 worktree 目录名 (文件名不能含 /)
DIR_NAME="${BRANCH//\//-}"
WT_PATH="$WORKTREE_ROOT/$DIR_NAME"

# 先更新本地基线分支, 保证 worktree 基于最新集成态
echo ">> 更新基线 $BASE_BRANCH ..."
git fetch origin "$BASE_BRANCH:$BASE_BRANCH" 2>/dev/null || git fetch origin "$BASE_BRANCH"

mkdir -p "$WORKTREE_ROOT"

if [ -d "$WT_PATH" ]; then
  echo ">> worktree 已存在: $WT_PATH  (跳过创建, 直接建环境)" >&2
else
  # -b: 若分支不存在则从 BASE_BRANCH 创建; 若已存在则直接 checkout
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo ">> 分支已存在, 直接 add worktree: $BRANCH"
    git worktree add "$WT_PATH" "$BRANCH"
  else
    echo ">> 从 $BASE_BRANCH 创建分支 $BRANCH + worktree"
    git worktree add -b "$BRANCH" "$WT_PATH" "$BASE_BRANCH"
  fi
fi

echo ">> 在 worktree 内 uv sync (python $PYTHON_VER) ..."
(
  cd "$WT_PATH"
  if [ -d ".venv" ]; then
    echo ">> .venv 已存在, 跳过 uv sync"
  else
    uv sync --python "$PYTHON_VER"
  fi
)

echo
echo "✅ 完成。进入 worktree 开发:"
echo "   cd $WT_PATH"
echo "   source .venv/bin/activate"
echo
echo "清理 (开发完): git worktree remove $WT_PATH && git branch -d $BRANCH"
echo "定期清理失效 worktree: git worktree prune"
