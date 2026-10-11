# Worktree 开发工作流

> 用 **git worktree** 让每个 feature 分支有独立的工作目录，主仓库只留给 `dev/main`。
> 配合 **uv** 每个 worktree 独立 `.venv`，靠全局缓存复用，建环境秒级、互不污染。

---

## 1. 目录约定

| 位置 | 用途 |
|------|------|
| `/root/workspace/vla/starvla` | **主仓库**，checkout 固定留在 `dev/main` |
| `../starvla.worktrees/<分支名-连字符>` | **worktree**，每个 feature 分支一个 |

- 分支名里的 `/` 转连字符做目录名（文件名不能含 `/`）：`feat/my-topic` → `starvla.worktrees/feat-my-topic`。
- **worktree 不要建在主仓库目录树内部**（历史曾出现过 `.claude/worktrees/...`，应移出）。

---

## 2. 一键建 worktree（推荐）

```bash
bash scripts/dev/new-worktree.sh feat/my-topic
```

脚本自动完成：
1. 更新本地 `dev/main` 到最新；
2. 从 `dev/main` 创建分支 `feat/my-topic` + worktree `../starvla.worktrees/feat-my-topic`；
3. 在 worktree 内 `uv sync --python 3.10` 建独立 `.venv`（已存在则跳过）。

完成后：
```bash
cd ../starvla.worktrees/feat-my-topic
source .venv/bin/activate
```

---

## 3. uv 环境策略

- **每个 worktree 独立 `.venv`**（隔离性好，不同分支可装不同依赖版本互不影响）。
- **uv 全局缓存**（默认 `~/.cache/uv`）用硬链接复用已下载的包，所以新建 `.venv` 很快（秒级，不会每次重下 torch）。
- 确保缓存目录在**大盘**上：`export UV_CACHE_DIR=/mnt/cfs/.uv-cache`（或类似），否则多 worktree 会撑爆系统盘。
- 重建环境：`cd <worktree> && rm -rf .venv && uv sync --python 3.10`。

> CI / `train.sh` 走的是另一套（按 commit 从 BOS 拉 zip + `uv sync`，见 [`ci_and_training_source.md`](ci_and_training_source.md)），与 worktree 的本地 `.venv` 互不影响。

---

## 4. 常用操作

| 操作 | 命令 |
|------|------|
| 列出所有 worktree | `git worktree list` |
| 开发完移除 worktree | `git worktree remove ../starvla.worktrees/feat-my-topic` |
| 删本地分支 | `git branch -d feat/my-topic` |
| 清理失效 worktree 记录 | `git worktree prune` |

> 移除 worktree 前确认改动已 push / 合并。`worktree prune` 清理那些目录已被删但 git 仍有记录的项（`git worktree list` 里标 `prunable` 的）。

---

## 5. 注意事项

- 主仓库 `/root/workspace/vla/starvla` **不要**用来开 feature 分支，保持在 `dev/main`，避免与 worktree 抢分支。
- worktree 与主仓库**共享同一个 `.git`**（对象库），所以 `fetch` 一次全可见，但 `.venv` / 构建产物各 worktree 独立。
- 同一分支**不能**同时被两个 worktree checkout（git 会拒绝）。

---

## 相关文档
- [`upstream_sync_and_isolation.md`](upstream_sync_and_isolation.md) — 分支模型与隔离策略
- [`testing_layers.md`](testing_layers.md) — worktree 内的本地测试
- [`git_merge_rebase_guide.md`](git_merge_rebase_guide.md) — 合回 dev/main
- `scripts/dev/new-worktree.sh` — 建工作树脚本
