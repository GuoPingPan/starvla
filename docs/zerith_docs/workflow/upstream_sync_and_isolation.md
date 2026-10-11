# 上游同步与隔离开发规范

> 本仓库基于开源 [starVLA](https://github.com/starVLA/starVLA)（上游分支 `starVLA_dev`）。
> 本文件规定：**如何持续同步上游**，以及**如何让自有改动尽量不污染上游代码**，从而把每次同步的冲突面降到最小。
>
> 个人分支 → `dev/main` 的 rebase/MR 流程见 [`git_merge_rebase_guide.md`](git_merge_rebase_guide.md)；本文件只讲**上游 → `dev/main`** 与**隔离策略**。

---

## 1. 分支模型

| 分支 / 远端 | 角色 | 谁能动 |
|-------------|------|--------|
| `github/starVLA_dev`（远端 `github`） | 开源上游，**只读镜像**。只 `fetch`，从不 commit | 无人 |
| `dev/main`（远端 `origin`） | **长期集成分支** = 上游 + 我们全部补丁的合并态 | 所有人（经 MR） |
| `feat/<topic>` / `dev/<人>` | feature 分支，从 `dev/main` 拉出，在 worktree 里开发 | 个人 |

```
github/starVLA_dev ──(fetch, 只读)──►  上游
        │
        │  merge (每月级；dev/main 长期共享, 保留双历史, 不 rebase)
        ▼
     dev/main   =  上游 + 我们的补丁   ◄── feature 分支合并到此
        ▲
        │  merge / squash merge  (feature 分支先 rebase 到 dev/main)
   feat/<topic>   ──► worktree 开发
```

> `origin`（GitLab `gitlab.zerith.com`）是内部远端；`github` 是开源上游远端。两者都要保留。

---

## 2. 上游 → `dev/main`：merge 同步流程（每月级）

**用 `merge` 而非 `rebase`**：`dev/main` 是长期共享集成分支，rebase 会改写所有人依赖的历史；每月级同步的上游 delta 很大，merge 保留双历史、冲突边界清晰（只在登记过的文件里）。

```bash
# 1. 更新本地 dev/main 到最新
git fetch origin
git checkout dev/main
git pull origin dev/main

# 2. 拉上游
git fetch github

# 3. 合并上游 (产生 merge commit; 冲突只在登记过的上游文件)
git merge github/starVLA_dev
```

### 冲突处理
冲突**只会**出现在 [`upstream_patches.md`](upstream_patches.md) 登记的 16 个文件里：

```bash
git status                          # 看冲突文件
# 逐个对照 upstream_patches.md: 该文件理应在这里 (若不在 → 有漏登记, 先补登记)
# 保留上游未冲突部分 + 保留我们的 ZERITH PATCH 区块; 同区域都改了则按区块语义手动合并
git add <file>
git commit                          # 完成 merge
```

### 同步后必做
1. 更新 `upstream_patches.md` 顶部"最近核对的上游 SHA"为本轮上游 HEAD。
2. 跑测试分层验证（见 [`testing_layers.md`](testing_layers.md)）：至少 `uv run pytest tests/` + 一次 overfit smoke。
3. `git push origin dev/main`。

> 若冲突量大，可先在一个临时分支 `chore/sync-upstream-<date>` 上做 merge，验证通过后再合回 `dev/main`。

---

## 3. 隔离策略：增量优先

目标：让自有改动**尽量**是新增文件（A），而非改上游既有文件（M）——前者零冲突，后者每次同步都要人工 merge。

### 3.1 必须改上游文件时
- **先查 [`upstream_patches.md`](upstream_patches.md)**：该文件是否已登记？没登记就新增一行（Patch ID / 文件 / 原因 / 负责人）。
- **改动收敛在 `ZERITH PATCH` 标记区块内**（约定见 upstream_patches.md「标记区块约定」）。
- **改完自检**：`python scripts/dev/check_patches.py`（报出本次改动里未登记的上游文件）。

### 3.2 新增能力优先用扩展点（零冲突）
| 需求 | 推荐做法（增量） | 不要做（污染上游） |
|------|------------------|--------------------|
| 新框架 | 在 `starVLA/model/framework/VLM4A/` 新建文件，用 `@FRAMEWORK_REGISTRY.register()` 注册 | 改 `PI0.py` / `PI05.py` 等既有框架核心 |
| 新数据集 | 注册到 `examples/<bench>/train_files/data_registry/` | 改共享 dataloader 接口 |
| 新 transform | 新建独立文件，在 `transform/__init__.py` 注册 | 改既有 transform 文件 |
| 新 benchmark / 机器人 | 在 `examples/<bench>/` 新建目录 | 改 `examples/` 既有目录 |

> 现状好范例：UMI 的 5 个新 transform（`normalization.py` / `pose_rotation.py` / `quaternion_to_6d.py` / `relative_pose.py` / `rotation_utils.py`）全部以新文件落地，没改上游 transform 文件本身。

### 3.3 现存的上游修改
`dev/main` 当前对上游有 **16 个 M 文件**（详见 upstream_patches.md；已从 21 个降至 16 个，5 个补丁已增量化，见 [`patch_reduction_plan.md`](patch_reduction_plan.md)）。这些是历史既成的核心行为改动（action mode、rotation_6d、UMI 数据布局、trainer overfit 模式等），多数无法纯增量替代。治理方式：登记 + 标记区块 + 每次同步重点人工 review（其中 `datasets.py`、`train_starvla.py`、`state_action.py`、`PI0.py`、`trainer_tools.py` 改动最重）。

---

## 4. feature 分支生命周期

1. 建 worktree（从 `dev/main` 拉）：`bash scripts/dev/new-worktree.sh feat/<topic>`（见 [`worktree_dev.md`](worktree_dev.md)）。
2. 开发，用测试分层自验（见 [`testing_layers.md`](testing_layers.md)）。
3. 合回前 rebase 到最新 `dev/main`：`git rebase dev/main`（流程见 `git_merge_rebase_guide.md`）。
4. 跑 `python scripts/dev/check_patches.py`：确认改动的上游文件都已登记。
5. 发 MR → `dev/main`（Squash Merge）。

---

## 5. 速查

| 操作 | 命令 |
|------|------|
| 同步上游 | `git fetch github && git checkout dev/main && git merge github/starVLA_dev` |
| 更新本地 dev/main | `git fetch origin dev/main:dev/main`（非 ff 自动拒绝，安全） |
| 检查未登记上游改动 | `python scripts/dev/check_patches.py` |
| 上游改动登记表 | `docs/zerith_docs/workflow/upstream_patches.md` |
| 个人分支 → dev/main | 见 `git_merge_rebase_guide.md`（rebase + force-with-lease + MR） |

---

## 相关文档
- [`upstream_patches.md`](upstream_patches.md) — 上游补丁登记表
- [`patch_reduction_plan.md`](patch_reduction_plan.md) — 补丁瘦身方案（M→A 转化）
- [`git_merge_rebase_guide.md`](git_merge_rebase_guide.md) — 个人分支合并流程（rebase）
- [`worktree_dev.md`](worktree_dev.md) — worktree + uv 开发工作流
- [`testing_layers.md`](testing_layers.md) — 测试分层
