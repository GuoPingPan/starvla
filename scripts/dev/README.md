# scripts/dev/ — 开发 & agent 自动化工具

> 本目录是 **开发期工具**，供开发者和 AI coding agent 使用。
> **不属于集群运行时**：`scripts/train.sh`（集群训练流水线）不会调用这里的任何脚本。
> 因此这里脚本的变更不影响训练 pipeline。

## 与 `scripts/`（根）的分工

| 位置 | 用途 | 谁用 |
|------|------|------|
| `scripts/`（根） | 运行时 / CI / 集群 / 数据工具 | 任务队列、人（跑训练/拉数据） |
| `scripts/dev/`（本目录） | 开发工作流：worktree、测试、补丁自检 | 开发者、AI agent |

## 脚本一览

| 脚本 | 作用 | 典型调用 |
|------|------|---------|
| `new-worktree.sh` | 为 feature 分支建 git worktree + uv 环境 | `bash scripts/dev/new-worktree.sh feat/my-topic` |
| `check_patches.py` | 自检本分支改动的上游文件是否都登记 | `python scripts/dev/check_patches.py` |
| `smoke.sh` | 模块级 smoke（单框架/数据 loader 独立跑） | `bash scripts/dev/smoke.sh QwenOFT <yaml>` |
| `overfit.sh` | overfit smoke（1 episode、几百步、1 GPU） | `bash scripts/dev/overfit.sh <yaml> 300` |

测试命令直接调用本目录脚本（`uv run pytest tests/` / `bash scripts/dev/smoke.sh` / `bash scripts/dev/overfit.sh`），不修改上游 `Makefile`。

## 相关文档

- 开发流程总览：`docs/zerith_docs/workflow/upstream_sync_and_isolation.md`
- worktree 工作流：`docs/zerith_docs/workflow/worktree_dev.md`
- 测试分层：`docs/zerith_docs/workflow/testing_layers.md`
- 上游补丁登记：`docs/zerith_docs/workflow/upstream_patches.md`
