# zerith 内部文档索引

> 本目录是 zerith 团队的**内部文档**（非上游 starVLA 自带），按主题分 4 个子目录。
> 上游同步与隔离规则、通用 agent 指引见仓库根 [`agent.md`](../../agent.md)。

## 分类总览

| 目录 | 主题 | 文件数 |
|------|------|--------|
| [`workflow/`](workflow/) | 开发流程与上游治理（同步 / 隔离 / worktree / 测试 / 分支 / CI） | 7 |
| [`umi/`](umi/) | UMI · PI05 子项目（训练计划 / 评测 / 冒烟 / 部署） | 5 |
| [`infra/`](infra/) | 基础设施与运维（train.sh 流水线 / BOS / 代理 / 安装） | 4 |
| [`guides/`](guides/) | 教程与参考（bash / VSCode 调试 / 项目总结） | 3 |

---

## workflow/ — 开发流程与上游治理

- [`upstream_sync_and_isolation.md`](workflow/upstream_sync_and_isolation.md) — 上游 `starVLA_dev` → `dev/main` 的 merge 同步流程与隔离策略
- [`upstream_patches.md`](workflow/upstream_patches.md) — 上游补丁登记表（M 文件清单 + 标记区块约定）
- [`patch_reduction_plan.md`](workflow/patch_reduction_plan.md) — 上游补丁瘦身方案（M→A 转化分析与路线图）
- [`worktree_dev.md`](workflow/worktree_dev.md) — git worktree + uv 开发工作流
- [`testing_layers.md`](workflow/testing_layers.md) — 测试分层（单元 / smoke / overfit / 短train / 完整train）
- [`git_merge_rebase_guide.md`](workflow/git_merge_rebase_guide.md) — 个人分支 → `dev/main` 的 rebase + MR 流程
- [`ci_and_training_source.md`](workflow/ci_and_training_source.md) — CI 打包 / commit ↔ BOS zip 关系

## umi/ — UMI · PI05 子项目

- [`umi_training_plan.md`](umi/umi_training_plan.md) — UMI 训练计划（**单一真实来源**：数据流 / normalizer / 架构）
- [`umi_non_zumi_code_changes.md`](umi/umi_non_zumi_code_changes.md) — UMI 工作对 `examples/realRobots/zerith_pi05/` 之外代码的改动登记
- [`umi_pi05_open_eval.md`](umi/umi_pi05_open_eval.md) — UMI · PI05 开环评测（open-eval）说明
- [`umi_pi05_smoke_test.md`](umi/umi_pi05_smoke_test.md) — UMI PI05 冒烟测试记录
- [`run_umi_pi05_serve.md`](umi/run_umi_pi05_serve.md) — UMI · PI05 部署启动命令

> `启动命令.md`（UMI 训练启动命令速查）尚未提交，提交后应归入本目录。

## infra/ — 基础设施与运维

- [`train_sh_run_flow.md`](infra/train_sh_run_flow.md) — `scripts/train.sh` 训练流水线运行流程详解
- [`bos_ckpt.md`](infra/bos_ckpt.md) — `bos_ckpt.sh`（BOS checkpoint 上传/下载/列举）说明
- [`proxy_issue_fix.md`](infra/proxy_issue_fix.md) — 代理冲突问题排查与修复记录
- [`install.sh`](infra/install.sh) — 一键安装脚本

## guides/ — 教程与参考

- [`StarVLA_项目总结.md`](guides/StarVLA_项目总结.md) — StarVLA 项目总结（总览）
- [`bash_basics.md`](guides/bash_basics.md) — Bash 基础语法速学手册（以 `train.sh` 为例）
- [`vscode_debug_launch.md`](guides/vscode_debug_launch.md) — VSCode `launch.json` 调试配置详解

---

## 新增文档须知
- 新文档归入对应子目录（主题不明确时先放 `workflow/` 或征询）。
- `.gitignore` 的 `*/**/*.md` 规则会忽略子目录下新建的 `.md`，**添加时需 `git add -f`**。
- 文档间互链：同目录用相对路径（`xxx.md`），跨目录用根相对路径（`docs/zerith_docs/<桶>/xxx.md`）。
