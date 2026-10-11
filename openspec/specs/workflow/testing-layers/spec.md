# 测试分层（Testing Layers）

## Purpose

把「测试 = 跑 `scripts/train.sh`」拆成分层，让绝大多数改动用远比集群流水线更轻的层验证。`train.sh` 是按 commit 拉 BOS 代码 zip → uv sync → 拉数据 → 拉预训练 → 训练 → 上传的完整集群流水线，每次跑要多小时，不是测试。完整叙述见 `docs/zerith_docs/workflow/testing_layers.md`。

## Requirements

### Requirement: 五个测试层

系统 SHALL 定义五个测试层，每层有明确的命令、耗时、资源与验证目标：单元测试（`uv run pytest tests/`，秒，CPU，验证数学/transform/config）；模块 smoke（`bash scripts/dev/smoke.sh <框架> <yaml>`，秒~分，1 GPU，验证单框架 forward/dataloader 形状）；Overfit smoke（`bash scripts/dev/overfit.sh <yaml>`，分，1 GPU，验证 1 episode 几百步 loss 能否→0）；短 train.sh（`MAX_STEPS=50 ... --local`，~10 分，1 GPU，验证 BOS 全链路）；完整 train.sh（多卡，小时，正式实验）。

#### Scenario: 选层依据资源与目标
- **WHEN** 开发者要验证 transform/数学改动
- **THEN** 跑单元测试 + overfit smoke（1 GPU），不上集群

#### Scenario: 完整实验才上集群
- **WHEN** 需要正式实验或复现
- **THEN** 才使用完整 `train.sh`（多卡、小时级）

### Requirement: 改动类型到测试层的映射

不同改动类型 SHALL 对应必须跑的层：transform/数学/config → 单元 + overfit；框架 forward/dataloader 入口 → 模块 smoke + overfit；trainer/训练入口/overfit 逻辑 → overfit；`train.sh`/CI/数据下载 → 短 train.sh；新数据集注册 → overfit；正式实验 → 完整。

#### Scenario: 改了共享 dataloader 接口
- **WHEN** 改动触及框架 forward 或 dataloader 入口
- **THEN** 必须跑模块 smoke 与 overfit，确认形状与拟合链路未被破坏

### Requirement: overfit 结果只能是拟合证据

Overfit smoke 与单 episode 过拟合的结果 MUST 只作为 pipeline 状态和 fitting evidence，**不能作为泛化证据**；任何基于训练样本的诊断结论都不得表述为「未见数据泛化」。

#### Scenario: 报告 overfit 结论
- **WHEN** 单 episode 过拟合 loss 下降
- **THEN** 结论只能记为「pipeline 通过 / 正向拟合证据」，泛化维度记为「未评估」

### Requirement: 本地 PR 门禁

提交 PR 前本地 MUST 通过：`black --check`、`ruff check`（仅限改动的 `.py`）、`pytest tests/`；若改了上游文件还需 `python scripts/dev/check_patches.py`。

#### Scenario: 改了上游文件
- **WHEN** PR 涉及上游既有文件
- **THEN** 门禁额外要求 `check_patches.py` 通过，确认改动已登记

### Requirement: 本地与集群的分工

开发机有本地 GPU 时，单元/模块 smoke/overfit SHALL 在 worktree 的本地 `.venv` 跑；只有短 train.sh 与完整 train.sh 需要走 BOS（拉数据/预训练），其中短 train.sh 也可用 `--local` + `MAX_STEPS=50` 在本地 1 GPU 冒烟。

#### Scenario: 短 train.sh 本地冒烟
- **WHEN** 需要验证 BOS 全链路但不想要完整训练
- **THEN** 用 `MAX_STEPS=50 BATCH=2 ... --local` 在本地 1 GPU 跑约 10 分钟冒烟

## References

- 完整分层与映射表：`docs/zerith_docs/workflow/testing_layers.md`
- CI 与代码来源：`docs/zerith_docs/workflow/ci_and_training_source.md`
- 测试入口脚本：`scripts/dev/smoke.sh`、`scripts/dev/overfit.sh`
