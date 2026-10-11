# CI 与训练代码来源（commit ↔ BOS zip）

## Purpose

明确训练用的代码不是本地工作区，而是「GitLab CI 打包 → 上传 BOS → `train.sh` 段 2 下载」的 zip。`push` 是触发 CI 的事件，BOS 是产物存放处。完整叙述见 `docs/zerith_docs/workflow/ci_and_training_source.md`。

## Requirements

### Requirement: 训练代码来自 BOS zip

`train.sh` 段 2 SHALL 从 **BOS** 按 commit 拉取源码 zip，**不直接访问 GitLab、不看本地 git**；「本地有代码」≠「能训练」，必须 BOS 上有对应 commit 的 zip。

#### Scenario: 本地改了但没 push
- **WHEN** 本地 commit 未 push
- **THEN** GitLab 无记录 → CI 不跑 → BOS 无 zip → `train.sh` 段 2 报「找不到 commit xxx 的 zip」

### Requirement: push 触发 CI 流水线

CI SHALL 由「push 事件本身」触发：GitLab 服务器收到新提交后自动读 `.gitlab-ci.yml` 创建 pipeline；当前 `rules` 被注释，**每次 push 都跑**（任何分支）。

#### Scenario: 收窄触发分支
- **WHEN** 需要限制只在特定分支跑 CI
- **THEN** 取消 `.gitlab-ci.yml` 中 `rules` 注释并按需改，如 `if: $CI_COMMIT_BRANCH == "dev/Alex"`

### Requirement: zip 命名规则

CI 产出的源码 zip SHALL 命名为 `starvla_<ref>_<时间戳>_<commit短号>.zip`，其中分支里的 `/` 会被替换成 `-`（文件名不能含 `/`），例如 `starvla_dev-Alex_20260806102232_22cd18a0.zip`。

#### Scenario: 分支名含斜杠
- **WHEN** 分支为 `dev/Alex`
- **THEN** zip 名里 ref 段为 `dev-Alex`

### Requirement: 没 push 时的三条出路

本地 commit 没 push 时 SHALL 可：①用已打包的旧 commit 训练（`bash scripts/train.sh <yaml> <旧commit>`）；②push 等 CI 打包后用新 commit；③绕过 train.sh 直跑本地工作区代码（`ACCEL_BIN`/`REPO_ROOT` 覆盖 launcher）。

#### Scenario: 立刻训练本地未 push 代码
- **WHEN** 需要立刻训练本地改动且不想等 CI
- **THEN** 用方式 ③：`env ACCEL_BIN=$PWD/.venv/bin/accelerate REPO_ROOT=$PWD ...` 直跑 launcher

### Requirement: 检查 commit 是否在 BOS

SHALL 可用 `bos_io.py ls` 列目录 + grep commit 号判断（有输出=存在），或直接试下载（`bos_io.py src-zip <commit>`，能下载=存在）；注意 `bos_io.py` 用 `bos://`（双斜杠），`bcecmd` 用 `bos:`（单斜杠）。

#### Scenario: 判断 commit 可训练
- **WHEN** 想确认某 commit 能否被 `train.sh` 训练
- **THEN** 用 `bos_io.py ls "bos://umi-gripper/starvla_res/" | grep <commit短号>`，有输出即可训练

## References

- 完整叙述：`docs/zerith_docs/workflow/ci_and_training_source.md`
- 流水线段 2 详解：`docs/zerith_docs/infra/train_sh_run_flow.md`
- CI 配置：`.gitlab-ci.yml`
