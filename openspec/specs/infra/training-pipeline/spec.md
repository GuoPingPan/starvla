# train.sh 训练流水线（Training Pipeline）

## Purpose

`scripts/train.sh` 6 段集群训练流水线的行为合同：从「敲下回车到训练结束上传」每一步做什么、产物落在哪里、参数如何覆盖。核心点：训练不是直接在当前仓库跑，而是按 commit 从 BOS 拉 CI 产出的源码 zip。完整叙述见 `docs/zerith_docs/infra/train_sh_run_flow.md`。

## Requirements

### Requirement: 一条命令 = 6 段流水线

`train.sh` SHALL 是 6 段接力：①配置环境（凭证/路径/ENV）→ ②拉代码（zip→`repo_<sha>`）→ ③拉数据（按 yaml）→ ④拉预训练 → ⑤训练（accelerate）→ ⑥上传全部产物（BOS）。

#### Scenario: 训练在解压代码上跑
- **WHEN** 运行 `train.sh`
- **THEN** 训练不是在当前工作区跑，而是段 2 从 BOS 拉 zip 解压到 `/root/data/code/repo_<sha>` 建 venv 后，`cd` 进去训练，当前工作区改动不影响本次训练

### Requirement: 段 2 按 commit 拉代码并建环境

段 2 SHALL 用 `bos_io.py src-zip <commit>` 按 commit 前缀匹配下载源码 zip（命名 `starvla_<ref>_<ts>_<sha>.zip`），从 zip 名末段解析真实 sha；`flock` 串行化「解压 + uv sync」（同 commit 多任务排队），`.venv` 已存在则直接复用；最后 `cd` 进 `repo_<sha>`、激活 venv、补装 boto3、设 `PYTHONPATH`。

#### Scenario: 同 commit 并发任务
- **WHEN** 多个任务用同一个 commit
- **THEN** `flock` 串行化，第二个任务拿到锁后看到 `.venv` 已存在 → 直接复用，跳过解压/uv sync

#### Scenario: zip 不存在
- **WHEN** BOS 上找不到该 commit 的 zip（未 push 等 CI）
- **THEN** 报 `ERROR: 没拉到源码 zip (先跑 GitLab CI 上传产物到 starvla_res/)` 并退出

### Requirement: 段 3 按 yaml 拉数据（幂等）

段 3 SHALL 跑 `scripts/download_data.py <CONFIG_YAML>`：用 ast（不 import）从 `data_config.py` 抽取 `DATASET_NAMED_MIXTURES`，取 `data_mix` 的条目；每条看元组第 4 元素（`bos_uri`），有则下载 zip→解压到 `<data_root>/<dataset_id>/`（幂等标志 = `meta/info.json` 存在即跳过，先下临时目录再解压防半成品，`flatten_if_wrapped` 统一布局），无（空串）则打 `⚠` 警告并跳过。

#### Scenario: 本地派生数据无 bos_uri
- **WHEN** forceclose 数据集第 4 元素是空串
- **THEN** 段 3 跳过下载并打警告；该数据必须已预置在 `/root/data/z-umi/data/<id>/`，否则段 5 训练报「找不到数据集」

### Requirement: 段 4 递归拉预训练（幂等）

段 4 SHALL 用 `bos_io.py get-dir` 把 `Orion/pretrained/` 整目录递归下载到 `/root/data/pretrain_models/`（`--exclude '.hfd/*'`）；幂等规则：本地文件已存在**且大小一致** → 跳过（16GB 预训练不重复下）；重建相对路径使远端/本地目录树一一对应。

#### Scenario: 重复跑 train.sh
- **WHEN** 预训练已存在且大小一致
- **THEN** 段 4 秒过，不重复下载

### Requirement: 段 5 参数优先级 env > launcher 默认 > yaml

`train.sh` 导出的 env（`BATCH`/`MAX_STEPS`/`SAVE_EVERY`/`NUM_WORKERS`/`PREFETCH`/`WANDB_PROJECT` 等）SHALL 优先级高于 launcher 默认，launcher 默认高于 yaml；launcher 把这些组装成 `--datasets.vla_data.*` / `--trainer.*` CLI 覆盖传给 `train_starvla.py`。

#### Scenario: BATCH 覆盖 yaml
- **WHEN** 命令带 `BATCH=2`，yaml `per_device_batch_size: 4`
- **THEN** 实际以 `--datasets.vla_data.per_device_batch_size 2` 覆盖 yaml，单卡 batch=2

### Requirement: RUN_ID 派生与产物目录

`RUN_ID` SHALL 默认 = `<yaml文件名>_<MAX_STEPS>step`，可被 `RUN_ID` env 覆盖；输出目录 `output_dir=/root/data/model_outputs/<RUN_ID>/`（非 resume 带时间戳子目录 `<MMDD_HHMMSS>/`），launcher 把脚本与 yaml 拷进去留档；`REPO_ROOT` 显式传解压目录（该目录无 `.git`），`ACCEL_BIN` 指向 venv 里的 accelerate。

#### Scenario: RUN_ID 默认派生
- **WHEN** 不传 `RUN_ID`，yaml 文件名为 `run_umi_xxx`、`MAX_STEPS=20000`
- **THEN** `RUN_ID=run_umi_xxx_20000step`，产物落到 `/root/data/model_outputs/run_umi_xxx_20000step/`

### Requirement: 段 5 accelerate + DeepSpeed

段 5 SHALL 用 `accelerate launch --config_file deepspeed_zero2.yaml --num_processes <GPU数>` fork N 个进程跑 `train_starvla.py`；DeepSpeed 内层 `ds_config.yaml`：ZeRO-2、bf16 开、fp16 关、`gradient_accumulation_steps: auto`（运行时值）、`gradient_clipping: 1.0`；单机只加 `--num_processes`，多机才追加 `--num_machines/--machine_rank/--main_process_ip`。

#### Scenario: --local 走 SwanLab 本地模式
- **WHEN** 命令带 `--local`
- **THEN** `SWANLAB_MODE=local` 传到 `train_starvla.py`，`swanlab.init(mode="local")` 不需登录，日志写到 `output_dir/swanlog/`，可用 `swanlab watch` 看面板

### Requirement: 段 6 上传全部产物

段 6 SHALL 用 `bos_io.py put-dir` 把整个 run 目录（含时间戳子目录、checkpoints、config、swanlog）递归上传到 `bos://umi-gripper/Orion/ckpts/<RUN_ID>/`，防止训练中断后无法恢复。

#### Scenario: 上传保留目录结构
- **WHEN** 上传 run 目录
- **THEN** 递归遍历本地所有文件，按相对路径重建云端 key，目录结构一一对应

## References

- 完整流水线详解：`docs/zerith_docs/infra/train_sh_run_flow.md`
- CI 与代码来源：`docs/zerith_docs/workflow/ci_and_training_source.md`
- 主入口：`scripts/train.sh`、`scripts/bos_io.py`、`scripts/download_data.py`
