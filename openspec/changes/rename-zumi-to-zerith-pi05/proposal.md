# 更名 z_umi → zerith_pi05（UMI 子项目示例目录）

## Why

`examples/realRobots/z_umi/` 的目录名不自解释——不知道它是什么机器人、跑什么模型。该目录实为
**zerith UMI 数据 + PI05** 的训练/评测/部署完整子项目（UMI 子项目实施区，规则见 `agent.md` 附录 A）。
更名为 `zerith_pi05` 后，从目录名即可知道数据来源（zerith UMI）与模型（PI05），降低新人与新 agent
的理解成本；同时为后续同数据不同模型（如 QwenPI_v3）的命名留下清晰对照。

## What Changes

- 目录更名：`examples/realRobots/z_umi/` → `examples/realRobots/zerith_pi05/`（`git mv`，历史可追溯）。
- 全仓引用同步：脚本路径、Python 模块 import（`examples.realRobots.z_umi.*` → `examples.realRobots.zerith_pi05.*`，
  含 data-analyze 与 RoboDojo data_registry 的跨目录 import）、yaml/sh 注释、docs 全部路径引用。
- **保持不变（物理/外部标识，改名会断链）**：
  - 本地/共享盘数据根 `/root/data/z-umi`（及 `/data/z-umi`、`/nix/data/z-umi` 等机器变体）；
  - 云端数据集前缀 `datasets/z-umi/`（OSS/BOS URI，data_config 元组第 4 元素）；
  - wandb 默认项目名 `pi05_z_umi_relative_6d`（`run_umi_pi05_relative_6d.sh` 默认值，保持训练曲线面板历史连续）；
  - 数据集 ID、robot_type 名（`umi_2gripper_relative_6d`）、类名（`UMILeRobotDataset`）——均不含 z_umi。
- 兼容 shim：新增 `examples/realRobots/z_umi/train_files/run_umi_pi05_relative_6d.sh` 转发到新 launcher。
  原因：AIHC 旧镜像内置的 bootstrap `scripts/train.sh` 硬编码旧 launcher 路径，且段 5 从解压 repo 相对执行；
  不加 shim 则旧镜像 + 新 commit zip 的所有队列任务直接失败。镜像 bootstrap 更新后删除 shim。
  （2026-09-21 后续：shim 已删除，队列启动改为 CFS 上的 train.sh——`launch_train.sh` 提交时同步
  `scripts/train.sh + cloud_io.py` 到 `/mnt/cfs/code/`，与镜像解耦。）
- spec / 文档同步：`agent.md`（目录树、子项目表、附录 A 标题 + 命名变更记录）、
  `openspec/specs/umi/pi05-open-eval` 与 `umi/umi-code-changes` 的 Purpose/References 路径引用、
  `docs/zerith_docs/umi/umi_training_plan.md` 顶部更名说明、`docs/zerith_docs/` 其余文档路径引用。

不改动：任何训练/评测/部署行为逻辑、数据管线、transform 链、模型代码。纯重命名 + 引用同步 + 兼容 shim。

## Capabilities

### New Capabilities

（无——本变更不新增行为合同。）

### Modified Capabilities

- `umi/pi05-open-eval`：无 Requirement 级语义变更；仅 Purpose 与 References 中的路径引用
  从 `examples/realRobots/z_umi/` 更新为 `examples/realRobots/zerith_pi05/`（随本变更直接落实）。
- `umi/umi-code-changes`：同上，Purpose 中历史路径表述（`examples/Z-umi/`、`z_umi`）更新为新目录名
  并保留更名记录。
- `infra/training-pipeline`：**不改**。其引用的 `/root/data/z-umi/data/<id>/` 是物理数据根，
  属"保持不变"项。

## Impact

- **受影响文件**：约 50 个（目录整体 `git mv` + agent.md + scripts/ + examples/ 跨目录 import + docs + openspec）。
- **兼容性风险**：
  - AIHC 旧镜像 bootstrap 硬编码旧 launcher 路径 → 已用 shim 兜底；新镜像应改指新路径后删 shim。
  - 外部分支（如 `robodojo`）merge 本变更时，其 data_registry 的 `from examples.realRobots.z_umi...`
    import 需同步解冲突为 `zerith_pi05`。
  - 提交机上本地 checkout 的旧脚本/笔记若硬编码旧路径，需自行更新；shim 只兜底 AIHC bootstrap 一条路径。
- **验证**：`git grep z_umi` 残留仅为白名单（wandb 项目名字符串、物理数据路径、shim、历史记录）；
  三个 data_registry 的 `data_config.py` 与 `ddl_to_data_config.py` 编译/导入通过；registry 自动发现
  （按 `examples/*/train_files/data_registry/` 扫描，与目录名无关）不受影响。
