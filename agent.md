# agent.md — starVLA 项目开发指引

## 项目概述

starVLA 是基于 Qwen VL 的视觉-语言-动作 (VLA) 模型框架，支持多种机器人数据集训练。

当前工作分支: `dev/luke`
主分支: `starVLA_dev`

## 开发规范 (CRITICAL)

### 1. 先读计划

在进行任何 Z-umi 相关开发前，必须先阅读：

```text
docs/umi_training_plan.md
```

该计划是 Z-umi 训练流程的数据流、normalizer 语义和技术架构的唯一真实来源。

### 2. OpenSpec 标准流程

所有开发工作遵循 OpenSpec 规范驱动流程：

```text
/opsx:propose "变更描述"   ->  创建提案，生成 spec
/opsx:apply                ->  按 spec 实施代码
/opsx:archive              ->  完成后归档
```

OpenSpec 配置: `openspec/config.yaml`
技能/命令: `.claude/skills/`, `.claude/commands/`

### 3. 同步更新规则 (CRITICAL)

如果后续任何修改影响了 Z-umi 实施方案，必须同步更新以下文件：

| 文件 | 更新内容 |
|------|---------|
| `docs/umi_training_plan.md` | 计划变更、新增步骤、设计决策 |
| `openspec/specs/` | 对应的 spec 文档 |
| `CLAUDE.md` | Claude/Codex 相关开发规则 |
| `agent.md` | 本文件 |

禁止修改代码但不同步更新计划文档。

所有因 UMI 工作产生、但位于 `examples/Z-umi/` 之外的代码修改，还必须同步登记到 `docs/umi_non_zumi_code_changes.md`，说明原有问题、修改内容、影响和验证结果。

如果修改 `CLAUDE.md` 或 `agent.md` 任一文件，必须检查并同步另一个文件，避免不同开发人员或不同 coding agent 读到不一致的规则。

### 4. Z-umi YAML 参数所有权

`examples/Z-umi/train_files/run_umi_train.sh` 只能负责环境激活、运行时环境变量、进程数和选择 `CONFIG_YAML`。模型、数据、训练超参、run metadata 和 wandb 参数必须以 YAML 为准，禁止在 launcher 中通过 dotlist 或额外 CLI 参数覆盖 YAML。

开环拟合评测必须明确区分训练分布内拟合和未见 episode 泛化。使用训练数据集中的样本时，只能报告 pipeline 状态和 fitting evidence，不能报告泛化成功。

回放 `umi-test` episode 时必须复用训练统计，禁止从测试数据重算 normalizer。动作必须按 checkpoint 保存的 action mode 恢复 absolute pose/gripper 后再与目标帧原数据比较；开环误差不能表述为闭环任务成功率。

固定单 episode 过拟合诊断使用 `examples/Z-umi/train_files/train_umi_overfit.yaml`；单 action chunk 诊断使用 `examples/Z-umi/train_files/train_umi_overfit_anchor.yaml`。`datasets.vla_data.overfit_episode_id` 只能通过 YAML 显式开启，`overfit_anchor_index` 只能在 episode 模式下选择一个无需尾部填充的完整 action chunk；两种模式都必须复用完整训练集统计和 action transform 语义，并使用 `num_workers: 0` 保证变换后样本只缓存一份。相关结果只能作为 pipeline/fitting evidence，不能作为泛化证据。

上述 cached 模式是历史 4090 诊断。需要保持 `train_umi_A800.yaml` 的模型、DataLoader 和 trainer 参数，但让数据与统计都只来自一条轨迹时，必须使用独立数据集 `umi_20260720_overfit_ep0` 和独立配置 `examples/Z-umi/train_files/train_umi_A800_overfit_ep0.yaml`；正式 A800 YAML 不得修改。目标数据集不得复制源 statistics/step cache，训练 episode 回放仍只能作为 fitting evidence。

四卡 A800 UMI 使用 decord 时，内部解码并发也必须由 YAML 显式声明：

```yaml
video_backend: decord
video_backend_kwargs:
  num_threads: 1
```

禁止在通用视频代码中为所有数据集硬编码该值。修改外层 `num_workers`、进程数或平台 CPU quota 后，必须重新检查 worker thread count、CPU PSI 和有效 step throughput；训练日志中的 `data_times` 不包含 persistent worker 的完整后台解码开销，不能单独作为验收依据。

### 5. UMI action mode 合同

`train_umi_A800.yaml` 使用独立的 `action_mode: relative`。pose 必须通过显式
`action_mode_pose_pairs` 计算 `T_relative = inverse(T_base) @ T_target`；同一
16-step chunk 共享当前 state anchor。quaternion 只能通过旋转矩阵乘法求相对旋转，
禁止逐元素相减。pose pair 之外的 gripper 保持绝对值。

旧 `abs`、`rel`、`delta` 行为必须保持不变。relative quaternion 在 action-mode
处理后再由现有 transform 转 `rotation_6d`；PyTorch3D 6D 使用旋转矩阵前两行，
固定 `[-1,1]` min-max 是恒等映射。改变 action mode/pairs 后必须删除或失效旧
`stats_gr00t.json` 并重算；评测必须先按 checkpoint 保存的 mode 逆变换到 absolute
pose/gripper 再比较。

### 6. 环境

```bash
conda activate starVLA
```

Python 3.10，依赖以当前 `starVLA` conda 环境为准。

### 7. 关键路径

| 模块 | 路径 |
|------|------|
| 数据集 | `/root/workspace/data/lerobot_dataset_test/` |
| 计划文件 | `docs/umi_training_plan.md` |
| Agent 指引 | `agent.md` |
| Claude 指引 | `CLAUDE.md` |
| 实施代码 | `examples/Z-umi/` |
| 数据管线 | `starVLA/dataloader/gr00t_lerobot/` |
| 模型框架 | `starVLA/model/framework/VLM4A/` |
| 训练入口 | `starVLA/training/train_starvla.py` |
| 开环评测入口 | `starVLA/training/eval_starvla.py` (多卡, 镜像 train 入口) |
| OpenSpec | `openspec/` |
