# agent.md — StarVLA AI Agent 开发指引（单一文档）

> 本文件是 **所有 AI coding agent（Claude Code、Codex、Copilot、opencode 等）在本仓库工作的唯一规则与技术参考来源**。
> 已废弃并删除 `CLAUDE.md`，其内容已并入本文件；如需调整，直接修改本文件即可，无需多文件同步。

## 1. 项目概述

StarVLA 是一个"乐高式"的视觉-语言-动作（VLA）模型开发平台，核心设计理念是**高内聚、低耦合、可插拔**。每个功能组件（模型、数据、训练器、配置、评估）都遵循自顶向下、直观分离，实现即插即用与独立调试。

- **模块化框架**：所有 VLA 框架变体共享统一数据接口与基础设施，仅 action head 不同（`VLM4A/`、`WM4A/`）。
- **显式模型边界**：Dataloader 只返回模型无关的原始字典；`framework.forward()` / `framework.predict_action()` 直接消费原始输入。
- **统一训练基础设施**：支持 SFT、多模态多目标协同训练、跨形态协同训练。

## 2. 核心目录结构

```
starVLA/
├── model/                           # 模型相关
│   ├── framework/                   # 框架实现（VLA 模型的入口）
│   │   ├── base_framework.py        # 基础框架抽象类
│   │   ├── VLM4A/                   # 基于 VLM 的 VLA 框架
│   │   │   ├── PI0.py              # Physical Intelligence π0 实现
│   │   │   ├── PI05.py             # π0.5 实现（继承自 PI0）
│   │   │   ├── QwenGR00T.py        # Qwen + GR00T 双系统架构
│   │   │   ├── QwenOFT.py          # Qwen + OFT（Outlier Feature Transformer）
│   │   │   └── LangForce.py        # LangForce 框架
│   │   └── WM4A/                    # 基于世界模型的 VLA 框架
│   ├── modules/                     # 可插拔模块
│   │   ├── vlm/                     # VLM 接口（Qwen、Gemma、Florence-2 等）
│   │   ├── action_model/           # Action heads（MLP、DiT、Flow-Matching、OpenPI 等）
│   │   ├── projector/              # 视觉投影器
│   │   └── dino_model/             # DINO 特征提取器
│   └── tools.py                     # 工具函数和 FRAMEWORK_REGISTRY
│
├── dataloader/                      # 数据加载
│   ├── gr00t_lerobot/              # LeRobot 格式数据集（NVIDIA GR00T 协议）
│   │   ├── datasets.py             # LeRobotSingleDataset、LeRobotMixtureDataset
│   │   ├── data_config.py          # 数据配置和统计
│   │   ├── transform/              # 数据转换（归一化、旋转等）
│   │   └── mixtures.py             # 数据集混合配置
│   ├── vlm_datasets.py              # VLM 数据集（LLaVA-JSON 格式）
│   └── lerobot_datasets.py          # LeRobot 数据集入口
│
├── training/                        # 训练相关
│   ├── train_starvla.py             # VLA 训练主脚本
│   ├── train_starvlm.py             # VLM 训练主脚本
│   ├── train_starvla_cotrain.py     # VLA+VLM 协同训练
│   ├── trainer_utils/               # 训练工具
│   └── eval_starvla.py              # VLA 开环评测入口（多卡，镜像 train 结构）
│
├── eval/                            # 评测工具
│   ├── core/                        # 评测核心（metrics、transforms、IO）
│   └── viz/                         # 评测可视化（Rerun 日志）
│
└── config/                          # 配置文件
    ├── deepseeds/                  # DeepSpeed 配置
    │   ├── deepspeed_zero2.yaml    # ZeRO-2 配置
    │   └── zero3.yaml              # ZeRO-3 配置
    └── training/                   # 训练默认配置

examples/                            # 示例和 benchmark
├── simBenchmarks/                  # 模拟 benchmark
│   ├── LIBERO/                     # LIBERO benchmark
│   ├── SimplerEnv/                 # SimplerEnv benchmark
│   ├── Robocasa_365/               # RoboCasa-365 benchmark
│   └── ...
├── realRobots/                      # 真实机器人示例
│   ├── Franka/                     # Franka 机器人
│   ├── zerith_pi05/                # zerith UMI 数据 · PI05（原 z_umi，UMI 子项目，规则见附录 A）
│   └── ...
└── modelExtensions/                # 模型扩展示例
    ├── CoTrainVLM/                  # VLM 协同训练
    └── ...

docs/                                # 文档
├── starVLA_guideline.md             # 快速入门指南
├── faq.md                           # 常见问题
├── PR_readme.md                     # PR 指南
└── branching_strategy.md           # 分支策略
```

## 3. Agent 工作流硬规则（CRITICAL）

以下规则在任何任务中都必须遵守，违反会破坏可复现性或他人协作。

### 3.1 先读计划，再动代码
涉及任一子项目（目前主要是 UMI / zerith_pi05）的开发前，必须先阅读对应的唯一真实来源文档：

| 子项目 | 单一真实来源 |
|--------|--------------|
| UMI / zerith_pi05 | `docs/zerith_docs/umi/umi_training_plan.md`（数据流、normalizer 语义、技术架构） |

该计划是数据流与 normalizer 语义的权威；代码必须与之一致，禁止凭直觉修改。

### 3.2 文档同步义务
代码改动若影响了任一子项目的实施方案，**必须同步更新**：

| 文件 | 更新内容 |
|------|---------|
| `docs/zerith_docs/umi/umi_training_plan.md` | 计划变更、新增步骤、设计决策 |
| `docs/zerith_docs/umi/umi_non_zumi_code_changes.md` | 因 UMI 工作产生、但位于 `examples/realRobots/zerith_pi05/` 之外的代码修改（说明原问题、改动、影响、验证） |
| `agent.md` | 通用规则变更（本文件为单一来源，改这里即可） |

**禁止：修改了代码但不同步更新计划与登记文档。**

**新增文档统一放 `docs/zerith_docs/` 下**（按域选子目录：`infra` / `umi` / `workflow` / `guides`），不要散落在 `examples/` 或仓库其他位置；历史遗留的散落文档随目录清理逐步迁入（如 2026-09 清理 `examples/realRobots/H1_ddl_d18480a6`，其 H1 数据集类早已迁至 `examples/data-analyze/train_files/data_registry/h1_dataset.py`）。

### 3.3 文件隔离原则
StarVLA 架构强调**文件级隔离**：
- 新框架 → 在 `starVLA/model/framework/` 下新建模块，勿改既有框架核心逻辑。
- 新 benchmark → 在 `examples/<benchmark>/` 下新建目录。
- 新数据集 → 注册到 `examples/<benchmark>/train_files/data_registry/`，勿改共享 dataloader 接口（除非充分论证）。
- Bug 修复 → 精准修改受影响文件，**不要顺手重构/格式化无关代码**。

### 3.4 提交与改动纪律
- **未经明确请求，绝不 commit / push / 建 PR / amend。**
- 改动完成后必须运行 lint（见 §4），确保改动的文件通过 `black` + `ruff`。
- 改动尽量聚焦、可验证；优先复用仓库已有的库与模式，不要假设某库可用。

## 4. 环境与常用命令

### 环境
```bash
conda activate starVLA      # Python 3.10，依赖以当前 conda 环境为准
```

### 代码质量（line-length = 121）
```bash
# 仅检查/格式化你本次修改的文件（PR 推荐）
black --check path/to/file.py
ruff check path/to/file.py
black path/to/file.py && ruff check --fix path/to/file.py

# 全仓（参考用；历史 lint 欠账可能导致 make check 整体失败，PR 只需自身文件干净）
make check        # black --check . + ruff check（不改文件）
make autoformat   # black + ruff --fix-only（改文件）
make clean        # 清理 pyc/pycache
```

### 训练 / 评测入口
```bash
# 训练（标准 accelerate 启动）
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml <your_config.yaml>

# 评测（多卡，镜像 train 结构）
accelerate launch ... starVLA/training/eval_starvla.py --config_yaml <your_config.yaml>
```

### 独立模块 smoke test
每个框架/dataloader 文件可独立运行快速调试：
```bash
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml path/to/config.yaml
python starVLA/dataloader/lerobot_datasets.py --config_yaml path/to/config.yaml
```

### 单元测试
```bash
pytest tests/    # 如 test_quaternion_to_6d_transform.py、test_config_overrides.py 等
```

## 5. 关键路径（已校验存在）

| 模块 | 路径 |
|------|------|
| 训练入口 | `starVLA/training/train_starvla.py` |
| 评测入口（多卡，镜像 train） | `starVLA/training/eval_starvla.py` |
| 模型框架 | `starVLA/model/framework/`（`VLM4A/`、`WM4A/`、`base_framework.py`） |
| 框架注册表 | `starVLA/model/tools.py`（`FRAMEWORK_REGISTRY`、`build_framework`） |
| 数据管线 | `starVLA/dataloader/`（入口 `lerobot_datasets.py`，核心 `gr00t_lerobot/`） |
| DeepSpeed 配置 | `starVLA/config/deepseeds/` |
| 快速入门 | `docs/starVLA_guideline.md` |
| 开发工具脚本（dev/agent 用） | `scripts/dev/`（`new-worktree.sh`、`check_patches.py`、`smoke.sh`、`overfit.sh`） |
| 内部开发规范 | `docs/zerith_docs/`（上游同步 / worktree / 测试分层 / 上游补丁登记） |
| zerith_pi05（UMI）实施 | `examples/realRobots/zerith_pi05/`（`train_files/`、`eval_files/`、`deploy_files/`） |

> 数据根目录是机器/配置相关的运行时路径，统一通过 YAML 的 `data_root_dir` 声明，不要在文档里硬编码绝对路径。

## 6. 分支模型

### 6.1 上游（开源 starVLA）
| 分支 | 用途 |
|------|------|
| `starVLA_dev` | 上游活跃开发分支（远端 `github`，**只读镜像**，只 fetch） |
| `starVLA` | 上游稳定发布分支 |

### 6.2 本仓库（zerith 内部）
| 分支 | 用途 |
|------|------|
| `dev/main` | **长期集成分支** = 上游 + 我们全部补丁的合并态；MR 的 target |
| `feat/<topic>` / `dev/<人>` | feature 分支，从 `dev/main` 拉出，在 worktree 开发 |

- 上游 → `dev/main`：**merge**（每月级；详见 `docs/zerith_docs/workflow/upstream_sync_and_isolation.md`）。
- feature → `dev/main`：先 rebase 再 MR（详见 `docs/zerith_docs/workflow/git_merge_rebase_guide.md`）。
- 当前工作分支请以 `git branch --show-current` 为准（勿在文档中固化具体分支名）。
- 分支命名：`feat/`、`fix/`、`docs/`、`refactor/`、`exp/`、`hotfix/`，小写连字符。

### 6.3 内部开发流程（详见 zerith_docs，agent 改本仓库代码必读）
- **隔离原则**：自有改动尽量是**新增文件**（零冲突）；必须改上游既有文件时，登记到 `docs/zerith_docs/workflow/upstream_patches.md` 并加 `ZERITH PATCH` 标记区块，改完跑 `python scripts/dev/check_patches.py` 自检。
- **worktree 开发**：feature 分支在 `../starvla.worktrees/<分支>` 开发，主仓库只留 `dev/main`。一键建：`bash scripts/dev/new-worktree.sh <分支>`（见 `docs/zerith_docs/workflow/worktree_dev.md`）。
- **测试分层**：单元(`uv run pytest tests/`) / 模块smoke(`bash scripts/dev/smoke.sh`) / overfit(`bash scripts/dev/overfit.sh`) / 短train / 完整train。绝大多数改动不必上 `scripts/train.sh`（见 `docs/zerith_docs/workflow/testing_layers.md`）。
- **开发工具脚本**统一放 `scripts/dev/`（不进集群运行时）；`scripts/` 根目录是人/集群侧脚本（见 `scripts/dev/README.md`）。

## 7. 配置系统

StarVLA 使用 OmegaConf/YAML 配置系统，所有配置通过 `cfg` 对象传递。

### 典型配置结构
```yaml
run_id: starvla
run_root_dir: playground/Checkpoints
framework:
  name: QwenGR00T              # 框架名称（必须）
  qwenvl:                      # VLM 配置
    base_vlm: path/to/vlm
  action_model:                # Action head 配置
    action_dim: 7
    action_horizon: 8
datasets:
  vla_data:                    # VLA 数据配置
    dataset_py: lerobot_datasets
    data_root_dir: path/to/data
    per_device_batch_size: 16
  vlm_data:                    # VLM 数据配置（可选）
    dataset_py: vlm_datasets
trainer:
  max_train_steps: 100000
  learning_rate:               # 模块级学习率
    base: 1e-5
    action_model: 1e-4
  freeze_modules: "qwen_vl_interface"  # 冻结模块（逗号分隔）
```

### 配置覆盖
```bash
# 通过命令行覆盖配置
--framework.name QwenGR00T
--framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct
--trainer.freeze_modules "qwen_vl_interface,dino_encoder"
```

## 8. 核心数据流

### 训练数据流
1. **DataLoader** 返回原始字典（模型无关）：
   ```python
   {
       "image": List[PIL.Image],
       "lang": str,
       "action": np.ndarray[T, action_dim],
       "state": Optional[np.ndarray[..., state_dim]]
   }
   ```

2. **Framework.forward()** 处理原始输入，返回损失：
   ```python
   out = framework.forward(batch)
   # out 必须包含 "action_loss" 键
   ```

3. **Trainer** 处理分布式、优化器、调度器等

### 推理数据流
1. **Framework.predict_action()** 接收观测，预测动作：
   ```python
   out = framework.predict_action(examples)
   # out 必须包含 "normalized_actions" 键（推理时需按 model.norm_stats 反归一化）
   ```

Action 训练时归一化、推理时反归一化；统计量存于 `model.norm_stats`。

## 9. 重要设计模式

### 9.1 Framework Registry
所有 VLA 框架通过 `@FRAMEWORK_REGISTRY.register()` 装饰器注册：
```python
@FRAMEWORK_REGISTRY.register("QwenGR00T")
class QwenGR00T(baseframework):
    ...
```

使用 `build_framework(cfg)` 构建模型。

### 9.2 多标签训练支持
框架通过 `supports_training_tag()` 和 `compute_loss()` 支持多数据源协同训练：
- `"vla"` → `framework.forward(batch)`
- `"vlm"` → `framework.forward_vlm(batch)`

### 9.3 Action 归一化
- 训练时：actions 被归一化（基于数据集统计）
- 推理时：输出需要反归一化
- 统计量存储在 `model.norm_stats` 中

### 9.4 模块冻结与不同学习率
```yaml
trainer:
  freeze_modules: "qwen_vl_interface,dino_encoder"  # 正则匹配
  learning_rate:
    base: 1e-5
    qwen_vl_interface: 1e-5    # 模块名: 学习率
    action_model: 1e-4
```

## 10. 训练调试技巧

### 10.1 检查模型结构
```python
from starVLA.model.framework.base_framework import build_framework
model = build_framework(cfg)
print(model)  # 打印模块路径（用于 freeze_modules 匹配）
```

### 10.2 验证 dataloader
```python
from starVLA.dataloader import build_dataloader
dataloader = build_dataloader(cfg, dataset_py="lerobot_datasets")
batch = next(iter(dataloader))
print(batch.keys())  # 检查返回的键
```

### 10.3 快速 smoke test
每个框架文件可以直接运行：
```bash
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml config.yaml
```

### 10.4 配置验证
运行训练前检查未使用的配置字段（trainer 会自动警告）：
```python
warn_unused_yaml_fields(cfg)
```

## 11. 常见任务

### 添加新的 VLA 框架
1. 在 `starVLA/model/framework/VLM4A/` 创建新文件
2. 继承 `baseframework` 类
3. 实现 `forward()` 和 `predict_action()` 方法
4. 使用 `@FRAMEWORK_REGISTRY.register()` 注册
5. 在 `examples/` 添加配置示例

### 集成新数据集
1. 将数据转换为 LeRobot 格式
2. 在 `examples/<benchmark>/train_files/data_registry/data_config.py` 配置数据统计和转换
3. 创建训练配置 YAML
4. 参考 `docs/agent_skills/integrate-starvla-dataset/` 或使用 agent skill

### 调试训练问题
1. 检查 `accelerate` 状态（训练脚本会打印）
2. 检查数据集统计量是否正确
3. 验证 `freeze_modules` 匹配正确的模块路径
4. 使用 `make check` 检查代码风格

## 12. 架构事实与提醒

### 自定义脚本位置
任何 `**/bar/` 目录下的文件会被 git 忽略，可用于放置自定义脚本：
```bash
examples/simBenchmarks/LIBERO/train_files/bar/my_train.sh
```

### NPU 支持
代码自动检测 `torch_npu`，无需修改即可在 Ascend NPU 上运行。

### 多进程安全
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 已在训练脚本中设置
```

---

## 附录 A：UMI / zerith_pi05 子项目硬规则

以下规则在涉及 `examples/realRobots/zerith_pi05/` 时强制生效；权威解释以 `docs/zerith_docs/umi/umi_training_plan.md` 为准。

> **命名变更记录（2026-09）**：`examples/realRobots/z_umi/` 已更名为 `examples/realRobots/zerith_pi05/`
> （目录名不再自解释模型；PI05 即其训练的模型）。以下内容**保持不变**（物理/外部标识，改名会断链）：
> 本地与共享盘数据根 `/root/data/z-umi`（及 `/data/z-umi`、`/nix/data/z-umi`）、云端数据前缀
> `datasets/z-umi/`（OSS/BOS URI）、wandb 默认项目名 `pi05_z_umi_relative_6d`。
> 兼容 shim 已删除（2026-09-21）：AIHC 队列启动不再依赖镜像内置 bootstrap——`launch_train.sh`
> 提交时把仓库当前 `scripts/train.sh + cloud_io.py` 同步到 `/mnt/cfs/code/`，容器直接执行 CFS
> 上的版本（与镜像解耦，旧镜像也能跑新流水线）。

### A.1 Action mode 合同
- relative pose 必须通过显式 `action_mode_pose_pairs` 计算 `T_relative = inverse(T_base) @ T_target`；同一 chunk 共享当前 state anchor。
- **四元数只能通过旋转矩阵乘法求相对旋转，禁止逐元素相减。** pose pair 之外的 gripper 保持绝对值。
- 旧 `abs`/`rel`/`delta` 行为必须保持不变。relative 四元数在 action-mode 处理后，再由现有 transform 转 `rotation_6d`；PyTorch3D 6D 取旋转矩阵前两行，固定 `[-1,1]` min-max 是恒等映射。
- 改变 action mode / pairs 后，**必须删除或失效旧 `stats_gr00t.json` 并重算**。
- 评测必须先按 checkpoint 保存的 mode 逆变换到 absolute pose/gripper，再与目标比较。

### A.2 归一化与评测纪律
- 回放测试 episode 时**必须复用训练统计，禁止从测试数据重算 normalizer**。
- 开环误差**不能表述为闭环任务成功率**。
- 使用训练数据集中的样本做开环拟合评测时，只能报告 pipeline 状态与 fitting evidence，**不能报告泛化成功**。

### A.3 Overfit 诊断纪律
- 单 episode / 单 action chunk 过拟合诊断**只能通过 YAML 显式开启**（如 `overfit_episode_id`；`overfit_anchor_index` 仅在 episode 模式下选一个无需尾部填充的完整 chunk）。
- 两种诊断模式都必须**复用完整训练集统计与 action transform 语义**，并用 `num_workers: 0` 保证变换后样本只缓存一份。
- 过拟合结果**只能作为 pipeline / fitting evidence，不能作为泛化证据**。
- 需要在 A800 等正式配置下做单轨迹诊断时，必须使用独立数据集与独立 overfit 配置 YAML，**正式 A800 YAML 不得修改**；目标数据集不得复制源 `statistics/` 与 step cache。

### A.4 Launcher 与 YAML 参数所有权
- launcher 脚本（`run_umi_*.sh`）只负责：环境激活、运行时环境变量、进程数、选择 `CONFIG_YAML`。
- 模型、数据、训练超参、run metadata、wandb 参数**必须以 YAML 为准**，禁止在 launcher 中通过 dotlist 或额外 CLI 覆盖 YAML。

### A.5 视频解码并发
- 四卡 A800 使用 decord 时，内部解码并发必须由 YAML 显式声明：
  ```yaml
  video_backend: decord
  video_backend_kwargs:
    num_threads: 1
  ```
- **禁止在通用视频代码中为所有数据集硬编码该值。**
- 修改外层 `num_workers`、进程数或平台 CPU quota 后，必须重新检查 worker thread count、CPU PSI 与有效 step throughput；训练日志中的 `data_times` 不含 persistent worker 的完整后台解码开销，**不能单独作为验收依据**。

## 相关文档
- `docs/starVLA_guideline.md` — 快速入门
- `docs/branching_strategy.md` — 分支与 PR 策略
- `docs/PR_readme.md` — PR 提交与审查流程
- `docs/faq.md` — 常见问题
- `docs/zerith_docs/umi/umi_training_plan.md` — UMI 训练计划（单一真实来源）
- `docs/zerith_docs/umi/umi_non_zumi_code_changes.md` — UMI 跨目录改动登记
- `docs/model_zoo.md` — 模型列表
- `docs/agent_skills/integrate-starvla-dataset/` — 数据集集成 agent skill

### zerith 内部规范（`docs/zerith_docs/`，按桶分类，见 `docs/zerith_docs/README.md`）
- `workflow/upstream_sync_and_isolation.md` — 上游同步与隔离开发
- `workflow/upstream_patches.md` — 上游补丁登记表
- `workflow/patch_reduction_plan.md` — 上游补丁瘦身方案（M→A 转化分析）
- `workflow/worktree_dev.md` — worktree + uv 开发工作流
- `workflow/testing_layers.md` — 测试分层
- `workflow/git_merge_rebase_guide.md` — 个人分支合并（rebase）
- `workflow/ci_and_training_source.md` — CI 打包 / commit ↔ 云端 zip (OSS/BOS 双云, cloud_io.py)
- `umi/umi_training_plan.md` — UMI 训练计划（单一真实来源）
- `umi/umi_non_zumi_code_changes.md` — UMI 跨目录改动登记
