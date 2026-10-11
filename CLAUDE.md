# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

StarVLA 是一个"乐高式"的视觉-语言-动作（VLA）模型开发平台，核心设计理念是**高内聚、低耦合、可插拔**。每个功能组件（模型、数据、训练器、配置、评估等）都遵循自顶向下、直观分离的原则，实现即插即用设计和独立调试。

### 核心架构特点

1. **模块化框架设计**：所有 VLA 框架变体共享统一的数据接口和基础设施，仅 action head 不同
   - VLM4A/：基于视觉-语言模型的 VLA 框架（PI0、PI05、QwenGR00T、QwenOFT 等）
   - WM4A/：基于世界模型的 VLA 框架（WM4A）

2. **显式模型边界**：
   - Dataloader 仅返回模型无关的原始字典
   - `framework.forward()` 和 `framework.predict_action()` 直接处理原始输入
   - 训练/测试边界清晰，易于调试

3. **统一训练基础设施**：支持多种训练范式
   - SFT（监督微调）
   - 多模态多目标协同训练
   - 跨形态协同训练

## 常用开发命令

### 环境安装
环境安装流程、一键脚本（`docs/z-docs/install.sh`）与已知问题排错见 [docs/z-docs/install.md](docs/z-docs/install.md)。

### 代码质量检查
```bash
# 格式化整个项目（注意：仅用于参考，PR 中应只格式化修改的文件）
make autoformat                # Black 格式化 + Ruff 自动修复

# 检查代码风格
make check                      # Black 检查 + Ruff lint

# 仅检查/格式化特定文件（推荐用于 PR）
black --check path/to/file.py
black path/to/file.py
ruff check path/to/file.py
ruff check --fix path/to/file.py

# 清理临时文件
make clean                      # 删除 pyc/pycache 文件
```

### 训练
```bash
# 标准 accelerate 启动流程
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml

# 多节点训练
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $SLURM_PROCID \
  --num_machines $SLURM_NNODES \
  --num_processes=${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml <config.yaml>
```

### 独立模块测试
StarVLA 强调模块化设计，每个主要框架文件可独立运行进行快速调试：
```bash
# 测试模型模块
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml path/to/config.yaml

# 测试 dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml path/to/config.yaml
```

## 核心目录结构

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
│   ├── z_umi/                      # z-UMI 机器人
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

## 配置系统

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

## 核心数据流

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
   # out 必须包含 "normalized_actions" 键
   ```

## 重要设计模式

### 1. Framework Registry
所有 VLA 框架通过 `@FRAMEWORK_REGISTRY.register()` 装饰器注册：
```python
@FRAMEWORK_REGISTRY.register("QwenGR00T")
class QwenGR00T(baseframework):
    ...
```

使用 `build_framework(cfg)` 构建模型。

### 2. 多标签训练支持
框架通过 `supports_training_tag()` 和 `compute_loss()` 支持多数据源协同训练：
- `"vla"` → `framework.forward(batch)`
- `"vlm"` → `framework.forward_vlm(batch)`

### 3. Action 归一化
- 训练时：actions 被归一化（基于数据集统计）
- 推理时：输出需要反归一化
- 统计量存储在 `model.norm_stats` 中

### 4. 模块冻结与不同学习率
```yaml
trainer:
  freeze_modules: "qwen_vl_interface,dino_encoder"  # 正则匹配
  learning_rate:
    base: 1e-5
    qwen_vl_interface: 1e-5    # 模块名: 学习率
    action_model: 1e-4
```

## 训练调试技巧

### 1. 检查模型结构
```python
from starVLA.model.framework.base_framework import build_framework
model = build_framework(cfg)
print(model)  # 打印模块路径（用于 freeze_modules 匹配）
```

### 2. 验证 dataloader
```python
from starVLA.dataloader import build_dataloader
dataloader = build_dataloader(cfg, dataset_py="lerobot_datasets")
batch = next(iter(dataloader))
print(batch.keys())  # 检查返回的键
```

### 3. 快速 smoke test
每个框架文件可以直接运行：
```bash
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml config.yaml
```

### 4. 配置验证
运行训练前检查未使用的配置字段（trainer 会自动警告）：
```python
warn_unused_yaml_fields(cfg)
```

## 常见任务

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
4. 参考 `docs/integrate_your_dataset.md` 或使用 agent skill

### 调试训练问题
1. 检查 `accelerate` 状态（训练脚本会打印）
2. 检查数据集统计量是否正确
3. 验证 `freeze_modules` 匹配正确的模块路径
4. 使用 `make check` 检查代码风格

## 重要提醒

### 1. 分支策略
- `starVLA_dev`：活跃开发分支，新功能首先合并到这里
- `starVLA`：稳定发布分支，经过验证的代码
- PR 应该 target `starVLA_dev`

### 2. 自定义脚本位置
任何 `**/bar/` 目录下的文件会被 git 忽略，可用于放置自定义脚本：
```bash
examples/simBenchmarks/LIBERO/train_files/bar/my_train.sh
```

### 3. NPU 支持
代码自动检测 `torch_npu`，无需修改即可在 Ascend NPU 上运行。

### 4. 多进程安全
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 已在训练脚本中设置
```

## 相关文档

- **快速入门**：`docs/starVLA_guideline.md`
- **FAQ**：`docs/faq.md`
- **PR 指南**：`docs/PR_readme.md`
- **分支策略**：`docs/branching_strategy.md`
- **模型列表**：`docs/model_zoo.md`
- **集成数据集**：`docs/integrate_your_dataset.md`

## Agent Skills

StarVLA 内置 agent skills 用于 AI 辅助开发：
- `docs/agent_skills/integrate-starvla-dataset/`：自动化数据集集成
- 适用于 Claude Code、VS Code Copilot 等 AI 编程助手
