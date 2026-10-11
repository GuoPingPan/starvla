# StarVLA 项目总结

> 生成日期：2026-08-05 ｜ 当前分支：`dev/train_anylze`

本文档是对 StarVLA 代码库的整体总结，涵盖核心架构、模块划分、数据流、训练/评测/部署流程，以及当前开发分支的重点工作。目标读者为需要快速了解该项目的开发人员。

---

## 1. 项目概览

**StarVLA** 是一个"乐高式"（Lego-like）的视觉-语言-动作（VLA，Vision-Language-Action）模型开发平台，核心理念是 **高内聚、低耦合、可插拔**。每个功能组件（模型、数据、训练器、配置、评测）都遵循自顶向下、直观分离的原则，支持即插即用设计与独立调试。

- 项目主页：https://starvla.github.io/
- 技术报告：arXiv 2604.05014（ECCV 2026）
- 代码最早 fork 自 InternVLA-M1，并借鉴 LeRobot / NVIDIA GR00T / Qwen-VL / DeepSpeed 等开源项目。
- 许可证：MIT。

**核心设计：所有 VLA 框架变体共享统一的数据接口和训练基础设施，仅 action head 不同。**

支持的框架变体：

| 框架 | 说明 |
| --- | --- |
| StarVLA-FAST | 自回归离散 action token（fast tokenizer，π₀-fast 风格） |
| StarVLA-OFT | MLP head 并行连续 action 解码（OpenVLA-OFT 风格） |
| StarVLA-PI | Flow-Matching action 专家，扩散式连续 action（π₀ 风格） |
| StarVLA-GR00T | 双系统架构：VLM 为 System 2 + Flow-Matching 为 System 1（GR00T 风格） |

支持的训练范式：SFT、多模态多目标协同训练（Co-Training）、跨形态协同训练、RL 后训练（经 RLinf 集成）。

---

## 2. 目录结构总览

```
starVLA/
├── model/                            # 模型层（核心）
│   ├── framework/                    # 框架实现（VLA 模型的入口）
│   │   ├── base_framework.py         # 基础框架抽象类 + build_framework 入口
│   │   ├── share_tools.py            # 配置兼容/读取等共享工具
│   │   ├── tools.py                  # FRAMEWORK_REGISTRY 注册表 + 工具函数
│   │   ├── VLM4A/                    # 基于 VLM 的 VLA 框架（19 个）
│   │   │   ├── PI0.py / PI05.py      # π0 / π0.5
│   │   │   ├── QwenGR00T.py          # Qwen + Flow-Matching 双系统
│   │   │   ├── QwenOFT.py / QwenFast.py / QwenPI.py / QwenPI_v3.py ...
│   │   │   ├── Gemma4GR00T.py / Gemma4PI.py / MiniCPM* / ABot_M0.py / LangForce.py ...
│   │   └── WM4A/                     # 基于世界模型的 VLA（Cosmos / Wan 视频生成模型）
│   │       ├── CosmoPredict2GR00T.py / CosmoPredict2OFT.py / CosmoPredict2PI.py
│   │       └── WanGR00T.py / WanOFT.py / WanPI.py
│   ├── modules/                      # 可插拔模块
│   │   ├── vlm/                      # VLM 接口（QWen2_5 / QWen3 / QWen3_5 / Gemma4 / Florence2 / MiniCPM_V / Molmo2 / OpenPIPaliGemma / CosmosReason2 ...）
│   │   ├── action_model/             # Action heads（MLP / DiT / flow_matching / GR00T / FAST / discrete_diffusion / OpenPI / LayerwiseFM ...）
│   │   ├── projector/                # 视觉投影器
│   │   ├── dino_model/               # DINO 特征提取器
│   │   └── world_model/              # 世界模型 backbone
│   └── tools.py                       # has_flash_attn / Registry / CrossAttention / 图像预处理
│
├── dataloader/                       # 数据层
│   ├── gr00t_lerobot/                # LeRobot 格式数据集（NVIDIA GR00T 协议）
│   │   ├── datasets.py               # LeRobotSingleDataset / LeRobotMixtureDataset
│   │   ├── data_config.py            # 基础 DataConfig（OxeDroid / OxeBridge ...）
│   │   ├── registry.py               # 自动发现 examples/**/data_registry/ 并合并
│   │   ├── mixtures.py               # 数据集混合配置（libero / oxe / fourier ...）
│   │   ├── schema.py / embodiment_tags.py / video.py / transform/ ...
│   ├── lerobot_datasets.py           # VLA 数据入口（collate_fn / get_vla_dataset）
│   ├── vlm_datasets.py               # VLM 数据（LLaVA-JSON 格式）
│   └── qwenvl_llavajson/             # QwenVL JSON 数据集实现
│
├── training/                         # 训练层
│   ├── train_starvla.py              # VLA 训练主脚本（VLATrainer）
│   ├── train_starvlm.py              # VLM 训练主脚本
│   ├── train_starvla_cotrain.py      # VLA+VLM 协同训练
│   ├── train_starvln.py              # 原生（NPU）训练入口
│   ├── eval_starvla.py               # 多卡开环评测入口（VLAEvaluator）
│   └── trainer_utils/                # TrainerUtils / optimizer / freeze / config_tracker / logger / perf_analyzer
│
├── eval/                             # 评测工具
│   ├── core/                         # checkpoint_utils / io / metrics / transforms
│   └── viz/                          # Rerun 日志可视化
│
├── config/                           # 默认配置
│   ├── deepseeds/                    # deepspeed_zero2.yaml / zero3.yaml
│   └── training/                     # starvla_cotrain_libero.yaml / oxe / discrete_diffusion ...
│
├── examples/                         # 示例与 benchmark
│   ├── simBenchmarks/                # LIBERO / SimplerEnv / Robocasa / RoboTwin / DOMINO / Behavior / calvin / MetaWorld / VLA-Arena / VLN-CE ...
│   ├── realRobots/                   # Franka / zerith_pi05 / UnitreeG1 / RoboChallenge
│   └── modelExtensions/              # CoTrainVLM / DiscreteDiffusion / Gemma4 / MiniCPM / NeuralVLA
│
├── deployment/                       # 模型服务部署
│   ├── model_server/                 # server_policy / server_umi_pi05_rio / policy_wrapper ...
│   ├── docker/                       # Dockerfile.train / server / robocasa
│   └── upload/                       # push_model_to_hf.py
│
├── scripts/                          # 训练流水线脚本（本分支核心）
│   ├── train.sh                      # 6 段式训练流水线（BOS 存储 + NFS）
│   ├── bos_io.py                     # BOS（百度云对象存储）读写工具
│   ├── download_data.py              # 按 yaml 从 BOS 拉数据
│   └── bos_ckpt.sh                   # checkpoint 管理
│
├── tests/                            # pytest 测试
├── docs/                             # 文档（guideline / faq / model_zoo / WM4A / agent_skills ...）
└── Makefile / pyproject.toml / uv.lock / .gitlab-ci.yml
```

---

## 3. 核心架构设计

### 3.1 显式模型边界

StarVLA 强调三层边界分离：

1. **Dataloader 只返回模型无关的原始字典**，不做任何模型专属预处理（tokenizer、图像编码等）。单个样本结构：
   ```python
   {
       "image": List[PIL.Image],
       "lang": str,
       "action": np.ndarray[T, action_dim],
       "state": Optional[np.ndarray[..., state_dim]],
   }
   ```
2. **`framework.forward()` / `framework.predict_action()` 直接处理原始输入**，训练/测试边界清晰。
3. **`baseframework` 统一损失接口** `compute_loss(tag, batch)` 按 tag 分发到 `forward`（vla）或 `forward_vlm`（vlm），天然支持多标签协同训练。

### 3.2 Framework Registry 机制

所有框架通过装饰器注册到全局注册表，`build_framework(cfg)` 是唯一的构建入口：

```python
@FRAMEWORK_REGISTRY.register("QwenGR00T")
class QwenGR00T(baseframework):
    ...
```

- 注册表位于 `starVLA/model/tools.py` 的 `Registry` 类。
- `base_framework.py:60` 的 `_auto_import_framework_modules()` 会递归扫描 `framework/` 下所有模块自动完成 import 注册。
- 扩展新框架 = 在 `VLM4A/` 或 `WM4A/` 新建文件 + 继承 `baseframework` + 实现 `forward()`/`predict_action()` + 注册。

### 3.3 baseframework 抽象

`base_framework.py:113` 的 `baseframework(PreTrainedModel)` 提供：

- `from_pretrained(pretrained_checkpoint, config_overrides)`：从 checkpoint 恢复（读 config + norm_stats + 权重，strict 加载）。
- `supports_training_tag(tag)` / `compute_loss(tag, batch)`：多标签训练路由。
- `forward_vlm(batch)`：默认委托给 `qwen_vl_interface`。
- 可训练模块发现工具（`auto_get_trainable_modules` / `print_freeze_status`）。

---

## 4. 配置系统

使用 OmegaConf（YAML）+ 命令行点号覆盖。

### 4.1 典型配置结构

```yaml
run_id: starvla
run_root_dir: playground/Checkpoints
framework:
  name: QwenGR00T            # 框架名称（必须）
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
  action_model:
    action_dim: 7
    action_horizon: 8
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/...
    data_mix: libero_goal
    per_device_batch_size: 16
  vlm_data:                 # 可选，协同训练
    dataset_py: vlm_datasets
trainer:
  max_train_steps: 100000
  learning_rate:            # 模块级学习率
    base: 2.5e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
  freeze_modules: ''        # 冻结模块（dotted-path，逗号分隔）
  loss_scale: {vla: 1.0, vlm: 0.1}
```

### 4.2 配置系统特色

- **CLI 覆盖**：`--framework.name QwenGR00T --trainer.freeze_modules "..."` 经 `normalize_dotlist_args` 转成 OmegaConf dotlist 合并。
- **`AccessTrackedConfig`**（`trainer_utils/config_tracker.py`）：包装 OmegaConf 并拦截字段访问，训练结束产出 `config.yaml`（只含实际用到的字段）与 `config.full.yaml`（全部字段），方便复现与 diff。
- **`apply_config_compat`**（`framework/share_tools.py`）：旧 yaml 字段名自动升级到新 schema（version_id="0.21"），幂等。
- **死字段告警**：`warn_unused_yaml_fields` 检测配了但不被读取的字段（如 `action_type`，实际由 `action_mode` + DataConfig transform 控制）。

---

## 5. 数据层

### 5.1 数据集类（gr00t_lerobot/datasets.py）

- `LeRobotSingleDataset`：单个数据集，支持 LeRobot v2.0 / v3.0，多模态（video/state/action/language），支持 parquet 读取、视频帧抽取（decord / torchvision_av）、数据增强 transform 链、统计量计算与缓存。
- `LeRobotMixtureDataset`：多数据集混合，支持按数据集 / 轨迹加权采样，训练/评测按比例确定性划分（逐轨迹取模）。

### 5.2 数据注册表与自动发现（registry.py）

`registry.py` 自动扫描 `examples/**/train_files/data_registry/` 目录：

- `ROBOT_TYPE_CONFIG_MAP`：robot_type → DataConfig 实例（定义 modality keys、transform 链、embodiment_tag）。
- `DATASET_NAMED_MIXTURES`：mixture 名 → [(dataset, weight, robot_type, [bos_uri])]。

**集成新数据集** = 在对应 benchmark 的 `data_registry/` 下写 `data_config.py`（+ 可选自定义 Dataset 子类），自动被全局发现，无需改核心代码。

### 5.3 Transform 体系（transform/）

ComposedModalityTransform 把多个 ModalityTransform 串成一条链，训练/评测共用：

- **视频**：VideoToTensor / VideoCrop / VideoResize / VideoColorJitter / VideoToNumpy。
- **state/action**：StateActionToTensor、NormalizationTransform（min_max / q99 / binary / mean_std）、PerStepNormalizationTransform（逐时间步归一化）、StateActionSinCosTransform。
- **几何变换**：RelativePoseTransform（SE(3) 相对位姿，几何正确）、QuaternionTo6DRotation（四元数→6D 旋转表示，提升到共享 transform 模块）。
- 支持 `apply` / `unapply` 双向操作——评测时用 `unapply` 把归一化预测精确还原成原始物理量。

### 5.4 UMI 数据集特殊处理（本分支核心）

`examples/realRobots/zerith_pi05/train_files/data_registry/umi_dataset.py` 的 `UMILeRobotDataset` 继承 `LeRobotSingleDataset`，处理 UMI 数据特性：

1. **无独立 action 列**：从 state 合成"绝对下一帧"action（`_get_action_from_state`，末帧 hold 而非补 0），供 RelativePoseTransform 做 SE(3) 相对化。
2. **一维列升维**：夹爪状态等一维数组 reshape 成 2D。
3. **无语言注释**：用默认空串占位。
4. **缺失 modality.json 自动生成**：从 `umi_modality_template.json` 原子写拷贝。
5. **在线 norm 统计**：`set_transforms_metadata` 中按 stride 采样、走完整 transform 链（归一化 passthrough）计算 pre-normalization 统计量，多 rank 只让 rank0 算+写缓存；评测模式可用 `STARVLA_EVAL_PER_KEY_JSON` 直接用模型固化统计。
6. **`action_pad` mask**：轨迹尾部凑不齐 action_horizon 的 timestep 标记为 False，loss 忽略。

---

## 6. 训练系统

### 6.1 训练入口（train_starvla.py）

主装配流程（`main()`，顺序严格）：

```
wrap_config → warn_unused_yaml_fields → setup_directories
→ build_framework（registry 实例化模型）
→ prepare_data（train + eval 两路 dataloader）
→ setup_optimizer_and_scheduler（多 LR 分组）
→ VLATrainer.prepare_training() → train()
```

技术栈：PyTorch + Accelerate + DeepSpeed（ZeRO-2/3），自动支持 Ascend NPU（`torch_npu` 自动检测）。

### 6.2 VLATrainer 关键机制

- **`prepare_training()`** 顺序敏感：设种子（每 rank 不同）→ 存初始配置 → checkpoint 加载（resume vs pretrained 区分）→ LR scheduler 重放 → 冻结模块 → `accelerator.prepare` → 初始化 tracker。
- **`_train_step()`**：`accelerator.accumulate(model)` 内 forward + backward + step，`torch.autocast` bf16 混合精度，梯度裁剪（DeepSpeed 下用 `get_global_grad_norm`），LR scheduler 只在 `sync_gradients=True` 时 step。
- **`eval_action_model()`**：在线 sanity eval，从 eval 划分（确定性采样）取 batch 算 MSE，不占用训练 batch。
- **checkpoint**：模型单独存 `.pt`/`.safetensors`（不存 optimizer 状态，省显存/磁盘），resume 时用最新 `steps_N`。

### 6.3 TrainerUtils 通用工具（trainer_utils/trainer_tools.py）

- 参数冻结：`freeze_backbones`（dotted-path，非正则）。
- 模块级 LR：`build_param_lr_groups`（按 `trainer.learning_rate` 下的 key 分组）。
- checkpoint 加载：`load_pretrained_backbones`（全量 / 按子模块 `reload_modules` 部分加载）。
- 梯度手术（multi-task）：`compute_grad_angle_with_stats`、`pcgrad_project`（PCGrad）。
- 其他：`_reset_dataloader`（epoch 推进 + `sampler.set_epoch`）、`eval_qwenpi`（IoU + action 距离）、`extract_json_from_string`。

### 6.4 性能分析器

`trainer_utils/perf_analyzer.py` 的 `PerformanceAnalyzer`：默认零开销，`STARVLA_PROFILE=1` 开启后分段计时（forward/backward/optimizer、data_time/model_time），输出"通信 vs 计算"瓶颈诊断报告。

### 6.5 其他训练入口

- `train_starvlm.py`：纯 VLM 训练。
- `train_starvla_cotrain.py`：VLA+VLM 多目标协同训练。
- `train_starvln.py`：NPU 原生训练入口。

---

## 7. 评测系统

### 7.1 在线开环评测（eval_starvla.py）

`VLAEvaluator` 镜像 `VLATrainer` 结构（1:1 对称，便于维护）：

- **数据通路与训练完全一致**：同一 dataset 类、同一 collate_fn、同一 transforms。
- **表示无关评测**：`_build_action_spec` 从 dataset 元数据推出每个 action key 的（norm_dim, raw_dim, rotation_type），`_decode_to_raw` 复用训练 transform 链的 `unapply` 把归一化预测还原成原始物理量（位姿/四元数/6D/关节角通用）；四元数符号对齐 `align_quaternion_sign_slice`。
- **多卡支持**：`accelerate launch --multi-gpu`，每 episode 内 anchor 经 DistributedSampler 切分到各 rank，`all_gather_object` 收集（变长数据不用 gather tensor）。全局 seed（不加 rank 偏移）保证 flow-matching 噪声可复现。
- **多 episode**：`--episodes 0 1 2`（0 基位置索引，自动映射真实 episode_id）。
- **产物 schema v4**：每 episode 子目录下 `predictions.npz` + `metrics.json`（含 representation 元数据）+ 可选 `obs_images.npz`（Rerun 可视化用）。
- **模型 norm 固化**：`STARVLA_EVAL_PER_KEY_JSON` 强制用模型固化的 per-subkey norm，脱离数据集 stats。
- 复用 `baseframework.from_pretrained` 加载 ckpt（自带 config.yaml，与 merged cfg 解耦）。

### 7.2 评测核心（eval/core/）

- `checkpoint_utils.py`：ckpt 路径发现/解析、`pick_mixture_entry`、dtype 工具。
- `io.py`：write_predictions / write_metrics / write_obs_images。
- `metrics.py`：指标计算。
- `transforms.py`：四元数符号对齐等。

### 7.3 可视化

`eval/viz/rerun_logger.py`：把 predictions / 观测图像转成 Rerun 日志（rrd），用于 Stage 2 可视化。

---

## 8. 部署系统（deployment/）

- `model_server/`：`server_policy.py`（通用策略服务）、`server_umi_pi05_rio.py`（UMI PI05 实时部署，dict I/O + 绝对 action decode）、`policy_wrapper.py`、`policy_norm_processor.py`（归一化处理器）。
- `docker/`：`Dockerfile.train` / `Dockerfile.server` / `Dockerfile.robocasa` + docker-compose。
- `upload/push_model_to_hf.py`：上传模型到 HuggingFace。

---

## 9. 训练流水线（scripts/，本分支核心）

### 9.1 train.sh 六段式流水线

`scripts/train.sh` 面向任务队列/后台拉起，共 6 段：

1. **配置环境**：source BOS/SwanLab 凭证（不入库）、挂载 `/mnt/cfs` NFS 共享盘、建 `/root/data`、`/root/code` 软链。
2. **拉取代码**：按 commit sha（任意位数前缀）或 latest 从 BOS 拉 CI 产物 zip → 解压 → `uv sync --python 3.10` 建 venv；`repo_<sha>` 目录 + flock 保证并发安全与 venv 复用。
3. **拉取数据**：`download_data.py` 按 yaml + data_config 的第 4 元素 bos_uri 拉数据。
4. **拉取预训练**：`bos_io.py get-dir`。
5. **训练**：复用 launcher `run_umi_pi05_relative_6d.sh`，导出 NUM_GPUS 自动检测等参数。
6. **上传产物**：整个 run 目录上传到 `bos://<bucket>/<namespace>/ckpts/<RUN_ID>/`。

### 9.2 bos_io.py（BOS 存储工具）

- `src-zip`：拉源码 CI 产物。
- `get-dir` / `put-dir`：目录递归下载/上传。
- 特性：bcecmd 优先、boto3 SIGALRM 看门狗（防下载卡死）、断点续传。

### 9.3 数据集合成与异常轨迹剔除

本分支在 `data_registry/data_config.py` 为 UMI 配置了多组 mixture：

- `shenzhen_relative_6d` / `_2`：深圳站点数据。
- `shenzhen_relative_6d_2_nooutlier`：剔除 2 条异常长轨迹（ep58 794帧、ep81 998帧）派生。
- `shenzhen_relative_6d_2_nooutlier_forceclose`：夹爪 force-close hack（峰后线性降到 0）。
- `hefei_relative_6d` / `_no280`：合肥站点数据（剔 ep280 异常长轨迹）。

这类派生数据集第 4 元素 bos_uri 留空，训练侧忽略、仅 train.sh 的 download_data.py 读取。

---

## 10. 当前分支工作重点（dev/train_anylze）

从 git 历史看，当前分支聚焦 **UMI 真实机器人数据的端到端训练流水线**：

- **存储切换**：`/nix` 本地盘 → `/mnt/cfs` NFS 共享盘（f0e9f47）。
- **BOS 可靠性**：bcecmd 优先 + boto3 看门狗 + 断点续传（6a7d246）。
- **数据卫生**：hefei/shenzhen 异常长轨迹剔除、force-close 夹爪 hack（683c946、bba394c）。
- **多进程安全**：modality.json 原子写（9b1cec0）、norm 统计多 rank 缓存锁（6a72960）、视频文件索引修复（0f2d6d9）。
- **评测**：`--episodes` 位置索引语义（00ada63）、eval 启动崩溃修复（259861e）、维度广播修复（dfd355c）。
- **SwanLab**：默认云端 + `--local` 切换（cc46a2b）、无 TTY 非交互登录（b8a8bba）。
- **配置与 bugfix**：train.sh 支持 config/commit 参数（fb5f213）、训练配置持续迭代。

---

## 11. 常用开发命令

### 环境安装
```bash
# 一键安装（见 docs/zerith_docs/infra/install.sh）
bash docs/zerith_docs/infra/install.sh
```

### 代码质量
```bash
make check                  # Black 检查 + Ruff lint
make autoformat             # Black + Ruff 自动修复（PR 只格式化改动文件）
black --check path/to/file.py
ruff check path/to/file.py
```

### 训练
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml

# UMI 流水线（本分支）
bash scripts/train.sh examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d.yaml [commit] [--local]
```

### 独立模块 smoke test
```bash
# 模型模块
python starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml path/to/config.yaml
# Dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml path/to/config.yaml
# 评测
python -m starVLA.training.eval_starvla --run-dir <run_dir> --eval-yaml <eval.yaml>
```

### 测试
```bash
pytest tests/
```

---

## 12. 相关文档索引

| 文档 | 路径 |
| --- | --- |
| 快速入门指南 | `docs/starVLA_guideline.md` |
| 常见问题 FAQ | `docs/faq.md` |
| 模型列表 | `docs/model_zoo.md` |
| WM4A 架构 | `docs/WM4A.md` |
| 集成数据集 | `docs/integrate_your_dataset.md` |
| 分支策略 | `docs/branching_strategy.md` |
| PR 指南 | `docs/PR_readme.md` |
| 环境安装 | `docs/zerith_docs/infra/install.sh` |
| Agent Skills | `docs/agent_skills/` |
| UMI 训练计划 | `docs/zerith_docs/umi/umi_training_plan.md` |
| UMI / z-UMI 代码差异 | `docs/zerith_docs/umi/umi_non_zumi_code_changes.md` |
| 评测协议 | `examples/eval_protocol.md` |

---

## 13. 扩展指南（速查）

### 添加新 VLA 框架
1. 在 `starVLA/model/framework/VLM4A/` 创建新文件。
2. 继承 `baseframework`，实现 `forward()` 和 `predict_action()`。
3. 用 `@FRAMEWORK_REGISTRY.register("Name")` 注册。
4. 在 `examples/` 添加配置示例。
5. `framework.name` 即注册名。

### 集成新数据集
1. 数据转换为 LeRobot 格式。
2. 在 `examples/<bench>/train_files/data_registry/data_config.py` 配置 DataConfig（modality keys + transform 链）与 mixture。
3. 创建训练配置 YAML。
4. 参考 `docs/integrate_your_dataset.md`。

### 冻结与学习率
```yaml
trainer:
  freeze_modules: "qwen_vl_interface,dino_encoder"
  learning_rate:
    base: 1e-5
    qwen_vl_interface: 1e-5
    action_model: 1e-4
```

---

## 14. 注意事项

1. **分支策略**：PR 应 target `starVLA_dev`；`starVLA` 为稳定分支。
2. **`**/bar/` 目录 git-ignored**：可放自定义脚本。
3. **NPU 支持**：代码自动检测 `torch_npu`，无需修改。
4. **不存 optimizer 状态**：resume 只恢复 model + LR 位置。
5. **冻结在 accelerate.prepare 之前**：避免冻结参数被 DeepSpeed 包装进优化器。
