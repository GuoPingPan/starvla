# UMI LeRobot 数据集到 starVLA QwenPI_v3 训练计划

## 当前结论

Z-umi 训练链路采用 UMI-compatible 的动作表示：

1. 原始 action 列离线生成，直接复制同帧 observation，不做时移、不做减法。
2. dataloader 按 action chunk 读取后，对 position 和 gripper 做 chunk 首帧相对化。
3. normalizer 的统计对象是所有训练样本的 relative action chunk，而不是 raw absolute action。
4. normalizer 是训练集固定统计，不做每个 chunk 自己的动态 min/max。
5. gripper 是连续量，使用 `min_max`，不使用 `binary`。
6. rotation 保持绝对，源数据四元数为 `xyzw`，进入 PyTorch3D 前转为 `wxyz`，再转 `rotation_6d`。

动态 per-chunk min/max 不采用：训练时能看到真实 action chunk，但推理时没有真实未来 chunk 的 min/max，无法反归一化。正确做法是先构造 relative chunk，再在所有训练 chunk 和所有 horizon step 上汇总固定 min/max。

## 数据集事实

| 属性 | 值 |
|------|-----|
| 数据路径 | `/root/workspace/data/lerobot_dataset_test` |
| 格式 | LeRobot v3.0 |
| Episodes | 153 |
| 帧数 | 61,115 |
| FPS | 60 |
| 相机 | `cam_wrist_0`，2704x2028，H.264 |
| pose 状态列 | `observation.state.end_0_pose`，`[x, y, z, qx, qy, qz, qw]` |
| gripper 状态列 | `observation.state.effector_0_joint_state`，标量 float32 |
| action pose 列 | `action.end_0_pose`，由 observation pose 同帧复制 |
| action gripper 列 | `action.effector_0_joint_state`，由 observation gripper 同帧复制 |
| 任务文本 | `meta/tasks.parquet` 中的中文任务 |

当前 A800 配置 `umi_20260718_combined` 使用另一组同格式 UMI v3 数据：

| 数据集 | Episodes | 帧数 | parquet | video |
|--------|----------|------|---------|-------|
| `20260718120227_lerobot_a76c9d3a` | 242 | 68,360 | 1 个，5.82 MiB | 19 个，共 3.52 GiB |
| `20260718120617_lerobot_b09b1e28` | 353 | 99,073 | 1 个，7.92 MiB | 28 个，共 5.14 GiB |

两者均为 `2704x2028@60fps H.264`。一个 parquet 文件包含多个 episode，而视频按约 200 MiB 聚合文件存储；这一布局与常见 LeRobot v2 的“每 episode 一个 parquet/video”不同，是 DataLoader 内存治理必须覆盖的正式数据事实。

## 数据流

### 1. 离线预处理

文件：`examples/Z-umi/preprocess_umi_data.py`

生成 action 列：

```python
action.end_0_pose[t] = observation.state.end_0_pose[t]
action.effector_0_joint_state[t] = observation.state.effector_0_joint_state[t]
```

这里不做相对化。相对化必须放在 dataloader 的 chunk 读取阶段，因为 base 是每个 sampled chunk 的首帧。

### 2. modality 拆分

文件：`examples/Z-umi/generate_modality.py`

`end_0_pose` 是 7 维 list 列：

```text
position: [0:3]
rotation: [3:7]
```

gripper 是单独的标量列，所以切片必须是：

```json
{"start": 0, "end": 1}
```

不能写成 `7:8`；`7:8` 只存在于概念拼接向量里，不存在于 gripper 原始列里。

### 3. chunk 内相对动作

YAML：

```yaml
datasets:
  vla_data:
    action_mode: rel
    action_mode_apply_keys:
      - action.end_0_position
      - action.effector_0_joint_state
```

对每个训练样本：

```text
position_rel[t] = action.position[t] - state.position[0]
gripper_rel[t] = action.gripper[t] - state.gripper[0]
rotation[t]     = action.rotation[t]
```

rotation 不参与相对减法，因为四元数减法没有几何意义。

### 4. normalizer 统计

正确语义：

```text
for each training sample i:
  build action chunk A_i[0:H]
  convert selected keys to relative values

for each dim d:
  min[d] = min over all i,t of A_rel[i,t,d]
  max[d] = max over all i,t of A_rel[i,t,d]
```

这与 UMI 原始实现一致：先由 dataset 生成训练使用的 action chunk 表示，再遍历 dataset 拟合 normalizer。UMI 对 action position 和 gripper 使用 range normalizer，对 rotation 使用 identity/range-safe 表示。

当前 starVLA 对应实现：

- `calculate_rel_action_statistics()` 会按 action indices 构造 chunk，再减 chunk 首帧 state。
- `StateActionTransform` 对 position/gripper 使用 `min_max`。
- rotation 转 `rotation_6d` 后使用固定 `[-1, 1]` 范围。
- `stats_gr00t.json` cache 必须包含 action mode、apply keys、state map、indices 和 modality 指纹，避免语义变化后复用旧统计。

### 5. transform 后模型输入

UMI state/action 维度：

| 分量 | Raw | Transform 后 |
|------|-----|--------------|
| position | 3 | 3 |
| rotation | quaternion 4 | rotation_6d 6 |
| gripper | 1 | 1 |
| total | 8 | 10 |

模型 sample：

```python
{
    "action": np.ndarray,  # [16, 10]
    "state": np.ndarray,   # [1, 10]
    "image": [PIL.Image],
    "lang": "...",
    "robot_tag": "NEW_EMBODIMENT",
}
```

`include_state: true` 必须开启，否则 QwenPI_v3 无法把离散化 state token 注入 instruction。

## 关键文件

| 文件 | 作用 |
|------|------|
| `examples/Z-umi/preprocess_umi_data.py` | 生成 action 列 |
| `examples/Z-umi/generate_modality.py` | 生成 `meta/modality.json` |
| `examples/Z-umi/train_files/data_registry/data_config.py` | UMI DataConfig 和 transforms |
| `examples/Z-umi/train_files/train_umi.yaml` | QwenPI_v3 训练配置 |
| `examples/Z-umi/train_files/run_umi_train.sh` | 训练启动脚本 |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | LeRobot 读取、action mode、statistics |
| `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` | 归一化、四元数顺序转换、rotation transform |
| `starVLA/dataloader/gr00t_lerobot/transform/concat.py` | state/action 拼接 |
| `starVLA/model/framework/VLM4A/QwenPI_v3.py` | QwenPI_v3 训练入口 |
| `docs/umi_dataloader_memory.md` | worker 内存根因、nvitop 语义和 benchmark 对比 |
| `openspec/specs/lerobot-worker-memory/spec.md` | worker 内存行为合同 |
| `openspec/specs/lerobot-video-decode-concurrency/spec.md` | 视频后端参数和解码并发合同 |

## 单卡 4090 首跑配置

当前 `examples/Z-umi/train_files/train_umi.yaml` 默认按单张 RTX 4090 首跑设置：

```yaml
run_id: umi_qwenpi_v3_4090_first_run
framework:
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-2B-Instruct
  action_model:
    action_dim: 10
    state_dim: 0
    repeated_diffusion_steps: 1
    diffusion_model_cfg:
      action_dit_hidden_dim: 768
datasets:
  vla_data:
    per_device_batch_size: 1
trainer:
  freeze_modules: 'qwen_vl_interface'
  gradient_accumulation_steps: 8
```

含义：

- 冻结 `qwen_vl_interface`，先只训练 `project_layers` 和 `action_model`，降低显存压力。
- 单卡 micro-batch 为 1，配合梯度累积 8，先验证数据流和 loss 是否稳定。
- `state_dim: 0` 只关闭 action head 的 raw state encoder；`include_state: true` 仍然保留，QwenPI_v3 会把 state 离散化后写入 instruction。
- `repeated_diffusion_steps: 1` 和 `action_dit_hidden_dim: 768` 是 pipeline-first 的保守显存设置。
- 如果首跑稳定，再考虑逐步增大 batch 或解冻部分 VLM，并把 `qwen_vl_interface` 学习率控制在 `1e-6 ~ 1e-5`。

`examples/Z-umi/train_files/run_umi_train.sh` 是一键训练入口，会自动：

- 进入 starVLA repo 根目录。
- 激活 `CONDA_ENV=starVLA`。
- 设置 NCCL 和 `TOKENIZERS_PARALLELISM=false` 等运行环境变量。
- 默认 `NUM_PROCESSES=1`，适配单张 4090 首跑。
- 从所选 YAML 读取并打印 `base_vlm`、`data_root_dir`、`data_mix`、`run_root_dir`、`run_id` 等诊断信息。
- 仅调用 `train_starvla.py --config_yaml <yaml>`，不传入任何 OmegaConf dotlist 覆盖。

训练、模型、数据和 wandb 参数以 YAML 为唯一来源。禁止通过启动脚本覆盖 `train_umi.yaml` 内的参数；需要改实验参数时，修改 YAML 或通过 `CONFIG_YAML=<path>` 选择另一份 YAML。

首次使用在线 wandb 时需要先登录一次；默认 UMI YAML 使用 `wandb_mode: disabled`，pipeline 首跑不需要 wandb 登录：

```bash
conda activate starVLA
wandb login
```

之后可直接一键启动：

```bash
bash examples/Z-umi/train_files/run_umi_train.sh
```

短跑 smoke test 使用单独 YAML，不使用命令行覆盖：

```bash
SMOKE_TEST=1 bash examples/Z-umi/train_files/run_umi_train.sh
```

选择其它配置也必须显式切换 YAML：

```bash
CONFIG_YAML=examples/Z-umi/train_files/train_umi_smoke.yaml \
bash examples/Z-umi/train_files/run_umi_train.sh
```

## 验证步骤

1. 生成/验证 action 列：

```bash
conda run -n starVLA python examples/Z-umi/preprocess_umi_data.py
```

2. 生成/验证 modality：

```bash
conda run -n starVLA python examples/Z-umi/generate_modality.py
```

3. 跑 dataloader smoke。统计缓存已经包含 action mode、indices 和 modality 指纹，语义变化时会自动重算；仅在需要强制重算时手动移走旧缓存。

```bash
conda run -n starVLA python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/Z-umi/train_files/train_umi.yaml
```

4. 检查：

- `action.shape == (16, 10)`
- `state.shape == (1, 10)`
- position/gripper 是 chunk 首帧相对值
- gripper 为连续 `min_max`
- rotation 为绝对 `rotation_6d`
- exported statistics 的 action mask/stat 维度为 10

## 开环评测

**入口 (2026-07-22 重构)**: 多卡开环评测走 `starVLA/training/eval_starvla.py`, 结构跟 `train_starvla.py` 1:1 对称 (复用 `TrainerUtils` / `baseframework.from_pretrained` / `make_LeRobotSingleDataset` / `collate_fn`)。旧单卡入口 `examples/Z-umi/eval_files/eval_open_loop.py` 已删除, 共享 helpers 迁到 `starVLA/eval/core/checkpoint_utils.py`。详细变更见 `docs/umi_non_zumi_code_changes.md` 第 12 节。

### 快速开始

```bash
# 默认 (单卡, 用 launcher 默认 RUN_DIR / EVAL_YAML)
bash examples/Z-umi/eval_files/run_umi_open_eval.sh

# overfit run 快测 (8 anchor)
RUN_DIR=/nix/data/umi_qwenpi_v3_A800_overfit_ep0 \
EVAL_YAML=examples/Z-umi/eval_files/eval_umi_overfit_overlay.yaml \
  bash examples/Z-umi/eval_files/run_umi_open_eval.sh --max-anchors 8

# 跑完直接出 Rerun 可视化
bash examples/Z-umi/eval_files/run_umi_open_eval.sh --visualize
```

也可以直接调 python (launcher 只负责 env / symlink / PYTHONPATH):

```bash
PYTHONPATH=. python -m starVLA.training.eval_starvla \
  --run-dir /nix/data/umi_qwenpi_v3_A800_overfit_ep0 \
  --eval-yaml examples/Z-umi/eval_files/eval_umi_overfit_overlay.yaml \
  --data-root-override /nix/data \
  --checkpoint-step 10000 \
  --episode-id 0
```

### 多卡评测

设 `NUM_PROCESSES=N`, launcher 自动切到 `accelerate launch --multi-gpu`:

```bash
NUM_PROCESSES=4 bash examples/Z-umi/eval_files/run_umi_open_eval.sh
```

每个 rank 处理 episode 内不同 anchor, `dist.all_gather_object` 把结果拼回 rank 0 写盘。`set_seed` 用全局 seed 不加 rank 偏移, 保证不同 world_size 下同一 anchor 拿到同一 flow-matching 噪声, 结果可复现。

### 多 episode

`--episodes` 接受多个 episode id, 或 YAML 里写 `eval.episodes: [0, 1, 2]`:

```bash
bash examples/Z-umi/eval_files/run_umi_open_eval.sh --episodes 0 1 2
```

每个 episode 一个子目录: `<output>/<run>_step<N>_seed<S>/episode_<id>/{predictions.npz, metrics.json}`。

### 两段式工作流

**Stage 1** (推理, GPU): `eval_starvla.py` 跑完只写 artifacts, 不画图:
- `predictions.npz` (per-anchor raw 预测 + GT, schema v4, 字段定义于 `starVLA/eval/core/io.py`)
- `metrics.json` (eval 时用的 yaml 配置 + run metadata, schema v4)

**Stage 2** (可视化, CPU): `stage2_visualize.py` 读 npz + json, 调 `starVLA.eval.viz.rerun_logger.log_episode_to_rerun` 写 `episode.rrd`:

```bash
# Stage 1 跑过, 单独做可视化
STAGE2_INPUT_DIR=playground/Evals/<run>_step10000_seed42 \
  bash examples/Z-umi/eval_files/run_umi_open_eval.sh --stage2-only

# 或直接调 stage2 (支持多 dir)
PYTHONPATH=. python examples/Z-umi/eval_files/stage2_visualize.py \
  playground/Evals/<run>_step10000_seed42/episode_0 \
  playground/Evals/<run>_step10000_seed42/episode_1

# 打开 .rrd
rerun playground/Evals/<run>_step10000_seed42/episode_0/episode.rrd
```

### predictions.npz schema v4

| key | shape | 说明 |
|------|-------|------|
| `__schema_version` | () | = 4, Stage 2 读时强校验 |
| `anchors` | (N,) int64 | anchor frame 索引 (episode 内) |
| `state_at_anchor` | (N, 8) | raw 绝对 pose 在 frame `anchor`: `[x,y,z,qx,qy,qz,qw,gripper]` |
| `action_gt` | (N, 8) | raw 绝对 pose 在 frame `anchor+1` (= 下一个 state, 跟 obs 错位 1 帧) |
| `action_pred_chunks` | (N, 16, 8) | raw 绝对预测 chunk: frame `anchor..anchor+15` 的 pose |
| `latency_seconds` | (N,) | per-sample 推理延迟 (含 cuda.synchronize) |

模型输出 (normalized, 10D: `position(3) + rotation_6d(6) + gripper(1)`) 经 `transforms.unapply` 反 norm + rot6d→quat, 再经 `dataset.invert_action_mode` 把 pose-pair relative 还原成绝对 pose (CLAUDE.md 第 5 节)。Quat 用 `align_quaternion_sign` 对齐到 GT 半球避免双覆盖假阳性。

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--run-dir` | (必填) | 训练 run 目录 (含 `config.yaml` + `checkpoints/`) |
| `--eval-yaml` | (必填) | Eval overlay YAML, 跟 ckpt config 合并 |
| `--data-root-override` | None | 评测机数据路径 (跟训练机不同时用) |
| `--checkpoint-step` | 最新 | 指定 ckpt step, 不指定用最新 |
| `--episode-id INT` | eval.yaml | 单 episode |
| `--episodes INT INT ...` | eval.yaml | 多 episode |
| `--max-anchors INT` | 全量 | smoke test 用, 限制每个 episode 的 anchor 数 |
| `--batch-size` | eval.yaml (4) | DataLoader batch size |
| `--seed` | eval.yaml (42) | flow-matching 噪声 seed |
| `--device` | cuda | `cuda` 或 `cpu` |
| `--dtype` | bf16 | `bf16` / `fp16` / `fp32` |
| `--output-dir` | 自动派生 | 覆盖默认 `<output_root>/<run>_step<N>_seed<S>/` |

优先级: CLI > `eval:` YAML > 代码默认。

### Rerun viz 解读

打开 `.rrd` 后看到 8 个面板 (pos_x, pos_y, pos_z, qx, qy, qz, qw, gripper), x 轴是 frame index。每个面板:

- **obs (蓝)**: `state[i]` 在 frame `i` 的真实 pose (1 点/anchor)
- **gt (红)**: `action_gt[i]` 在 frame `i` 显示, 数值是 frame `i+1` 的真实 pose (下一个 state, 跟 obs 错位 1 帧)
- **pred (绿, 7 组)**: `pred_chunks[i]` 在 frame `i..i+15` 显示, 模型预测的 16-step chunk

overfit run 下, pred chunk 应当沿 obs/gt 平滑滑动; 位置维 (pos_x/y/z) 跟 obs/gt 数值差应 < 1cm。quaternion 维可能因双覆盖呈现镜像, 已通过 `align_quaternion_sign` 对齐到 GT 半球。

### 开环评测证据分类

按 CLAUDE.md 第 4 节, 开环评测必须区分训练分布内拟合 vs 未见 episode 泛化:
- 评测**训练 mixture 内**数据集 → 只能报告 pipeline 状态和 fitting evidence
- 评测**留出 `umi-test`** → 可作为 generalization candidate, 但开环误差 ≠ 闭环任务成功率, `closed_loop_success_rate` 始终 `not_evaluated`

回放训练 episode 时必须复用训练 normalizer (从 `run_dir/checkpoints/` 自动加载), 禁止从测试数据重算。Action 必须按 ckpt 保存的 action mode 逆变换到 absolute pose/gripper 后再比较。

### 历史结果 (episode 0 / frame 50, 旧版 eval, seeds 42/43/44 均值)

| checkpoint | normalized RMSE | position ADE | rotation mean | gripper MAE |
|------------|-----------------|--------------|---------------|-------------|
| 5000 | 0.067324 | 0.030654 | 6.442 deg | 0.004509 |
| 10000 | 0.053200 | 0.018819 | 3.969 deg | 0.004710 |
| 20000 | 0.028654 | 0.013500 | 2.228 deg | 0.002017 |
| 30000 | 0.017247 | 0.007836 | 2.236 deg | 0.000783 |

最终 checkpoint 相比 step 5000 的 normalized RMSE 改善 75.00%, 分类 `pipeline_pass=true / fitting_evidence=positive / generalization=not_evaluated`。该结果基于旧版 `eval_open_loop.py` 单样本拟合诊断, 入口已替换; 趋势作为历史参考, 新入口下需重跑生成可比数据。

## 固定单 episode 过拟合诊断

当需要验证训练代码改动是否破坏拟合链路时，使用独立配置：

```bash
CONFIG_YAML=examples/Z-umi/train_files/train_umi_overfit.yaml \
bash examples/Z-umi/train_files/run_umi_train.sh
```

该配置基于正式 `train_umi.yaml`，只做三项诊断隔离：

- `run_id: umi_qwenpi_v3_4090_overfit_ep0`，避免覆盖正式训练结果。
- `datasets.vla_data.overfit_episode_id: 0`，固定使用 episode 0。
- `datasets.vla_data.num_workers: 0`，只在主进程中保留一份样本缓存。

episode 0 有 317 帧。启动阶段会按锚点 0-316 逐条执行现有数据链路，包括视频读取、尾部填充、position/gripper 相对化、图像 resize、state/action normalization 和模型样本打包。317 条变换后样本只构造一次，之后所有 epoch 直接复用内存缓存，不再重新读取视频或执行 dataloader transforms。

最后 15 个锚点保持正式训练语义：超出 episode 末尾的 absolute action 使用最后一帧填充，然后再按当前锚点 state 做 relative position/gripper 转换。单 episode 只限制优化样本，不改变 normalizer；归一化和 run 目录导出的 `dataset_statistics.json` 仍来自完整训练集统计。

缓存完成后日志会打印 `payload_mib`、`process_rss_before_mib`、`process_rss_after_mib` 和 `process_rss_delta_mib`。episode 0 的确定性数组/图像负载为：

| 项目 | 计算 | 字节 |
|------|------|------|
| RGB 图像 | `317 * 224 * 224 * 3` | 47,717,376 |
| float16 action | `317 * 16 * 10 * 2` | 202,880 |
| float16 state | `317 * 1 * 10 * 2` | 6,340 |
| 合计（不含文本和 Python 容器） |  | 47,926,596，约 45.71 MiB |

因此长期保留的 resized cache 约 46 MiB，算上任务文本、PIL/list/dict 对象开销后仍应在几十 MiB 量级。不要把训练进程总 RSS 当成缓存大小：实测缓存构建中主进程曾达到约 7.17 GiB RSS，但其中同时包含已加载的 4B 模型、视频解码器和运行时分配器。正式日志中的 RSS delta 用来观察本机本次运行的额外常驻开销；`payload_mib` 才是缓存内容的可复现下界。

训练计划保持正式配置不变：30000 optimizer steps、1000 warmup steps、batch size 1、gradient accumulation 8、每 5000 step 保存 checkpoint，并保留 flow-matching 随机性和 dropout。该模式不缓存 Qwen processor 输入或 VLM hidden states，模型内部视觉输入处理仍逐 step 执行。

### 单 action chunk 过拟合诊断

需要先验证模型能否记住一个确定训练样本时，使用最小差异配置：

```bash
CONFIG_YAML=examples/Z-umi/train_files/train_umi_overfit_anchor.yaml \
bash examples/Z-umi/train_files/run_umi_train.sh
```

`train_umi_overfit_anchor.yaml` 使用独立 `run_id`，并通过 `datasets.vla_data.overfit_anchor_index: 50` 选择唯一训练样本。这里的 anchor 50 表示以 episode 0 的第 50 帧作为当前观测/状态，构造 action delta 0-15，即帧 50-65 的完整 16-step action chunk，最终训练数据集长度为 1。

该配置保留正式实验的 30000 optimizer steps、1000 warmup steps、5000-step checkpoint、学习率、cosine scheduler、noise/time 采样和冻结 Qwen VLM 的策略。为了把这一层定义为“极限记忆”而不是正式训练参数复现，单独关闭 action head dropout 和 optimizer weight decay，将 `gradient_accumulation_steps` 设为 1，并将 `num_inference_timesteps` 提高到 20。单样本 DataLoader 长度为 1；使用当前 Accelerate/DeepSpeed 训练循环时，累计值 8 会让 trainer/scheduler 的 step 与 DeepSpeed 实际 optimizer update 边界不一致，因此该配置必须使用累计 1。20-step 参数只影响 action 回放积分，不改变 flow-matching 训练 loss。

单 action chunk 诊断刻意禁止尾部填充，以避免“同一个训练目标”中混入由末帧复制得到的动作。episode 0 长度为 317，当前 action horizon 为 16，因此合法 anchor 范围是 0-301。已用真实数据确认 anchor 50 生成一个 state `[1,10]`、action `[16,10]` 的样本；anchor 302 会因需要帧 317 而在训练前直接报错。正式训练参数的复现实验仍由单 episode 配置承担；该配置只用于检验单样本极限记忆能力。

过拟合证据使用现有开环脚本，不修改评测接口。分别对 episode 0 的 frame 50、150、250 比较 checkpoints 5000/10000/20000/30000。验收要求为训练 loss 有限且总体下降、三个位置的平均 normalized RMSE 在最终 checkpoint 相比早期 checkpoint 改善。所有结论只能表述为 pipeline/fitting evidence，不能表述为未见数据泛化。

### 当前验证状态（2026-07-17）

已经完成：

- OpenSpec strict validation、Python 编译、YAML 参数检查和 `git diff --check`。
- 真实 episode 0 缓存验证：317 个 anchor，图像 `224x224`、state `[1,10]`、action `[16,10]`，最后 15 个 anchor 保持正式尾部填充语义。
- 真实单 action chunk 验证：episode 0 / anchor 50 仅缓存 1 个样本，action 对应帧 50-65；anchor 302 因 action chunk 越界而被明确拒绝。
- 缓存 DataLoader 重复迭代两轮后，source transforms 调用数保持 317，样本 fingerprint 不变。
- 两步全链路 smoke：保留的日志记录了有限 loss，并确认当时成功保存 step 1/2 checkpoint 和 final model；训练阶段 `timing/data` 约为 0.0003-0.0007 秒/步，证明训练迭代直接命中内存缓存。当前 smoke 模型目录已不在工作区，现存凭据是 `playground/Evals/umi_overfit_ep0_validation/smoke_train.log`。

尚未完成：正式 30000 步训练和 frame 50/150/250 的三组评测。2026-07-17 启动过一次正式命令，但按人工接管要求在缓存阶段中止，退出码为 130，未产生 optimizer step；`playground/Evals/umi_overfit_ep0_validation/full_train.log` 只能作为中止记录，不能作为正式实验结果。

### 手动完成正式实验

从仓库根目录启动训练。配置的 `wandb_mode` 为 `online`，运行前需确认当前环境已经登录且网络可用：

```bash
set -o pipefail
CONFIG_YAML=examples/Z-umi/train_files/train_umi_overfit.yaml \
bash examples/Z-umi/train_files/run_umi_train.sh 2>&1 | \
tee playground/Evals/umi_overfit_ep0_validation/manual_full_train.log
```

本次中止发生在缓存阶段且没有 checkpoint；重新执行会从头构建缓存并从 step 0 开始。训练完成后检查退出状态、缓存诊断、有限 loss 和四个目标 checkpoint：

```bash
rg "Cached overfit episode|payload_mib|Training Configuration|Step [0-9]+, Loss|Training complete" \
  playground/Evals/umi_overfit_ep0_validation/manual_full_train.log

for step in 5000 10000 20000 30000; do
  test -f "playground/Checkpoints/umi_qwenpi_v3_4090_overfit_ep0/checkpoints/steps_${step}_pytorch_model.pt" ||
  test -f "playground/Checkpoints/umi_qwenpi_v3_4090_overfit_ep0/checkpoints/steps_${step}_model.safetensors" ||
  { echo "missing checkpoint ${step}"; exit 1; }
done
```

然后分别运行三个固定 anchor 的现有开环评测：

```bash
for anchor in 50 150 250; do
  output_dir="playground/Evals/umi_overfit_ep0_anchor${anchor}"
  PYTHONPATH=. python examples/Z-umi/eval_files/eval_open_loop.py \
    --run-dir playground/Checkpoints/umi_qwenpi_v3_4090_overfit_ep0 \
    --data-root-override /nix/data \
    --dataset-name 20260718120617_lerobot_b09b1e28 \
    --dataset-path /nix/data/20260718120617_lerobot_b09b1e28 \
    --checkpoint-step 30000 \
    --episode-id 0 \
    --max-anchors 1 \
    --horizon-step 15 \
    --device cuda --dtype bf16 \
    --output-dir "${output_dir}" 2>&1 | tee "${output_dir}.log"
done
```

> **migration note**: 旧版 `eval_open_loop.py` 使用 `--frame-index` + `--checkpoint-steps` (多 ckpt) 做单点拟合。新脚本统一用 anchor-based episode replay, 上述命令通过 `--max-anchors 1` 把回放限制到单个 anchor 等价近似原拟合点;如需多 ckpt 趋势比较需外层 bash 循环或后续给脚本加多 ckpt 支持(见"仍待优化")。

每个输出目录应包含 `metrics.json`、`checkpoint_trend.png`、`normalized_action_fit.png`、`decoded_action_fit.png` 和 `input.png`。最终报告应同时给出三个 `metrics.json` 的 `pipeline_pass`、`fitting_evidence` 和 normalized RMSE 趋势；只有三处都能证明最终 checkpoint 相比早期 checkpoint 改善时，才把本次诊断记为正向拟合证据。

### umi-test 整段 episode 回放

step 30000 checkpoint 可以逐帧回放 `/root/workspace/data/umi-test` 中任意 episode：

```bash
PYTHONPATH=. python examples/Z-umi/eval_files/eval_open_loop.py \
  --run-dir playground/Checkpoints/umi_qwenpi_v3_4090_first_run \
  --data-root-override /nix/data \
  --dataset-name 20260718120617_lerobot_b09b1e28 \
  --dataset-path /root/workspace/data/umi-test \
  --episode-id 0 \
  --checkpoint-step 30000 \
  --horizon-step 15 \
  --stride 1 \
  --device cuda \
  --dtype bf16 \
  --output-dir playground/Evals/umi_test_episode_000_step30000
```

`umi-test` 只有 observation 列，没有 action 列、`modality.json` 或 StarVLA statistics。回放脚本不得修改测试数据集，而是在内存中把未来 observation 作为 source action，并始终复用训练 run 对应数据集的 modality 和 normalization statistics，避免测试统计泄漏。

默认选择 action chunk 的 horizon 15：anchor `t` 的预测与 source frame `t+15` 对齐。模型输出中的 relative position/gripper 加回 anchor state，恢复为原数据绝对坐标；absolute quaternion 在绘图前与 ground truth 做等价符号对齐。最终把 `x/y/z/qx/qy/qz/qw/gripper` 八个维度画入同一张 4x2 图，并保存 JSON 指标和 NPZ 数组。

episode 0 完整回放覆盖 370 个 anchor（目标帧 15-384），结果为：

| metric | value |
|--------|-------|
| position ADE | 0.012420 |
| rotation mean / RMSE / max | 2.968 / 3.298 / 6.851 deg |
| gripper MAE / RMSE | 0.001212 / 0.001787 |
| normalized horizon-15 RMSE | 0.028412 |
| mean inference latency | 0.1855 s |

这些结果是独立测试数据上的开环 episode 拟合度；它们不等价于闭环机器人任务成功率。

使用固定随机种子 `20260713` 从 episode 1-152 中无放回抽取 `31, 144, 140, 88, 33`，额外完成 5 个完整 episode 回放：

| episode | anchors | position ADE | rotation mean / max | gripper MAE | normalized RMSE |
|---------|---------|--------------|---------------------|-------------|-----------------|
| 31 | 368 | 0.012864 | 2.016 / 4.925 deg | 0.001584 | 0.027541 |
| 144 | 389 | 0.013166 | 2.133 / 4.524 deg | 0.001619 | 0.026921 |
| 140 | 389 | 0.010041 | 1.751 / 5.183 deg | 0.001303 | 0.023118 |
| 88 | 406 | 0.015794 | 3.095 / 9.642 deg | 0.001823 | 0.035861 |
| 33 | 365 | 0.009225 | 1.514 / 4.337 deg | 0.001480 | 0.025214 |
| mean | 1917 total | 0.012218 | 2.102 / 5.722 deg | 0.001562 | 0.027731 |

5 个 episode 均通过输入输出 shape、有限数值和 action inverse 检查。episode 88 是该样本中误差最大的一条，主要误差出现在快速姿态变化区间。

对同一组 5 个随机 episode、相同 anchor 和 seed，把固定比较位置从 horizon 15 改为 horizon 1 后，配对均值如下：

| metric | horizon 1 | horizon 15 | reduction |
|--------|-----------|------------|-----------|
| position ADE | 0.002907 | 0.012218 | 76.2% |
| rotation mean | 1.338 deg | 2.102 deg | 36.4% |
| gripper MAE | 0.000388 | 0.001562 | 75.1% |
| normalized RMSE | 0.011542 | 0.027731 | 58.4% |

horizon 1 的逐帧尾部误差为 position P95/P99/max `0.005252/0.006098/0.009811`、rotation `2.385/2.972/3.881 deg`、gripper `0.001171/0.001999/0.004294`。这证明短期预测显著优于 horizon 15。

但是 60 FPS 下 horizon 1 只对应约 16.7 ms，而当前平均推理延迟约 190 ms。真机不能在推理完成后直接执行已经过期的 horizon 1；必须按 observation timestamp 和端到端延迟跳过 stale action，选择约 horizon 11-12，或者对完整 chunk 做时间对齐的 temporal ensemble。

## A800 persistent worker 内存与解码并发治理

`train_umi_A800.yaml` 为吞吐使用：

```yaml
per_device_batch_size: 16
num_workers: 12
prefetch_factor: 4
persistent_workers: true
worker_memory_trim_interval: 1
ordered_sample_dispatch: true
video_backend: decord
video_backend_kwargs:
  num_threads: 1
```

这里保持当前 12 workers、prefetch 4 和 persistent workers，在每个 worker 处理等价
的 16 个 sample 后建立显式回收边界：

1. batch 保持 live reference，构造过程中的 PyAV/torchvision/Pandas 临时对象已不可达。
2. full `gc.collect()` 回收老生代循环对象。
3. glibc 可用时调用 `malloc_trim(0)`，请求把空闲 arena 页归还给 OS。
4. 每个底层 `LeRobotSingleDataset` 只缓存当前 parquet 文件；切换文件时替换，不随历史访问数量增长。

新参数默认 `0`，因此 LIBERO 和其他未显式配置的 benchmark 不执行额外 batch-boundary trim。A800 UMI 为本次已确认的高分辨率 workload，显式设为 `1`。

该行为不改变采样索引、图像/动作/状态值、normalizer 或训练随机数。真实数据验证中，trim 开关前后的 6 个 worker 样本逐字段 exact equal，两个 UMI 数据集的 parquet cache/fresh read 也 exact equal。单 worker 的 anonymous PSS 在 warm-up 后 60 个随机样本仅增加约 0.3 MiB；batch size 16 连续四批的 anonymous PSS 为 `346.7, 349.5, 349.7, 349.6 MiB`。

### Decord 内部线程限制

`decord.VideoReader` 在没有 `num_threads` 参数时使用默认值 `0`，含义是自动选择内部解码线程，不是零线程。该默认值适合少量 reader 的孤立解码，但当前 4 rank、每 rank 16 workers 已经有 64 路外层并发。实测活跃 worker 通常有 20–21 条线程，而训练 cgroup 只有约 62 核 CPU quota，最终形成近千条线程竞争 CPU。

保持 `action_dit_hidden_dim=768`、`repeated_diffusion_steps=1`、batch、worker、prefetch 和 persistent 设置不变的 10 分钟对比为：

| 后端 | 有效耗时 | CPU PSI some/full | cgroup 读盘 |
|------|----------|-------------------|-------------|
| `torchvision_av` | `3.53 s/step` | `0.033 / 0.033 s/s` | `0.10 MiB/s` |
| decord 默认自动线程 | `7.19 s/step` | `0.881 / 0.511 s/s` | `0.00 MiB/s` |
| `decord(num_threads=1)` | `3.35 s/step` | `0.065 / 0.065 s/s` | `0.01 MiB/s` |

两轮都实际消耗约 62 个 CPU cores，但自动线程使 runnable tasks 长时间排队。磁盘 I/O pressure 接近零，GPU 采样反复出现一个 rank 落后、其他 rank 在 NCCL collective 等待，因此根因是嵌套解码并发而不是存储。

`timing/data` 只记录训练主进程从预取队列执行 `next()` 的同步时间，不包含 persistent worker 的后台解码；即使它接近零，worker 仍可能占满 CPU，并连带拉长记录 forward/backward/optimizer wall time 的 `timing/model`。

通用 dataset factory 现在把 `datasets.vla_data.video_backend_kwargs` 透传到 `LeRobotSingleDataset`；字段缺省时传空字典，保持原后端默认行为。A800 UMI 显式使用单线程 reader，使内部解码并行与 64 个外层 worker 匹配。该设置不改变请求的帧索引或训练 RNG。两个真实 UMI 子数据集各抽取帧索引 `0/60/300`，`num_threads=0` 与 `1` 的输出 shape 均为 `[3,2028,2704,3]`、dtype 均为 `uint8`，像素 exact equal、最大差值为 0。

重启后的 10 分钟、600-sample 监控确认 4 个 rank 各有 16 个 worker，worker `NLWP` 从自动线程时的 20–21 降为 4–5，每 rank 16 个 worker 合计只有 77–80 条线程。有效时间从 `7.19` 降到 `3.35 s/step`，缩短 53.5%，并比 `torchvision_av` 基线快约 5.1%；CPU PSI some/full 分别下降约 92.6%/87.3%，四卡平均 GPU utilization 恢复到约 70.3%，与 `torchvision_av` 的约 70.7% 基本一致。

线程超额竞争已经解决，但按序 DataLoader 仍在每 16 step 的 worker 轮转边界出现 head-of-line straggler。单线程 worker 串行完成一个 16-sample 高分辨率随机 seek batch 时，最慢 rank 会让其他 rank 在 NCCL collective 等待；本窗口最长同一步停留约 56 秒。它解释了为何正常 `model_times` 已恢复到约 0.63–0.81 秒，而总体速度仍是 `3.35 s/step`。该长尾属于下一项独立吞吐优化，不能通过重新开启大量 decord 内部线程规避。

### Prepared batch 的 ordered sample dispatch

`ordered_sample_dispatch: true` 不改变 batch size，也不使用 `in_order=False`。训练
DataLoader 先正常经过 Accelerate/DeepSpeed preparation，确保 DeepSpeed 仍看到
micro-batch 16，并由 `BatchSamplerShard` 先确定每个 rank 的原始 batch。

随后包装器只把该 batch 的 16 个 indices 展开成单 sample worker 任务，并携带
`batch_id/position/batch_size`。主进程按原 position 严格重组；任何乱序、不完整或
跨 batch token 都直接报错。训练循环收到的 per-rank index、batch 边界和 sample
顺序保持不变。

预取和回收参数按 sample 数保持等价：

```text
原预取：4 batches/worker * 16 samples = 64 samples/worker
新预取：64 single-sample tasks/worker
原 trim：每 1 batch = 每 16 samples
新 trim：每 16 single-sample tasks
```

四 rank CPU 对照中，12-worker 原调度的最慢 rank 稳定平均/最大长尾为
`6.49/24.08 s`，新调度为 `3.67/7.03 s`；平均改善约 43.5%，最大长尾下降约
70.8%。16 个真实 UMI 样本的图像像素及 `action/state/lang/robot_tag` 全字段 exact
equal，主进程 Python/NumPy/Torch RNG 推进一致。48-batch memory soak 的 aggregate
worker `Pss_Anon` 在 warm-up 后保持约 6 GiB 平台。

真实 GPU/DDP 训练不会热加载该修改；下一次正常启动仍需在稳定窗口验证有效
`s/step`、GPU 空转、CPU PSI 和至少 1000 step 的 worker PSS slope。由于 UMI
transform 是确定性的，worker assignment 改变不改变当前训练样本；包含 worker-local
随机 augmentation 的其他数据集默认不启用。
包装器当前显式要求 `gradient_accumulation_steps: 1`；若未来要与梯度累计同时使用，
必须先恢复等价的 Accelerate end-of-dataloader 状态传播并完成新的训练语义验证。

这里的“训练语义不变”只指 worker trim、bounded parquet cache 和 decord 线程调度。当前 A800 YAML 相对 Git 基线还把 micro-batch 从 4 改为 16、gradient accumulation 从 8 改为 1；每 rank 每 optimizer update 的样本数从 32 变为 16，4 rank global batch 从 128 变为 64。这些既有超参数变化会影响训练轨迹，不能把整份 YAML 的前后 run 当作仅内存/解码并发修复的 A/B。`data_root_dir` 也从本机不存在的 `/root/workspace/data` 切到实际数据目录 `/root/workspace/luke/data`；比较结果时必须确认数据内容一致。

当前 ordered sample dispatch 配置的完整 4-rank、48-worker 重跑仍是最终验收项。
在至少 1000 optimizer step 的监控窗口内必须同时满足：

- aggregate worker `Pss_Anon` warm-up 后不再保持正线性斜率。
- cgroup `memory.events:oom_kill` 不增加。
- rank PSS 保持平台。
- loss、样本 shape 和有限数值检查正常。
- 活跃 worker 不再普遍显示 20–21 条线程，CPU PSI 显著低于 decord 自动线程基线。
- 在相同模型和 DataLoader 参数下，10 分钟有效 step 时间优于 `7.19 s/step`。

nvitop 单进程 RSS 是真实 resident pages，但 fork 共享页会在多个 PID 中重复出现，且 `%MEM` 以宿主机总内存为分母，不代表 500 GiB cgroup 的剩余额度。正式验收以 cgroup `memory.current/memory.stat/memory.events` 和 worker `/proc/<pid>/smaps_rollup` 的 PSS/Pss_Anon 为准。

详细故障分析和 UMI/LIBERO 对比见 `docs/umi_dataloader_memory.md`；OpenSpec 合同见
`openspec/specs/lerobot-worker-memory/spec.md`、
`openspec/specs/lerobot-video-decode-concurrency/spec.md` 和
`openspec/changes/optimize-lerobot-ordered-sample-dispatch/specs/lerobot-ordered-sample-dispatch/spec.md`；
运行时验收和 archive 后，最后一项会由 OpenSpec CLI 合并到 canonical specs。

## 文档同步要求

任何影响 Z-umi 数据语义、normalizer、训练配置、OpenSpec 流程或开发规则的修改，必须同步更新：

| 文件 | 要求 |
|------|------|
| `docs/umi_training_plan.md` | 更新设计决策、数据流、验证步骤 |
| `openspec/specs/` | 更新或新增对应 requirement |
| `CLAUDE.md` | 更新 Claude/Codex 相关开发规则 |
| `agent.md` | 同步通用 agent 开发规则 |
| `docs/umi_non_zumi_code_changes.md` | 登记所有 `examples/Z-umi/` 之外的 UMI 代码修改、影响和验证结果 |

`CLAUDE.md` 和 `agent.md` 必须保持语义同步；修改任一文件时必须检查另一个文件。

## 参考

- UMI dataset normalizer: https://github.com/real-stanford/universal_manipulation_interface/blob/main/diffusion_policy/dataset/umi_dataset.py
- UMI range normalizer: https://github.com/real-stanford/universal_manipulation_interface/blob/main/diffusion_policy/common/normalize_util.py
- UMI task config: https://github.com/real-stanford/universal_manipulation_interface/blob/main/diffusion_policy/config/task/umi.yaml
