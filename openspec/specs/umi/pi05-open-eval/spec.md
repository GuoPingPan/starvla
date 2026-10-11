# UMI · PI05 开环评测（Open-Eval）

## Purpose

UMI · PI05 开环评测入口（`examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh` 及配套脚本）的行为合同：launcher 与 eval 分离、配置优先级、数据集挑选、多卡复现与两段式产物。完整叙述见 `docs/zerith_docs/umi/umi_pi05_open_eval.md`。

## Requirements

### Requirement: launcher 与 eval 分离

Launcher SHALL 只负责 uv 环境、`PYTHONPATH`、路径 preflight、参数透传、Stage1→Stage2 串联，**不把训练参数 hardcode 进脚本**；所有评测参数通过 `EVAL_YAML` overlay + CLI 传；绕开 launcher 直接跑 Stage 1（`python -m starVLA.training.eval_starvla`）也是合法用法。

#### Scenario: PI05 无 base_vlm symlink
- **WHEN** saved config 没有 `framework.qwenvl.base_vlm`（PI05/paligemma）
- **THEN** launcher 的 base_vlm symlink 处理 try/except 返回空，跳过 symlink

### Requirement: 配置优先级

参数优先级 SHALL 为 **CLI > eval_yaml > 代码默认**；`load_merged_config` 把 ckpt 的 `config.full.yaml` 与 eval overlay YAML 做 `OmegaConf.merge`（overlay 覆盖同名项），再套 CLI 覆盖。

#### Scenario: CLI 覆盖 overlay
- **WHEN** CLI 传 `--checkpoint-step` 而 eval.yaml 也写了该字段
- **THEN** CLI 值胜出，覆盖 overlay 与代码默认

### Requirement: mixture 多 entry 时必须显式指定数据集

`pick_mixture_entry` 逻辑 SHALL：未指定 `--dataset-name` 且 mixture 只有 1 个 entry → 自动挑唯一 entry；未指定且 mixture 有 >1 个 entry → **抛错，拒绝「隐式挑第一个」**；指定的名字不在 mixture → 抛错并列出可用项。

#### Scenario: mixture 有 2 个 entry
- **WHEN** 本 run 的 `data_mix` 有 2 个活跃 entry 且未传 `--dataset-name`
- **THEN** 抛 `ValueError ... has 2 datasets ...; pass --dataset-name to pick one.`

### Requirement: episodes 是位置索引

`--episodes` SHALL 指的是 **eval 列表（split 后升序）里的位置索引**（0 基），不是原始 episode_id，内部自动映射到真实 episode_id；startup 打印 `位置索引→真实 episode_id` 映射，越界给清晰报错。

#### Scenario: 指定第 0、1 条 eval 轨迹
- **WHEN** 传 `--episodes 0 1`
- **THEN** 评测 eval 划分里位置索引 0、1 对应的真实 episode，而非原始 episode_id 0、1

### Requirement: 跨数据集评测的统计量口径

换不同分布数据集（如 `hefei_relative_6d`）时，SHALL 仍用模型固化的 `_per_key` 统计（训练时统计），与数据集自己的 stats 不一致 → 开环 MSE/ADE **不可定量比较**，仅适用于 pipeline 验证 / Rerun 可视化 / 同 embodiment 泛化烟测；要可比的跨数据集数字需额外对齐 norm_stats。

#### Scenario: 跨数据集结果不可比
- **WHEN** 用 hefei 数据评测 shenzhen 训练的模型
- **THEN** norm 口径不一致，MSE/ADE 不能与同分布评测定量比较，只能作 pipeline/烟测证据

### Requirement: 多卡与复现

`NUM_PROCESSES=N` 时 launcher SHALL 自动切 `accelerate launch --multi-gpu`；单 episode 内 anchor 跨卡切分，`dist.all_gather_object` 回 rank 0 聚合；全局 seed（默认 42）**不加 rank 偏移**，保证不同 world_size 下同一 anchor 拿到同一 flow-matching 噪声。

#### Scenario: 不同卡数结果一致
- **WHEN** 同一 anchor、同一 seed，分别用 1 卡与多卡评测
- **THEN** flow-matching 噪声相同，predictions 逐元素相等（latency 允许差异）

### Requirement: 两段式产物

output_dir SHALL 自动派生为 `<output_root>/<run_basename>_step<N>_seed<S>/`，每 episode 一个子目录 `episode_<id>/`；Stage 1 产物：`predictions.npz`（schema v4）、`metrics.json`（含 eval_config / run metadata / `representation` 块）、`obs_images.npz`（可选 sidecar）；Stage 2 读 npz+json 写 `episode.rrd`；换数据集（`--data-mix`）时 output_dir 自动追加 `_mix-<name>`，不覆盖原数据集结果。

#### Scenario: 跨数据集结果隔离
- **WHEN** 用 `--data-mix hefei_relative_6d` 评测
- **THEN** 产物落到 `..._mix-hefei_relative_6d/` 子目录，不覆盖原 shenzhen 数据集结果

### Requirement: eval 用裸 Accelerator

`eval_starvla.py` SHALL 用裸 `Accelerator`（不带 DeepSpeed plugin），因为推理无 backward，DeepSpeed 会要求 dataloader 有 `batch_size` 属性而报错；并强制 `SWANLAB_MODE=disabled`，避免残留环境变量污染。

#### Scenario: 推理不挂 DeepSpeed
- **WHEN** 构造 eval accelerator
- **THEN** 不带 DeepSpeed plugin，避免 `train_micro_batch_size_per_gpu` 缺失错误

### Requirement: ckpt 布局兼容

`checkpoint_utils` SHALL 兼容扁平导出的两种 ckpt 布局（`.pt` 直接平铺在 `run_dir/checkpoints/` 或 `run_dir/`），无需手动 mv。

#### Scenario: 两种 ckpt 布局都能读
- **WHEN** ckpt 平铺在 `run_dir/` 或在 `run_dir/checkpoints/`
- **THEN** 都能被自动发现，无需手动调整目录

## References

- 完整说明：`docs/zerith_docs/umi/umi_pi05_open_eval.md`
- Stage 1 主入口：`starVLA/training/eval_starvla.py`
- Overlay 配置：`examples/realRobots/zerith_pi05/eval_files/eval_umi_pi05_relative_6d_overlay.yaml`
- 数据集注册表：`examples/realRobots/zerith_pi05/train_files/data_registry/data_config.py`
