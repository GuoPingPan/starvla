# UMI 非 zerith_pi05 代码改动（UMI Pipeline Code Changes）

## Purpose

登记为跑通 UMI 2B/10D 与 PI05 pipeline 而对 `examples/realRobots/zerith_pi05/`（原 `z_umi`，2026-09 更名，见 agent.md 附录 A）、`openspec/` 之外代码做的修改：每个改动的原因、内容、影响与验证。完整叙述见 `docs/zerith_docs/umi/umi_non_zumi_code_changes.md`。

## Requirements

### Requirement: SE(3) relative pose action mode

系统 SHALL 新增独立 `action_mode: relative`（不改旧 `abs/rel/delta`）：通过 `action_mode_pose_pairs` 显式配置 action/state 的 position、rotation 与 quaternion 顺序，用旋转矩阵计算 `p_rel = R_base^T(p_target-p_base)`、`R_rel = R_base^T R_target`；一个 chunk 全部 horizon step 共享 `state[0]`；提供正向 `apply_action_mode()` 与逆向 `invert_action_mode()`；`relative` 明确拒绝与 `action_mode_apply_keys` 混用，pose pair 之外的 action 字段（如 gripper）不处理。

#### Scenario: rotation_6d 归一化恒等
- **WHEN** rotation 转 rotation_6d 后归一化
- **THEN** PyTorch3D rotation_6d 用旋转矩阵前两行，固定 `[-1,1]` min-max 为恒等映射；`StateActionTransform` 不得拒绝 relative rotation 的归一化

#### Scenario: round-trip 精度
- **WHEN** 对真实 UMI 数据做正向 + 逆向
- **THEN** 手工公式误差约 `1e-9`，round trip 误差不超过 `3.7e-9`

### Requirement: YAML 参数真正生效

多处历史代码让 YAML 参数失效，MUST 修正：`train_starvla.py` 在 YAML load 之后再创建 `Accelerator`（传入 `gradient_accumulation_steps`），且只在 `sync_gradients` 时增 `completed_steps`/eval/log/save/step scheduler/梯度裁剪；`total_loss = action_loss * trainer.loss_scale.vla`；`gradient_clipping` 空时 fallback `max_grad_norm`；`wandb.init(..., mode=cfg.wandb_mode)`；`QWen3.py` 删除强制 `sdpa` 覆盖（让 `attn_implementation` 由 YAML 控制）；`ds_config.yaml` 的 `gradient_accumulation_steps` 改 `auto`；`QwenPI_v3.py` 的 `repeated_diffusion_steps` 读取优先级为 `framework.action_model` > `trainer` > 默认 4。

#### Scenario: 梯度累计由 YAML 控制
- **WHEN** YAML 写 `gradient_accumulation_steps: 8`
- **THEN** 训练运行时真正按 8 累计，日志的 total batch size / scheduler step / eval 触发点与真实 optimizer step 对齐

#### Scenario: flash_attention_2 生效
- **WHEN** YAML 设 `attn_implementation: flash_attention_2` 且环境有 `flash_attn`
- **THEN** Qwen3 实际使用 flash_attention_2；不可 import 时打印 warning 并 fallback sdpa，不崩溃

### Requirement: dataloader 并发参数从 YAML 读

`_build_vla_dataloader_kwargs()` SHALL 从 `datasets.vla_data` 读 `num_workers`/`prefetch_factor`/`persistent_workers`/`pin_memory`，缺省保持原项目默认（16/4/true/true）；`num_workers=0` 时不传 `prefetch_factor`/`persistent_workers`，避免 PyTorch 校验错误。

#### Scenario: 单卡 4090 首跑降内存
- **WHEN** UMI 4090 首跑配置
- **THEN** 用 `num_workers: 2 / prefetch_factor: 2 / persistent_workers: false`，预取 batch 数从约 64 降到约 4，优先解决 CPU/容器 OOM kill

### Requirement: LeRobot v3 数据适配

`datasets.py` SHALL 多项适配：`_column_to_2d_array()` 统一把 scalar/list parquet 列转 `[T,D]`（scalar gripper `[T]→[T,1]`，非 2D 报错）；`_detect_lerobot_version()` 优先用显式配置否则读 `meta/info.json`；`_build_stats_cache_config()` 把 action/state indices、apply keys、state map、key 顺序、modality 指纹加入 cache；`_pack_sample()` 优先用 transform 已产出的 `video`/`action`/`state`，否则才 fallback 逐 modality concat；`_extract_normalization_statistics()` 对 rotation_6d 保存 6D 统计而非原始 4D。

#### Scenario: scalar gripper 不再错切
- **WHEN** parquet gripper 是标量 float
- **THEN** 经 `_column_to_2d_array()` 转成 `[T,1]`，后续按 `[T,D]` 切片不再出错

### Requirement: rotation_6d concat 维度

`concat.py` rotation key 的允许维度 SHALL 从「原始维度或 3」扩展为「原始维度、3、6」；`apply()` 记录每个 key 的实际 transformed 维度，`unapply()` 优先用实际维度拆分；inverse 后校验是否刚好消费完整个拼接 tensor，禁止静默遗留/错位。

#### Scenario: 10D action 正确拆分
- **WHEN** UMI action 经 transform 后为 `(16,10)`
- **THEN** 正确拆分为 `3+6+1`（position+rotation_6d+gripper），不再被错误拆成 `3+4+1`

### Requirement: quaternion 源顺序配置

`state_action.py` SHALL 新增 `source_rotation_orders`，支持 `wxyz`/`xyzw`；forward 时先转 `wxyz` 再做 rotation transform 再归一化，inverse 反向；`model_dump(mode="json")` 包含 source order，使 cache fingerprint 感知该语义。

#### Scenario: xyzw 源顺序被识别
- **WHEN** 数据源 quaternion 是 `xyzw`
- **THEN** 配置 `source_rotation_orders` 后先转 `wxyz` 再做 rotation transform，cache fingerprint 感知顺序变化

### Requirement: worker 线性内存回收

`lerobot_datasets.py` collate SHALL 增加 opt-in `worker_memory_trim_interval`：只在 worker 内计数，到 interval 执行 full `gc.collect()` 与（libc 可用时）`malloc_trim(0)`，默认 0；`datasets.py` 正确保存当前 trajectory id 并为每个 `LeRobotSingleDataset` 缓存当前 parquet 文件（切换时覆盖，不随历史增长）；该行为不改变 sampling/transform/normalizer/模型/loss/optimizer/scheduler。

#### Scenario: trim 不改变数据
- **WHEN** 对比 trim interval 0 与 1 的 worker 输出
- **THEN** `image/action/state/lang/robot_tag` 逐字段 exact equal

### Requirement: 视频后端参数透传

`_get_video_backend_kwargs()` SHALL 从 `datasets.vla_data.video_backend_kwargs` 读 backend-agnostic 参数，把 OmegaConf/AccessTrackedConfig mapping 规范化为 dict（非 mapping 抛 `TypeError`），缺省返回空 dict 保持原默认；A800 UMI 显式 `video_backend: decord` + `num_threads: 1`，不硬编码为全项目默认。

#### Scenario: decord 单线程像素一致
- **WHEN** 比较帧索引 0/60/300 的 decord `num_threads=0` 与 `1`
- **THEN** shape/dtype 相同，像素 exact equal，最大差值为 0

### Requirement: ordered sample dispatch

opt-in `ordered_sample_dispatch` SHALL：原 DataLoader 先正常 `accelerator.prepare()`（保留 DeepSpeed micro-batch 与 `BatchSamplerShard`），再把 batch 展开为带 `batch_id/position/batch_size` 的单 sample token，worker 每次只执行一个 `dataset[index]`，主进程严格按 token 恢复原 batch（乱序/跨 batch/不完整直接报错）；启用时要求 `gradient_accumulation_steps: 1`；sample prefetch factor 与 trim interval 自动乘以 batch size 保持等价。

#### Scenario: 最慢 rank 长尾下降
- **WHEN** 4 rank、12 worker 高分辨率随机 seek batch
- **THEN** 最慢 rank 平均从约 6.49 降到 3.67 s/step，最大长尾从 24.08 降到 7.03 s

### Requirement: 两段式 eval 框架与 schema v4

eval SHALL 拆为 Stage 1（推理，GPU，`eval_starvla.py` 写 `predictions.npz`+`metrics.json`）与 Stage 2（可视化，CPU，`stage2_visualize.py` 写 `.rrd`）；`predictions.npz` schema v4 含 `__schema_version(=4)`/`anchors`/`state_at_anchor(N,8)`/`action_gt(N,8)`/`action_pred_chunks(N,16,8)`/`latency_seconds`，Stage 2 读时强校验版本；decode 复用训练 transform 链的精确逆 `unapply`，并用 `align_quaternion_sign` 对齐 GT 半球。

#### Scenario: 多卡复现
- **WHEN** 多卡评测
- **THEN** `set_seed` 用全局 seed 不加 rank 偏移，不同 world_size 下同一 anchor 拿到同一 flow-matching 噪声，结果可复现

## References

- 完整改动登记：`docs/zerith_docs/umi/umi_non_zumi_code_changes.md`
- UMI 训练计划：`docs/zerith_docs/umi/umi_training_plan.md`
- worker 内存根因：`docs/umi_dataloader_memory.md`
