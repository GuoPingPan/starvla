# UMI Pipeline 非 Z-umi 代码修改说明

本文只说明本次为了跑通 UMI 2B/10D pipeline 而修改的非 `examples/Z-umi/`、非 `openspec/` 代码。Z-umi 配置、启动脚本、OpenSpec 变更文档不在本文范围内。

## SE(3) relative pose action mode（2026-07-20）

### 涉及文件

- `starVLA/dataloader/gr00t_lerobot/action_modes.py`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `starVLA/dataloader/gr00t_lerobot/transform/state_action.py`
- `tests/test_lerobot_relative_pose_action_mode.py`
- `tests/test_lerobot_action_mode_regression.py`

### 原有问题

原有 `action_mode: rel` 和 `delta` 把配置字段当作普通欧氏向量做减法，只适合
position、joint 或 gripper；对 quaternion 逐元素相减没有旋转几何意义。数据管线
也缺少一个能把 action position/rotation 与明确 state anchor 配对的配置，无法表达：

```text
T_relative = inverse(T_base) @ T_target
```

旧统计路径无法为每个 episode 的所有 chunk 生成同一语义的 relative pose 统计，
评测端也没有 action-mode 层的统一逆变换。`StateActionTransform` 还会拒绝标记为
relative 的 rotation 转换后归一化，即使 rotation_6d 每个分量天然在 `[-1,1]`。

### 修改内容

新增独立 `action_mode: relative`，不修改旧 `abs/rel/delta`：

- 通过 `action_mode_pose_pairs` 显式配置 action/state position、rotation 和
  quaternion 顺序，不依赖字段名猜测旋转语义。
- `xyzw/wxyz` quaternion 先校验并归一化，再通过旋转矩阵计算
  `p_rel = R_base^T(p_target-p_base)` 和 `R_rel = R_base^T R_target`。
- 一个 action chunk 的全部 horizon step 共享 `state[0]`；新增正向
  `apply_action_mode()` 和逆向 `invert_action_mode()`，逆向使用
  `T_target = T_base @ T_rel`。
- pose pair 之外的 action 字段不处理，因此 UMI gripper 可保持未来绝对值。
  `relative` 明确拒绝与 `action_mode_apply_keys` 混用。
- statistics 先按 `episode_index` 分组，使用 first/last tail padding，并按 episode
  批量计算所有 anchor 的 relative pose；不会跨 episode 构造 chunk。未配对 action
  列继承 raw absolute 统计。
- raw `LeRobotModalityMetadata` 继续把 parquet pose 标为 absolute，并独立决定
  action-mode 之前的 `first_last` tail padding；transform 使用的
  `DatasetMetadata` 则可把配对 action pose 标为 relative。两层元数据不得混用，
  否则 episode 尾部 quaternion 会在 SE(3) 相对化前被零填充。
- cache fingerprint 增加 `relative_pose_version` 和完整 pose pairs，避免新旧 action
  语义复用同一个 `stats_gr00t.json`。
- rotation 转换后的固定边界对 absolute/relative 一致。PyTorch3D
  `rotation_6d` 使用旋转矩阵前两行，固定 `min=-1,max=1` 的 min-max 公式严格为
  恒等映射。

### 影响与验证

模型输入接口不变：绝对 state 仍是 `[1,10]`，action 仍是 `[16,10]`，只改变显式
选择新模式的 action target 语义。旧模式有 exact regression，未配置
`relative` 的数据集不进入新路径。

已完成 15 项单元/回归测试，覆盖 SE(3) 正逆、`q/-q` 等价、无效 quaternion、
batch/runtime 一致、共享 anchor、tail padding、episode 边界、absolute gripper、
cache fingerprint、rotation_6d 行顺序和恒等归一化。两套真实 UMI 数据的手工公式
误差约 `1e-9`，round trip 误差不超过 `3.7e-9`；完整 DataLoader 输出
state `[1,10]`、action `[16,10]` 且 finite。单卡 A800 真实 4B 模型 2-step
训练完成 forward/backward/optimizer，并保存 final model。

## 固定单 episode 过拟合缓存补充

### 涉及文件

- `starVLA/dataloader/__init__.py`

### 原有问题

正常 LeRobot mixture 会在每次 `__getitem__` 时重新选择 episode/anchor，并重复视频读取、尾部填充、相对动作转换、图像 resize、normalization 和样本打包。固定一个 episode 做 30000-step 过拟合诊断时，这些确定性数据操作没有必要反复执行，也难以证明每轮看到的是完全相同的训练样本集合。

### 修改内容

新增显式 opt-in 的 `datasets.vla_data.overfit_episode_id` 路径。启用时要求 mixture 只有一个底层数据集且 `num_workers=0`，一次性读取目标 episode 的 parquet rows，按选定锚点调用现有 `get_step_data()`、`transforms()` 和 `_pack_sample()`，再用内存样本列表构建 DataLoader。episode id 使用 `is not None` 判定，因此 episode 0 可正常选择。缓存构造完成后释放 source dataset 持有的 episode parquet table，并执行一次垃圾回收。

在 episode 配置下可进一步设置 `datasets.vla_data.overfit_anchor_index`。未设置时仍缓存 episode 的全部锚点并保留原有尾部填充；设置后只缓存该锚点对应的一个训练样本，并检查 action 的全部 `delta_indices` 均落在 episode 内。单 action chunk 模式不允许尾部填充，越界锚点会直接报错；只配置 anchor 而未配置 episode 也会直接报错。

完整 mixture 的 `save_dataset_statistics()` 在训练数据集被缓存列表替换前执行，保证单 episode 优化仍导出并使用完整训练集统计。未配置 `overfit_episode_id` 时保持原 mixture dataloader 路径。

缓存完成日志同时输出缓存样本数/shape、数组与图像的 payload 大小，以及缓存构建前后的当前进程 RSS。RSS 只用于观察实际进程，因为其中包含模型和视频解码器；episode 0 的 `224x224 RGB + float16 state/action` 可复现负载约为 45.71 MiB，不含少量文本和 Python/PIL 容器开销。

### 影响与验证

该改动只影响显式开启的诊断配置，不修改模型、优化器、scheduler、普通 mixture 采样或评测接口。已验证 episode 0 的 317 条缓存样本、尾部填充、重复迭代不再触发 transforms，以及两步全链路 smoke。另已用真实数据验证 anchor 50 只生成一个 state `[1,10]`、action `[16,10]` 的训练样本，对应 episode 帧 50-65；anchor 302 因需要帧 302-317、超出有效范围 0-316 而被拒绝。30000-step 正式诊断和固定 frame 的 checkpoint 趋势由人工按 `docs/umi_training_plan.md` 中的命令完成；在这些结果产出前不能声称完整过拟合实验已经通过。

## 修改范围

涉及文件：

- `starVLA/training/train_starvla.py`
- `starVLA/dataloader/__init__.py`
- `starVLA/dataloader/lerobot_datasets.py`
- `starVLA/dataloader/ordered_sample_dispatch.py`
- `starVLA/model/modules/vlm/QWen3.py`
- `starVLA/config/deepseeds/ds_config.yaml`
- `starVLA/dataloader/gr00t_lerobot/action_modes.py`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `starVLA/dataloader/gr00t_lerobot/transform/concat.py`
- `starVLA/dataloader/gr00t_lerobot/transform/state_action.py`
- `starVLA/model/framework/VLM4A/QwenPI_v3.py`
- `tests/test_lerobot_video_backend_kwargs.py`
- `tests/test_lerobot_relative_pose_action_mode.py`
- `tests/test_lerobot_action_mode_regression.py`
- `tests/test_ordered_sample_dispatch.py`

这些修改的共同目的：让训练实际遵循 YAML 参数，并让 UMI LeRobot v3 数据能稳定
产出 QwenPI_v3 需要的 10D state/action 输入。训练入口和 dataloader 文件负责运行时
配置、调度与内存生命周期，其余核心文件负责数据语义、统计缓存、变换维度和
QwenPI_v3 action head 参数归属。

## 1. `train_starvla.py`

### 原有问题

训练入口在 import 阶段就创建了全局 `Accelerator`：

```python
deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
```

这会导致一个问题：当 YAML 还没有加载时，`Accelerator` 已经初始化完成，因此 `trainer.gradient_accumulation_steps` 无法参与实际的 Accelerate/DeepSpeed 初始化。结果是 YAML 里写了梯度累计，但训练运行时可能仍按 Accelerate/DeepSpeed 默认值执行。

训练 step 里还有两个问题：

- `optimizer.zero_grad()` 放在 `accelerator.accumulate()` block 开头，会破坏梯度累计语义。
- `eval/log/save` 在非真实 optimizer step 时也可能触发，尤其在 gradient accumulation 场景下容易让 step 计数、日志和 scheduler 行为不一致。

此外，`loss_scale.vla` 和 `max_grad_norm` 这类 YAML 参数之前没有完整生效：

- `total_loss` 固定等于 `action_loss`，没有读取 `trainer.loss_scale.vla`。
- 梯度裁剪只读 `gradient_clipping`，没有 fallback 到 `max_grad_norm`。
- `wandb_mode` 主要依赖环境变量，不能完全由 YAML 控制。

### 修改内容

新增 `setup_accelerator(cfg)`，在 YAML load、CLI merge、compat 处理之后再创建 `Accelerator`：

```python
Accelerator(
    deepspeed_plugin=DeepSpeedPlugin(),
    gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
)
```

训练循环改为只在 `accelerator.sync_gradients` 为真时：

- 增加 `completed_steps`
- 更新 progress bar
- 执行 eval/log/save
- step scheduler
- 做梯度裁剪

训练 step 改为：

- 不在 accumulate block 开头清梯度
- backward 后执行 optimizer step
- scheduler 只在同步 step 执行
- 最后 `optimizer.zero_grad(set_to_none=True)`

同时：

- `total_loss = action_loss * trainer.loss_scale.vla`
- `gradient_clipping` 为空时 fallback 到 `max_grad_norm`
- `wandb.init(..., mode=cfg.wandb_mode)`，让 wandb mode 从 YAML 生效

### 修改影响

修改后，`trainer.gradient_accumulation_steps` 才是真正的训练行为来源，而不只是配置文件里的记录。日志中的 total batch size、scheduler step、eval/save 触发点也与真实 optimizer step 对齐。

这对 UMI pipeline 很关键，因为单卡 4090 必须依赖 batch size 1 + 梯度累计来做首跑验证。

## 2. `dataloader/__init__.py`

### 原有问题

VLA dataloader 的并发参数之前在代码里固定为：

```python
num_workers=16
pin_memory=True
persistent_workers=True
prefetch_factor=4
```

这会让 16 个 worker 每个预取 4 个 batch，训练过程中最多提前准备约 64 个 batch。对 UMI 这类包含视频/图像解码、parquet 读取、PIL resize 的数据，CPU 内存峰值会很高；本次正式训练中已经出现 `DataLoader worker ... is killed by signal: Killed`，并且容器 `memory.events` 记录了 `oom_kill 1`。

### 修改内容

新增 `_build_vla_dataloader_kwargs()`，从 `datasets.vla_data` 读取：

- `num_workers`
- `prefetch_factor`
- `persistent_workers`
- `pin_memory`

如果 YAML 不设置这些字段，默认保持原项目行为：`num_workers=16`、`prefetch_factor=4`、`persistent_workers=true`、`pin_memory=true`。如果 `num_workers=0`，代码不会向 PyTorch DataLoader 传入 `prefetch_factor` 和 `persistent_workers`，避免触发 PyTorch 参数校验错误。

### 修改影响

UMI YAML 现在可以直接控制 dataloader 的内存和吞吐权衡。单卡 4090 首跑配置继续使用更保守的：

```yaml
num_workers: 2
prefetch_factor: 2
persistent_workers: false
pin_memory: true
```

这样预取 batch 数从约 64 降到约 4，优先解决 CPU/容器内存 OOM kill。

A800 当前配置显式使用 12 workers、prefetch 4 和 persistent workers。新增的
`worker_memory_trim_interval` 单独处理 worker heap 生命周期；完整设计和验证见
本文第 9、11 节及 `docs/umi_dataloader_memory.md`。

## 3. `QWen3.py`

### 原有问题

原代码先从 YAML 读取：

```python
attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
```

但下一行直接覆盖为：

```python
attn_implementation = "sdpa"
```

这会导致 YAML 中写 `flash_attention_2` 永远不生效。也就是说，即使配置里设置了 `framework.qwenvl.attn_implementation: flash_attention_2`，Qwen3 实际加载时仍然使用 `sdpa`。

### 修改内容

删除强制覆盖，只保留已有的安全 fallback 逻辑：

- YAML 请求 `flash_attention_2`
- 如果 `flash_attn` 可 import，则使用 `flash_attention_2`
- 如果不可用，则打印 warning 并 fallback 到 `sdpa`

### 修改影响

修改后，Qwen3 的 attention 实现真正由 YAML 控制。这样 UMI 的 `train_umi.yaml` 可以明确声明期望使用 `flash_attention_2`，同时在环境缺少 FlashAttention 时不会直接崩溃。

## 4. `ds_config.yaml`

### 原有问题

DeepSpeed 配置中写死了：

```json
"gradient_accumulation_steps": 1
```

这和 YAML 中的 `trainer.gradient_accumulation_steps` 是两套来源。对于 UMI 首跑，YAML 需要设置为 8；如果 DeepSpeed 仍固定为 1，就会产生配置与实际执行不一致的风险。

### 修改内容

改为：

```json
"gradient_accumulation_steps": "auto"
```

让 Accelerate/DeepSpeed 使用运行时传入的累计步数，而不是在全局 DeepSpeed 配置里固定写死。

### 修改影响

DeepSpeed 不再覆盖 YAML/Accelerator 的梯度累计设置。UMI smoke YAML 可以用 1，正式 `train_umi.yaml` 可以用 8，两者都通过 YAML 控制，不需要改全局 DeepSpeed 配置。

## 5. `datasets.py`

### 原有问题

UMI LeRobot v3 数据和当前 dataloader 的几个假设不完全一致：

- 低维 parquet 列可能是标量 float，例如 gripper；原逻辑多处直接 `np.stack` / `np.vstack`，对标量列容易得到 `[T]`，后续按 `[T, D]` 切片时会出错。
- 数据集版本之前主要依赖 `data_cfg.lerobot_version`，对 LeRobot v3 的 `meta/info.json`、`meta/tasks.parquet` 自动识别不足。
- relative action statistics 的 cache key 只包含 `action_mode`，没有包含 action/state indices、apply keys、state map、modality metadata 等语义信息。修改相对化语义或 modality 后，可能错误复用旧的 `stats_gr00t.json`。
- transform 之后的数据可能已经被 concat 成 `video`、`state`、`action` 总键，但 `_pack_sample()` 仍按原始 modality key 逐个读取，导致 pack 后 sample 缺字段或 shape 不符合 QwenPI_v3。
- rotation 从 quaternion 变成 `rotation_6d` 后，保存/export 的 statistics 仍可能按原始 quaternion 4D 统计，造成 action/state 统计维度和真实模型输入 10D 不一致。
- v3 metadata 的 task 表读取后没有稳定设置 `task_index` 为索引，任务文本映射可能不符合后续按 task index 查找的逻辑。

### 修改内容

新增 `_column_to_2d_array()`，统一把 scalar/list parquet column 转成 `[T, D]`：

- scalar gripper: `[T] -> [T, 1]`
- list pose/quaternion: `[T, D]`
- 非 2D 结构直接报错，避免静默产生错误 shape

统计和数据读取路径中把直接 `np.stack` 的位置改为 `_column_to_2d_array()`，覆盖：

- `calculate_dataset_statistics()`
- `calculate_delta_action_statistics()`
- `calculate_rel_action_statistics()`
- `_get_data_by_modality()`

新增 `_detect_lerobot_version()`：

- 优先使用显式 `data_cfg.lerobot_version`
- 否则读取 `meta/info.json` 的 `codebase_version`
- 再根据 `meta/tasks.parquet` / v2 tasks 文件存在性 fallback

扩展 `_build_stats_cache_config()`，把以下内容加入 cache config：

- action/state indices
- `action_mode_apply_keys`
- `action_mode_state_map`
- full action/state key 顺序
- modality metadata fingerprint

这样相对化或 modality 语义变化时会重算 stats，而不是复用旧缓存。

重写 `_pack_sample()` 的兼容逻辑：

- 如果 transform 已经产出 `video`，直接从 packed video 取第一步图像。
- 如果 transform 已经产出 `action` / `state`，直接使用 packed tensor/array。
- 否则才 fallback 到原始 modality keys 逐个 concat。
- tensor 会先 detach/cpu 转 numpy，保证 sample 输出是训练框架期望的 numpy/PIL 格式。

导出统计时新增 `_extract_normalization_statistics()`：

- 从 `StateActionTransform` 的 normalizer 中读取真实 transform 后统计。
- 对 rotation_6d 这类 representation-changing transform，优先保存 6D 统计，而不是原始 4D quaternion 统计。
- `generate_action_mask_for_used_keys()` 支持根据 normalization statistics 判断真实维度。

### 修改影响

修改后，UMI 的 scalar gripper、LeRobot v3 metadata、relative action stats cache、packed sample 输出和 rotation_6d 统计可以对齐到同一个 10D state/action 合同：

```text
position(3) + rotation_6d(6) + gripper(1) = 10
```

这直接影响 QwenPI_v3 的训练输入。如果没有这些修改，pipeline 可能在 dataloader 初始化、stats 计算、sample packing 或统计维度校验阶段失败，或者更糟糕地复用过期 stats 造成训练语义错误。

## 6. `concat.py`

### 原有问题

`ConcatTransform` 对 rotation action 的维度校验只允许原始维度或 axis-angle 3D：

```python
target_shapes.append(3)
```

UMI 当前需要把 quaternion rotation 转成 `rotation_6d`，因此 rotation action 的最后一维会从 4 变成 6。原校验不接受 6D，会在 concat 阶段误判为维度不匹配。

此外，`ConcatTransform.unapply()` 始终使用 metadata 中的原始维度拆分拼接向量。UMI metadata 中 rotation 是 quaternion 4D，但正向 transform 后实际拼接的是 rotation_6d 6D。因此 `(16, 10)` 模型输出会被错误拆成 `3+4+1`，rotation 少取两维、gripper 通道错位，并在 rotation normalizer inverse 时报维度不匹配。

### 修改内容

rotation key 的允许维度从：

```text
原始维度, 3
```

扩展为：

```text
原始维度, 3, 6
```

并把 assert 信息改为显示实际维度和允许列表，方便定位。

新增运行时 transformed dimension 记录：

- `apply()` 在校验后记录每个 action/state key 的实际最后一维。
- `unapply()` 优先使用实际 transformed dimension 拆分；没有正向记录时保持 metadata 原始维度 fallback。
- inverse 完成后校验所有维度是否刚好消费完整个拼接 tensor，禁止静默遗留或错位通道。

### 修改影响

`rotation_6d` 可以正常参与 action concat，最终 action 维度能正确组成 10D。该修改只放宽 rotation key 的合法 transform 后维度，不改变非 rotation key 的维度校验。

完成正向 transform 的 UMI action 现在可从 `(16, 10)` 正确拆分为 `3+6+1`，再经
rotation inverse 恢复为 `(16, 8)` 的 action-mode 层表示。新 `relative` 模式对应
relative position/quaternion + absolute gripper；旧 `rel` 配置对应 relative
position + absolute quaternion + relative gripper。未改变表示维度的现有数据仍使用
原 metadata dimension。

## 7. `state_action.py`

### 原有问题

UMI 原始 pose quaternion 是 `xyzw` 顺序，而 PyTorch3D/常见 rotation transform 通常期望 `wxyz`。如果直接把 `xyzw` 当成 `wxyz` 输入 rotation converter，转换出的 `rotation_6d` 会错误，但 shape 看起来仍然合法，属于难发现的数据语义 bug。

同时，`StateActionTransform.model_dump(mode="json")` 没有包含 quaternion source order 信息。如果后续统计缓存依赖 transform config，source order 改变也不一定会触发 cache 失效。

### 修改内容

新增 `source_rotation_orders` 配置：

```python
source_rotation_orders: dict[str, str]
```

支持：

- `wxyz`
- `xyzw`

新增校验：

- 配置值不在上述集合时报错。
- 只有 quaternion key 才允许设置 source order。

新增 `_reorder_quaternion()`：

- `xyzw -> wxyz`
- `wxyz -> xyzw`

forward 时：

1. 如果 key 配置了 source order，先转为 `wxyz`。
2. 再执行 rotation transform。
3. 再执行 normalizer。

inverse 时反向处理：

1. 先 inverse normalizer/rotation transform。
2. 如果配置了 source order，再从 `wxyz` 转回源顺序。

`model_dump(mode="json")` 也加入 `source_rotation_orders`，使统计 cache fingerprint 能感知该语义。

### 修改影响

UMI quaternion 会按正确的 `xyzw -> wxyz -> rotation_6d` 路径处理。这样 rotation 的 6D 表示才是几何正确的，不只是维度正确。

## 8. `QwenPI_v3.py`

### 原有问题

`QwenPI_v3.forward()` 之前从 `self.config.trainer.repeated_diffusion_steps` 读取重复 diffusion 次数：

```python
self.config.trainer.get("repeated_diffusion_steps", 16)
```

但该参数属于 action head / diffusion action model 的行为，UMI YAML 中也放在：

```yaml
framework:
  action_model:
    repeated_diffusion_steps: ...
```

因此旧读取路径会导致 YAML 中的 `framework.action_model.repeated_diffusion_steps` 不生效，或者只能靠 trainer 下的非标准字段兜底。

### 修改内容

读取优先级改为：

1. `config.framework.action_model.repeated_diffusion_steps`
2. `config.trainer.repeated_diffusion_steps`
3. 默认 `4`

并对缺失的 `framework` / `action_model` / `trainer` 做 None-safe fallback。

### 修改影响

UMI YAML 中的 action model 配置现在能真正控制 `repeated_diffusion_steps`。这对 4090 首跑很重要：从较大的重复次数降到 `1` 可以显著降低单步显存和计算开销。

## 9. LeRobot worker 线性内存回收

### 涉及文件

- `starVLA/dataloader/__init__.py`
- `starVLA/dataloader/lerobot_datasets.py`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`

### 原有问题

A800 配置以 4 个 rank、每 rank 16 个 persistent worker 读取 `2704x2028` UMI 视频。失败前训练 rank PSS 基本稳定，worker PSS 合计却从约 155.9 GiB 增长到 227.4 GiB，最终由 500 GiB cgroup OOM killer 杀死 worker。原有每 worker 1000 sample 才执行一次 `gc.collect()` 的逻辑在 step 383 前尚未触发，也不会让 glibc 主动归还已空闲 arena 页。

UMI LeRobot v3 的两个子数据集分别把 242/353 个 episode 放在单个 5.82/7.92 MiB parquet 文件中。旧代码没有在 `get_step_data()` 更新 `curr_traj_id`，每个 sample 都会重新物化整个文件再过滤 episode。它与高分辨率 PyAV/PyTorch decode/resize temporary allocation 共同放大了 worker private anonymous memory。

### 修改内容

`lerobot_datasets.py` 的 collate 增加 opt-in `worker_memory_trim_interval`：

- 只在 DataLoader worker 内计数，`num_workers: 0` 的主进程 collate 不执行。
- 到达 interval 后执行 full `gc.collect()`。
- 当前 libc 提供 `malloc_trim` 时继续执行 `malloc_trim(0)`。
- 默认 interval 为 `0`，未配置的社区 benchmark 不增加 GC pause。

`dataloader/__init__.py` 把 YAML 参数传入 collate；A800 UMI 配置显式使用：

```yaml
worker_memory_trim_interval: 1
```

`datasets.py` 正确保存当前 trajectory id，并为每个 `LeRobotSingleDataset` 增加当前 parquet 文件缓存。切换文件时覆盖旧缓存，不建立随历史文件数量增长的字典；LeRobot v3 过滤 episode 后仍执行 `.copy()`。

### 修改影响与验证

该修改不改变 sampling、transform、normalizer、模型、loss、optimizer 或 scheduler。真实 UMI 数据上已经完成：

- trim interval `0` 与 `1` 的独立单 worker DataLoader 各取 3 个 batch、每 batch 2 个样本；`image/action/state/lang/robot_tag` 逐字段 exact equal。
- 两个 UMI 子数据集的 parquet cache hit 与 forced fresh read 逐字段 exact equal，action/state shape 均为 `[16,10]` / `[1,10]`。
- warm-up 10 个样本后继续 60 个随机样本，worker `Pss_Anon 342.6 -> 342.9 MiB`。
- batch size 16 的四个连续 worker batch，anonymous PSS 为 `346.7, 349.5, 349.7, 349.6 MiB`，第二批后进入平台。

可能变化的是 wall-clock throughput，不保证两次分布式 GPU 训练 checkpoint 字节级一致；数据与训练数值语义不变。修复后的完整 4-rank、64-worker A800 长跑尚未重启，需在正式训练中继续验收 aggregate worker PSS slope 和 cgroup `oom_kill`。

详细根因、nvitop 指标含义和 UMI/LIBERO 数据对比见 `docs/umi_dataloader_memory.md`。行为合同见 `openspec/specs/lerobot-worker-memory/spec.md`。

## 10. LeRobot 视频后端参数透传

### 涉及文件

- `starVLA/dataloader/lerobot_datasets.py`
- `tests/test_lerobot_video_backend_kwargs.py`

### 原有问题

底层 `gr00t_lerobot/video.py` 已支持 `decord.VideoReader(video_path, **video_backend_kwargs)`，`LeRobotSingleDataset` 也保存并向视频读取函数传递该 mapping；但上层 `make_LeRobotSingleDataset()` 只从 YAML 读取 `video_backend`，没有读取或传递 `video_backend_kwargs`。

因此，A800 UMI 即使选择 decord，也只能使用 `num_threads=0` 的自动线程策略。当前 64 个外层 DataLoader worker 中，活跃 worker 实测通常有 20–21 条线程；在约 62 核 cgroup CPU quota 下形成严重超额并发。10 分钟窗口的有效速度为 `7.19 s/step`，比相同 `768 / repeated=1` 模型配置的 `torchvision_av` 基线 `3.53 s/step` 慢约 2.04 倍。

### 修改内容

新增 `_get_video_backend_kwargs()`：

- 从 `datasets.vla_data.video_backend_kwargs` 读取 backend-agnostic 参数。
- 把 OmegaConf/AccessTrackedConfig mapping 规范化为普通 dict。
- 对非 mapping 值抛出明确 `TypeError`，避免配置被静默忽略。
- 字段缺省时返回空 dict，保持修改前的后端默认行为。
- 默认 `LeRobotSingleDataset` 路径和 opt-in `DataConfig.make_dataset` 路径都接收同一 mapping。

UMI A800 YAML 显式设置：

```yaml
video_backend: decord
video_backend_kwargs:
  num_threads: 1
```

该值没有硬编码为全项目默认值；其他数据集和硬件可以根据外层 worker 数及 CPU quota 独立选择。

### 修改影响与验证

修改只改变视频 reader 的线程调度，不修改 dataset sampling、帧索引、transform、normalizer、训练 RNG、模型、loss、optimizer 或 scheduler。已完成：

- 5 个单元测试覆盖 OmegaConf/AccessTrackedConfig 到普通 dict 的转换、默认 dataset、自定义 factory、字段缺省兼容和非法类型拒绝。
- 两个真实 UMI 子数据集各比较帧索引 `0/60/300`；decord `num_threads=0` 与 `1` 的 shape/dtype 相同，像素 exact equal，最大差值为 0。
- YAML 解析确认 `video_backend_kwargs.num_threads == 1`。

修改前已启动的 persistent workers 不会热更新代码或 reader 参数。使用新配置重启后的完整 10 分钟验收已经完成：

- 4 个 rank 各有 16 个 worker，worker `NLWP` 从 20–21 降为 4–5；每 rank worker 总线程为 77–80。
- 有效速度从 decord 自动线程的 `7.19 s/step` 改善到 `3.35 s/step`，缩短 53.5%，并比 `torchvision_av` 的 `3.53 s/step` 快约 5.1%。
- CPU PSI some/full 从 `0.881/0.511` 降到 `0.065/0.065 s/s`。
- 四卡平均 GPU utilization 从自动线程的约 46.5% 恢复到约 70.3%，与 `torchvision_av` 的约 70.7% 基本一致。
- 磁盘 I/O PSI 仍约为零，因此改善来自解除 CPU 线程超额竞争。

剩余每 16 step 的 ordered DataLoader head-of-line straggler 仍会产生最长约 56 秒的等待；正常 model step 已恢复到约 0.63–0.81 秒。这是高分辨率随机 seek batch 的独立长尾，不属于本次线程透传缺陷。

## 11. LeRobot prepared batch 的 ordered sample dispatch

### 涉及文件

- `starVLA/dataloader/ordered_sample_dispatch.py`
- `starVLA/training/train_starvla.py`
- `tests/test_ordered_sample_dispatch.py`

### 原有问题

PyTorch map-style DataLoader 以完整 batch 作为 worker 任务。UMI 的 batch size 为 16，
所以一个 worker 要串行完成 16 次 `2704x2028` H.264 随机 seek。视频最长关键帧间隔
约 4 秒，不同 batch 的 GOP 解码量差异大；ordered delivery 每轮遇到最慢 worker
时形成 20–30 秒队头阻塞。

增加 Decord 内部线程会在 62 核 quota 下恶化抢占；增加到 16 workers 只能推迟尖峰。
真实视频实验也证明 timestamp cache 不是有效方案，因为 `VideoReader` 初始化/GOP
解码远大于 timestamp 数组读取。

### 修改内容

新增 opt-in `ordered_sample_dispatch`：

- 原 DataLoader 先正常传给 `accelerator.prepare()`，保留 DeepSpeed micro-batch 和
  Accelerate per-rank `BatchSamplerShard`。
- 把 prepared batch 展开为带 `batch_id/position/batch_size` 的单 sample token。
- worker 每次只执行一个 `dataset[index]`；主进程严格按 token 恢复原 batch。
- 显式保持 ordered delivery，乱序、跨 batch 或不完整结果立即报错。
- sample prefetch factor 自动乘以训练 batch size；worker trim interval 同样乘以
  batch size，保持修改前的排队 payload 和 sample 级回收频率。
- 启用时要求 `gradient_accumulation_steps: 1`，避免改变 epoch 末尾累计边界。
- 字段默认关闭；只在已验证为 index-deterministic 的 UMI A800 配置启用。

### 修改影响与验证

不修改 MP4/Parquet、dataset index、rank 分片、batch 内容/边界、transform、
normalizer、模型、loss、optimizer、scheduler、effective batch 或 checkpoint 格式。
真实 UMI 16-sample 对照中，`image/action/state/lang/robot_tag` 全字段 exact equal，
图像逐像素一致，主进程 Python/NumPy/Torch RNG 推进一致。

12 项目标/回归单测通过，包含真实 Accelerate-prepared loader 的 batch/RNG
等价检查。四 rank CPU 供数对照把最慢 rank 稳定平均从
`6.49` 降到 `3.67 s/step`，最大长尾从 `24.08` 降到 `7.03 s`。12-worker、
48-batch soak 的 aggregate worker `Pss_Anon` 在 step 16–48 保持约 6 GiB 平台，
没有重新出现线性增长。

当前结论只覆盖完整、健康且 transform 确定的 UMI 数据。异常文件触发随机 retry、
或社区数据集使用 worker-local 随机 augmentation 时，需要独立等价验证后才能启用。
下一次 GPU/DDP 正式启动仍需补充有效 step、GPU 空转和 1000-step PSS slope 验收。

## 验证结果

已完成以下验证：

```bash
bash -n examples/Z-umi/train_files/run_umi_train.sh
python -m py_compile starVLA/training/train_starvla.py starVLA/dataloader/__init__.py starVLA/model/modules/vlm/QWen3.py
openspec validate --changes harden-umi-2b-yaml-owned-pipeline
SMOKE_TEST=1 bash examples/Z-umi/train_files/run_umi_train.sh
```

`SMOKE_TEST=1` 已真实完成 2 个训练 step，并保存最终模型到：

```text
playground/Checkpoints/umi_qwenpi_v3_4090_smoke/final_model
```

日志确认：

- 加载的是 `Qwen3-VL-2B-Instruct`
- UMI action/state 维度为 10
- `qwen_vl_interface` 被冻结
- smoke YAML 的 `gradient_accumulation_steps: 1` 实际生效
- DeepSpeed 配置中的 `gradient_accumulation_steps` 由 `auto` 解析为运行时值

UMI action inverse round-trip 额外验证结果：

```text
normalized action shape:        (16, 10)
decoded action key shapes:      (16, 3), (16, 4), (16, 1)
position max absolute error:    5.55e-17
rotation matrix max abs error:  4.44e-16
gripper max absolute error:     6.94e-18
```

使用修复后的 inverse 路径完成 episode 0/frame 50、四个 checkpoint、三个固定随机种子的端到端开环评测：

| checkpoint | normalized RMSE | position ADE | rotation mean | gripper MAE |
|------------|-----------------|--------------|---------------|-------------|
| 5000 | 0.067324 | 0.030654 | 6.442 deg | 0.004509 |
| 10000 | 0.053200 | 0.018819 | 3.969 deg | 0.004710 |
| 20000 | 0.028654 | 0.013500 | 2.228 deg | 0.002017 |
| 30000 | 0.017247 | 0.007836 | 2.236 deg | 0.000783 |

最终 checkpoint 的 normalized RMSE 相比 step 5000 改善 75.00%。评测通过 shape、有限数值和 inverse 检查，结果分类为训练样本上的正向拟合证据；未评估未见 episode 泛化。完整产物位于 `playground/Evals/umi_open_loop_ep0_f50/`。

后续新增的 `umi-test` 整段 episode 回放只增加 `examples/Z-umi/eval_files/replay_umi_test_episode.py` 和配套文档，没有产生新的非 `examples/Z-umi/` 核心代码修改。episode 0 的 370-anchor 真实回放已完成，产物位于 `playground/Evals/umi_test_episode_000_step30000/`。

**2026-07-21 eval 脚本合并 (Z-umi 内部重构)**: 把 `replay_umi_test_episode.py` 重命名为 `eval_open_loop.py` 作为唯一开环评测入口, 删除原单样本拟合脚本 `eval_open_loop.py` (旧版), 共享 utilities 抽到 `examples/Z-umi/eval_files/eval_utils.py`。同步新增 launcher `examples/Z-umi/eval_files/run_umi_open_eval.sh`。本次重构未产生新的非 `examples/Z-umi/` 代码修改; `docs/umi_training_plan.md` 已同步更新评测章节。

**2026-07-21 两段式 eval 框架 + Rerun 可视化 (新增 starVLA/eval 包)**: 把 eval 拆成 Stage 1 (推理, GPU) + Stage 2 (可视化, CPU) 两阶段。新增 `starVLA/eval/` 顶层包 (位于 `examples/Z-umi/` 之外, 即本文件登记范围):

- `starVLA/eval/core/io.py`: schema-stable 的 `predictions.npz` + `metrics.json` 读写, 都带 `__schema_version`, v1。
- `starVLA/eval/core/metrics.py`: `calculate_episode_metrics` + `SOURCE_LABELS`, 可被训练侧 validation 复用。
- `starVLA/eval/core/transforms.py`: `rotation_error_degrees` / `restore_absolute_action` / `align_quaternion_sign` 等共享数学工具。
- `starVLA/eval/viz/rerun_logger.py`: `log_episode_to_rerun`, 把 results/metrics/metadata 流式 log 到 `.rrd` 文件, 用于离线 Rerun viewer 检阅。

`examples/Z-umi/eval_files/` 同步新增 `stage2_visualize.py` (读 Stage 1 产物 → 写 `.rrd`), 移除 `eval_open_loop.py` 内的 matplotlib 绘图代码 (Stage 1 不再产 PNG plot), `run_umi_open_eval.sh` 加 `--visualize` / `--stage2-only` 两个 launcher-level 开关。Stage 1 产物从 `{metrics.json, predictions.npz, episode_*_action_fit.png, first_input.png}` 改为 `{metrics.json, predictions.npz, first_input.png}` (PNG plot 删除, 由 Stage 2 的 `.rrd` 替代)。

依赖: `rerun-sdk==0.34.1` + `numpy==2.2.6` (starVLA env, 2026-07-21 升级)。

**升级副作用 (已知、用户确认接受)**:
- `lerobot 0.3.3` 要求 `rerun-sdk<0.23.0,>=0.21.0`, 现状 0.34.1 违反该约束。lerobot 当前不用于 UMI 训练/评测, 影响范围有限, 需要时单独升 lerobot 或单独 env。
- `numba 0.59.1` 要求 `numpy<1.27,>=1.22`, 现状 2.2.6 违反该约束。numba 在主训练/评测路径上未直接调用, 但若有依赖它的子模块需重新验证。
- rerun 0.34 API breaking changes 已适配: `rr.set_time_sequence(name, value)` → `rr.set_time(name, sequence=value)`, `rr.Scalar(value)` → `rr.Scalars([value])`。改动覆盖 `starVLA/eval/viz/rerun_logger.py` 和 `examples/Z-umi/eval_files/stage2_visualize.py`。

`docs/umi_training_plan.md` 已同步更新评测章节说明两段式工作流。

## 12. 多卡开环评测入口 (eval_starvla.py)

### 涉及文件

- `starVLA/training/eval_starvla.py` (NEW)
- `starVLA/eval/core/checkpoint_utils.py` (NEW)
- `starVLA/eval/core/io.py` (新增 `json_ready` 公开别名)
- `examples/Z-umi/eval_files/eval_open_loop.py` (DELETE)
- `examples/Z-umi/eval_files/eval_utils.py` (DELETE)
- `examples/Z-umi/eval_files/replay_umi_train_episode.py` (import 路径更新)
- `examples/Z-umi/eval_files/run_umi_open_eval.sh` (改用新入口 + `NUM_PROCESSES` 多卡)

### 原有问题

旧 UMI 开环评测入口 `eval_open_loop.py` 是纯单卡实现:
`baseframework.from_pretrained(...).to(device)` 绕开 Accelerator, DataLoader 用
`Subset` + `num_workers=0` 没走 DistributedSampler, 多卡场景结构上无法扩展。
Helpers (`checkpoint_path` / `pick_mixture_entry` 等) 散落在 `examples/Z-umi/eval_files/`,
无法被 starVLA 内部模块复用。

### 修改内容

新增 `starVLA/training/eval_starvla.py`, 结构跟 `train_starvla.py` 1:1 对称:

- `VLAEvaluator(TrainerUtils)` 镜像 `VLATrainer`, 去掉 optimizer/scheduler/wandb/resume
- `setup_eval_accelerator()` 不带 DeepSpeed plugin (eval 无 backward, DeepSpeed 反而
  触发 `train_micro_batch_size_per_gpu` 缺失错误, 详见加速器构造处注释)
- 复用 `baseframework.from_pretrained` / `make_LeRobotSingleDataset` / `collate_fn` /
  `starVLA.eval.core.{io,transforms}` / `TrainerUtils` 静态工具
- 多卡机制: 每 episode 一个 `_IndexedSubset` (注入 traj_id/anchor 到 sample dict,
  因为 `_pack_sample` 不返回索引字段, 多卡下 DistributedSampler 不保证顺序) →
  `accelerator.prepare(loader)` 注入 DistributedSampler → `predict_action` 推理 →
  `dist.all_gather_object` 跨 rank 拼接 → rank 0 按 anchor 排序写盘

`starVLA/eval/core/checkpoint_utils.py` 接收 5 个从 `eval_utils.py` 迁过来的 helper
(`torch_dtype` / `to_numpy` / `discover_checkpoint_steps` / `checkpoint_path` /
`pick_mixture_entry`), 让 starVLA 内部代码不再依赖 `examples/Z-umi/`。

`starVLA/eval/core/io.py` 把私有 `_json_ready` 暴露为公开 `json_ready` 别名, 供
`replay_umi_train_episode.py` 复用同一 JSON 序列化 helper, 不再重复实现。

`run_umi_open_eval.sh` 改用 `python -m starVLA.training.eval_starvla`, 新增
`NUM_PROCESSES` 环境变量切换 `accelerate launch --multi-gpu` 多卡路径; `--visualize`
路径适配新的 per-episode 子目录布局 (`episode_*/episode.rrd`)。

### 影响与验证

- 评测 schema v4 不变, Stage 2 (`stage2_visualize.py`) 无需修改
- 多卡下 `set_seed` 用全局 seed 不加 rank 偏移, 保证不同 world_size 同一 anchor
  拿到同一 flow-matching 噪声, 评测可复现
- 单卡 smoke test (8 anchor, overfit_ep0 ckpt 10000): schema v4 OK, 5 个必填 key 齐,
  shape 跟旧 `eval_open_loop.py` 完全一致; 同 seed 重跑 predictions 逐元素相等
  (latency 允许差异)
- Stage 2 schema 兼容性: 新输出跑 `stage2_visualize.py` 成功生成 `.rrd`, 无版本错误
- 多卡确定性 (1 vs 2 GPU) 需在多 GPU 机器上由用户验证, 本机单 4090 无法测试

不动 train 侧任何文件 (`train_starvla.py` / `trainer_tools.py` / `base_framework.py` /
`dataloader/`)。`eval_open_loop.py` 的所有功能性代码 (config merge / dataset build /
episode range / action invert / per-sample decode) 都迁移或重写进了新文件。

## 审核重点

建议重点审核以下问题：

1. `train_starvla.py` 中 scheduler/log/eval/save 是否只应在 `sync_gradients` 时触发。
2. `optimizer.step()` 在 Accelerate accumulate 场景下是否符合当前 Accelerate/DeepSpeed 版本的行为预期。
3. `QWen3.py` 删除强制 `sdpa` 后，是否需要在无 FlashAttention 的环境中继续接受自动 fallback。
4. 全局 `ds_config.yaml` 改为 `"auto"` 是否会影响其它训练脚本；当前目标是让运行时配置成为梯度累计的来源。
