# UMI 训练计划（UMI LeRobot → starVLA）

## Purpose

UMI LeRobot 数据集到 starVLA（QwenPI_v3 / PI05）训练链路的**单一真实来源**：动作表示、数据流、normalizer 语义、维度合同与验证步骤。完整叙述见 `docs/zerith_docs/umi/umi_training_plan.md`。

## Requirements

### Requirement: action 列由 observation 同帧复制

原始 action 列离线生成时，SHALL 直接复制同帧 observation（pose 与 gripper），**不做时移、不做减法**；相对化必须放到 dataloader 的 chunk 读取阶段，因为 base 是每个 sampled chunk 的首帧。

#### Scenario: 离线预处理
- **WHEN** 运行 `examples/Z-umi/preprocess_umi_data.py`
- **THEN** `action.end_0_pose[t] = observation.state.end_0_pose[t]`，`action.effector_0_joint_state[t] = observation.state.effector_0_joint_state[t]`

### Requirement: chunk 首帧相对化

dataloader 按 action chunk 读取后，SHALL 对 **position 和 gripper** 做 chunk 首帧相对化：`position_rel[t] = action.position[t] - state.position[0]`，`gripper_rel[t] = action.gripper[t] - state.gripper[0]`；**rotation 不参与相对减法**（四元数减法无几何意义），保持绝对。

#### Scenario: rotation 保持绝对
- **WHEN** 对 chunk 做 relative 化
- **THEN** position/gripper 相对化，rotation 维持绝对值

### Requirement: normalizer 用训练集固定统计

normalizer 的统计对象 MUST 是**所有训练样本的 relative action chunk**，在所有训练 chunk 和所有 horizon step 上汇总**固定** min/max，**不做每个 chunk 自己的动态 per-chunk min/max**；理由：推理时没有真实未来 chunk 的 min/max，无法反归一化。

#### Scenario: 拒绝 per-chunk 动态归一化
- **WHEN** 有人提议按每个 chunk 算 min/max
- **THEN** 不采用；正确做法是先构造 relative chunk，再在全局训练集上汇总固定 min/max

### Requirement: gripper 连续 min_max

gripper 是连续量，SHALL 使用 `min_max` 归一化，**不使用 `binary`**。

#### Scenario: gripper 不二值化
- **WHEN** 配置 gripper 归一化
- **THEN** 走连续 `min_max`，不出现 binary 离散化

### Requirement: rotation 表示转换链

源数据四元数为 **`xyzw`** 顺序，SHALL 在进入 PyTorch3D 前转为 **`wxyz`**，再转 **`rotation_6d`**；rotation_6d 使用旋转矩阵前两行，固定 `min=-1, max=1` 的 min-max 公式严格为恒等映射。

#### Scenario: xyzw 直接当 wxyz 是 bug
- **WHEN** 把 `xyzw` 直接当成 `wxyz` 输入 rotation converter
- **THEN** 转换出的 rotation_6d 数值错误（但 shape 看起来合法），属于难发现的数据语义 bug，必须走 `xyzw → wxyz → rotation_6d` 路径

### Requirement: 10D 维度合同

transform 后的 state/action 维度合同 MUST 为 `position(3) + rotation_6d(6) + gripper(1) = 10`；模型 sample 的 `action` shape 为 `[16, 10]`，`state` shape 为 `[1, 10]`。

#### Scenario: 验证 dataloader 输出
- **WHEN** 跑 dataloader smoke
- **THEN** 检查 `action.shape == (16, 10)`、`state.shape == (1, 10)`、position/gripper 是 chunk 首帧相对值、gripper 连续 min_max、rotation 绝对 rotation_6d

### Requirement: include_state 必须开启

`include_state: true` MUST 开启，否则 QwenPI_v3 无法把离散化 state token 注入 instruction。

#### Scenario: 关闭 include_state 致错
- **WHEN** `include_state` 被关闭
- **THEN** QwenPI_v3 无法注入离散化 state token，破坏训练语义

### Requirement: normalizer cache 指纹

`stats_gr00t.json` cache MUST 包含 action mode、apply keys、state map、indices 和 modality 指纹；语义（相对化/modality/旋转顺序）变化时自动重算，避免复用过期统计。

#### Scenario: 改了 action mode 后复用旧 cache
- **WHEN** 相对化或 modality 语义变化
- **THEN** cache fingerprint 不匹配 → 自动重算统计，不复用旧 `stats_gr00t.json`

### Requirement: 固定单 episode 过拟合诊断语义

`overfit_episode_id` SHALL 一次性读取目标 episode 全部锚点并复用内存缓存；`overfit_anchor_index` 进一步只缓存单个训练样本；单 episode 只限制优化样本，**不改变 normalizer**，归一化和导出的 `dataset_statistics.json` 仍来自完整训练集统计；单 action chunk 模式禁止尾部填充，越界锚点直接报错。

#### Scenario: overfit 缓存复用
- **WHEN** episode 0（317 帧）缓存完成
- **THEN** 后续所有 epoch 直接复用内存缓存，不再重读视频或执行 dataloader transforms，resized cache 约 46 MiB

### Requirement: A800 worker 内存与解码并发治理

A800 高分辨率（2704x2028）workload SHALL 使用 `worker_memory_trim_interval`（每 N sample 做 `gc.collect()` + `malloc_trim(0)`）、单线程 decord（`video_backend_kwargs.num_threads: 1`）和 `ordered_sample_dispatch`；这些参数默认关闭，未配置的社区 benchmark 不执行额外行为，且不改变采样/transform/normalizer/训练 RNG。

#### Scenario: 解码线程超额
- **WHEN** 64 个外层 worker 各开 decord 自动线程（每 worker 20-21 条）
- **THEN** 在约 62 核 cgroup 下形成近千条线程竞争，有效速度退化到约 7.19 s/step；显式 `num_threads: 1` 后降到约 3.35 s/step

### Requirement: 文档同步要求

任何影响 Z-umi 数据语义、normalizer、训练配置的修改，MUST 同步更新 `umi_training_plan.md`、`agent.md` 与 `umi_non_zumi_code_changes.md`。

#### Scenario: 改了 normalizer 语义
- **WHEN** 修改影响数据语义或 normalizer
- **THEN** 三份文档同步更新，避免单一真实来源失同步

## References

- 完整训练计划：`docs/zerith_docs/umi/umi_training_plan.md`
- 非 Z-umi 代码改动：`docs/zerith_docs/umi/umi_non_zumi_code_changes.md`
- worker 内存合同：`openspec/specs/lerobot-worker-memory/spec.md`
- 视频解码并发合同：`openspec/specs/lerobot-video-decode-concurrency/spec.md`
