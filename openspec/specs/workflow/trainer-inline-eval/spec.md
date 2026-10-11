# 训练循环内在线 Eval（Trainer Inline Eval）

## Purpose

约束 `train_starvla.py::VLATrainer.eval_action_model` 这类「训练循环内的在线 action eval」。
该 eval 是轻量 sanity check：从 eval 划分取 1 个 batch，跑 `predict_action` 算 normalized
空间 MSE，结果写 SwanLab 的 `eval/mse` 曲线。它不替代离线开环评测
（`eval_starvla.py`），只用于训练过程中观察泛化趋势与发散预警。

本 spec 把历次改动中踩过的坑固化为强制约束，防止回归。

## Requirements

### Requirement: eval_interval 须为 logging_frequency 的整数倍

在线 eval 算出的 `eval/mse` 经 `VLATrainer._log_metrics` 写 tracker，而 `_log_metrics`
受 `logging_frequency` 门控（仅 `completed_steps % logging_frequency == 0` 时写）。
若 `eval_interval` 不是 `logging_frequency` 的整数倍，eval 会在某些 step 算出指标却
不进 tracker，导致 SwanLab 曲线缺点、不可解释。

- **WHEN** 配置 `eval_interval`
- **THEN** 该值 MUST 是 `logging_frequency` 的正整数倍；禁用 eval 时统一设 `100000000`

#### Scenario: 配置 yaml 须显式标注门控

- **WHEN** 任何训练 yaml 设置 `eval_interval`
- **THEN** 该字段附近 MUST 带注释说明「须为 `logging_frequency` 整数倍，否则
  `_log_metrics` 门控会让 eval 算了却不写 tracker」

### Requirement: eval 必须 RNG 隔离

`predict_action` 内部的 flow-matching 噪声（`sample_action_noise`，PI0.py）会消费
GPU RNG。不隔离会扰动后续训练的 dropout / 数据增强随机流，破坏可复现性。

#### Scenario: eval 前后保存恢复 RNG

- **WHEN** `eval_action_model` 执行
- **THEN** 进入 eval 前 MUST 快照 CPU + 所有 CUDA device 的 RNG state；无论 eval
  正常返回还是抛异常，都 MUST 在 `finally` 中恢复原 state

### Requirement: eval 必须 NaN 安全

bf16 训练可能发散（见 `docs/pi05_training_nan_debug.md`），权重变 NaN 后 eval 的
MSE 也是 NaN。`_SwanLabTracker.log` 的 `try/except` 遇 nan 会 disable **整个**
tracker（`logger.py`），一次发散 eval 会把训练日志全断。

#### Scenario: NaN 时不写 key

- **WHEN** eval 算出的 MSE 为 NaN
- **THEN** MUST 不把该 key 写入 metrics（而非写入 nan 值）；SwanLab 上该 step 的
  `eval/mse` 点缺失即视为发散早期预警——比 `grad_norm` 可信（后者被 DeepSpeed
  剪枝掩盖成 1.0）

### Requirement: predict_action 须在 autocast 下调用且参数须匹配目标 framework

不同 framework 的 `predict_action` 识别不同 kwargs。PI0 遗留的 `use_ddim` /
`num_ddim_steps` 对 PI05 无效（PI05 只认 `num_steps`，见 `PI0.py::predict_action`），
传无效参数虽不报错但被静默忽略，掩盖实际推理步数。

更关键：`predict_action` **必须**在 `torch.autocast("cuda", bfloat16)` 下调用，与训练
forward（`_train_step` 的 autocast）保持同一精度上下文。训练环境里模型权重 dtype 状态
（DeepSpeed bf16 包装 + `to_precision`）与纯推理不同，不加 autocast 会在 `sample_actions →
denoise_step → gemma expert` 的 linear 上撞 `BFloat16 != float` dtype mismatch。历史上 UMI
yaml 把 `eval_interval` 设成 `100000000` 禁用了训练内 eval，导致 `predict_action` **从未在
训练循环里被调用过**，这个 dtype 坑一直潜伏，直到启用 eval 才暴露。

#### Scenario: 不传遗留无效参数 + autocast 包裹

- **WHEN** 训练内 eval 调用 `predict_action`
- **THEN** MUST 在 `torch.autocast("cuda", bfloat16)` 上下文里调用；且 MUST 不传目标
  framework 不识别的遗留参数；推理步数由模型自身的 `num_inference_steps`（config 层）决定

#### Scenario: predict_action 副作用必须清理, 避免污染后续训练

- **WHEN** 训练内 eval 调用 `predict_action`
- **THEN** eval 后 MUST 恢复 `predict_action` 修改的状态并释放残留显存：
  （1）恢复 `vlm.language_model.config._attn_implementation`（`sample_actions` PI0.py 会改成
  `"eager"` 且不恢复）；（2）`torch.cuda.empty_cache()` 释放 `forward_prefix(use_cache=True)`
  的 KV cache 等中间态残留。predict_action 的副作用若不清理，叠加训练常驻会在后续 backward
  顶穿显存（启用 eval 前 `eval_interval=1e8` 从不调用 predict_action，此问题未暴露）。
  eval 用与训练相同的 batchsize 即可——predict_action 是 `@torch.inference_mode()`，峰值低于
  train forward+backward

### Requirement: 改动须在能跑通训练的环境做过载 smoke

改 trainer eval 逻辑属 trainer 改动，按 `workflow/testing-layers` 须跑 overfit
smoke。且 smoke MUST 实际触发 eval 路径才有意义。

#### Scenario: smoke 步数须覆盖 eval

- **WHEN** 为 eval 改动跑 overfit smoke
- **THEN** smoke 的 `max_train_steps` MUST 大于 `eval_interval`，或经 CLI 覆盖
  `eval_interval` 到小值（如 50），确保 eval 在 smoke 内至少触发一次；并从日志
  确认 `eval/mse` 被算出且未引入异常

#### Scenario: 单卡 ZeRO-2 对大模型会 OOM

- **WHEN** 在单卡跑 PI05（4.14B）等大模型 smoke
- **THEN** 须注意 ZeRO-2 在 `world_size=1` 时不分片，优化器状态全压单卡会 OOM
  （见 `docs/pi05_training_nan_debug.md` 附录）；smoke 应在多卡环境，或开启
  optimizer CPU offload，或用更小 framework 验证 eval 逻辑

## References

- 实现：`starVLA/training/train_starvla.py::VLATrainer.eval_action_model`
- tracker 门控：`starVLA/training/train_starvla.py::_log_metrics`
- NaN 根因：`docs/pi05_training_nan_debug.md`
- 测试分层：`openspec/specs/workflow/testing-layers/spec.md`
