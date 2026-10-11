# 上游补丁登记表（Upstream Patches Registry）

> 本文件登记**对上游 `starVLA_dev` 既有文件的所有修改**（即上游同步时会产生冲突的文件）。
> 纯新增文件（additive）不在此表，列在文末"纯新增（低风险）"区，仅备查。
>
> **用途**：每次从 `github/starVLA_dev` 同步到 `dev/main` 时，逐行核对这张表 —— 冲突只会出现在这里的文件。
> 配套工具：`scripts/dev/check_patches.py`（自动检查本次改动里是否有未登记的上游文件）。

**最近核对的上游 SHA**：`3974599`（= 当前 `dev/main` 所含上游点，2026-08 核对）

---

## 关键概念：M vs A

- **M（Modified）**：修改了上游已有的文件 → 上游同步时**会冲突** → **必须登记 + 加标记区块**。
- **A（Added）**：新建文件 → 上游永远不会动 → **零冲突风险** → 不登记，是推荐的增量方式。

> 现状亮点：UMI 的 transform 全部以**新文件**方式落地（`normalization.py` / `pose_rotation.py` / `quaternion_to_6d.py` / `relative_pose.py` / `rotation_utils.py`），没有改上游 transform 文件本身 —— 这是"增量优先"的好范例。

---

## 修改的上游文件（M = 同步冲突面，当前 17 个）

> 类型说明：`增强` = 在原能力上扩展；`接入` = 为新功能加挂载点；`配置` = 参数/依赖变更。

### 数据管线（dataloader）

| Patch ID | 文件 | 原因 | 类型 | 规模 | 负责人 |
|----------|------|------|------|------|--------|
| P001 | `starVLA/dataloader/gr00t_lerobot/datasets.py` | UMI LeRobot v3 数据布局（多 episode/parquet、视频按 ~200MiB 聚合）、action mode、normalizer 统计复用 | 增强 | ~528 | orion |
| P003 | `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` | action mode（abs/rel/delta）、relative pose、gripper、chunk 首帧相对化 | 增强 | ~474 | orion |
| P004 | `starVLA/dataloader/gr00t_lerobot/video.py` | decord 后端、`num_threads` 显式声明、视频解码并发治理 | 增强 | ~97 | orion |
| P006 | `starVLA/dataloader/gr00t_lerobot/schema.py` | schema 扩展（pose/gripper 列） | 增强 | ~6 | orion |
| P007 | `starVLA/dataloader/__init__.py` | 暴露新 dataloader 接口 | 接入 | ~7 | orion |
| P008 | `starVLA/dataloader/lerobot_datasets.py` | 入口适配新数据/统计语义 | 增强 | ~9 | orion |
| P019 | `starVLA/dataloader/gr00t_lerobot/transform/__init__.py` | re-export `VideoCenterCrop`（配合 video.py 的 CenterCrop pipeline，修 commit 96c987c 遗漏的 `__init__` 导出） | 接入 | ~3 | orion |

### 模型框架（model）

| Patch ID | 文件 | 原因 | 类型 | 规模 | 负责人 |
|----------|------|------|------|------|--------|
| P009 | `starVLA/model/framework/VLM4A/PI0.py` | action mode 集成、rotation_6d、state anchor | 增强 | ~463 | orion |
| P010 | `starVLA/model/framework/VLM4A/PI05.py` | 同 PI0（继承） | 增强 | ~77 | orion |
| P011 | `starVLA/model/modules/action_model/OpenPI_ActionHead.py` | action head 适配 action mode/6d | 增强 | ~15 | orion |
| P012 | `starVLA/model/framework/share_tools.py` | 共享工具调整 | 增强 | ~20 | orion |

### 训练 / 评测（training）

| Patch ID | 文件 | 原因 | 类型 | 规模 | 负责人 |
|----------|------|------|------|------|--------|
| P013 | `starVLA/training/train_starvla.py` | 训练入口：评测集成、配置、UMI 支持、overfit 模式 | 增强 | ~506 | orion |
| P014 | `starVLA/training/train_starvla_cotrain.py` | 协同训练适配 | 增强 | ~97 | orion |
| P015 | `starVLA/training/train_starvlm.py` | VLM 训练适配 | 增强 | ~53 | orion |
| P016 | `starVLA/training/trainer_utils/trainer_tools.py` | trainer：overfit 模式、评测、stats、logger | 增强 | ~331 | orion |
| P017 | `starVLA/training/trainer_utils/overwatch.py` | 分布式 watch 调整 | 增强 | ~26 | orion |

### 配置 / 依赖

| Patch ID | 文件 | 原因 | 类型 | 规模 | 负责人 |
|----------|------|------|------|------|--------|
| P018 | `pyproject.toml` | uv 化、boto3 extra、依赖 | 配置 | ~85 | orion |

> 当前 **16 个** M 文件（原 21 个；P002/P005/P019/P020/P021 已增量化，见下）。其中 `datasets.py`、`train_starvla.py`、`state_action.py`、`PI0.py`、`trainer_tools.py` 是改动最重的 5 个，是每次同步时最需要重点人工 review 的文件。

### 已增量化（本次重构，不再算补丁）

| 原 Patch | 原文件 | 处理 | 替代（新增文件） |
|----------|--------|------|------------------|
| P002 | `data_config.py`（末尾追加的 `Relative6DPoseDataConfig` 示例） | 回退上游（全仓库无 robot_type 使用 `relative_6d_example`，死示例） | `starVLA/dataloader/gr00t_lerobot/zerith_example_relative_6d.py`（opt-in 示例，import 即注册） |
| P005 | `transform/__init__.py`（新 transform 重导出） | 回退上游（无消费者经包导入；都用直接子模块导入） | 测试改直导：`transform.quaternion_to_6d import QuaternionTo6DRotation` |
| P019 | `requirements.txt` (+sentencepiece) | 回退上游（sentencepiece 已在 pyproject，冗余） | — |
| P020 | `deepspeed_zero3.yaml` | 回退上游（zerith 不用 zero3，且原改动指向不存在的 `ds_config_zero3.yaml`） | — |
| P021 | `ds_config.yaml` (grad_accum→auto) | 回退上游；zerith 走独立配置 | `zerith_ds_config.yaml` + `zerith_deepspeed_zero2.yaml`，zerith_pi05 launcher 已改指 |

---

## 标记区块约定（Marker Convention）

对上游文件的每一处改动，**尽量**收敛在可见标记内，便于 review 与 merge 时定位：

```python
# --- ZERITH PATCH P003: relative action mode (见 upstream_patches.md) ---
...我们的修改...
# --- END ZERITH PATCH P003 ---
```

- 标记里的 `P003` 对应上表 Patch ID。
- **新代码务必先加标记再提交**（`scripts/dev/check_patches.py` 会提醒未登记的文件，但不会强制加标记——靠自觉与 review）。
- 历史已有改动若未加标记，不强制回填，但**后续改动**与**解决上游冲突时**应顺手补上。

---

## 上游同步时如何用这张表

1. `git fetch github && git checkout dev/main && git merge github/starVLA_dev`
2. 若冲突：`git status` 看冲突文件 → 对照本表确认它**应该**在这里（若不在，说明有漏登记，先补登记）。
3. 逐个解决：保留上游未冲突部分 + 保留我们的 `ZERITH PATCH` 区块；如上游在同一区域也改了，按区块语义手动合并。
4. 解决后更新本表"最近核对的上游 SHA"为本轮上游 HEAD。
5. 跑测试分层（见 `testing_layers.md`）验证未破坏 pipeline。

---

## 纯新增文件（A = 低风险增量，仅备查）

以下为 `dev/main` 相对上游**新增**的文件，上游不会动它们，同步零冲突。无需加标记。

**新 transform**（UMI 旋转/归一化，增量范例）：
- `dataloader/gr00t_lerobot/transform/normalization.py`
- `dataloader/gr00t_lerobot/transform/pose_rotation.py`
- `dataloader/gr00t_lerobot/transform/quaternion_to_6d.py`
- `dataloader/gr00t_lerobot/transform/relative_pose.py`
- `dataloader/gr00t_lerobot/transform/rotation_utils.py`

**zerith 示例 DataConfig**（从上游 data_config.py 外移，opt-in）：
- `dataloader/gr00t_lerobot/zerith_example_relative_6d.py`

**DeepSpeed 配置**（zerith 专用，不改上游 `ds_config.yaml` / `deepspeed_zero2.yaml`）：
- `config/deepseeds/zerith_ds_config.yaml`（grad_accum=auto）
- `config/deepseeds/zerith_deepspeed_zero2.yaml`（accelerate 入口，指向上者）

**评测工具**（全新 `eval/` 模块）：
- `eval/core/`（`checkpoint_utils.py`、`io.py`、`metrics.py`、`transforms.py`、`__init__.py`）
- `eval/viz/rerun_logger.py`、`eval/__init__.py`、`eval/viz/__init__.py`

**部署**：
- `deploy/robodojo_pi05/`（`model.py`、`__init__.py`）、`deploy/__init__.py`

**训练/评测入口新增**：
- `training/eval_starvla.py`、`training/trainer_utils/logger.py`、`training/trainer_utils/perf_analyzer.py`

**CI / 运维脚本**（`scripts/`，不属上游）：
- `scripts/train.sh`、`scripts/bos_ckpt.sh`、`scripts/bos_io.py`、`scripts/download_data.py`、`scripts/ddl_to_data_config.py`、`.gitlab-ci.yml`

**测试**：
- `tests/test_quaternion_to_6d_transform.py`、`tests/test_rerun_logger_dims.py`
