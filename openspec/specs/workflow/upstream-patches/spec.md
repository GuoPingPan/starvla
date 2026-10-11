# 上游补丁登记表（Upstream Patches Registry）

## Purpose

规定对上游 `starVLA_dev` 既有文件的所有修改如何登记、标记与隔离，把每次上游同步的冲突面降到最小，并让 `scripts/dev/check_patches.py` 能自动发现未登记的上游改动。完整叙述见 `docs/zerith_docs/workflow/upstream_patches.md`。

## Requirements

### Requirement: M 与 A 的分类

系统 SHALL 维护两类改动：**M（Modified）**= 修改上游既有文件，上游同步时会冲突，**必须登记并加标记区块**；**A（Added）**= 新建文件，上游永远不会动，**零冲突风险，不登记**。

#### Scenario: 新增文件不登记
- **WHEN** 开发者新增一个独立文件（例如新 transform）
- **THEN** 该文件不进入补丁登记表，上游同步时不会产生冲突

#### Scenario: 修改上游文件必须登记
- **WHEN** 开发者修改了上游既有文件
- **THEN** 该文件必须作为一条 Patch 记录补登进登记表，否则 `check_patches.py` 会报出未登记

### Requirement: 登记表条目完整性

登记表的每个 M 文件 SHALL 包含：Patch ID、文件路径、原因、类型（增强/接入/配置）、规模（行数）、负责人；登记表顶部维护「最近核对的上游 SHA」。

#### Scenario: 查表核对冲突面
- **WHEN** 从 `github/starVLA_dev` 同步到 `dev/main` 后发生冲突
- **THEN** 冲突文件必须能在登记表中找到；若找不到说明有漏登记，需先补登记

### Requirement: 标记区块约定

对上游文件的每一处改动 SHALL 尽量收敛在可见的 `ZERITH PATCH <ID>` 标记区块内，便于 review 与 merge 时定位；标记里的 ID 对应登记表的 Patch ID；新代码务必先加标记再提交。

#### Scenario: 历史未加标记的改动
- **WHEN** 历史已有改动未加标记
- **THEN** 不强制回填，但后续改动与解决上游冲突时应顺手补上

### Requirement: 增量优先的扩展点

新增能力 SHALL 优先用「新增文件 + 注册」的增量方式，而非改上游既有文件：新框架在 `model/framework/` 新建并用 `@FRAMEWORK_REGISTRY.register()` 注册；新数据集注册到 `data_registry/`；新 transform 用独立文件注册；新 benchmark 在 `examples/<bench>/` 新建目录。

#### Scenario: UMI transform 增量范例
- **WHEN** UMI 引入 5 个新 transform（normalization/pose_rotation/quaternion_to_6d/relative_pose/rotation_utils）
- **THEN** 全部以新文件落地，不改上游 transform 文件本身，属于零冲突增量

### Requirement: 提交前自检

开发者改了上游文件后，提交/发 MR 前 MUST 运行 `python scripts/dev/check_patches.py`，确认本次改动的上游文件都已登记。

#### Scenario: 漏登记被发现
- **WHEN** 开发者改了上游文件但未登记
- **THEN** `check_patches.py` 报出未登记的上游文件，提醒补登记

### Requirement: 已增量化补丁的回退

当某项历史修改可被纯新增文件替代时，SHALL 回退上游文件改动（恢复上游原状）并把替代实现落到新文件；已增量化条目从 M 列表移除，记录在「已增量化」表中。

#### Scenario: data_config 示例外移
- **WHEN** `data_config.py` 末尾的 `Relative6DPoseDataConfig` 示例是死代码
- **THEN** 回退上游 `data_config.py`，把 opt-in 示例放到新文件 `zerith_example_relative_6d.py`（import 即注册）

## References

- 完整登记表与 M 文件清单：`docs/zerith_docs/workflow/upstream_patches.md`
- 同步流程：`docs/zerith_docs/workflow/upstream_sync_and_isolation.md`
- 自检工具：`scripts/dev/check_patches.py`
