# 上游补丁瘦身方案（Patch Reduction Plan）

> 目标：把对上游 `starVLA_dev` 既有文件的修改（M）尽量转成**新增文件**（A），让每次上游同步的冲突面最小，且**保证兼容性**（zerith 行为不变、上游文件回到上游原生态）。
>
> 进展：M 文件已从 **21 → 16**（P002/P005/P019/P020/P021 本轮已增量化，见 [`upstream_patches.md`](upstream_patches.md)）。
> 本文件给出剩余 16 个的定性分析与分批处置建议。

---

## 0. 处置策略分级

| 策略 | 含义 | 风险 | 适用 |
|------|------|------|------|
| **A. 增量化** | 新建 zerith_ 文件 / 子类 / 包装，回退上游文件 | 低~中 | 配置、可注册的新能力、纯追加代码 |
| **B. 贡献上游** | 改动向后兼容，提 PR 给开源仓，合并后自动消失 | 低（不合并就维持现状） | 小的 forward-compatible 增强 |
| **C. 保留+标记** | 无法外置的核心行为改动，加 `ZERITH PATCH` 标记 | —（同步时人工 merge） | 深度内联的逻辑改动 |

---

## 1. 已完成（本轮，A 增量化）

| 原补丁 | 处置 | 兼容性 |
|--------|------|--------|
| P019 `requirements.txt`(+sentencepiece) | 回退上游（pyproject 已含） | ✅ 冗余，无影响 |
| P020 `deepspeed_zero3.yaml` | 回退上游（zerith 不用 zero3；原改动指向不存在的文件） | ✅ 死改动 |
| P021 `ds_config.yaml`(grad_accum→auto) | 回退上游 + 新增 `zerith_ds_config.yaml` / `zerith_deepspeed_zero2.yaml`；zerith_pi05 launcher 改指 | ✅ zerith 走独立配置，上游 launcher 回原生行为 |
| P002 `data_config.py`(末尾示例) | 回退上游（`Relative6DPoseDataConfig`/`relative_6d_example` 全仓无人用，死示例） | ✅ 移到 `zerith_example_relative_6d.py`（opt-in） |
| P005 `transform/__init__.py`(重导出) | 回退上游（无消费者经包导入，都用直接子模块导入）；测试改直导 | ✅ 零影响 |

> 验证：zerith_pi05 训练链 `run_umi_pi05_relative_6d.sh → zerith_deepspeed_zero2.yaml → zerith_ds_config.yaml(auto)` 完整；上游 launcher（Gemma4/MiniCPM/VLN-CE，经核实为上游原样）继续用 `deepspeed_zero2.yaml → ds_config.yaml(1)`，互不影响。

---

## 2. 推荐增量化（A）—— 下一批，低风险

### P018 `pyproject.toml`（安装配置）
**现状**：zerith 填了完整 pinned `dependencies` + `attention` extra + uv cu124 索引/sources。
**建议**：拆分以缩小 diff——
- uv 的 `[[tool.uv.index]]`（aliyun 镜像）可移到独立的 **`uv.toml`**（uv 原生支持项目级 `uv.toml`，不进 pyproject）。
- `[tool.uv.sources]`（torch→cu124）**必须**留在 pyproject（uv 限制），无法外置。
- `dependencies` 列表是 zerith 让 `uv sync` 工作的必需项（上游 `dependencies = []`），**保留**；可考虑作为 **B（贡献上游）**——上游空依赖本就不便用户安装。
**结论**：部分增量化（移走 mirror index），剩余保留。风险低。

### P002 `data_config.py` / P005 `transform/__init__.py`（注册胶水）
**现状**：注册 zerith_pi05 数据集 / 新 transform 的挂载点。
**建议**：注册本就是设计的扩展点，diff 应最小化（只加 import + 一行注册）。若当前 diff 含其它逻辑，把非注册逻辑抽出。
**结论**：维持注册行（属设计内扩展），确保 diff 只有"注册"无"逻辑"。风险低。

---

## 3. 推荐贡献上游（B）—— forward-compatible 增强

这些改动**向后兼容**（新增可选参数 / 类型放宽 / 容忍额外元素），上游接受后即可永久移出补丁面：

| 补丁 | 改动 | 为何兼容 |
|------|------|---------|
| P006 `schema.py` | `list[float]`→`Any` + 支持 2D stats | 类型放宽，旧用法不受影响 |
| P007 `dataloader/__init__.py` | `build_dataloader(..., mode="train")` 新增可选参 | 有默认值，旧调用不变 |
| P008 `lerobot_datasets.py` | `split_mode` 参 + mixture 容忍第 4 元素 `*_` | 可选参 + 向后兼容解包 |
| P011 `OpenPI_ActionHead.py` | action head 适配（~15 行） | 小幅增强 |
| P012 `share_tools.py` | 共享工具（~20 行） | 小幅增强 |
| P017 `overwatch.py` | 分布式 watch（~26 行） | 小幅增强 |

**建议**：把这些打包成 1~2 个"兼容性增强"PR 提给上游。合并前维持现状（已登记+标记）。

---

## 4. 必须保留+标记（C）—— 核心内联行为改动

以下改动深嵌在共享代码路径里，无法用子类/包装外置（涉及数据读取、归一化、action mode、训练循环核心），**维持标记区块**，每次同步重点人工 review：

| 补丁 | 文件 | 规模 | 可外置性 | 备注 |
|------|------|------|---------|------|
| P001 | `datasets.py` | ~528 | 低 | UMI 数据布局、stats 缓存内嵌在 `LeRobotSingleDataset` |
| P003 | `state_action.py` | ~474 | 低 | 15 个新方法直接加进 `Normalizer`/`StateActionTransform` 等既有类 |
| P009 | `PI0.py` | ~463 | 中 | 部分可用 `ZerithPI0(PI0)` 子类外置，但 config/forward 深度耦合 |
| P010 | `PI05.py` | ~77 | 中 | 同 PI0（继承） |
| P013 | `train_starvla.py` | ~506 | 中 | `VLATrainer` 改动；可考虑 zerith 训练入口（新文件）只 inherit override |
| P014 | `train_starvla_cotrain.py` | ~97 | 中 | 同上 |
| P015 | `train_starvlm.py` | ~53 | 中 | 同上 |
| P016 | `trainer_tools.py` | ~331 | 低 | 纯内联修改既有函数（无新函数），最难外置 |

### 子类化外置的机会（中长期，需训练验证）
- **训练入口**：新增 `starVLA/training/zerith_train_starvla.py`，`class ZerithVLATrainer(VLATrainer)` 只 override 需要的方法；逐步把 P013/P014/P015 的改动迁过去，回退上游入口。**前提**：override 能覆盖所需行为，且评测/overfit 全链路验证通过。
- **框架**：`ZerithPI0(PI0)` / `ZerithPI05(PI05)` 经 `FRAMEWORK_REGISTRY` 注册，把 action mode/6d 逻辑收进子类。
- **数据/transform**：`state_action.py` 的 15 个新方法若能聚合成新 transform（已部分如此：`relative_pose.py` 等是独立文件），尽量把剩余内联也抽成新 transform 文件。

> ⚠️ 这些重构**有回归风险**，必须逐项做并跑 overfit smoke + 短训练验证（见 [`testing_layers.md`](testing_layers.md)）。**不建议**一次性大改。

---

## 5. 优先级路线图

| 优先级 | 动作 | 收益 | 风险 |
|--------|------|------|------|
| ✅ 已做 | P002/P005/P019/P020/P021 增量化 | -5 文件（21→16） | 已验证 |
| 🔜 高 | P018 拆 uv mirror 到 `uv.toml` | 缩小 pyproject diff | 低 |
| 🔜 中 | P006/P007/P008/P011/P012/P017 打包提上游 PR | 合并后 -6 文件 | 低（不合并维持现状） |
| 🔜 低（长期） | 训练入口/框架子类化（P013/P014/P015/P009/P010） | -5 文件 | 高，需逐项 overfit 验证 |
| ⛔ 维持 | P001/P003/P004/P016（深内联，但 P001/P003/P009 含大量可独立的新代码） | — | 标记区块 + 同步 review |

### 内联补丁的 +/- 数据（决定可外置性）

> `删-≈0` 的纯追加文件几乎不与上游冲突（追加在文件末尾），优先级实际偏低；真正需关注的是 `删-` 多的内联修改。

| 补丁 | 文件 | 增+ | 删- | 性质 |
|------|------|-----|-----|------|
| P018 | pyproject.toml | 81 | 0 | 纯追加（低冲突） |
| P017 | overwatch.py | 18 | 0 | 纯追加（低冲突） |
| P006 | schema.py | 4 | 2 | 微改 |
| P008 | lerobot_datasets.py | 6 | 3 | 微改 |
| P007 | dataloader/__init__.py | 5 | 2 | 微改 |
| P011 | OpenPI_ActionHead.py | 12 | 1 | 近纯追加 |
| P012 | share_tools.py | 14 | 2 | 近纯追加（但被上游 tools.py/base_framework.py 调用，不可重路由） |
| P010 | PI05.py | 60 | 1 | 近纯追加 |
| P015 | train_starvlm.py | 33 | 17 | 内联 |
| P014 | train_starvla_cotrain.py | 50 | 44 | 内联 |
| P004 | video.py | 65 | 15 | 内联（新 cache 函数可挪新文件，命中检查需保留） |
| P009 | PI0.py | 385 | 17 | 增量为主、少量内联（子类化潜力大） |
| P003 | state_action.py | 363 | 42 | 内联（15 个新方法加进既有类） |
| P013 | train_starvla.py | 360 | 105 | 重度内联 |
| P016 | trainer_tools.py | 210 | 82 | 重度内联（最难外置） |
| P001 | datasets.py | 398 | 84 | 重度内联 |

---

## 相关文档
- [`upstream_patches.md`](upstream_patches.md) — 补丁登记表（M/A 清单与标记约定）
- [`upstream_sync_and_isolation.md`](upstream_sync_and_isolation.md) — 同步流程与隔离策略
- [`testing_layers.md`](testing_layers.md) — 重构后的验证手段
