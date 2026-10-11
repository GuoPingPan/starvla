# 测试分层

> 把"测试 = 跑 `scripts/train.sh`"拆成分层。`train.sh` 是**完整集群训练流水线**（按 commit 拉 BOS 代码 zip → uv sync → 拉数据 → 拉预训练 → 训练 → 上传），不是测试，每次跑要多小时。绝大多数改动用下面更轻的层验证即可，只有正式实验才上 `train.sh`。
>
> CI / 代码来源（commit ↔ BOS zip）见 [`ci_and_training_source.md`](ci_and_training_source.md)；UMI smoke 记录见 [`umi_pi05_smoke_test.md`](../umi/umi_pi05_smoke_test.md)。本文件只定义**分层与映射**。

---

## 1. 测试分层

| 层 | 命令 | 耗时 | 资源 | 验证什么 |
|----|------|------|------|---------|
| **单元测试** | `uv run pytest tests/` | 秒 | CPU | 数学 / transform / config 校验（`tests/`） |
| **模块 smoke** | `bash scripts/dev/smoke.sh <框架> <yaml>` | 秒~分 | 1 GPU | 单框架 forward / dataloader 形状正确 |
| **Overfit smoke** | `bash scripts/dev/overfit.sh <yaml>` | 分 | 1 GPU | 1 episode、几百步、loss 能否→0（核心"我没改坏"验证） |
| **短 train.sh** | `MAX_STEPS=50 ... bash scripts/train.sh <yaml> <commit> --local` | ~10 分 | 1 GPU | BOS 全链路（拉代码/数据/预训练）能跑通 |
| **完整 train.sh** | `bash scripts/train.sh <yaml> <commit>` | 小时 | 多卡 | 正式实验 |

---

## 2. 改动类型 → 应跑哪层

| 改了什么 | 单元 | 模块smoke | overfit | 短train | 完整 |
|---------|:----:|:--------:|:------:|:------:|:----:|
| transform / 数学（rotation_6d 等） | ✅ | — | ✅ | — | — |
| config / data_config / schema | ✅ | — | ✅ | — | — |
| 框架 forward / dataloader 入口 | — | ✅ | ✅ | — | — |
| trainer / 训练入口 / overfit 逻辑 | — | — | ✅ | — | — |
| `scripts/train.sh` / CI / 数据下载 | — | — | — | ✅ | — |
| 新数据集注册 | — | — | ✅ | — | — |
| 正式实验 / 复现 | — | — | — | — | ✅ |

---

## 3. 命令速查

### 单元测试
```bash
uv run pytest tests/             # 全部
uv run pytest tests/test_quaternion_to_6d_transform.py   # 单个
```

### 模块 smoke（单框架独立跑）
```bash
bash scripts/dev/smoke.sh QwenOFT examples/.../xxx.yaml
# 也可直接指定文件: bash scripts/dev/smoke.sh starVLA/dataloader/lerobot_datasets.py <yaml>
```

### Overfit smoke
```bash
bash scripts/dev/overfit.sh examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml
# 前提: yaml 里已用 overfit_episode_id 开启单 episode 过拟合 (语义见 agent.md 附录 A.3)
```
> overfit 结果**只能作为 pipeline / fitting evidence，不能作为泛化证据**（见 `agent.md` 附录 A.2）。

### 短 train.sh（全链路冒烟）
```bash
MAX_STEPS=50 BATCH=2 SAVE_EVERY=25 \
  bash scripts/train.sh examples/.../xxx.yaml latest --local
```

### PR 门禁（本地，必过）
```bash
black --check <你改的 .py>
ruff check <你改的 .py>
pytest tests/
python scripts/dev/check_patches.py   # 改了上游文件的话, 确认已登记
```

---

## 4. 本地 vs 集群

- 开发机**有本地 GPU**：单元 / 模块 smoke / overfit 都在 worktree 的本地 `.venv` 里跑（见 [`worktree_dev.md`](worktree_dev.md)）。
- 只有**短 train.sh** 和**完整 train.sh** 需要走 BOS（因为要拉数据/预训练）；其中短 train.sh 也可本地 1 GPU 跑（`--local` + `MAX_STEPS=50`）做冒烟。
- 集群多卡正式训练才用 `train.sh` 默认形态。

---

## 相关文档
- [`ci_and_training_source.md`](ci_and_training_source.md) — CI 打包 / commit ↔ BOS zip
- [`umi_pi05_smoke_test.md`](../umi/umi_pi05_smoke_test.md) — UMI PI05 smoke 实操记录
- [`worktree_dev.md`](worktree_dev.md) — 本地 worktree + uv 环境
- `scripts/dev/smoke.sh` / `overfit.sh` — 测试入口脚本
