# UMI · PI05 开环评测 (open-eval) 说明

> 本文档讲解 UMI · PI05 的开环评测入口 `examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh` 及其配套脚本，并以一条真实评测指令为例逐项拆解，最后列出常见坑与注意事项。

---

## 1. 评测指令（以本次为例）

```bash
RUN_DIR=/root/data/model_outputs/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0805_214632/ \
  DATA_ROOT_OVERRIDE=/root/data/z-umi/data \
    bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh \
      --checkpoint-step 10000 --episodes 0 1 \
      --dataset-name 20260803035455_lerobot_5852395f_noep81_forceclose
```

**含义一句话**：用 `0805_214632` 这个 run 的 **step10000** 权重，从 mixture `shenzhen_relative_6d_2_nooutlier_forceclose`（含 2 个 entry，必须显式挑）里选 **`5852395f_noep81_forceclose`** 这份数据，评测其 eval 划分里的 **第 0、1 两条** eval 轨迹（位置索引），评测机数据根指向 `/root/data/z-umi/data`。

---

## 2. 相关文件与职责分工

| 文件 | 职责 |
| --- | --- |
| `examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh` | **Launcher**：管 uv 环境 / PYTHONPATH / preflight / 参数透传 / Stage1+Stage2 串联，不硬编码训练参数 |
| `starVLA/training/eval_starvla.py` | **Stage 1**：多卡开环评测主入口，复刻 `train_starvla.py` 结构，推理 → decode → 写盘 |
| `examples/realRobots/zerith_pi05/eval_files/eval_umi_pi05_relative_6d_overlay.yaml` | **Eval overlay**：只放 eval-specific 字段，与 ckpt 的 `config.full.yaml` merge 使用 |
| `examples/realRobots/zerith_pi05/eval_files/stage2_visualize.py` | **Stage 2**：读 `predictions.npz` + `metrics.json` 写 Rerun `.rrd` |
| `starVLA/eval/core/checkpoint_utils.py` | ckpt 发现 / step 定位 / mixture 选数据集 (`pick_mixture_entry`) |
| `starVLA/eval/core/io.py` | Stage 1 产物的 schema-stable 读写（`predictions.npz` / `metrics.json` / `obs_images.npz`） |
| `examples/realRobots/zerith_pi05/train_files/data_registry/data_config.py` | `DATASET_NAMED_MIXTURES` 注册表（数据集 mix → 数据集列表） |

### 为什么 launcher 与 eval 分离

Launcher 只负责：uv 环境、`PYTHONPATH`、路径 preflight、参数透传、Stage1→Stage2 串联；所有评测参数都通过 `EVAL_YAML` overlay + CLI 传，不把训练参数 hardcode 进脚本。想绕开 launcher 直接跑 Stage 1 也是合法的（见 §5 用法 B）。

---

## 3. 指令逐项拆解

### 3.1 环境变量

| 变量 | 本次值 | 作用 |
| --- | --- | --- |
| `RUN_DIR` | `/root/data/model_outputs/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0805_214632/` | 实例 ckpt 目录（含 `config.full.yaml` + `checkpoints/`）。脚本默认指向 `/root/.models/run_umi_shenzhen_relative_6d_20000step_0727_100151`，**必须显式覆盖**。⚠️ 注意 `/root/data` 本身是**软链接 → `/mnt/cfs/data`**（百度 CFS 网络盘挂载点），权重 `.pt` 是落在 CFS 上的实体文件（非指向别处模型的软链接） |
| `DATA_ROOT_OVERRIDE` | `/root/data/z-umi/data` | 覆盖 ckpt config 里的 `data_root_dir`（训练 config 里存的路径可能在本机失效） |
| `DATA_MIX_OVERRIDE` | （未设） | 留空 = 沿用 ckpt 的 `data_mix`（本次为 `shenzhen_relative_6d_2_nooutlier_forceclose`） |
| `EVAL_YAML` | （未设，用默认） | overlay 配置文件 |
| `NUM_PROCESSES` | （未设） | >1 时自动切 `accelerate launch --multi-gpu` 多卡 |

### 3.2 CLI 参数（透传给 Stage 1 的 `eval_starvla.py`）

| 参数 | 本次值 | 含义 |
| --- | --- | --- |
| `--checkpoint-step 10000` | 10000 | 用 `checkpoints/steps_10000_pytorch_model.pt`。不指定则取 eval.yaml 或自动发现的最新 step（本次 0805_214632 下有 5000/10000/15000/20000 四个） |
| `--episodes 0 1` | 0 1 | **eval 列表的 0 基位置索引**（非原始 episode_id），内部自动映射到真实 episode_id |
| `--dataset-name 20260803035455_lerobot_5852395f_noep81_forceclose` | 必填 | 本 run 的 mixture 现有 **2 个 entry**，`pick_mixture_entry` 在 mixture >1 且未指定时会抛错，必须用此参数显式挑一个。本例挑 `5852395f_noep81_forceclose`（剔 ep81 + force-close），见 §7.1 |

### 3.3 本次 run 的关键配置（来自 `config.full.yaml`）

| 配置项 | 值 | 说明 |
| --- | --- | --- |
| `framework.name` | PI05 | paligemma 3B + action expert，pi0.5 |
| `data_mix` | `shenzhen_relative_6d_2_nooutlier_forceclose` | 训练用数据集 mix |
| `action_dim / action_horizon` | 32 / 50 | action padding 宽度 / chunk 长度 |
| `split` | 90% train / 10% eval（交错式） | eval 都是训练未见过的轨迹 |
| `image_keys` | base + 左右 wrist 共 3 个 camera | 评测时每 anchor 取 wrist 帧写 sidecar |

---

## 4. 执行流程

```
run_umi_pi05_open_eval.sh
  ├─ preflight: 检查 uv / RUN_DIR / EVAL_YAML / STAGE2_SCRIPT / config 文件存在
  ├─ base_vlm symlink 处理: 只有 saved config 真有 framework.qwenvl.base_vlm 才 symlink
  │   （PI05/paligemma 没有该键 → try/except 返回空 → 跳过）
  ├─ 扫 launcher-only flag: --visualize / --no-visualize / --stage2-only，其余全透传
  │
  ├─ Stage 1: uv run python -m starVLA.training.eval_starvla
  │     --run-dir <RUN_DIR> --eval-yaml <EVAL_YAML>
  │     [--data-root-override] [--data-mix] <STAGE1_ARGS...>
  │   输出 tee 到临时日志（Stage 2 据日志里的 __EVAL_OUTPUT_DIR__ 标记定位 output_dir）
  │   （NUM_PROCESSES>1 时切 accelerate launch --multi-gpu）
  │
  └─ Stage 2（默认开）: stage2_visualize.py <output_dir>/episode_*/  → 写 episode.rrd
```

### Stage 1 内部（`eval_starvla.py`）

1. **加载配置**：`load_merged_config` 把 ckpt `config.full.yaml` 与 eval overlay YAML 做 `OmegaConf.merge`（overlay 覆盖同名项）；再套 CLI 覆盖（优先级 **CLI > eval_yaml > 代码默认**）。
2. **构建数据集**：`build_dataset` 走与训练同一条路径 `make_LeRobotSingleDataset`，`pick_mixture_entry` 从 `DATASET_NAMED_MIXTURES` 挑 `(dataset_name, robot_type)`。
3. **resolve 参数**：ckpt step / seed / batch_size / episodes 位置索引 → 真实 episode_id。
4. **加载模型**：`baseframework.from_pretrained` 读 ckpt 自带的 config 构建 PI05，`.to(cuda, bf16).eval()`。
5. **评测循环**：每 episode 一个 `_IndexedSubset` + DataLoader（anchor 跨 rank 切分）→ `unwrap_model().predict_action()` → 归一化输出经**训练 transform 链的精确逆 (unapply)** 还原成 raw 物理量 → 四元数符号对齐 → 写 `predictions.npz` + `metrics.json` + `obs_images.npz`。
6. 主进程末尾打印 `__EVAL_OUTPUT_DIR__ <dir>`，供 launcher 精确定位。

> 关键点：decode 复用训练 transform 链的 `unapply`（min_max → rotation_6d→quat → SE(3) 相对→绝对），不依赖 `invert_action_mode`，对 quaternion / 6d / euler / 关节角等任意表示都成立。

### Stage 2 内部（`stage2_visualize.py`）

读 `predictions.npz` + `metrics.json`（缺 `obs_images.npz` 则退化成无图 rrd）→ `log_episode_to_rerun` 写 `episode.rrd`。Rerun viewer 里：

- 7 个 dim group（pos_xyz + euler_rpy + gripper），每 dim 一条 obs / gt / pred
- chunks 按 anchor 分 7 个 bin，点击切换
- 拖动时间轴可见对应 anchor 的输入图像（有 obs_images sidecar 时）

> 需要 `rerun-sdk`：`uv pip install rerun-sdk`（launcher 里注明）。

---

## 5. 常用用法速查

```bash
# 默认（最新 step，全量 eval）+ Stage 2 写 rrd
bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh

# smoke：每 episode 只跑前 5 个 anchor
bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh --max-anchors 5

# 跳过 .rrd 生成
bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh --no-visualize

# 指定 episode / step
bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh --episode-id 3 --checkpoint-step 20000

# 多卡（NUM_PROCESSES>1 自动切 accelerate launch）
NUM_PROCESSES=4 bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh

# 换数据集评测（跨分布，产物自动落到 ..._mix-<name>/ 隔离）
DATA_MIX_OVERRIDE=hefei_relative_6d \
  bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh --max-anchors 2

# Stage 2 单独跑（Stage 1 已跑过）
STAGE2_INPUT_DIR=playground/Evals/<run>_step20000_seed42 \
  bash examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh --stage2-only

# 绕开 launcher 直接跑 Stage 1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m starVLA.training.eval_starvla \
  --run-dir <RUN_DIR> \
  --eval-yaml examples/realRobots/zerith_pi05/eval_files/eval_umi_pi05_relative_6d_overlay.yaml \
  --episodes 0 --max-anchors 2
```

---

## 6. 产物说明

output_dir 自动派生：`<output_root>/<run_basename>_step<N>_seed<S>/`

```bash
playground/Evals/0805_214632_step10000_seed42/          # 本次
└── episode_<id>/                                        # 每条 eval 轨迹一个子目录
    ├── predictions.npz      # schema v4：anchors / state_at_anchor /
    │                        #   action_gt / action_pred_chunks / latency_seconds
    ├── metrics.json         # eval_config / episode_id / checkpoint_step / run_dir /
    │                        #   artifacts / representation（表示元数据）
    └── obs_images.npz       # sidecar（可选）：每 anchor wrist 帧，rrd 带图用
```

- 换数据集（`--data-mix`）时 output_dir 自动追加 `_mix-<name>`，不覆盖原数据集结果。
- `metrics.json` 的 `representation` 块记录每个 action key 的维度 / 旋转类型 / action_horizon，供下游 metrics/viz 表示无关地解读。

---

## 7. 注意事项与坑

### 7.1 `--dataset-name`：mixture >1 时必须显式指定（否则抛错）

本 run 的 `data_mix = shenzhen_relative_6d_2_nooutlier_forceclose`，该 mixture 在 `examples/realRobots/zerith_pi05/train_files/data_registry/data_config.py:166` 里现有 **2 个活跃 entry**：

| entry | 说明 |
| --- | --- |
| `20260726133821_lerobot_7d6be9ce_noep58_forceclose` | 7d6be9ce 剔 ep58 后 force-close |
| `20260803035455_lerobot_5852395f_noep81_forceclose` | 5852395f 剔 ep81 后 force-close（**本次评测目标**） |

因此 **`--dataset-name` 是必填的**。`pick_mixture_entry`（`starVLA/eval/core/checkpoint_utils.py:148`）逻辑：

- 未指定 `--dataset-name` 且 mixture 只有 1 个 entry → 自动挑唯一 entry；
- 未指定且 mixture 有 >1 个 entry → 抛错，**拒绝"隐式挑第一个"**：
  ```
  ValueError: data_mix='shenzhen_relative_6d_2_nooutlier_forceclose' has 2 datasets
  ['20260726133821_lerobot_7d6be9ce_noep58_forceclose', '20260803035455_lerobot_5852395f_noep81_forceclose'];
  pass --dataset-name to pick one.
  ```
- 指定的名字不在 mixture 里 → 抛 `--dataset-name=... not in mixture '...'; available: [...]`。

> 历史：该 entry（`5852395f_noep81_forceclose`）此前在 `data_config.py:176` 被**注释掉**，当时 mixture 只有 1 个 entry、不用 `--dataset-name` 也能跑；现在已**重新激活**，评测时**必须显式带 `--dataset-name`**。本地数据 `/root/data/z-umi/data/20260803035455_lerobot_5852395f_noep81_forceclose` 已就位。

### 7.2 `--episodes` 是位置索引，不是原始 episode_id

`--episodes 0 1` 指的是 **eval 列表（split 后升序）里的第 0、1 个**，内部映射到真实 episode_id。startup 会打印 `[eval] episodes 位置索引→真实 episode_id: [0→epX, 1→epY]`。越界给清晰报错。

### 7.3 跨数据集评测的统计量口径

换不同分布数据集（如 `hefei_relative_6d`）时，**norm 仍用模型固化的 `_per_key` 统计**（训练时 shenzhen 统计），与数据集自己的 stats 不一致 → 开环 MSE/ADE **不可定量比较**，仅适用于 pipeline 验证 / Rerun 可视化 / 同 embodiment 泛化烟测。要可比的跨数据集数字需额外对齐 norm_stats。

### 7.4 多卡与复现

- 多卡：`NUM_PROCESSES=4` → launcher 自动切 `accelerate launch --multi-gpu`；单 episode 内 anchor 跨卡切分，`gather_object` 回 rank 0 聚合。
- 可复现：全局 seed（默认 42）**不加 rank 偏移**，不同 world_size 下同一 anchor 拿同一 flow-matching 噪声。

### 7.5 环境

- 环境统一走 `uv run`（项目 `.venv`），不再用 mamba/conda；Stage 2 另需 `rerun-sdk`。
- eval 纯推理不接 tracker；`eval_starvla.py` 会强制 `SWANLAB_MODE=disabled`，避免残留环境变量污染。
- Eval 用裸 `Accelerator`（不带 DeepSpeed plugin），因为推理无 backward，DeepSpeed 会要求 dataloader 有 `batch_size` 属性而报错。

### 7.6 ckpt 布局

本实例 ckpt 是**扁平导出**（`.pt` 直接平铺在 `run_dir/checkpoints/` 或 `run_dir/` 两种布局都支持），`checkpoint_utils` 已兼容，无需手动 mv。

---

## 8. 相关文档

- Stage 1 主入口：`starVLA/training/eval_starvla.py`（顶部 docstring 有完整用法）
- Overlay 配置：`examples/realRobots/zerith_pi05/eval_files/eval_umi_pi05_relative_6d_overlay.yaml`
- 训练流水线：`docs/zerith_docs/infra/train_sh_run_flow.md`
- UMI PI05 训练冒烟测试：`docs/zerith_docs/umi/umi_pi05_smoke_test.md`
- 数据集注册表：`examples/realRobots/zerith_pi05/train_files/data_registry/data_config.py`
