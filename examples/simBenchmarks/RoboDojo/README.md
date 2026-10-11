# RoboDojo · PI05 (bimanual, z-umi relative-6d 模板)

基于 starVLA 的 **PI0.5** 在 **RoboDojo** 双臂仿真 benchmark 上做监督微调，数据管线与训练配置
适配自 `examples/realRobots/zerith_pi05`（SE(3) 相对位姿 + rotation_6d）。

## 概览

- **数据**：RoboDojo `RoboDojo_ee_lerobot_v30_video`（LeRobot **v3.0**，双臂 bimanual）。
  - state/action 各 **16D** = 左臂[pos(3)+quat_wxyz(4)+gripper(1)] + 右臂同结构。
  - 3 相机：`cam_high` + `cam_left_wrist` + `cam_right_wrist`（av1 视频）。
  - 35 任务 / 3500 episodes / 25fps，episode 按 task_index 严格连续（task N = eps N×100 ~ N×100+99）。
- **动作表征**（沿用 z-umi relative-6d，扩展为双臂）：
  - `RelativePoseTransform` 把 action 位姿相对到当前 state[t0]（SE(3) 群运算，左右臂分别处理）。
  - `QuaternionTo6DRotation(input_convention="wxyz")` 四元数→rotation_6d（RoboDojo 是 wxyz，几何正确）。
  - 变换后 action/state 各 **20D**（左[pos3+rot6d6+grip1]+右同），pad 到 32 匹配 PI05 预训练 ckpt。
- **模型**：PI0.5 base checkpoint（`/mnt/cfs/data/pretrain_models/pi05_base_starvla`）+ PaliGemma tokenizer。

## 文件结构

```
examples/simBenchmarks/RoboDojo/
├── train_files/
│   ├── data_registry/
│   │   ├── robodojo_modality_template.json   # 双臂 16D 拆分 modality 模板（+ annotation）
│   │   ├── robodojo_dataset.py               # RoboDojoLeRobotDataset（语言/任务过滤/per-task缓存）
│   │   └── data_config.py                    # robot_type=robodojo_bimanual_relative_6d + transforms + mixtures
│   ├── robodojo_pi05.yaml                    # PI05 训练配置
│   ├── run_robodojo_pi05.sh                  # 训练启动脚本（单卡）
│   └── smoke_test_dataloader.py              # dataloader 冒烟测试
└── eval_files/
    ├── eval_robodojo_pi05_overlay.yaml       # 开环评测 overlay
    └── run_robodojo_pi05_eval.sh             # 评测启动脚本
```

数据下载脚本：`/nix/data/robodojo/download_robodojo.py`。

## 环境关键点

- venv：`/root/workspace/vla/starvla/.venv`（torch 2.6+cu124, accelerate, decord, pyav）。
- editable 安装指向主仓库 `/root/workspace/vla/starvla`，**训练/评测脚本显式 export
  `PYTHONPATH=<robodojo-worktree>`**，使 `starVLA` 与 `examples/` 解析到本 worktree
  （registry 的 `discover_and_merge` 才能扫到 RoboDojo 的 data_config）。

## 数据准备

```bash
# parquet（state/action，全 35 任务，0.24GB）+ meta（info/tasks/episodes）
# 视频：data/RoboDojo_ee_lerobot_v30_video/videos/observation.images.<cam>/chunk-000/file-NNN.mp4
# 单任务约 3-4GB，全量 ~113GB。
python /nix/data/robodojo/download_robodojo.py --workers 8
```

下载完成后目录布局（LeRobot v3.0）：
```
/nix/data/robodojo/RoboDojo_ee_lerobot_v30_video/
├── meta/{info.json, tasks.parquet, stats.json, episodes/chunk-000/file-000.parquet}
├── data/chunk-000/file-{000,001,002}.parquet
└── videos/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/chunk-000/file-NNN.mp4
```

## 训练

```bash
# 单任务（task 24 = Pour Liquid Into Cup，Generalization 中等难度）
bash examples/simBenchmarks/RoboDojo/train_files/run_robodojo_pi05.sh

# 换任务 / 多任务 / 改参数（环境变量覆盖）
TASK_INDEX=30 bash .../run_robodojo_pi05.sh     # task 30 = Stack Bowls
TASK_INDEX=all bash .../run_robodojo_pi05.sh    # 多任务（需全量视频）
MAX_STEPS=20000 BATCH=4 bash .../run_robodojo_pi05.sh
```

输出：`/nix/data/model_outputs/robodojo_pi05_*_task<N>_<steps>step/<timestamp>/`（checkpoints/ + final_model/）。

### ⚠️ 全参微调，不要冻结 VLM

所有配置均为 `freeze_modules: ''`（全参微调）。**不要改回 `freeze_modules: vlm`**：
冻结 VLM 已实测显著掉点。早期 NaN 调试阶段曾用冻结方案临时规避梯度 NaN，该结论已过时，勿再沿用。

## 开环评测

```bash
RUN_DIR=/nix/data/model_outputs/robodojo_pi05_full_task24_10000step/<timestamp>
# smoke
bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh $RUN_DIR --episodes 2400 2401 --max-anchors 5
# 全量（task24 held-out episodes）
bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh $RUN_DIR
```

输出 `playground/Evals/<run>/`：开环 MSE / 轨迹对比 / predictions.npz。

## 闭环评测（可选，重）

RoboDojo 是 **eval-only** 仿真器（Isaac Sim + XPolicyLab），闭环成功率评估需另装仿真栈
（见 https://github.com/RoboDojo-Benchmark/RoboDojo）。开环指标正常后再考虑。

## 已知问题 / 备注

- **任务选择**：JS 榜单拿不到精确 pi0.5 各任务成功率；按能力维度 + 任务性质选了 Generalization 类的
  Pour Liquid (task24) / Stack Bowls (task30) 等中等难度任务。换任务只需改 `task_index`。
- **PI0.py 兜底**：`forward_from_observation` 对 v_t 加了 `nan_to_num`（安全网，正常训练不触发）。
- **缓存隔离**：`RoboDojoLeRobotDataset` 把 norm stats (`stats_gr00t_task<N>.json`) 和 steps index
  (`steps_data_index_task<N>.pkl`) 按 task 隔离，避免不同任务串用。
