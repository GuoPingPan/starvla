#!/usr/bin/env python3
"""Stage 2: 用 Rerun 可视化 RoboDojo 双臂开环评测 artifacts。

从 Stage 1 输出目录读 ``predictions.npz`` + ``metrics.json``, 写离线 ``.rrd``。
跟 UMI 版 (examples/realRobots/zerith_pi05/eval_files/stage2_visualize.py) 的唯一区别:
  传入双臂 16 维 dim_names (RoboDojo raw 布局), 复用维度无关的 log_episode_to_rerun。

Rerun viewer 里:
  - 16 个 dim plot = 左臂 8 (pos3 + quat_wxyz4 + gripper1) + 右臂 8 (同), 4 列 grid
  - 每个 dim 一条 obs + gt + pred; chunks 按 anchor index 分 7 个 bin, 点击切换
  - obs_image: 3 相机 (cam_high / cam_left_wrist / cam_right_wrist), 每相机独立视图

用法:
    PYTHONPATH=. python examples/simBenchmarks/RoboDojo/eval_files/stage2_visualize.py <output_dir> [<output_dir> ...]
    bash examples/simBenchmarks/RoboDojo/eval_files/run_robodojo_pi05_eval.sh --stage2-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from starVLA.eval.core.io import read_metrics, read_obs_images, read_predictions
from starVLA.eval.viz.rerun_logger import log_episode_to_rerun

# RoboDojo 双臂 raw 动作 16 维, 顺序对齐 npz 最后一维。
# 来源: examples/simBenchmarks/RoboDojo/train_files/data_registry/data_config.py:9-10
#   [left_pos(3) + left_quat_wxyz(4) + left_gripper(1)      # dim 0-7
#    + right_pos(3) + right_quat_wxyz(4) + right_gripper(1)] # dim 8-15
# ⚠️ 四元数顺序是 wxyz (w 在前), 由训练 transform QuaternionTo6DRotation(..., input_convention="wxyz")
#    的逆变换 _decode_to_raw 还原。切勿照搬 UMI 的 qx,qy,qz,qw (qw 在后) 顺序重新排序。
ROBODOJO_RAW_DIM_NAMES = [
    "left_pos_x", "left_pos_y", "left_pos_z",
    "left_quat_w", "left_quat_x", "left_quat_y", "left_quat_z",
    "left_gripper",
    "right_pos_x", "right_pos_y", "right_pos_z",
    "right_quat_w", "right_quat_x", "right_quat_y", "right_quat_z",
    "right_gripper",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_dirs",
        type=Path,
        nargs="+",
        help="Stage 1 输出目录 (含 predictions.npz + metrics.json)。",
    )
    parser.add_argument(
        "--application-id",
        type=str,
        default="starVLA.robodojo.open_loop",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="episode.rrd",
    )
    return parser.parse_args(argv)


def visualize_one(input_dir: Path, application_id: str, output_name: str) -> Path:
    """读一个目录的 artifacts, 写对应的 .rrd 文件。返回 .rrd 路径。"""
    results = read_predictions(input_dir)
    report = read_metrics(input_dir)
    # obs_images.npz 是可选 sidecar; 缺失 (None) → 走无图 rrd, 不报错。
    obs_images = read_obs_images(input_dir)
    output_path = input_dir / output_name
    log_episode_to_rerun(
        results=results,
        metadata=report,
        output_path=output_path,
        application_id=application_id,
        obs_images=obs_images,
        dim_names=ROBODOJO_RAW_DIM_NAMES,
    )
    return output_path


def main() -> None:
    args = parse_args()
    for input_dir in args.input_dirs:
        if not (input_dir / "predictions.npz").exists() or not (input_dir / "metrics.json").exists():
            print(f"SKIP {input_dir}: missing predictions.npz or metrics.json", file=sys.stderr)
            continue
        rrd = visualize_one(input_dir, args.application_id, args.output_name)
        print(f"wrote {rrd}")


if __name__ == "__main__":
    main()
