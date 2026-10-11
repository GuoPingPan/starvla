#!/usr/bin/env python3
"""Stage 2: 用 Rerun 可视化 UMI 开环评测 artifacts。

从 Stage 1 输出目录读 ``predictions.npz`` + ``metrics.json``, 写离线 ``.rrd``。
Rerun viewer 里:
  - 7 个 dim group (pos_xyz + euler_rpy + gripper), 每个 dim 一条 obs + gt + pred
  - chunks 按 anchor index 分 7 个 bin, 点击切换 bin 显示
  - obs_image: 若 obs_images.npz sidecar 在场, 每 anchor 的 wrist 帧 (跟 anchor_frame
    timeline 对齐, 拖动时间轴即看到对应输入图像); 缺失则无图

用法:
    PYTHONPATH=. python examples/realRobots/z_umi/eval_files/stage2_visualize.py <output_dir>
    bash examples/realRobots/z_umi/eval_files/run_umi_pi05_open_eval.sh --stage2-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from starVLA.eval.core.io import read_metrics, read_obs_images, read_predictions
from starVLA.eval.viz.rerun_logger import log_episode_to_rerun


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
        default="starVLA.umi.open_loop",
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
