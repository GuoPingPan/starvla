"""Rerun.io 日志器 (UMI 开环评测 Stage 2 可视化后端)。

输入是 ``starVLA.eval.core.io.read_predictions`` 返回的 dict, 输出离线 ``.rrd``。

布局: 8 个 dim plot 顺序排列, 3 列 grid。
  - 8 dim = pos_x, pos_y, pos_z, qx, qy, qz, qw, gripper (raw)
  - 每个 dim plot 里 (Spatial2DView + LineStrips2D + 显式 VisualBounds2D 避免"拍扁"):
      - obs  (蓝色) — 每个 anchor 的输入 state
      - gt   (红色) — 每个 anchor 的 GT (= next state, frame anchor+1, 跟 obs 错位 1 帧)
      - pred (绿色, 7 组从浅到深) — 每个 anchor 的 16-step 预测 chunk

分组方式: **stride=7 错位** — group g 包含 anchor i 其中 i % 7 == g。
  - group 0: anchors 0, 7, 14, 21, ...
  - group 1: anchors 1, 8, 15, 22, ...
  - ...
同一组用同一颜色 (相邻 anchor 不同组 → 颜色不同 → 易区分)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from starVLA.eval.core.io import METRICS_SCHEMA_VERSION, PREDICTIONS_SCHEMA_VERSION

N_BINS = 7

RAW_DIM_NAMES = ["pos_x", "pos_y", "pos_z", "qx", "qy", "qz", "qw", "gripper"]

# 颜色 (RGB, 0-255)
COLOR_OBS = [50, 100, 255]    # 蓝色 (obs/state)
COLOR_GT = [255, 50, 50]      # 红色 (gt)
# 7 组绿色: 从浅到深 (group 0 最浅, group 6 最深)
COLOR_PRED_BINS = [
    [180, 255, 180],  # group 0
    [140, 230, 140],
    [100, 200, 100],
    [60, 170, 60],
    [30, 140, 30],
    [0, 110, 0],
    [0, 80, 0],       # group 6
]


def log_episode_to_rerun(
    results: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_path: Path,
    *,
    application_id: str = "starVLA.umi.open_loop",
    horizon_step: int = 1,
    obs_images: dict[str, np.ndarray] | None = None,
) -> Path:
    """把 episode eval 的 trajectory + chunks 写入 Rerun 录制 + blueprint。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(application_id, default_enabled=True)
    rr.save(str(output_path))

    # ── 顶层元信息 ────────────────────────────────────────────────────
    rr.log("episode/info", rr.TextLog(_format_metadata(metadata)), static=True)
    rr.log(
        "episode/schema",
        rr.TextLog(
            f"predictions_schema_version={PREDICTIONS_SCHEMA_VERSION} "
            f"metrics_schema_version={METRICS_SCHEMA_VERSION} "
            f"n_bins={N_BINS} horizon_step={horizon_step} binning=stride7"
        ),
        static=True,
    )

    anchors = np.asarray(results["anchors"])                       # (N,)
    state_at_anchor = np.asarray(results["state_at_anchor"])      # (N, 8) raw
    action_gt = np.asarray(results["action_gt"])                  # (N, 8) raw
    pred_chunks = np.asarray(results["action_pred_chunks"])       # (N, H, 8) raw
    n_samples, horizon, _ = pred_chunks.shape

    # ── 8 个 dim 各画 obs / gt / pred_bin_X/anchor_Y ─────────────────
    # 用 send_columns 批量 log, 避免每帧一次 rr.log 的 Python 调用开销
    # (旧版 32k 调用 ~5 min, 批量后 ~1.8k 调用 ~5s)
    for dim_idx, dim_name in enumerate(RAW_DIM_NAMES):
        # obs + gt: 1 次 send_columns / dim, 索引 = anchors, 值 = state/gt 在该 dim 的取值
        anchor_index = rr.TimeColumn(timeline="anchor_frame", sequence=anchors)
        obs_col = _scalar_column(state_at_anchor[:, dim_idx].astype(np.float64))
        gt_col = _scalar_column(action_gt[:, dim_idx].astype(np.float64))
        rr.send_columns(f"/{dim_name}/obs", indexes=[anchor_index], columns=[obs_col])
        rr.send_columns(f"/{dim_name}/gt", indexes=[anchor_index], columns=[gt_col])

        # pred: 每个 anchor 一个 send_columns / entity, 索引 = anchor + arange(horizon)
        # entity path 按 i % 7 分 bin (stride=7 错位)
        for i in range(n_samples):
            anchor = int(anchors[i])
            bin_idx = i % N_BINS
            times = anchor + np.arange(horizon, dtype=np.int64)
            time_col = rr.TimeColumn(timeline="anchor_frame", sequence=times)
            value_col = _scalar_column(pred_chunks[i, :, dim_idx].astype(np.float64))
            rr.send_columns(
                f"/{dim_name}/pred/bin_{bin_idx}/anchor_{i}",
                indexes=[time_col],
                columns=[value_col],
            )

    # ── 标量 metadata ────────────────────────────────────────────────
    latency = np.asarray(results["latency_seconds"])
    rr.log("metrics/latency_mean_seconds", rr.Scalars([float(np.mean(latency))]), static=True)
    rr.log("metrics/latency_median_seconds", rr.Scalars([float(np.median(latency))]), static=True)
    rr.log("metrics/sample_count", rr.Scalars([float(n_samples)]), static=True)
    rr.log("metrics/horizon", rr.Scalars([float(horizon)]), static=True)

    # ── 观测图像 (可选 sidecar): 每 cam 一个 entity, 沿 anchor_frame timeline ──
    # 每个 anchor log 一帧 (跟 obs 曲线点对齐); viewer 拖动 anchor_frame 即看到对应输入图像。
    # rr.set_time(sequence=...) 是 set_time_sequence 在 rerun 0.34 的等价 API; 末尾 reset_time
    # 清掉线程级时间, 不污染后续 rr.log (send_columns / send_blueprint 不受 stateful time 影响)。
    has_obs_image = False
    if obs_images:
        for cam_key, frames in obs_images.items():
            frames = np.asarray(frames)
            if frames.shape[0] != n_samples:
                raise ValueError(
                    f"obs_images['{cam_key}'] has {frames.shape[0]} frames but results has "
                    f"{n_samples} anchors; sidecar 跟 predictions.npz 不同源"
                )
            # entity 名去掉 'video.' 前缀: video.cam_wrist_0 → cam_wrist_0
            entity = f"/obs_image/{cam_key.replace('video.', '')}"
            for i in range(n_samples):
                rr.set_time("anchor_frame", sequence=int(anchors[i]))
                rr.log(entity, rr.Image(frames[i]))
            has_obs_image = True
        rr.reset_time()

    # ── blueprint: 8 个 TimeSeriesView (+ 可选 obs_image) 3 列 grid ───
    # TimeSeriesView 无 aspect lock, y 轴自适应数据范围, 不会被 x 轴 scale 压扁
    contents = [
        rrb.TimeSeriesView(
            origin=f"/{dim_name}",
            contents=[f"$origin/**"],
            name=dim_name,
        )
        for dim_name in RAW_DIM_NAMES
    ]
    if has_obs_image:
        # 第 9 格: 图像视图。9 项 3 列 → 3 行, obs_image 落右下。
        contents.append(
            rrb.Spatial2DView(
                origin="/obs_image",
                contents=["$origin/**"],
                name="obs_image",
            )
        )
    blueprint = rrb.Blueprint(rrb.Grid(contents=contents, grid_columns=3))
    rr.send_blueprint(blueprint)

    return output_path


def _scalar_column(values: np.ndarray) -> rr.ComponentColumn:
    """把一组 scalar 值包成 ComponentColumn, 给 send_columns 用。

    ``rr.Scalars([...])`` 是 AsComponents, 内部有 1 个 component batch (即 scalars 数组)。
    我们取出它的 descriptor + batch, 包成 ComponentColumn。
    """
    archetype = rr.Scalars(values)
    batch = archetype.as_component_batches()[0]
    return rr.ComponentColumn(
        descriptor=batch.component_descriptor(),
        component_batch=batch,
    )


def _format_metadata(metadata: dict[str, Any]) -> str:
    """把 metadata dict 格式化成多行字符串, 用于 TextLog body。"""
    import json
    lines = []
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
