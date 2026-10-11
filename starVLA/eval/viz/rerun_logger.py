"""Rerun.io 日志器 (开环评测 Stage 2 可视化后端, 维度无关)。

输入是 ``starVLA.eval.core.io.read_predictions`` 返回的 dict, 输出离线 ``.rrd``。
默认按 UMI 单臂 8 维 (pos/quat/gripper) 布局; 通过 ``dim_names`` 可覆盖为任意维度
(如 RoboDojo 双臂 16 维), dim plot 数量 = len(dim_names)。

布局: 每个 dim 一个 TimeSeriesView, grid 列数 = grid_columns 或自动 (≤9 dim → 3 列, 否则 4 列)。
  - 每个 dim plot 里 (Spatial2DView + LineStrips2D + 显式 VisualBounds2D 避免"拍扁"):
      - obs  (蓝色) — 每个 anchor 的输入 state
      - gt   (红色) — 每个 anchor 的 GT (= next state, frame anchor+1, 跟 obs 错位 1 帧)
      - pred (绿色, 7 组从浅到深) — 每个 anchor 的 H-step 预测 chunk (H = pred chunk horizon)

obs_image: 每个相机 key 一个 Spatial2DView (修复多相机覆盖; UMI 1 相机 → 1 视图, 双臂 3 相机 → 3 视图)。

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

# 默认维度名 (UMI 单臂 raw): pos(3) + quat xyzw(4) + gripper(1) = 8。
# 仅当 log_episode_to_rerun 未传 dim_names 时使用; 其它 embodiment (如 RoboDojo 双臂 16 维)
# 由各自 stage2 传入显式 dim_names。
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
    dim_names: list[str] | None = None,
    grid_columns: int | None = None,
) -> Path:
    """把 episode eval 的 trajectory + chunks 写入 Rerun 录制 + blueprint。

    dim_names: 每个 raw 动作维度的名字 (顺序对齐 state/action 最后一维)。None → 用
        模块级 RAW_DIM_NAMES (UMI 8 维)。长度必须等于 npz 的 raw 维度, 否则抛 ValueError
        (杜绝静默截断/越界)。
    grid_columns: blueprint grid 列数。None → ≤9 维 3 列, 否则 4 列。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(application_id, default_enabled=True)
    rr.save(str(output_path))

    # ── 维度名解析 (None → UMI 默认 8 维) ─────────────────────────────
    names = list(dim_names) if dim_names is not None else list(RAW_DIM_NAMES)

    # ── 顶层元信息 ────────────────────────────────────────────────────
    rr.log("episode/info", rr.TextLog(_format_metadata(metadata)), static=True)
    rr.log(
        "episode/schema",
        rr.TextLog(
            f"predictions_schema_version={PREDICTIONS_SCHEMA_VERSION} "
            f"metrics_schema_version={METRICS_SCHEMA_VERSION} "
            f"n_dims={len(names)} n_bins={N_BINS} horizon_step={horizon_step} binning=stride7"
        ),
        static=True,
    )

    anchors = np.asarray(results["anchors"])                       # (N,)
    state_at_anchor = np.asarray(results["state_at_anchor"])      # (N, D) raw
    action_gt = np.asarray(results["action_gt"])                  # (N, D) raw
    pred_chunks = np.asarray(results["action_pred_chunks"])       # (N, H, D) raw
    n_samples, horizon, pred_dim = pred_chunks.shape

    # 维度校验: npz 的 raw 维度必须 == len(names), 否则 dim 索引越界或静默截断。
    _assert_dim(state_at_anchor, len(names), "state_at_anchor")
    _assert_dim(action_gt, len(names), "action_gt")
    if pred_dim != len(names):
        raise ValueError(
            f"action_pred_chunks 最后一维 ({pred_dim}) != len(dim_names) ({len(names)}); "
            f"dim_names 与 npz raw 维度不一致"
        )

    # ── 每个 dim 各画 obs / gt / pred_bin_X/anchor_Y ──────────────────
    # 用 send_columns 批量 log, 避免每帧一次 rr.log 的 Python 调用开销
    # (旧版 32k 调用 ~5 min, 批量后 ~1.8k 调用 ~5s)
    for dim_idx, dim_name in enumerate(names):
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
    cam_keys: list[str] = []
    if obs_images:
        for cam_key, frames in obs_images.items():
            frames = np.asarray(frames)
            if frames.shape[0] != n_samples:
                raise ValueError(
                    f"obs_images['{cam_key}'] has {frames.shape[0]} frames but results has "
                    f"{n_samples} anchors; sidecar 跟 predictions.npz 不同源"
                )
            # entity 名去掉 'video.' 前缀: video.cam_wrist_0 → cam_wrist_0
            cam = cam_key.replace("video.", "")
            entity = f"/obs_image/{cam}"
            for i in range(n_samples):
                rr.set_time("anchor_frame", sequence=int(anchors[i]))
                rr.log(entity, rr.Image(frames[i]))
            cam_keys.append(cam)
        rr.reset_time()

    # ── blueprint: 每个 dim 一个 TimeSeriesView (+ 每相机一个 obs_image 视图) ──
    # TimeSeriesView 无 aspect lock, y 轴自适应数据范围, 不会被 x 轴 scale 压扁
    ncols = grid_columns if grid_columns is not None else (3 if len(names) <= 9 else 4)
    contents = [
        rrb.TimeSeriesView(
            origin=f"/{dim_name}",
            contents=[f"$origin/**"],
            name=dim_name,
        )
        for dim_name in names
    ]
    # 每个相机单独一个 Spatial2DView (修复多相机在同一视图互相覆盖; 0 相机 → 不加)。
    for cam in cam_keys:
        contents.append(
            rrb.Spatial2DView(
                origin=f"/obs_image/{cam}",
                contents=["$origin/**"],
                name=cam,
            )
        )
    blueprint = rrb.Blueprint(rrb.Grid(contents=contents, grid_columns=ncols))
    rr.send_blueprint(blueprint)

    return output_path


def _assert_dim(arr: np.ndarray, expected: int, name: str) -> None:
    """校验 2D 数组第二维 == expected, 否则抛 ValueError。"""
    actual = arr.shape[1] if arr.ndim >= 2 else 1
    if actual != expected:
        raise ValueError(
            f"{name} 第二维 ({actual}) != len(dim_names) ({expected}); "
            f"dim_names 与 npz raw 维度不一致"
        )


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
