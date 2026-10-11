"""开环评测指标计算。

Stage 1 (推理) 调用,训练侧 validation 后续也可复用同一份指标定义。

Schema 版本: dict 形状一旦改就 bump METRICS_SCHEMA_VERSION (定义在 io.py)。
读取方应当显式拒绝版本不匹配的输入,而不是静默算错。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from starVLA.eval.core.io import METRICS_SCHEMA_VERSION
from starVLA.eval.core.transforms import rotation_error_degrees

# 8 维绝对 action 向量的标准维度名:
#   [x, y, z, qx, qy, qz, qw, gripper]
# 顺序跟 eval_starvla._invert_action_to_absolute 的拼接
# (按 dataset.modality_keys["action"]: pos3 + quat4 + gripper1) 保持一致,改这里要同步改。
SOURCE_LABELS = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]


def calculate_episode_metrics(results: dict[str, np.ndarray]) -> dict[str, Any]:
    """把 per-anchor 的 results 聚合成 per-episode 指标。

    Args:
        results: ``replay_episode`` 返回值, 必须包含:
            ``prediction``, ``ground_truth`` (N, 8) 绝对 action,
            ``normalized_prediction``, ``normalized_ground_truth`` (N, action_dim),
            ``latency_seconds`` (N,)。

    Returns:
        嵌套 dict。顶层 key:
            ``source_per_dimension``: 8 维绝对 action 每一维的 MAE/RMSE。
            ``position``: ADE / RMSE_xyz (单位米)。
            ``rotation``: mean / rmse / max 测地角 (单位度)。
            ``gripper``: MAE / RMSE。
            ``normalized_selected_horizon``: 选中 horizon step 的 normalized 空间 MAE/RMSE。
            ``latency_seconds``: mean / median / total 推理延迟。
    """
    prediction = results["prediction"]
    ground_truth = results["ground_truth"]
    error = prediction - ground_truth
    normalized_error = (
        results["normalized_prediction"] - results["normalized_ground_truth"]
    )
    # position L2 per-anchor, ADE 是它的平均
    position_l2 = np.linalg.norm(error[:, :3], axis=-1)
    # rotation 测地角 (degrees),双覆盖问题在 rotation_error_degrees 内部处理
    rotation_error = rotation_error_degrees(prediction[:, 3:7], ground_truth[:, 3:7])
    return {
        "__schema_version": METRICS_SCHEMA_VERSION,
        # 每一维分开报,便于看是哪一轴偏了 (比如只 z 漂)
        "source_per_dimension": {
            label: {
                "mae": float(np.mean(np.abs(error[:, dim]))),
                "rmse": float(np.sqrt(np.mean(np.square(error[:, dim])))),
            }
            for dim, label in enumerate(SOURCE_LABELS)
        },
        # position 重点指标: ADE = Average Displacement Error (整段轨迹平均 L2)
        "position": {
            "ade": float(np.mean(position_l2)),
            "rmse_xyz": np.sqrt(np.mean(np.square(error[:, :3]), axis=0)),
        },
        # rotation 重点指标: 测地角的 mean / rmse / max
        "rotation": {
            "mean_degrees": float(np.mean(rotation_error)),
            "rmse_degrees": float(np.sqrt(np.mean(np.square(rotation_error)))),
            "max_degrees": float(np.max(rotation_error)),
        },
        # gripper 一维,简单 MAE / RMSE
        "gripper": {
            "mae": float(np.mean(np.abs(error[:, 7]))),
            "rmse": float(np.sqrt(np.mean(np.square(error[:, 7])))),
        },
        # normalized 空间指标,跟训练 loss 空间一致, 用于跟训练 RMSE 对比
        # 注意: 这里只取 selected horizon_step 那一步, 不是全 chunk 平均
        "normalized_selected_horizon": {
            "mae": float(np.mean(np.abs(normalized_error))),
            "rmse": float(np.sqrt(np.mean(np.square(normalized_error)))),
        },
        # 延迟统计,用于评估推理是否达到部署要求 (UMI 一般 5-10Hz)
        "latency_seconds": {
            "mean": float(np.mean(results["latency_seconds"])),
            "median": float(np.median(results["latency_seconds"])),
            "total": float(np.sum(results["latency_seconds"])),
        },
    }
