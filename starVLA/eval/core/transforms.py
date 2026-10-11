"""评测用数学 / 几何变换工具。

metrics、viz 共用。所有函数都是无状态纯函数。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def rotation_error_degrees(pred_xyzw: np.ndarray, gt_xyzw: np.ndarray) -> np.ndarray:
    """两组 xyzw quaternion 之间的测地角(度数)。

    处理 quaternion 双覆盖问题: ``q`` 和 ``-q`` 表示同一个旋转,
    所以先对点积取绝对值再 arccos,避免符号差异被误判成 180 度误差。

    Args:
        pred_xyzw: (..., 4) 预测 quaternion,xyzw 顺序。
        gt_xyzw:   (..., 4) 真值 quaternion,同顺序。

    Returns:
        (...,) 形状的 float64 数组,每个元素是 [0, 180] 度之间的角度。
    """
    pred = pred_xyzw / np.clip(np.linalg.norm(pred_xyzw, axis=-1, keepdims=True), 1e-12, None)
    gt = gt_xyzw / np.clip(np.linalg.norm(gt_xyzw, axis=-1, keepdims=True), 1e-12, None)
    dots = np.clip(np.abs(np.sum(pred * gt, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def align_quaternion_sign_slice(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """对齐 quaternion 符号(双覆盖),输入是显式的 ``(..., 4)`` quaternion 切片。

    quaternion 双覆盖: ``q`` 和 ``-q`` 表示同一旋转, 4 个分量可能整体差一个负号。
    比较/绘图前先对齐符号, 否则会出现"几何等价但分量误差很大"的假阳性。

    跟 ``align_quaternion_sign`` 同语义, 但作用于任意 leading shape、末维 = 4 的
    quaternion 切片 —— **不假设外层是 8 维 ``[pos, quat, gripper]``**。表示无关的
    评测路径里, 调用方先按 representation spec 定位到 rotation 切片, 再调本函数。
    """
    aligned = prediction.copy()
    # 逐行点积 < 0 → 翻号。keepdims 让 sign 广播到末维 4, 对 (4,) 和 (H, 4) 都成立。
    sign = np.where(
        np.sum(aligned * ground_truth, axis=-1, keepdims=True) < 0, -1.0, 1.0
    )
    return aligned * sign


def align_quaternion_sign(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """[legacy] 对齐 8 维 absolute action 向量里 ``[3:7]`` 的 quaternion 符号。

    保留给外部 8 维 ``[pos(3), quat(4), gripper(1)]`` 的调用方; 内部委托给
    ``align_quaternion_sign_slice``。新代码(表示无关路径)请直接用切片版。
    """
    aligned = prediction.copy()
    aligned[3:7] = align_quaternion_sign_slice(prediction[3:7], ground_truth[3:7])
    return aligned


# 7 维物理量分组: pos(3) + euler(3) + gripper(1)
PHYSICAL_DIM_NAMES = ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw", "gripper"]


def action_to_physical_7d(actions_xyzw_gripper: np.ndarray) -> np.ndarray:
    """把 8 维 [x,y,z,qx,qy,qz,qw,gripper] 转成 7 维 [x,y,z,roll,pitch,yaw,gripper]。

    quaternion → euler (XYZ convention = roll/pitch/yaw) 转换, 用 scipy。
    输入 shape: (..., 8)。输出 shape: (..., 7)。

    用于可视化把 4 维 quaternion 压成 3 维 euler, 跟 pos/gripper 合成 7 组。
    """
    from scipy.spatial.transform import Rotation as R
    arr = np.asarray(actions_xyzw_gripper, dtype=np.float64)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected last dim 8, got {arr.shape}")
    pos = arr[..., :3]
    quat_xyzw = arr[..., 3:7]
    gripper = arr[..., 7:8]
    # scipy 要求 xyzw 顺序 (跟我们数据一致)
    euler = R.from_quat(quat_xyzw.reshape(-1, 4)).as_euler("xyz").reshape(quat_xyzw.shape[:-1] + (3,))
    return np.concatenate([pos, euler, gripper], axis=-1)
