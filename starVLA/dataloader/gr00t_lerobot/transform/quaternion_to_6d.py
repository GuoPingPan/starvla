# SPDX-FileCopyrightText: Copyright (c) 2025 starVLA community
# SPDX-License-Identifier: MIT

"""
Quaternion -> rotation_6d representation transform.

将分离的纯旋转四元数 (4D) 转换为 6D 连续旋转表示, 专门处理 *不含 position* 的
rotation key —— 与 :class:`PoseRotationTransform` 处理完整 7D pose (位置 + 旋转)
互补。6D 表示对 ``q`` / ``-q`` 的双覆盖不敏感, 更适合神经网络学习。

Forward:  ``[..., 4]`` quaternion  -> ``[..., 6]`` rotation_6d
Backward: ``[..., 6]`` rotation_6d -> ``[..., 4]`` quaternion  (canonicalized, w >= 0)

数据布局遵循 starVLA 的 FLAT 约定: ``data["state.end_0_rotation"]`` (整串带点的
key 即字典 key)。本转换可逆。
"""

from typing import List

import numpy as np
import torch
from pydantic import Field

from .base import InvertibleModalityTransform
from .rotation_utils import (
    quaternion_to_rotation_6d_jit,
    rotation_6d_to_quaternion_jit,
)


class QuaternionTo6DRotation(InvertibleModalityTransform):
    """将分离的四元数 (4D) 转换为 6D 旋转表示。

    专门处理不含 position 的纯 rotation key（与 :class:`PoseRotationTransform`
    处理完整 7D pose 不同）。

    Args:
        apply_to: 要转换的 key 列表。
        input_convention: 输入四元数的存储顺序。
            - "wxyz"（默认, **兼容旧行为**）: 直接传给 ``quaternion_to_rotation_6d_jit``。
              注意: 如果数据实际是 xyzw, 结果不是真正的几何 rotation_6d, 但仍在
              [-1, 1] 范围内, 可被 min_max 归一化。
            - "xyzw": 先把输入从 (x, y, z, w) 重排为 (w, x, y, z), 再调用 jit
              函数。这是几何正确的转换, 推荐新 pipeline 使用。
    """

    input_convention: str = Field(default="wxyz")

    def __init__(self, apply_to: List[str], input_convention: str = "wxyz", **kwargs):
        super().__init__(apply_to=apply_to, input_convention=input_convention, **kwargs)

    def model_dump(self, *args, **kwargs):
        """序列化转换配置。"""
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "input_convention"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    def _to_wxyz(self, q: torch.Tensor) -> torch.Tensor:
        if self.input_convention == "xyzw":
            return q[..., [3, 0, 1, 2]]
        return q

    def _from_wxyz(self, q: torch.Tensor) -> torch.Tensor:
        if self.input_convention == "xyzw":
            return q[..., [1, 2, 3, 0]]
        return q

    def apply(self, data: dict) -> dict:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if isinstance(value, torch.Tensor) and value.shape[-1] == 4:
                data[key] = quaternion_to_rotation_6d_jit(self._to_wxyz(value.float()))
        return data

    def unapply(self, data: dict) -> dict:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, torch.Tensor) and value.shape[-1] == 6:
                q_wxyz = rotation_6d_to_quaternion_jit(value.float())
                data[key] = self._from_wxyz(q_wxyz)
        return data
