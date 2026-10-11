# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import random
from typing import Any, ClassVar

import numpy as np
import pytorch3d.transforms as pt
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ..schema import DatasetMetadata, RotationType, StateActionMetadata
from .base import InvertibleModalityTransform, ModalityTransform


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    """数据归一化器，支持多种归一化模式。

    Norm 计算方式:
    1. min_max 模式 (UMI, RoboChallenge 等使用):
       - 公式: normalized = 2 * (x - min) / (max - min) - 1
       - 输出范围: [-1, 1]
       - 统计来源: calculate_dataset_statistics() 或 _calculate_umi_statistics()

    2. q99 模式:
       - 公式: normalized = 2 * (x - q01) / (q99 - q01) - 1
       - 输出范围: [-1, 1] (裁剪到 [-2.2, 2.2])
       - 更鲁棒，对异常值不敏感

    3. mean_std 模式:
       - 公式: normalized = (x - mean) / std
       - 输出范围: 无固定范围

    4. binary 模式:
       - 公式: normalized = (x > threshold)
       - 输出范围: {0, 1}
    """
    valid_modes = ["q99", "mean_std", "min_max", "binary"]

    def __init__(self, mode: str, statistics: dict, binary_threshold: float = 0.5):
        self.mode = mode
        self.statistics = statistics
        self.binary_threshold = binary_threshold
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -2.2, 2.2)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Min-Max 归一化模式 - UMI, RoboChallenge 等使用此模式
            # 输出范围: [-1, 1]
            # 公式: normalized = 2 * (x - min) / (max - min) - 1
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (
                max[..., mask] - min[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > self.binary_threshold).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化，将归一化后的数据恢复到原始范围。

        各模式的反归一化公式:
        1. min_max: x = (normalized + 1) / 2 * (max - min) + min
        2. q99: x = (normalized + 1) / 2 * (q99 - q01) + q01
        3. mean_std: x = normalized * std + mean
        4. binary: x = (normalized > threshold)

        用于推理时将模型输出的归一化 action 转换回实际 action。
        """
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean
        elif self.mode == "min_max":
            # Min-Max 反归一化
            # 公式: x = (normalized + 1) / 2 * (max - min) + min
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (max - min) + min
        elif self.mode == "binary":
            return (x > self.binary_threshold).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class PerStepNormalizer:
    """逐时间步、逐维度的 min_max 归一化器，统计量形状为 ``(H, D)``。

    与 :class:`Normalizer`（per-dim ``(D,)`` 统计量，所有时间步共用一组 min/max）的区别：
    action chunk 的每个时间步 h 有独立的 min/max，归一化在 ``(H, D)``（或带前导 batch 的
    ``(..., H, D)``）上做**纯元素级**运算。

    设计要点：
    - **不使用布尔 mask 高级索引**（``Normalizer`` 的 ``x[..., mask]`` 对 2D mask 会被
      当成 advanced indexing 压成 1D），改用 ``torch.where`` 做元素级 min_max，对任意
      前导维度（batch 等）广播。
    - ``min == max`` 的退化格：forward 输出 0，inverse 输出 min（无除法，安全），
      round-trip 一致（raw=min → 0 → min）。
    - 支持 ``min_max``（默认）与 ``q99``（q01/q99 边界，离群鲁棒）；二者 round-trip 都精确，
      ``q99`` 下落在 q01/q99 之外的离群点会映射到 [-1,1] 之外（不裁剪，保持可逆）。
    """

    def __init__(self, statistics: dict, mode: str = "min_max"):
        # statistics 形如 {"min": [H,D], "max": [H,D], "q01": ..., "q99": ..., ...}
        # mode: "min_max" 用 min/max 边界；"q99" 用 q01/q99 边界（离群鲁棒，主体铺满 [-1,1]）。
        assert mode in ("min_max", "q99"), f"unsupported per-step norm mode: {mode}"
        self.mode = mode
        self.statistics = {k: torch.as_tensor(v) for k, v in statistics.items()}

    def _bounds(self):
        """返回当前模式下的 (lo, hi) 边界张量。"""
        if self.mode == "q99":
            return self.statistics["q01"], self.statistics["q99"]
        return self.statistics["min"], self.statistics["max"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化：``2*(x-lo)/(hi-lo)-1``，逐 (h, d) 元素。lo/hi 由 mode 决定。"""
        assert isinstance(x, torch.Tensor), (
            f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        )
        lo, hi = self._bounds()
        lo = lo.to(x.dtype)
        hi = hi.to(x.dtype)
        denom = hi - lo  # (H, D)
        safe = denom != 0
        denom_safe = torch.where(safe, denom, torch.ones_like(denom))
        scaled = 2 * (x - lo) / denom_safe - 1
        return torch.where(safe, scaled, torch.zeros_like(x))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化：``(x+1)/2*(hi-lo)+lo``，纯广播、无除法（退化格安全），round-trip 精确。"""
        assert isinstance(x, torch.Tensor), (
            f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        )
        lo, hi = self._bounds()
        lo = lo.to(x.dtype)
        hi = hi.to(x.dtype)
        return (x + 1) / 2 * (hi - lo) + lo


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(
        default_factory=dict, description="The input dtypes for each state key."
    )
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    """状态/动作数据变换器 - 核心数据预处理组件。

    功能概述:
    -----------
    这是 VLA 模型数据预处理的核心组件，负责将原始的 state/action 数据
    转换为模型可用的格式。主要处理两个任务:

    1. 旋转表示转换 (Rotation Transform):
       - 支持多种旋转表示: axis_angle, quaternion, rotation_6d, euler_angles, matrix
       - 例: 将四元数转换为 rotation_6d (更适合神经网络学习)
       - 使用 RotationTransform 类实现双向转换

    2. 数据归一化 (Normalization):
       - min_max: 归一化到 [-1, 1]，UMI/RoboChallenge 使用
       - mean_std: 标准化，无固定范围
       - q99: 基于 1%/99% 分位数归一化，对异常值鲁棒
       - binary: 二值化，用于离散状态
       - 使用 Normalizer 类实现双向归一化

    使用流程:
    ---------
    1. 初始化: 指定 apply_to, normalization_modes, target_rotations
    2. set_metadata(): 加载统计信息 (mean, std, min, max 等)
    3. apply(): 训练时应用变换 (旋转 + 归一化)
    4. unapply(): 推理时逆向变换 (反归一化 + 逆旋转)

    数据流向:
    ---------
    原始数据 (numpy) → StateActionToTensor → StateActionTransform → 模型输入
                                            ↑
                                    旋转转换 + 归一化

    示例 (UMI):
    -----------
    >>> transform = StateActionTransform(
    ...     apply_to=["state.end_0_position", "action.end_0_position"],
    ...     normalization_modes={k: "min_max" for k in state_keys + action_keys},
    ... )
    >>> transform.set_metadata(dataset_metadata)  # 加载统计信息
    >>> transformed_data = transform.apply(data)  # 应用变换

    Args:
        apply_to (list[str]): 需要处理的键列表，如 ["state.joint_pos", "action.ee_pos"]
        normalization_modes (dict[str, str]): 每个键的归一化模式
            - "min_max": 归一化到 [-1, 1]
            - "mean_std": 标准化
            - "q99": 分位数归一化
            - "binary": 二值化
            如果键不在字典中，则不进行归一化
        target_rotations (dict[str, str]): 旋转表示的目标格式
            - "rotation_6d": 6D 连续旋转表示 (推荐)
            - "quaternion": 四元数
            - "axis_angle": 轴角
            - "euler_angles_XXX": 欧拉角 (XXX 为约定如 XYZ)
            如果键不在字典中，则不进行旋转转换
    """

    # ========== 可配置属性 (用户通过参数设置) ==========
    apply_to: list[str] = Field(..., description="需要处理的键列表，如 ['state.joint_pos', 'action.ee_pos']")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="每个键的归一化模式: {'state.joint_pos': 'min_max', ...}"
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="旋转表示的目标格式: {'state.ee_rot': 'rotation_6d', ...}"
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="每个键的统计信息 (由 set_metadata 自动填充)"
    )
    binary_threshold: float = Field(
        default=0.5, description="二值化模式的阈值"
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="模态元数据 (由 set_metadata 自动填充)"
    )

    # ========== 内部变量 (运行时自动初始化) ==========
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    """旋转转换器字典: {key: RotationTransform}"""
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    """归一化器字典: {key: Normalizer}"""
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)
    """输入数据类型字典: {key: dtype}，用于 unapply 时恢复原始类型"""

    # ========== 常量: 预定义的旋转表示统计范围 ==========
    # 这些是各种旋转表示的理论范围，用于归一化
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes", "target_rotations"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(
                    config, StateActionMetadata
                ), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        """验证归一化统计信息的完整性和一致性。

        检查:
        1. min_max 模式需要 min 和 max
        2. mean_std 模式需要 mean 和 std
        3. q99 模式需要 q01 和 q99
        4. binary 模式只需要单个值
        5. 统计量的维度必须一致

        Raises:
            AssertionError: 统计信息不完整或维度不一致
            ValueError: 不支持的归一化模式
        """
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode == "q99":
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """设置元数据并初始化变换器。

        此方法是数据预处理的关键初始化步骤，负责:
        1. 从 DatasetMetadata 中提取统计信息 (mean, std, min, max 等)
        2. 为需要旋转转换的键初始化 RotationTransform
        3. 为需要归一化的键初始化 Normalizer

        数据来源:
        - dataset_metadata.statistics: 包含从数据集计算的统计量
        - dataset_metadata.modalities: 包含每个模态的元数据

        初始化的变换器:
        - self._rotation_transformers: 旋转表示转换器字典
        - self._normalizers: 归一化器字典

        Args:
            dataset_metadata: 数据集元数据，包含统计信息和模态信息
                统计信息由 calculate_dataset_statistics() 或 _calculate_umi_statistics() 计算
                存储在 meta/stats_gr00t.json 文件中
        """
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # 步骤1: 验证并加载模态元数据
        # 确保所有 apply_to 中的键都有对应的元数据配置
        for key in self.apply_to:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, modality
                ), f"{state_key} config not found"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # 步骤2: 加载归一化所需的统计信息
        # 从 dataset_metadata 中提取 mean, std, min, max, q01, q99 等统计量
        for key in self.normalization_modes:
            split_key = key.split(".", 1)
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(
                dataset_statistics, modality
            ), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            # 提取统计信息: {mean: [...], std: [...], min: [...], max: [...], q01: [...], q99: [...]}
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
                state_key
            ].model_dump()

        # 步骤3: 初始化旋转转换器
        # 对于需要旋转表示转换的键 (如 quaternion -> rotation_6d)
        for key in self.target_rotations:
            # 获取原始旋转表示
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"

            # 获取目标旋转表示
            to_rep = RotationType(self.target_rotations[key])

            # 如果原始和目标不同，初始化转换器
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(
                    from_rep=from_rep.value, to_rep=to_rep.value
                )

        # 步骤4: 初始化归一化器
        # 为每个需要归一化的键创建 Normalizer 实例
        for key in self.normalization_modes:
            modality, state_key = key.split(".", 1)
            # 特殊处理: 如果涉及旋转转换，需要使用预定义的统计范围
            if key in self._rotation_transformers:
                # Case 1: 绝对旋转 (如末端姿态)
                if self.modality_metadata[key].absolute:
                    # 绝对旋转必须使用 min_max 模式
                    assert (
                        self.normalization_modes[key] == "min_max"
                    ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # 使用预定义的统计范围 (如 rotation_6d 的 min=[-1]*6, max=[1]*6)
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: 相对旋转 (不支持归一化)
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # 特殊处理: 离散状态必须使用 binary 模式
            elif (
                not self.modality_metadata[key].continuous
                and self.normalization_modes[key] != "binary"
            ):
                raise ValueError(
                    f"{key} is not continuous, so it should be normalized using `binary` mode"
                )
            # 标准处理: 使用从数据集计算的统计信息
            else:
                statistics = self.normalization_statistics[key]
            # 创建 Normalizer 实例并存储
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key], statistics=statistics,
                binary_threshold=self.binary_threshold,
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """应用数据变换 (训练时使用)。

        处理流程:
        1. 验证输入数据类型 (必须是 torch.Tensor)
        2. 应用旋转表示转换 (如果配置了 target_rotations)
        3. 应用归一化 (如果配置了 normalization_modes)

        变换顺序很重要: 先旋转，后归一化
        - 因为归一化使用的是旋转后表示的统计范围
        - 例如: quaternion 先转成 rotation_6d，再用 [-1, 1] 归一化

        Args:
            data: 字典，键为 "state.xxx" 或 "action.xxx"，值为 torch.Tensor
                输入数据通常已经是 tensor (由 StateActionToTensor 转换)

        Returns:
            dict[str, Any]: 变换后的数据字典，值被旋转和归一化

        示例:
            >>> data = {"state.end_0_position": tensor([[0.1, 0.2, 0.3]])}
            >>> transform.apply(data)
            # 如果 min=[0, 0, 0], max=[1, 1, 1]
            # 输出: {"state.end_0_position": tensor([[-0.8, -0.6, -0.4]])}
        """
        for key in self.apply_to:
            if key not in data:
                # 允许某些键缺失，只处理存在的键
                continue
            # 记录并验证输入数据类型
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"

            # 步骤1: 旋转表示转换 (如果需要)
            state = data[key]
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].forward(state)

            # 步骤2: 归一化 (如果需要)
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)

            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """逆向变换 (推理时使用)。

        将模型输出恢复到原始数据空间:
        1. 反归一化: 从 [-1, 1] 恢复到原始范围
        2. 逆旋转: 从目标表示恢复到原始表示
        3. 类型转换: 恢复到原始数据类型

        顺序与 apply 相反: 先反归一化，后逆旋转

        用于推理时将模型输出的归一化 action 转换为实际控制指令。

        Args:
            data: 字典，键为 "state.xxx" 或 "action.xxx"，值为归一化后的 tensor

        Returns:
            dict[str, Any]: 反变换后的数据字典，值在原始数据空间

        示例:
            >>> data = {"action.ee_pos": tensor([[-0.5, 0.0, 0.5]])}  # 归一化后
            >>> transform.unapply(data)
            # 如果 min=[0, 0, 0], max=[1, 1, 1]
            # 输出: {"action.ee_pos": tensor([[0.25, 0.5, 0.75]])}  # 原始范围
        """
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"

            # 步骤1: 反归一化 (从 [-1, 1] 恢复到原始范围)
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)

            # 步骤2: 逆旋转 (从目标表示恢复到原始表示)
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)

            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"

            # 步骤3: 恢复原始数据类型 (如果已知)
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class NormalizationTransform(InvertibleModalityTransform):
    """独立的归一化变换（只做归一化，不做旋转表示转换）。

    与 ``StateActionTransform`` 的区别：
    - ``StateActionTransform`` 同时处理 **旋转表示转换 + 归一化**
      （如 quaternion → rotation_6d 再 min_max）。
    - ``NormalizationTransform`` **只做归一化**，旋转表示转换交给其他
      transform（如 ``QuaternionTo6DRotation``、``RelativePoseTransform``）。

    典型使用场景：当数据 pipeline 已经用独立的 transform 完成了 6D 旋转 /
    SE(3) 相对位姿处理后，只需要一个纯粹的 min_max 归一化步骤时。

    支持的归一化模式（与 ``Normalizer`` 一致）：
    - ``min_max``: 归一化到 [-1, 1]，公式 ``2 * (x - min) / (max - min) - 1``
    - ``q99``: 基于 1%/99% 分位数归一化（对异常值鲁棒）
    - ``mean_std``: 标准化 ``(x - mean) / std``
    - ``binary``: 二值化 ``(x > threshold)``
    """

    apply_to: list[str] = Field(..., description="需要归一化的键列表，如 ['state.joint_pos', 'action.eef_pos']")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict,
        description="每个键的归一化模式: {'state.joint_pos': 'min_max', ...}。"
        "未在字典中出现的键不会被归一化。",
    )
    binary_threshold: float = Field(default=0.5, description="binary 模式的阈值")

    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    """归一化器字典: {key: Normalizer}，由 set_metadata 自动填充"""

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """从数据集元数据加载统计量并初始化 Normalizer。

        与 ``StateActionTransform.set_metadata`` 的区别：
        - 不读取 ``modality_metadata``（不需要 rotation_type / absolute 等信息）
        - 不构建 ``_rotation_transformers``
        - 只为 ``normalization_modes`` 中的键创建 ``Normalizer``
        """
        dataset_statistics = dataset_metadata.statistics

        for key, mode in self.normalization_modes.items():
            split_key = key.split(".", 1)
            assert len(split_key) == 2, f"State key should have 'modality.subkey' format: {key}"
            modality, subkey = split_key

            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            stats_per_modality = getattr(dataset_statistics, modality)
            assert subkey in stats_per_modality, (
                f"{subkey} statistics not found in {modality}. "
                f"Available: {list(stats_per_modality.keys())}"
            )
            statistics = stats_per_modality[subkey].model_dump()

            self._normalizers[key] = Normalizer(
                mode=mode,
                statistics=statistics,
                binary_threshold=self.binary_threshold,
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """对 apply_to 中的每个键应用归一化（训练时使用）。"""
        for key in self.apply_to:
            if key not in data or key not in self._normalizers:
                continue
            value = data[key]
            assert isinstance(value, torch.Tensor), (
                f"Expected torch.Tensor for {key}, got {type(value)}. "
                f"请先使用 StateActionToTensor 将 numpy 转为 tensor。"
            )
            data[key] = self._normalizers[key].forward(value)
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """反归一化（推理时使用），将模型输出恢复到原始数据范围。"""
        for key in self.apply_to:
            if key not in data or key not in self._normalizers:
                continue
            value = data[key]
            assert isinstance(value, torch.Tensor), (
                f"Expected torch.Tensor for {key}, got {type(value)}"
            )
            data[key] = self._normalizers[key].inverse(value)
        return data


class PerStepNormalizationTransform(InvertibleModalityTransform):
    """逐时间步 min_max 归一化（统计量 ``(H, D)``），用于 action chunk。

    结构镜像 :class:`NormalizationTransform`，但内部用 :class:`PerStepNormalizer`
    处理 ``(H, D)`` 统计量（每个时间步独立 min/max）。**不修改** ``NormalizationTransform``
    —— 需要传统 per-dim ``(D,)`` 归一化（state、或 A/B 对照）时仍用原类。

    ``_per_step_normalizers`` 为空时 ``apply``/``unapply`` 透传（与原类同机制），
    这保证在线算 norm 时（``set_metadata`` 尚未执行）``self.transforms.apply``
    仍输出 pre-normalization 值，供统计量计算使用。
    """

    apply_to: list[str] = Field(..., description="需要逐时间步归一化的键列表（通常是 action 子键）。")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict,
        description="每个键的归一化模式：'min_max' 或 'q99'（q01/q99 边界，离群鲁棒）。未出现的键不归一化。",
    )

    _per_step_normalizers: dict[str, PerStepNormalizer] = PrivateAttr(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """从数据集元数据加载 ``(H, D)`` 统计量并初始化 PerStepNormalizer。"""
        dataset_statistics = dataset_metadata.statistics
        for key, mode in self.normalization_modes.items():
            split_key = key.split(".", 1)
            assert len(split_key) == 2, f"State key should have 'modality.subkey' format: {key}"
            modality, subkey = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            stats_per_modality = getattr(dataset_statistics, modality)
            assert subkey in stats_per_modality, (
                f"{subkey} statistics not found in {modality}. "
                f"Available: {list(stats_per_modality.keys())}"
            )
            statistics = stats_per_modality[subkey].model_dump()
            self._per_step_normalizers[key] = PerStepNormalizer(statistics=statistics, mode=mode)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """逐时间步归一化（训练时使用）。"""
        for key in self.apply_to:
            if key not in data or key not in self._per_step_normalizers:
                continue
            value = data[key]
            assert isinstance(value, torch.Tensor), (
                f"Expected torch.Tensor for {key}, got {type(value)}. "
                f"请先使用 StateActionToTensor 将 numpy 转为 tensor。"
            )
            data[key] = self._per_step_normalizers[key].forward(value)
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """逐时间步反归一化（推理时使用）。"""
        for key in self.apply_to:
            if key not in data or key not in self._per_step_normalizers:
                continue
            value = data[key]
            assert isinstance(value, torch.Tensor), (
                f"Expected torch.Tensor for {key}, got {type(value)}"
            )
            data[key] = self._per_step_normalizers[key].inverse(value)
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(
        ..., description="Standard deviation of the noise to be added to the state or action."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data
