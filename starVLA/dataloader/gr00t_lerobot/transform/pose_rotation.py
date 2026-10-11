# SPDX-FileCopyrightText: Copyright (c) 2025 starVLA community
# SPDX-License-Identifier: MIT

"""
Pose rotation representation transforms.

Converts between quaternion/euler and rotation_6d/rotation_9d representations.
Compatible with starVLA's ModalityTransform architecture.
"""

from typing import Literal
from copy import deepcopy

import torch
from pydantic import Field

from .base import InvertibleModalityTransform
from ..schema import RotationType
from .rotation_utils import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_rotation_9d,
    rotation_9d_to_matrix,
    quaternion_to_rotation_6d_jit,
    quaternion_to_rotation_9d_jit,
    rotation_6d_to_quaternion_jit,
)


class PoseRotationTransform(InvertibleModalityTransform):
    """
    Transform pose rotation representations between quaternion/euler and rotation_6d/rotation_9d.

    Forward: input_rotation_type → rotation_6d/rotation_9d
    Backward: rotation_6d/rotation_9d → input_rotation_type

    Input pose layout:
      - 7D (quaternion): [x, y, z, quat_x, quat_y, quat_z, quat_w]
      - 6D (euler): [x, y, z, euler_x, euler_y, euler_z]

    Output pose layout:
      - 9D (rotation_6d): [x, y, z, rotation_6d]
      - 12D (rotation_9d): [x, y, z, rotation_9d]

    Note: Gripper should be stored in a separate key (e.g., left_gripper), not in ee_pose.
    This transform is invertible.

    Args:
        apply_to: List of keys to apply this transform to (e.g., ["state.eef_pose", "action.eef_pose"]).
        rotation_type: Target rotation representation ("rotation_6d" or "rotation_9d").
        input_rotation_type: Input rotation representation ("quaternion" or "euler").
        euler_convention: Euler angle convention (e.g., "XYZ"). Only used if input_rotation_type="euler".
        fast_forward: Whether to use JIT-compiled forward functions (default: True).
    """

    # Configuration fields
    rotation_type: Literal["quaternion", "rotation_6d", "rotation_9d"] = Field(
        ...,
        description="Target rotation representation.",
    )
    input_rotation_type: Literal["quaternion", "euler"] = Field(
        default="quaternion",
        description="Input rotation representation.",
    )
    euler_convention: str = Field(
        default="XYZ",
        description="Euler angle convention (e.g., 'XYZ'). Only used if input_rotation_type='euler'.",
    )
    fast_forward: bool = Field(
        default=True,
        description="Whether to use JIT-compiled functions for forward/backward.",
    )

    # Rotation dimension mappings: (total_pose_dim, rot_dim)
    _ROT_DIMS = {
        "quaternion": (7, 4),
        "euler": (6, 3),
        "rotation_6d": (9, 6),
        "rotation_9d": (12, 9),
    }

    def model_dump(self, *args, **kwargs):
        """Serialize transform configuration."""
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "rotation_type", "input_rotation_type", "euler_convention"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include)

    def apply(self, data: dict) -> dict:
        """Apply rotation transformation to all specified keys."""
        out_data = deepcopy(data)
        for key in self.apply_to:
            # Handle nested structure: data["modality"]["full_key"]
            if "." in key:
                parts = key.split(".", 1)
                modality = parts[0]
                full_key = key
                if modality in out_data and full_key in out_data[modality]:
                    if self.fast_forward:
                        out_data[modality][full_key] = self._forward_fast(out_data[modality][full_key])
                    else:
                        out_data[modality][full_key] = self._forward(out_data[modality][full_key])
            elif key in out_data:
                if self.fast_forward:
                    out_data[key] = self._forward_fast(out_data[key])
                else:
                    out_data[key] = self._forward(out_data[key])
        return out_data

    def unapply(self, data: dict) -> dict:
        """Reverse rotation transformation."""
        out_data = deepcopy(data)
        for key in self.apply_to:
            # Handle nested structure: data["modality"]["full_key"]
            if "." in key:
                parts = key.split(".", 1)
                modality = parts[0]
                full_key = key
                if modality in out_data and full_key in out_data[modality]:
                    if self.fast_forward and out_data[modality][full_key].ndim <= 2:
                        out_data[modality][full_key] = self._backward_fast(out_data[modality][full_key])
                    else:
                        out_data[modality][full_key] = self._backward(out_data[modality][full_key])
            elif key in out_data:
                if self.fast_forward and out_data[key].ndim <= 2:
                    out_data[key] = self._backward_fast(out_data[key])
                else:
                    out_data[key] = self._backward(out_data[key])
        return out_data

    def _forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Convert pose from input_rotation_type to rotation_type.

        Args:
            pose: (..., 6) or (..., 7)
                  6D (euler): [x, y, z, euler_x, euler_y, euler_z]
                  7D (quaternion): [x, y, z, quat_x, quat_y, quat_z, quat_w]

        Returns:
            rotated_pose: (..., 9) for rotation_6d or (..., 12) for rotation_9d
        """
        if self.input_rotation_type == "quaternion":
            assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        elif self.input_rotation_type == "euler":
            assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"

        if self.rotation_type == "quaternion":
            return pose

        position = pose[..., :3]

        if self.input_rotation_type == "quaternion":
            # Convert (x, y, z, qx, qy, qz, qw) to (w, x, y, z) for pytorch3d
            quaternion = pose[..., [6, 3, 4, 5]]
            matrix = quaternion_to_matrix(quaternion)
        elif self.input_rotation_type == "euler":
            euler_angles = pose[..., 3:6]
            matrix = euler_angles_to_matrix(euler_angles, self.euler_convention)
        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

        if self.rotation_type == "rotation_6d":
            rotation = matrix_to_rotation_6d(matrix)
        elif self.rotation_type == "rotation_9d":
            rotation = matrix_to_rotation_9d(matrix)
        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")

        return torch.cat([position, rotation], dim=-1)

    def _backward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation: convert from rotation_type to input_rotation_type.

        Args:
            pose: (..., 9) for rotation_6d or (..., 12) for rotation_9d

        Returns:
            original_pose: (..., 6) for euler or (..., 7) for quaternion
        """
        if self.rotation_type == "quaternion":
            return pose

        if self.rotation_type == "rotation_6d":
            rot_dim = 6
            expected_dim = 9
        elif self.rotation_type == "rotation_9d":
            rot_dim = 9
            expected_dim = 12
        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")

        assert pose.shape[-1] == expected_dim, f"Expected {expected_dim}D pose, got {pose.shape}"

        position = pose[..., :3]
        rotation = pose[..., 3 : 3 + rot_dim]

        if self.rotation_type == "rotation_6d":
            matrix = rotation_6d_to_matrix(rotation)
        elif self.rotation_type == "rotation_9d":
            matrix = rotation_9d_to_matrix(rotation)

        if self.input_rotation_type == "quaternion":
            quaternion = matrix_to_quaternion(matrix)
            # Convert (w, x, y, z) to (x, y, z, qx, qy, qz, qw)
            quaternion = quaternion[..., [1, 2, 3, 0]]
            return torch.cat([position, quaternion], dim=-1)
        elif self.input_rotation_type == "euler":
            euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
            return torch.cat([position, euler_angles], dim=-1)
        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

    def _forward_fast(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward with JIT-compiled rotation conversions.
        For euler input, falls back to non-JIT path.
        """
        if self.input_rotation_type == "quaternion":
            assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        elif self.input_rotation_type == "euler":
            assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"

        if self.rotation_type == "quaternion":
            return pose

        position = pose[..., :3]

        if self.input_rotation_type == "quaternion":
            # Convert (x, y, z, qx, qy, qz, qw) to (w, x, y, z) for JIT
            quaternion = pose[..., [6, 3, 4, 5]]

            if self.rotation_type == "rotation_6d":
                rotation = quaternion_to_rotation_6d_jit(quaternion)
            elif self.rotation_type == "rotation_9d":
                rotation = quaternion_to_rotation_9d_jit(quaternion)
            else:
                raise NotImplementedError

            return torch.cat([position, rotation], dim=-1)

        elif self.input_rotation_type == "euler":
            euler_angles = pose[..., 3:6]
            matrix = euler_angles_to_matrix(euler_angles, self.euler_convention)

            if self.rotation_type == "rotation_6d":
                rotation = matrix_to_rotation_6d(matrix)
            elif self.rotation_type == "rotation_9d":
                rotation = matrix_to_rotation_9d(matrix)
            else:
                raise NotImplementedError

            return torch.cat([position, rotation], dim=-1)

        else:
            raise ValueError(f"Unknown input_rotation_type: {self.input_rotation_type}")

    def _backward_fast(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Optimized backward with JIT-compiled rotation conversions.
        For euler input/output, falls back to non-JIT path.
        """
        if self.rotation_type == "quaternion":
            return pose

        if self.rotation_type == "rotation_6d":
            rot_dim = 6
            expected_dim = 9
        elif self.rotation_type == "rotation_9d":
            rot_dim = 9
            expected_dim = 12
        else:
            raise NotImplementedError

        assert pose.shape[-1] == expected_dim, f"Expected {expected_dim}D pose, got {pose.shape}"

        position = pose[..., :3]
        rotation = pose[..., 3 : 3 + rot_dim]

        if self.rotation_type == "rotation_6d":
            # Use JIT for quaternion conversion
            quaternion = rotation_6d_to_quaternion_jit(rotation)

            if self.input_rotation_type == "quaternion":
                # Convert (w, x, y, z) to (x, y, z, qx, qy, qz, qw)
                quaternion = quaternion[..., [1, 2, 3, 0]]
                return torch.cat([position, quaternion], dim=-1)
            elif self.input_rotation_type == "euler":
                matrix = quaternion_to_matrix(quaternion)
                euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
                return torch.cat([position, euler_angles], dim=-1)

        elif self.rotation_type == "rotation_9d":
            matrix = rotation_9d_to_matrix(rotation)

            if self.input_rotation_type == "quaternion":
                quaternion = matrix_to_quaternion(matrix)
                quaternion = quaternion[..., [1, 2, 3, 0]]
                return torch.cat([position, quaternion], dim=-1)
            elif self.input_rotation_type == "euler":
                euler_angles = matrix_to_euler_angles(matrix, self.euler_convention)
                return torch.cat([position, euler_angles], dim=-1)

        else:
            raise NotImplementedError(f"Unknown rotation_type: {self.rotation_type}")


class EulerPoseToQuaternionTransform(InvertibleModalityTransform):
    """
    Convert Euler-angle pose format to quaternion format before RelativePoseTransform.

    Forward:  [..., 6]  [x, y, z, euler_x, euler_y, euler_z]
           → [..., 7]  [x, y, z, quat_x, quat_y, quat_z, quat_w]

    Backward: [..., 7] → [..., 6]

    Use case: OXE Euler-angle datasets should convert to quaternions before
    RelativePoseTransform, preserving geometric correctness of relative pose
    computation instead of using approximate elementwise differences.

    This transform is invertible.

    Args:
        apply_to: List of keys to apply this transform to.
        euler_convention: Euler angle convention (default: "XYZ").
    """

    euler_convention: str = Field(
        default="XYZ",
        description="Euler angle convention (e.g., 'XYZ').",
    )

    def model_dump(self, *args, **kwargs):
        """Serialize transform configuration."""
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "euler_convention"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include)

    def apply(self, data: dict) -> dict:
        """Convert euler poses to quaternion poses."""
        out_data = deepcopy(data)
        for key in self.apply_to:
            # Handle nested structure: data["modality"]["full_key"]
            if "." in key:
                # key format: "state.eef_pose" or "action.eef_pose"
                # Data is stored as: data["state"]["state.eef_pose"]
                parts = key.split(".", 1)
                modality = parts[0]
                full_key = key  # "state.eef_pose"
                if modality in out_data and full_key in out_data[modality]:
                    out_data[modality][full_key] = self._euler_to_quat(out_data[modality][full_key])
            elif key in out_data:
                out_data[key] = self._euler_to_quat(out_data[key])
        return out_data

    def unapply(self, data: dict) -> dict:
        """Convert quaternion poses back to euler poses."""
        out_data = deepcopy(data)
        for key in self.apply_to:
            # Handle nested structure: data["modality"]["full_key"]
            if "." in key:
                parts = key.split(".", 1)
                modality = parts[0]
                full_key = key
                if modality in out_data and full_key in out_data[modality]:
                    out_data[modality][full_key] = self._quat_to_euler(out_data[modality][full_key])
            elif key in out_data:
                out_data[key] = self._quat_to_euler(out_data[key])
        return out_data

    def _euler_to_quat(self, pose: torch.Tensor) -> torch.Tensor:
        """[x,y,z,ex,ey,ez] → [x,y,z,qx,qy,qz,qw]"""
        assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"
        position = pose[..., :3]
        euler = pose[..., 3:6]
        matrix = euler_angles_to_matrix(euler, self.euler_convention)
        quat_wxyz = matrix_to_quaternion(matrix)
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        return torch.cat([position, quat_xyzw], dim=-1)

    def _quat_to_euler(self, pose: torch.Tensor) -> torch.Tensor:
        """[x,y,z,qx,qy,qz,qw] → [x,y,z,ex,ey,ez]"""
        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        position = pose[..., :3]
        quat_xyzw = pose[..., 3:7]
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        matrix = quaternion_to_matrix(quat_wxyz)
        euler = matrix_to_euler_angles(matrix, self.euler_convention)
        return torch.cat([position, euler], dim=-1)
