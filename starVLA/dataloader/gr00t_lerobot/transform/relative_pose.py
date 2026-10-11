# SPDX-FileCopyrightText: Copyright (c) 2025 starVLA community
# SPDX-License-Identifier: MIT

"""
Relative pose transforms using SE(3) group operations.

Implements geometrically correct relative pose computation using quaternion
multiplication instead of element-wise subtraction. Compatible with starVLA's
ModalityTransform architecture.
"""

from typing import List, Dict
from copy import deepcopy

import torch
from pydantic import Field

from .base import InvertibleModalityTransform
from .rotation_utils import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    quaternion_conjugate_jit,
    quaternion_multiply_jit,
    quaternion_rotate_vector_jit,
)


class RelativePoseTransform(InvertibleModalityTransform):
    """
    Convert absolute poses to relative poses w.r.t. the last state.

    Forward: pose_relative = T_base^{-1} @ pose_absolute
    Backward: pose_absolute = T_base @ pose_relative

    This uses SE(3) group operations for geometrically correct relative pose
    computation, not simple element-wise subtraction.

    Two input modes are supported:

    1. Combined pose key (default): A single key holds the concatenated
       ``[x, y, z, qx, qy, qz, qw]`` (or euler) pose. Use ``apply_to`` and
       ``state_keys_map`` as usual.

    2. Split position + rotation keys (``split_pose_keys``): The pose is
       stored as two separate keys in the dataset (e.g. UMI stores
       ``state.end_0_position`` (3,) and ``state.end_0_rotation`` (4,)).
       Provide a virtual pose name in ``apply_to`` and a full mapping in
       ``split_pose_keys``. The transform reads the four source keys
       (action_position / action_rotation / state_position / state_rotation),
       concatenates them internally, runs the SE(3) relative pose, then
       writes the relative position and rotation back to the original split
       keys. The virtual pose name itself is never materialised in ``data``.

    This transform is invertible.

    Args:
        apply_to: List of action keys (combined mode) or virtual pose names
            (split mode) to compute relative poses for.
        state_keys_map: Mapping from action key to corresponding state key
            (combined mode only). If None, assumes state key is action.key
            with "action." → "state." prefix.
        split_pose_keys: Mapping from virtual pose name (must be in
            ``apply_to``) to a spec with keys ``position``, ``rotation``,
            ``state_position``, ``state_rotation``. Each value is the actual
            key in ``data["action"]`` / ``data["state"]``.
        input_rotation_type: Input rotation representation ("quaternion" or "euler").
        euler_convention: Euler angle convention (default: "XYZ").
        fast_forward: Whether to use JIT-compiled functions (default: True).
    """

    state_keys_map: Dict[str, str] | None = Field(
        default=None,
        description="Mapping from action keys to state keys (combined mode). If None, auto-infers.",
    )
    split_pose_keys: Dict[str, Dict[str, str]] | None = Field(
        default=None,
        description=(
            "Mapping from virtual pose name to its split component keys. "
            "Enables split-key mode. Example: "
            '{"action.end_0_pose": {"position": "action.end_0_position", '
            '"rotation": "action.end_0_rotation", "state_position": "state.end_0_position", '
            '"state_rotation": "state.end_0_rotation"}}'
        ),
    )
    input_rotation_type: str = Field(
        default="quaternion",
        description="Input rotation representation ('quaternion' or 'euler').",
    )
    euler_convention: str = Field(
        default="XYZ",
        description="Euler angle convention for euler inputs.",
    )
    fast_forward: bool = Field(
        default=True,
        description="Whether to use JIT-compiled operations.",
    )

    def model_dump(self, *args, **kwargs):
        """Serialize transform configuration."""
        if kwargs.get("mode", "python") == "json":
            include = {
                "apply_to",
                "state_keys_map",
                "split_pose_keys",
                "input_rotation_type",
                "euler_convention",
            }
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include)

    def _get_state_key(self, action_key: str) -> str:
        """Get the corresponding state key for an action key (combined mode)."""
        if self.state_keys_map is not None and action_key in self.state_keys_map:
            return self.state_keys_map[action_key]
        # Auto-infer: action.xxx -> state.xxx
        return action_key.replace("action.", "state.")

    def _concat_pose(self, position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """Concatenate position + rotation into a single pose tensor."""
        return torch.cat([position, rotation], dim=-1)

    def _extract_last_pose(self, state_pose: torch.Tensor) -> torch.Tensor:
        """Extract the last frame (most recent observation) as the base pose."""
        if state_pose.ndim == 2:
            return state_pose[-1:]
        if state_pose.ndim == 3:
            flat = state_pose.reshape(-1, state_pose.shape[-1])
            return flat[-1:]
        return state_pose[..., -1:, :].reshape(-1, state_pose.shape[-1])

    def apply(self, data: dict) -> dict:
        """Apply relative pose transformation.

        Data layout: flat dict mapping full dotted keys (e.g.
        ``"action.end_0_position"``) to tensors, as produced by
        ``LeRobotSingleDataset.get_step_data``.
        """
        out_data = deepcopy(data)

        for action_key in self.apply_to:
            if action_key in self.split_pose_keys:
                self._apply_split(out_data, action_key)
            else:
                self._apply_combined(out_data, action_key)

        return out_data

    def unapply(self, data: dict) -> dict:
        """Reverse relative pose transformation (flat-data layout)."""
        out_data = deepcopy(data)

        for action_key in self.apply_to:
            if action_key in self.split_pose_keys:
                self._unapply_split(out_data, action_key)
            else:
                self._unapply_combined(out_data, action_key)

        return out_data

    def _apply_combined(self, out_data: dict, action_key: str) -> None:
        """Combined-mode apply: read/write a single pose key."""
        if action_key not in out_data:
            return

        state_key = self._get_state_key(action_key)
        if state_key not in out_data:
            return

        action_pose = out_data[action_key]
        state_pose = out_data[state_key]
        base_pose = self._extract_last_pose(state_pose)

        if self.fast_forward:
            out_data[action_key] = self._forward_fast(action_pose, base_pose)
        else:
            out_data[action_key] = self._forward(action_pose, base_pose)

    def _unapply_combined(self, out_data: dict, action_key: str) -> None:
        """Combined-mode unapply."""
        if action_key not in out_data:
            return

        state_key = self._get_state_key(action_key)
        if state_key not in out_data:
            return

        action_pose = out_data[action_key]
        state_pose = out_data[state_key]
        base_pose = self._extract_last_pose(state_pose)

        if self.fast_forward:
            out_data[action_key] = self._backward_fast(action_pose, base_pose)
        else:
            out_data[action_key] = self._backward(action_pose, base_pose)

    def _apply_split(self, out_data: dict, action_key: str) -> None:
        """Split-mode apply: read position+rotation, compute relative, write back split."""
        spec = self.split_pose_keys[action_key]
        pos_key = spec["position"]
        rot_key = spec["rotation"]
        state_pos_key = spec["state_position"]
        state_rot_key = spec["state_rotation"]

        action_pos = out_data.get(pos_key)
        action_rot = out_data.get(rot_key)
        state_pos = out_data.get(state_pos_key)
        state_rot = out_data.get(state_rot_key)

        # If any source is missing, leave keys untouched.
        if action_pos is None or action_rot is None or state_pos is None or state_rot is None:
            return

        action_pose = self._concat_pose(action_pos, action_rot)
        state_pose = self._concat_pose(state_pos, state_rot)
        base_pose = self._extract_last_pose(state_pose)

        if self.fast_forward:
            rel_pose = self._forward_fast(action_pose, base_pose)
        else:
            rel_pose = self._forward(action_pose, base_pose)

        # Split back: position keeps its own dim, rotation stays in input rotation type.
        rel_pos = rel_pose[..., : action_pos.shape[-1]]
        rel_rot = rel_pose[..., action_pos.shape[-1]:]
        out_data[pos_key] = rel_pos
        out_data[rot_key] = rel_rot

    def _unapply_split(self, out_data: dict, action_key: str) -> None:
        """Split-mode unapply: read relative position+rotation, recover absolute, write back split."""
        spec = self.split_pose_keys[action_key]
        pos_key = spec["position"]
        rot_key = spec["rotation"]
        state_pos_key = spec["state_position"]
        state_rot_key = spec["state_rotation"]

        rel_pos = out_data.get(pos_key)
        rel_rot = out_data.get(rot_key)
        state_pos = out_data.get(state_pos_key)
        state_rot = out_data.get(state_rot_key)

        if rel_pos is None or rel_rot is None or state_pos is None or state_rot is None:
            return

        rel_pose = self._concat_pose(rel_pos, rel_rot)
        state_pose = self._concat_pose(state_pos, state_rot)
        base_pose = self._extract_last_pose(state_pose)

        if self.fast_forward:
            abs_pose = self._backward_fast(rel_pose, base_pose)
        else:
            abs_pose = self._backward(rel_pose, base_pose)

        abs_pos = abs_pose[..., : rel_pos.shape[-1]]
        abs_rot = abs_pose[..., rel_pos.shape[-1]:]
        out_data[pos_key] = abs_pos
        out_data[rot_key] = abs_rot

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute relative pose: T_rel = T_base^{-1} @ T_abs.

        Args:
            pose: Action pose (..., T, 7) quaternion or (..., T, 6) euler.
            base_pose: Base state pose (..., 1, 7) or (..., 1, 6).

        Returns:
            Relative pose with same shape as input.
        """
        if self.input_rotation_type == "euler":
            return self._forward_euler(pose, base_pose)

        # Quaternion path
        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Expected 7D quaternion base pose, got {base_pose.shape}"

        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        return self._matrix_to_pose(pose_matrix)

    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Compute absolute pose: T_abs = T_base @ T_rel.

        Args:
            pose: Relative pose (..., T, 7) or (..., T, 6).
            base_pose: Base state pose (..., 1, 7) or (..., 1, 6).

        Returns:
            Absolute pose with same shape as input.
        """
        if self.input_rotation_type == "euler":
            return self._backward_euler(pose, base_pose)

        # Quaternion path
        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Expected 7D quaternion base pose, got {base_pose.shape}"

        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        return self._matrix_to_pose(pose_matrix)

    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
        """Convert pose (xyz + quaternion) to 4x4 homogeneous matrix."""
        position = pose[..., 0:3]
        quaternion = pose[..., [6, 3, 4, 5]]  # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0:3, 0:3] = rotation
        matrix[..., 0:3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor) -> torch.Tensor:
        """Convert 4x4 homogeneous matrix to pose (xyz + quaternion)."""
        position = matrix[..., 0:3, 3]
        rotation = matrix[..., 0:3, 0:3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]]  # (r i j k) to (i j k r)
        return torch.cat([position, quaternion], dim=-1)

    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor) -> torch.Tensor:
        """T_rel = T_base^{-1} @ T_abs

        Handles broadcasting when base_pose has fewer temporal dimensions.
        """
        # Ensure base_pose_matrix has same batch dimensions as pose_matrix
        if base_pose_matrix.ndim < pose_matrix.ndim:
            # Add leading dimensions to match
            shape_diff = pose_matrix.ndim - base_pose_matrix.ndim
            base_pose_matrix = base_pose_matrix.view(
                (1,) * shape_diff + base_pose_matrix.shape
            )

        # Broadcast base to match pose's temporal dimension
        if base_pose_matrix.shape[:-3] != pose_matrix.shape[:-3]:
            base_pose_matrix = base_pose_matrix.expand(
                pose_matrix.shape[:-3] + base_pose_matrix.shape[-3:]
            )

        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor) -> torch.Tensor:
        """T_abs = T_base @ T_rel

        Handles broadcasting when base_pose has fewer temporal dimensions.
        """
        # Ensure base_pose_matrix has same batch dimensions as pose_matrix
        if base_pose_matrix.ndim < pose_matrix.ndim:
            shape_diff = pose_matrix.ndim - base_pose_matrix.ndim
            base_pose_matrix = base_pose_matrix.view(
                (1,) * shape_diff + base_pose_matrix.shape
            )

        # Broadcast base to match pose's temporal dimension
        if base_pose_matrix.shape[:-3] != pose_matrix.shape[:-3]:
            base_pose_matrix = base_pose_matrix.expand(
                pose_matrix.shape[:-3] + base_pose_matrix.shape[-3:]
            )

        return base_pose_matrix @ pose_matrix

    def _forward_euler(self, pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward for euler poses using 3x3 tensor ops.

        Args:
            pose: (..., T, 6) euler pose.
            base_pose: (..., 1, 6) euler base.

        Returns:
            Relative euler pose.
        """
        assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"
        assert base_pose.shape[-1] == 6, f"Expected 6D euler base, got {base_pose.shape}"

        p_target = pose[..., :3]
        e_target = pose[..., 3:6]
        p_base = base_pose[..., :3]
        e_base = base_pose[..., 3:6]

        r_target = euler_angles_to_matrix(e_target, self.euler_convention)
        r_base = euler_angles_to_matrix(e_base, self.euler_convention)
        r_base_t = r_base.transpose(-1, -2)

        delta_p = (r_base_t @ (p_target - p_base).unsqueeze(-1)).squeeze(-1)
        r_rel = r_base_t @ r_target
        e_rel = matrix_to_euler_angles(r_rel, self.euler_convention)
        return torch.cat([delta_p, e_rel], dim=-1)

    def _backward_euler(self, pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Optimized backward for euler poses.

        Args:
            pose: (..., T, 6) relative euler pose.
            base_pose: (..., 1, 6) euler base.

        Returns:
            Absolute euler pose.
        """
        assert pose.shape[-1] == 6, f"Expected 6D euler pose, got {pose.shape}"
        assert base_pose.shape[-1] == 6, f"Expected 6D euler base, got {base_pose.shape}"

        p_rel = pose[..., :3]
        e_rel = pose[..., 3:6]
        p_base = base_pose[..., :3]
        e_base = base_pose[..., 3:6]

        r_rel = euler_angles_to_matrix(e_rel, self.euler_convention)
        r_base = euler_angles_to_matrix(e_base, self.euler_convention)

        p_target = (r_base @ p_rel.unsqueeze(-1)).squeeze(-1) + p_base
        r_target = r_base @ r_rel
        e_target = matrix_to_euler_angles(r_target, self.euler_convention)
        return torch.cat([p_target, e_target], dim=-1)

    def _forward_fast(self, pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward with JIT-compiled quaternion operations.

        For euler inputs, falls back to _forward_euler.
        """
        if self.input_rotation_type == "euler":
            return self._forward_euler(pose, base_pose)

        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Expected 7D quaternion base pose, got {base_pose.shape}"

        # Extract position and quaternion (i, j, k, r format from dataset)
        p_target = pose[..., :3]
        q_target = pose[..., 3:7]

        p_base = base_pose[..., :3]
        q_base = base_pose[..., 3:7]

        # Compute relative pose with JIT functions
        # p_rel = R_base^T * (p_target - p_base)
        q_base_conj = quaternion_conjugate_jit(q_base)
        delta_p = p_target - p_base
        p_rel = quaternion_rotate_vector_jit(q_base_conj, delta_p)

        # q_rel = q_base^{-1} * q_target
        q_rel = quaternion_multiply_jit(q_base_conj, q_target)

        return torch.cat([p_rel, q_rel], dim=-1)

    def _backward_fast(self, pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Optimized backward with JIT-compiled quaternion operations.

        For euler inputs, falls back to _backward_euler.
        """
        if self.input_rotation_type == "euler":
            return self._backward_euler(pose, base_pose)

        assert pose.shape[-1] == 7, f"Expected 7D quaternion pose, got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Expected 7D quaternion base pose, got {base_pose.shape}"

        # Extract position and quaternion (i, j, k, r format)
        p_rel = pose[..., :3]
        q_rel = pose[..., 3:7]

        p_base = base_pose[..., :3]
        q_base = base_pose[..., 3:7]

        # Compute absolute pose with JIT functions
        # p_target = R_base * p_rel + p_base
        p_target = quaternion_rotate_vector_jit(q_base, p_rel) + p_base

        # q_target = q_base * q_rel
        q_target = quaternion_multiply_jit(q_base, q_rel)

        return torch.cat([p_target, q_target], dim=-1)


class RelativeJointTransform(InvertibleModalityTransform):
    """
    Convert absolute joint positions to relative positions w.r.t. the last state.

    This is a simple subtraction transform for joint angles (no SE(3) needed).

    Forward: action_relative = action_absolute - state_last
    Backward: action_absolute = action_relative + state_last

    This transform is invertible.

    Args:
        apply_to: List of action keys to compute relative joints for.
        state_keys_map: Mapping from action keys to state keys (optional).
        fast_forward: Whether to use fast path (default: True).
    """

    state_keys_map: Dict[str, str] | None = Field(
        default=None,
        description="Mapping from action keys to state keys.",
    )
    fast_forward: bool = Field(
        default=True,
        description="Whether to use optimized path.",
    )

    def model_dump(self, *args, **kwargs):
        """Serialize transform configuration."""
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "state_keys_map"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include)

    def _get_state_key(self, action_key: str) -> str:
        """Get the corresponding state key for an action key."""
        if self.state_keys_map is not None and action_key in self.state_keys_map:
            return self.state_keys_map[action_key]
        return action_key.replace("action.", "state.")

    def apply(self, data: dict) -> dict:
        """Apply relative joint transformation."""
        if "action" not in data or "state" not in data:
            return data

        if self.fast_forward:
            return self._apply_fast(data)

        out_data = deepcopy(data)
        out_data["action"] = {}

        # Copy over keys not being transformed
        for key in data["action"]:
            if key not in self.apply_to:
                out_data["action"][key] = data["action"][key]

        # Process keys that need relative transformation
        for action_key in self.apply_to:
            if action_key not in data["action"]:
                continue

            state_key = self._get_state_key(action_key)
            if state_key not in data["state"]:
                out_data["action"][action_key] = data["action"][action_key]
                continue

            action = data["action"][action_key]
            state_last = data["state"][state_key]
            if state_last.ndim == action.ndim:
                state_last = state_last[..., -1:, :]
            else:
                state_last = state_last[-1:]

            out_data["action"][action_key] = action - state_last

        # Copy other modalities
        for modality in ["state", "video", "language"]:
            if modality in data:
                out_data[modality] = data[modality]

        return out_data

    def _apply_fast(self, data: dict) -> dict:
        """Optimized apply without deepcopy."""
        out_data = {
            "action": {},
            "state": data["state"],
        }

        # Copy over keys not being transformed
        for key in data["action"]:
            if key not in self.apply_to:
                out_data["action"][key] = data["action"][key]

        # Process keys that need relative transformation
        for action_key in self.apply_to:
            if action_key not in data["action"]:
                continue

            state_key = self._get_state_key(action_key)
            if state_key in data["state"]:
                state_last = data["state"][state_key]
                if state_last.ndim == data["action"][action_key].ndim:
                    state_last = state_last[..., -1:, :]
                else:
                    state_last = state_last[-1:]
                out_data["action"][action_key] = data["action"][action_key] - state_last
            elif action_key in data["action"]:
                out_data["action"][action_key] = data["action"][action_key]

        # Copy other modalities
        for modality in ["video", "language"]:
            if modality in data:
                out_data[modality] = data[modality]

        return out_data

    def unapply(self, data: dict) -> dict:
        """Reverse relative joint transformation."""
        if "action" not in data or "state" not in data:
            return data

        if self.fast_forward:
            return self._unapply_fast(data)

        out_data = deepcopy(data)
        out_data["action"] = {}

        # Copy over keys not being transformed
        for key in data["action"]:
            if key not in self.apply_to:
                out_data["action"][key] = data["action"][key]

        # Process keys
        for action_key in self.apply_to:
            if action_key not in data["action"]:
                continue

            state_key = self._get_state_key(action_key)
            if state_key not in data["state"]:
                out_data["action"][action_key] = data["action"][action_key]
                continue

            action = data["action"][action_key]
            state_last = data["state"][state_key]
            if state_last.ndim == action.ndim:
                state_last = state_last[..., -1:, :]
            else:
                state_last = state_last[-1:]

            out_data["action"][action_key] = action + state_last

        # Copy other modalities
        for modality in ["state", "video", "language"]:
            if modality in data:
                out_data[modality] = data[modality]

        return out_data

    def _unapply_fast(self, data: dict) -> dict:
        """Optimized unapply without deepcopy."""
        out_data = {
            "action": {},
            "state": data["state"],
        }

        # Copy over keys not being transformed
        for key in data["action"]:
            if key not in self.apply_to:
                out_data["action"][key] = data["action"][key]

        # Process keys
        for action_key in self.apply_to:
            if action_key not in data["action"]:
                continue

            state_key = self._get_state_key(action_key)
            if state_key in data["state"]:
                state_last = data["state"][state_key]
                if state_last.ndim == data["action"][action_key].ndim:
                    state_last = state_last[..., -1:, :]
                else:
                    state_last = state_last[-1:]
                out_data["action"][action_key] = data["action"][action_key] + state_last
            elif action_key in data["action"]:
                out_data["action"][action_key] = data["action"][action_key]

        # Copy other modalities
        for modality in ["video", "language"]:
            if modality in data:
                out_data[modality] = data[modality]

        return out_data
