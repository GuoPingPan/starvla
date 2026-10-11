"""RoboDojo observation/action keys for the 14D absolute-joint dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


JOINT_DIMS = {
    "left_arm_joint": 6,
    "left_gripper": 1,
    "right_arm_joint": 6,
    "right_gripper": 1,
}

XPOLICY_KEYS = {
    "left_arm_joint": "left_arm_joint_state",
    "left_gripper": "left_ee_joint_state",
    "right_arm_joint": "right_arm_joint_state",
    "right_gripper": "right_ee_joint_state",
}


def validate_training_keys(state_keys: Sequence[str], action_keys: Sequence[str]) -> None:
    """Fail early if a checkpoint uses a different layout or representation."""
    expected_state = [f"state.{subkey}" for subkey in JOINT_DIMS]
    expected_action = [f"action.{subkey}" for subkey in JOINT_DIMS]
    if list(state_keys) != expected_state or list(action_keys) != expected_action:
        raise ValueError(
            "RoboDojo joint adapter requires the 14D order "
            f"state={expected_state}, action={expected_action}; got "
            f"state={list(state_keys)}, action={list(action_keys)}"
        )


def observation_to_topics(obs_state: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Read current joints and the previous normalized gripper commands."""
    topics: dict[str, np.ndarray] = {}
    for subkey, dim in JOINT_DIMS.items():
        key = XPOLICY_KEYS[subkey]
        if key not in obs_state:
            raise KeyError(f"obs['state']['{key}'] missing")
        value = np.asarray(obs_state[key], dtype=np.float32).reshape(-1)
        if value.shape != (dim,) or not np.isfinite(value).all():
            raise ValueError(f"obs['state']['{key}'] must be finite ({dim},), got {value.shape}")
        topics[subkey] = value
    return topics


def validate_action_chunk(action_by_topic: Mapping[str, object], min_horizon: int) -> None:
    """Check decoded model output before sending any action to the simulator."""
    horizons = set()
    for subkey, dim in JOINT_DIMS.items():
        if subkey not in action_by_topic:
            raise KeyError(f"decoded action['{subkey}'] missing")
        value = np.asarray(action_by_topic[subkey])
        if value.ndim != 2 or value.shape[1] != dim or not np.isfinite(value).all():
            raise ValueError(f"decoded action['{subkey}'] must be finite (H, {dim}), got {value.shape}")
        horizons.add(value.shape[0])
    if len(horizons) != 1 or next(iter(horizons)) < min_horizon:
        raise ValueError(f"decoded action horizons {sorted(horizons)} are inconsistent or < {min_horizon}")


def action_at(action_by_topic: Mapping[str, object], index: int) -> dict[str, np.ndarray]:
    """Return one absolute joint target using the keys expected by RoboDojo."""
    action: dict[str, np.ndarray] = {}
    for subkey, dim in JOINT_DIMS.items():
        value = np.asarray(action_by_topic[subkey], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != dim or not 0 <= index < value.shape[0]:
            raise ValueError(f"decoded action['{subkey}'] has invalid shape/index: {value.shape}, {index}")
        target = value[index].copy()
        if not np.isfinite(target).all():
            raise ValueError(f"decoded action['{subkey}'][{index}] contains NaN or Inf")
        if "gripper" in subkey:
            # RoboDojo's joint branch consumes a normalized gripper opening.
            target = np.clip(target, 0.0, 1.0)
        action[XPOLICY_KEYS[subkey]] = target
    return action
