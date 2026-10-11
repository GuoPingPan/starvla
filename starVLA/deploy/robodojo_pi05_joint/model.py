"""PI05 closed-loop model for RoboDojo's 14D absolute joint actions.

The EE adapter supplies the common PI05 inference, image and batch loops. This
class changes only the checkpoint contract and the RoboDojo state/action map.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from starVLA.deploy.robodojo_pi05.model import Model as _EEModel
from starVLA.deploy.robodojo_pi05.model import _to_train_pil
from starVLA.deploy.robodojo_pi05_joint.mapping import (
    JOINT_DIMS,
    action_at,
    observation_to_topics,
    validate_action_chunk,
    validate_training_keys,
)


class Model(_EEModel):
    """Match PI05 joint dataset transforms to RoboDojo's joint API."""

    def __init__(self, model_cfg: dict):
        import torch

        from deployment.model_server.policy_norm_processor import PolicyNormProcessor
        from starVLA.model.framework.base_framework import baseframework
        from starVLA.model.framework.share_tools import read_mode_config

        self.model_cfg = dict(model_cfg)
        if self.model_cfg.get("action_type", "joint") != "joint":
            raise ValueError("RoboDojo PI05 joint adapter requires action_type='joint'")
        self.device = self.model_cfg.get("device", "cuda")
        self._torch = torch
        self.unnorm_key = self.model_cfg.get("unnorm_key") or "new_embodiment"
        image_size = self.model_cfg.get("image_size", [224, 224])
        self.image_resolution = (int(image_size[0]), int(image_size[1]))

        ckpt_path = self.model_cfg["checkpoint_path"]
        ckpt_cfg, norm_stats = read_mode_config(ckpt_path)
        if ckpt_cfg["framework"]["name"] != "PI05":
            raise ValueError("RoboDojo joint adapter requires a PI05 checkpoint")
        if ckpt_cfg["datasets"]["vla_data"].get("action_mode", "abs") != "abs":
            raise ValueError("RoboDojo joint adapter requires absolute joint actions")
        self.chunk_size = int(ckpt_cfg["framework"]["action_horizon"])
        if self.chunk_size <= 0:
            raise ValueError(f"Invalid framework.action_horizon={self.chunk_size}")
        self.execute_horizon = int(self.model_cfg.get("execute_horizon", 16))
        if not 1 <= self.execute_horizon <= self.chunk_size:
            raise ValueError(
                f"execute_horizon must be in [1, {self.chunk_size}], got {self.execute_horizon}"
            )

        self.norm_proc = PolicyNormProcessor(
            ckpt_path, unnorm_key=self.unnorm_key, cfg=ckpt_cfg, norm_stats=norm_stats
        )
        self.action_keys = self.norm_proc.action_keys
        self.state_keys = self.norm_proc.state_keys
        validate_training_keys(self.state_keys, self.action_keys)
        self.action_dim_actual = int(self.norm_proc.action_dim)
        if self.action_dim_actual != sum(JOINT_DIMS.values()):
            raise ValueError(f"Joint checkpoint action_dim={self.action_dim_actual}, expected 14")
        self._sa_transform = self.norm_proc.sa_transform
        self.model = baseframework.from_pretrained(ckpt_path).to(torch.bfloat16).to(self.device).eval()
        self._obs: dict | None = None
        self._obs_batch: list[dict] = []

        print(
            f"[starVLA_pi05_robodojo_joint] ready: action_dim=14, "
            f"chunk={self.chunk_size}, execute_horizon={self.execute_horizon}, "
            f"unnorm_key={self.unnorm_key}",
            flush=True,
        )

    def _forward_state(self, raw_topics: dict[str, np.ndarray]) -> np.ndarray:
        """Apply the joint dataset's state normalization before PI05 forward."""
        raw = {f"state.{subkey}": value[None, :] for subkey, value in raw_topics.items()}
        transformed = self._sa_transform.apply(raw)
        parts = []
        for full_key in self.state_keys:
            value = transformed[full_key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
        state = np.concatenate(parts).astype(np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"Normalized joint state must be finite (14,), got {state.shape}")
        return state

    def _prepare_example(self, obs: dict) -> tuple[dict, dict[str, np.ndarray]]:
        if obs is None:
            raise RuntimeError("update_obs(batch) must precede get_action(batch)")
        images = [
            self._extract_camera(obs, ["cam_head", "head_camera", "cam_high"]),
            self._extract_camera(obs, ["cam_left_wrist", "left_camera"]),
            self._extract_camera(obs, ["cam_right_wrist", "right_camera"]),
        ]
        pils = [_to_train_pil(image, self.image_resolution) for image in images]

        language: Any = obs.get("instruction") or obs.get("instructions")
        if isinstance(language, (list, tuple)):
            language = language[0] if language else ""
        language = str(language or self.model_cfg.get("task_name", "")).strip()

        if not obs.get("state"):
            raise KeyError("obs['state'] missing")
        raw_topics = observation_to_topics(obs["state"])
        state = self._forward_state(raw_topics)
        return {"image": pils, "lang": language, "state": state}, raw_topics

    def _infer_chunk(self, obs: dict) -> dict[str, np.ndarray]:
        action_by_topic = super()._infer_chunk(obs)
        validate_action_chunk(action_by_topic, self.execute_horizon)
        return action_by_topic

    def _infer_chunk_batch(self, obs_list: list[dict]) -> list[dict[str, np.ndarray]]:
        chunks = super()._infer_chunk_batch(obs_list)
        for chunk in chunks:
            validate_action_chunk(chunk, self.execute_horizon)
        return chunks

    def _action_dict_at(self, action_by_topic: dict, t: int) -> dict[str, np.ndarray]:
        return action_at(action_by_topic, t)
