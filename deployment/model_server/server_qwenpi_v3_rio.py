# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License (the "License");
"""data-analyze · relative-6d QwenPI_v3 checkpoint: Zenoh 服务 (对接 z-rio)。

与 ``server_umi_pi05_rio.py`` 同协议 (z-rio protocol v3, dict 进 dict 出) 的
QwenPI_v3 (LayerwiseFM + Qwen3.5-0.8B) 版本。复用 PI05 服务的 ``StarVlaZenohServer`` /
msgpack 编解码 / overlay 合并 (import, 不复制); 三个结构性差异:

  1. **state 双路**: PI05 服务里 state 只做 SE(3) 反归一化基准 (forward 时 0 填充);
     本服务 state 同时作为 forward 连续输入 (经训练 transform 归一化后进
     state_encoder) 与 post SE(3) 基准 —— 与训练时一致。
  2. **主干路径解析**: PaliGemma tokenizer.model → ``framework.qwenvl.base_vlm``
     本地目录 (Qwen processor 随目录加载)。优先级 CLI > deploy-yaml > ckpt > env。
  3. **metadata 如实上报**: ``framework="QwenPI_v3"`` + ``state_input="forward_and_post"``。

用法
====
  PYTHONPATH=/path/to/starvla python -m deployment.model_server.server_qwenpi_v3_rio serve \
      --checkpoint <run_dir_or_pt> \
      --deploy-yaml examples/data-analyze/deploy_files/deploy_qwenpi_v3_relative_6d.yaml
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.server_umi_pi05_rio import (
    PROTOCOL_VERSION,
    StarVlaZenohServer,
    merge_deploy_overlay,
)
from starVLA.eval.core.checkpoint_utils import resolve_checkpoint, torch_dtype

# ───────────────── Qwen 主干路径解析 (deploy-yaml > ckpt config > env) ─────────────────

_ENV_FALLBACK = "STARVLA_QWEN_BASE_VLM"


def _local_vlm_dir(path: str | None) -> bool:
    """主干目录本机可用 = 目录可读且含 ``config.json`` (``PermissionError`` 当 False,
    同 PI05 服务 ``_local_file_exists`` 的降级语义)。"""
    if not path:
        return False
    try:
        p = Path(str(path))
        return p.is_dir() and (p / "config.json").is_file()
    except (PermissionError, OSError):
        return False


def resolve_base_vlm_path(merged_cfg) -> str:
    """取 ``framework.qwenvl.base_vlm``: config 本地存在即用; 否则回退 env;
    都没有则抛带可操作提示的错误。"""
    from omegaconf import OmegaConf

    configured = OmegaConf.select(merged_cfg, "framework.qwenvl.base_vlm", default=None)
    if _local_vlm_dir(configured):
        return str(configured)
    env_path = os.environ.get(_ENV_FALLBACK)
    if _local_vlm_dir(env_path):
        print(
            f"[load] framework.qwenvl.base_vlm={configured!r} 本机缺失, "
            f"回退 env {_ENV_FALLBACK}={env_path}",
            flush=True,
        )
        return env_path
    raise FileNotFoundError(
        f"framework.qwenvl.base_vlm={configured!r} 本机不存在。在 deploy overlay 里设 "
        f"framework.qwenvl.base_vlm=<本地 Qwen 主干目录>, 或 export {_ENV_FALLBACK}=<path>。"
    )


def build_base_vlm_override(merged_cfg, ckpt_cfg: dict) -> list[str]:
    """ckpt 原主干路径本机缺失时, 给 ``from_pretrained`` 生成 config 覆盖参数。"""
    ckpt_vlm = (ckpt_cfg.get("framework", {}) or {}).get("qwenvl", {}).get("base_vlm")
    if _local_vlm_dir(ckpt_vlm):
        return []
    eff = resolve_base_vlm_path(merged_cfg)
    print(f"[load] ckpt base_vlm {ckpt_vlm!r} 本机缺失 → override -> {eff}", flush=True)
    return [f"framework.qwenvl.base_vlm={eff}"]


# ───────────────────────── 部署 config (overlay) ─────────────────────────


@dataclass
class QwenPIDeployConfig:
    """部署运行参数 (deploy-yaml 的 ``deploy:`` 段); 与 PI05 ``DeployConfig`` 同字段,
    仅 ``key_prefix`` 默认值区分服务实例, 避免与 PI05 服务并存期混淆。"""

    device: str = "cuda"
    dtype: str = "bf16"
    unnorm_key: str | None = None
    listen_addr: str = "127.0.0.1:5570"
    key_prefix: str = "z-rio/starvla-qwenpi-v3"
    shutdown_timeout: float = 5.0

    @classmethod
    def from_merged(cls, merged_cfg) -> "QwenPIDeployConfig":
        from omegaconf import OmegaConf

        section = dict(OmegaConf.select(merged_cfg, "deploy", default={}) or {})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in section.items() if k in known})


@dataclass(frozen=True)
class QwenPIDeployment:
    """一次加载的全部部署上下文 (镜像 PI05 ``Deployment``, tokenizer 覆盖换 base_vlm 覆盖)。"""

    ckpt_path: Path
    ckpt_cfg: dict
    merged_cfg: Any
    norm_stats: dict
    deploy: QwenPIDeployConfig
    base_vlm_override: list[str]

    @classmethod
    def load(cls, ckpt_arg, deploy_yaml: Path | None) -> "QwenPIDeployment":
        from starVLA.model.framework.share_tools import read_mode_config

        ckpt_path = resolve_checkpoint(ckpt_arg)
        ckpt_cfg, norm_stats = read_mode_config(str(ckpt_path))
        merged_cfg = merge_deploy_overlay(ckpt_cfg, deploy_yaml)
        return cls(
            ckpt_path=ckpt_path,
            ckpt_cfg=ckpt_cfg,
            merged_cfg=merged_cfg,
            norm_stats=norm_stats,
            deploy=QwenPIDeployConfig.from_merged(merged_cfg),
            base_vlm_override=build_base_vlm_override(merged_cfg, ckpt_cfg),
        )


# ───────────────────── policy: 模型加载 + z-rio 服务面 ──────────────────


class QwenPIv3Policy:
    """加载一个 data-analyze relative-6d QwenPI_v3 ckpt, 常驻提供 z-rio **绝对** action dict。

    predict 流程 (全部在 ``_lock`` 下串行):
      pre    图像按训练方式拉伸到 ``image_resolution`` (复刻 ``_pack_sample``,
             规避 letterbox OOD —— 与 PI05 服务同坑同修)
      fwd    state raw → 训练 transform 归一化 → 连续 state 输进 ``predict_action``
      post   ``PolicyNormProcessor.unapply_actions``: denorm + 相对→绝对 SE(3)
             (state 兼任基准) + topic 拆分, 输出绝对 action (旋转 quaternion xyzw)
    """

    def __init__(
        self,
        ckpt_path: Path,
        ckpt_cfg: dict,
        merged_cfg,
        norm_stats: dict,
        device: str = "cuda",
        dtype: str = "bf16",
        unnorm_key: str | None = None,
        config_overrides: Sequence[str] | None = None,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        from starVLA.model.framework.base_framework import baseframework

        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {self.ckpt_path}")
        self.device = device
        self._torch_dtype = torch_dtype(dtype)
        self._torch = torch

        ir = OmegaConf.select(merged_cfg, "framework.image_resolution", default=None)
        if ir is None:  # QwenPI_v3 无 PI0 式顶层 image_resolution; 用 Qwen 处理器默认方形输入
            ir = [224, 224]
        self.image_resolution = (int(ir[0]), int(ir[1]))

        self._norm_proc = PolicyNormProcessor(
            str(self.ckpt_path), unnorm_key=unnorm_key, cfg=ckpt_cfg, norm_stats=norm_stats
        )
        self.action_keys = self._norm_proc.action_keys
        self.state_keys = self._norm_proc.state_keys
        self.chunk_size = self._norm_proc.chunk_size
        self.action_dim_actual = self._norm_proc.action_dim
        self.state_dim_actual = sum(
            self._norm_proc._state_key_dims.get(k, 1) for k in self.state_keys
        )

        print(
            f"[load] ckpt={self.ckpt_path.name}\n"
            f"       action_keys={self.action_keys} action_dim={self.action_dim_actual} "
            f"state_keys={self.state_keys} state_dim={self.state_dim_actual} "
            f"chunk={self.chunk_size} image_resolution={self.image_resolution} "
            f"device={device} dtype={dtype}",
            flush=True,
        )

        print("[load] loading framework (torch.load state_dict) ...", flush=True)
        model = baseframework.from_pretrained(str(self.ckpt_path), config_overrides=config_overrides or None)
        self.model = model.to(device=self.device, dtype=self._torch_dtype).eval()
        self._lock = threading.Lock()
        self._infer_count = 0

    # ── pre ──

    def _to_train_pil(self, image: np.ndarray):
        """uint8 HWC RGB → 拉伸到 image_resolution 的 PIL (复刻训练 _pack_sample;
        与 PI05 服务 ``_to_train_pil`` 同实现, 规避 resize_with_pad 黑边 OOD)。"""
        from PIL import Image

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"image must be HWC RGB, got {arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got {arr.dtype}")
        return Image.fromarray(np.ascontiguousarray(arr)).resize(self.image_resolution)

    # ── fwd: raw state → normalized (1, D) ──

    def _normalize_state(self, state_raw: Dict[str, Any]) -> np.ndarray:
        """raw per-topic state (旋转 quaternion xyzw) → 训练 transform 归一化的 (1, D)。

        与 ``PolicyNormProcessor.unapply_actions`` 的 state 前置归一化同源 (同一
        ``transform.apply``), 按 ``state_keys`` 顺序拼接; 维度不符 / 非有限值抛错,
        不静默 0 填充。
        """
        import torch

        if not state_raw:
            raise ValueError(
                "state is required for QwenPI_v3 (forward continuous input + SE(3) base); got empty state"
            )
        state_dict = {}
        for sub, v in state_raw.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim < 2:
                arr = arr[None, :]
            state_dict[f"state.{sub}"] = arr
        missing = [k for k in self.state_keys if k not in state_dict]
        if missing:
            raise ValueError(f"state missing topics {missing}; expected keys {list(self.state_keys)}")
        state_norm = self._norm_proc.sa_transform.apply(
            {k: v for k, v in state_dict.items() if k.startswith("state.")}
        )
        parts = []
        for full_key in self.state_keys:
            t = state_norm[full_key]
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            parts.append(np.asarray(t, dtype=np.float32))
        state_arr = np.concatenate(parts, axis=-1)
        if state_arr.shape != (1, self.state_dim_actual):
            raise ValueError(
                f"normalized state shape {state_arr.shape} != (1, {self.state_dim_actual}); "
                f"state_keys={self.state_keys}"
            )
        if not np.isfinite(state_arr).all():
            raise ValueError("normalized state contains NaN or Inf")
        return state_arr

    # ── 服务面 ──

    @property
    def metadata(self) -> dict[str, Any]:
        """z-rio ``StarVla._validate_server_metadata`` 契约字段; framework/state_input 如实上报。"""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "checkpoint_path": str(self.ckpt_path),
            "framework": "QwenPI_v3",
            "state_input": "forward_and_post",
            "normalized_actions": False,
            "action_chunk_size": int(self.chunk_size),
            "action_mode": "absolute",
            "action_representation": "position_3+quaternion_xyzw_4+gripper_1",
            "state_representation": "position_3+quaternion_xyzw_4+gripper_1",
            "rotation_convention": "quaternion_xyzw",
            "gripper_unit": "meter",
            "obs_image_size": list(self.image_resolution),
            "image_count": 1,
            "default_unnorm_key": self._norm_proc.unnorm_key,
            "available_unnorm_keys": self._norm_proc.available_unnorm_keys,
            "action_keys": self.action_keys,
            "state_keys": self.state_keys,
        }

    def predict(self, obs: dict) -> dict:
        """dict 进 dict 出: 观测 (含 state, 旋转 quaternion xyzw) → 绝对动作 dict。"""
        video = obs.get("video") or {}
        image = video.get("cam_wrist_0")
        language = obs.get("language")
        state_raw = obs.get("state") or {}
        if image is None:
            raise KeyError("obs['video']['cam_wrist_0'] missing")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("obs['language'] must be a non-empty string")

        torch = self._torch
        with self._lock:
            t0 = time.perf_counter()
            pil = self._to_train_pil(np.ascontiguousarray(np.asarray(image)))
            state_norm = self._normalize_state(state_raw)  # fwd 路: 连续 state 输入
            example: dict[str, Any] = {"image": [pil], "lang": language, "state": state_norm}
            t_pre = time.perf_counter()
            with torch.inference_mode():
                out = self.model.predict_action(examples=[example])
            t_fwd = time.perf_counter()
            norm = np.asarray(out["normalized_actions"], dtype=np.float32)
            if norm.ndim == 3 and norm.shape[0] == 1:
                norm = norm[0]
            if norm.shape[-1] < self.action_dim_actual:
                raise ValueError(f"model output dim {norm.shape[-1]} < action_dim {self.action_dim_actual}")
            result = self._norm_proc.unapply_actions(
                {"org_action": norm[..., : self.action_dim_actual], "state": state_raw}
            )
            t_post = time.perf_counter()

        dt_pre, dt_fwd, dt_post = (t_pre - t0) * 1e3, (t_fwd - t_pre) * 1e3, (t_post - t_fwd) * 1e3
        self._infer_count += 1
        print(
            f"[infer #{self._infer_count}] total={dt_pre + dt_fwd + dt_post:.1f}ms "
            f"(pre={dt_pre:.1f} fwd={dt_fwd:.1f} post={dt_post:.1f}) lang={language!r}",
            flush=True,
        )
        return result

    def reset(self) -> None:
        """QwenPI_v3 推理无状态; 保留接口对齐 z-rio episode 切换协议。"""


# ──────────────────────────── CLI ────────────────────────────


def _serve_main(args: argparse.Namespace) -> None:
    try:
        import zenoh  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "zenoh 未安装 —— serve 需要 zenoh。用装有 zenoh 的环境运行 (见 PI05 服务用法说明)。"
        ) from e
    dep = QwenPIDeployment.load(args.checkpoint, args.deploy_yaml)
    policy = QwenPIv3Policy(
        dep.ckpt_path,
        dep.ckpt_cfg,
        dep.merged_cfg,
        dep.norm_stats,
        device=dep.deploy.device,
        dtype=dep.deploy.dtype,
        unnorm_key=dep.deploy.unnorm_key,
        config_overrides=dep.base_vlm_override,
    )
    server = StarVlaZenohServer(
        policy,
        listen_addr=dep.deploy.listen_addr,
        key_prefix=dep.deploy.key_prefix,
        shutdown_timeout=dep.deploy.shutdown_timeout,
    )
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="起 Zenoh 服务对接 z-rio (需要 zenoh)。")
    ps.add_argument("--checkpoint", required=True, help=".pt 文件 / run 目录 (扁平或 checkpoints/)。")
    ps.add_argument(
        "--deploy-yaml",
        type=Path,
        default=None,
        help="部署 overlay YAML: 跟 ckpt config OmegaConf.merge, 承载主干路径 + deploy: 段。",
    )
    ps.set_defaults(_fn=_serve_main)

    args = parser.parse_args(argv)
    args._fn(args)


if __name__ == "__main__":
    main()
