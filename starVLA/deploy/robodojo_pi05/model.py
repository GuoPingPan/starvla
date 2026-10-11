"""starVLA PI05 RoboDojo 闭环评测 adapter —— 核心实现。

本模块持有全部前后处理逻辑; XPolicyLab 侧 ``policy/starVLA_pi05_robodojo/model.py`` 是
re-export 薄壳 (``from starVLA.deploy.robodojo_pi05.model import Model``), 仅因
``XPolicyLab/setup_policy_server.py`` 按 ``XPolicyLab.policy.<name>.model`` 包路径加载
(硬编码)。改 PI05 逻辑只动本文件, RoboDojo / XPolicyLab 官方源码零改动。

进程内加载 PI05 + PolicyNormProcessor, 镜像 ``server_umi_pi05_rio.py:UMIPI05Policy``
但适配 XPolicyLab 协议 + RoboDojo 双臂 ee。

单 env (deploy.yml ``eval_batch: false``) 与 多 env batch (``eval_batch: true``) 两套路径:
  - 单 env: ``update_obs`` + ``get_action`` → ``_infer_chunk`` (``examples=[ex]``, B=1)。
  - 多 env: ``update_obs_batch`` + ``get_action_batch`` → ``_infer_chunk_batch``
    (``examples=[ex1..exN]`` 一次 batched forward → (N,50,32); unapply 按 env 循环)。
    PI0/PI05 ``predict_action`` 原生支持 B 维 (``_prepareExamples`` torch.stack 整个 list);
    ``PolicyNormProcessor.unapply_actions`` 强制 ``org_action.ndim==2`` (单 env (T,D)),
    故 unapply 按环境循环 —— forward 是 GPU 大头已 batch, unapply 是 numpy 后处理, N 次 cheap)。

与 z-umi 单臂的关键差异 (RoboDojo bimanual):
  - 3 相机 (cam_head / cam_left_wrist / cam_right_wrist → base_0_rgb / left_wrist_0_rgb /
    right_wrist_0_rgb); z-umi 只有 1 张 cam_wrist_0。
  - ``include_state=True`` → forward **要喂 transformed state** (raw → transform.apply →
    20D → _prepare_examples pad 到 32); z-umi forward 不喂 state (PI0 内部填 0)。
  - 双臂: state/action 各 6 个 topic (end_left/right_position/rotation + left/right_gripper)。
  - quaternion 约定 **wxyz** (z-umi 是 xyzw)。
  - unnorm_key = "new_embodiment"。

数据流 (get_action / get_action_batch):
  raw obs (3 图 + state 6 topic wxyz + lang)
    → 3 图拉伸 224² (复刻训练 _pack_sample, 非 letterbox)
    → state raw → ``_transform.apply`` → pack 20D → example["state"]
    → framework.predict_action → normalized_actions (B, 50, 32), 取前 20
    → ``PolicyNormProcessor.unapply_actions`` (传 raw state 做 SE(3) 相对→绝对 base)
    → 绝对 action by topic (quat wxyz) → 取前 execute_horizon 步 → XPolicyLab 双臂 ee action dict

⚠️ 不用 ``server_policy.py``/``PolicyServerWrapper``: 它 ``predict_action`` 传 ``state={}``,
   导致 RelativePoseTransform 跳过 SE(3) 合成, 对 relative_6d 模型致命 (输出仍是相对位姿)。
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from XPolicyLab.model_template import ModelTemplate

# RoboDojo 双臂 ee: starVLA state/action topic (sub-key, 去 state./action. 前缀), quat wxyz。
# 与 PolicyNormProcessor.action_keys / state_keys 完全一致 (探测确认)。
_LEFT_TOPICS = ("end_left_position", "end_left_rotation", "left_gripper")
_RIGHT_TOPICS = ("end_right_position", "end_right_rotation", "right_gripper")


def _to_train_pil(image: Any, image_resolution: tuple[int, int]):
    """uint8 HWC RGB → 拉伸到 image_resolution 的 PIL (复刻训练 _pack_sample 的 Image.resize)。

    训练用 ``Image.fromarray(img).resize((224,224))`` 拉伸; PI0 推理 ``_pil_to_tensor`` 用
    ``resize_with_pad`` (letterbox)。先按训练拉伸到目标尺寸, 让 PI0 的 letterbox 变 no-op,
    避免 16:9 图被加黑边 → OOD。 (z-umi 踩过的坑, server_umi_pi05_rio.py 注释详述。)
    """
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"image must be HWC RGB uint8, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr)).resize(image_resolution)


def _xpolicy_state_to_topics(obs_state: dict) -> dict[str, np.ndarray]:
    """XPolicyLab obs['state'] (双臂 ee) → starVLA raw state by topic (quat wxyz)。

    XPolicyLab: left_ee_pose(7)=[x,y,z,qw,qx,qy,qz] + left_ee_joint_state(gripper); 右臂同。
    starVLA:    end_left_position(3) + end_left_rotation(4, wxyz) + left_gripper(1); 右臂同。
    """
    def _pose_gripper(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = np.asarray(obs_state[f"{side}_ee_pose"], dtype=np.float32).reshape(-1)
        grip = np.asarray(obs_state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)
        if pose.shape[0] != 7:
            raise ValueError(f"{side}_ee_pose must be 7-D [x,y,z,qw,qx,qy,qz], got {pose.shape}")
        return pose[:3], pose[3:7], grip  # position, rotation(wxyz), gripper

    lp, lr, lg = _pose_gripper("left")
    rp, rr, rg = _pose_gripper("right")
    return {
        "end_left_position": lp, "end_left_rotation": lr, "left_gripper": lg,
        "end_right_position": rp, "end_right_rotation": rr, "right_gripper": rg,
    }


class Model(ModelTemplate):
    """starVLA PI05 (RoboDojo bimanual ee relative_6d) 的 XPolicyLab model adapter。

    单 env (eval_batch=false) 与 多 env batch (eval_batch=true) 双路径: 前者
    ``update_obs``→``get_action``→``_infer_chunk``; 后者 ``update_obs_batch``→
    ``get_action_batch``→``_infer_chunk_batch`` (batched forward + per-env unapply)。
    """

    def __init__(self, model_cfg: dict):
        import torch
        from starVLA.model.framework.base_framework import baseframework
        from deployment.model_server.policy_norm_processor import PolicyNormProcessor

        self.model_cfg = dict(model_cfg)
        self.device = self.model_cfg.get("device", "cuda")
        self._torch = torch

        # ── 配置 ──────────────────────────────────────────────────────────
        ckpt_path = self.model_cfg["checkpoint_path"]
        self.unnorm_key = self.model_cfg.get("unnorm_key") or "new_embodiment"
        ir = self.model_cfg.get("image_size", [224, 224])
        self.image_resolution = (int(ir[0]), int(ir[1]))
        self.execute_horizon = int(self.model_cfg.get("execute_horizon", 16))

        # ── 1. framework (进程内加载 PI05) ────────────────────────────────
        # tokenizer.model_path 在 ckpt config 里指向本机存在的路径, 无需 override。
        print(f"[starVLA_pi05_robodojo] loading PI05 framework from {ckpt_path} ...", flush=True)
        framework = baseframework.from_pretrained(ckpt_path)
        self.model = framework.to(torch.bfloat16).to(self.device).eval()

        # ── 2. PolicyNormProcessor (重建训练 transform 链; 内部 read_mode_config) ──
        self.norm_proc = PolicyNormProcessor(ckpt_path, unnorm_key=self.unnorm_key)
        self.action_keys = self.norm_proc.action_keys
        self.state_keys = self.norm_proc.state_keys
        self.chunk_size = int(self.norm_proc.chunk_size)        # 50
        self.action_dim_actual = int(self.norm_proc.action_dim)  # 20
        self._transform = self.norm_proc._transform             # ComposedModalityTransform
        if self.execute_horizon > self.chunk_size:
            self.execute_horizon = self.chunk_size

        # ── 3. 观测缓存 (单 env: _obs; 多 env batch: _obs_batch 与 env_idx_list 同序) ──
        self._obs: dict | None = None
        self._obs_batch: list[dict] = []

        print(
            f"[starVLA_pi05_robodojo] ready: action_dim={self.action_dim_actual} "
            f"chunk={self.chunk_size} execute_horizon={self.execute_horizon} "
            f"unnorm_key={self.unnorm_key}\n"
            f"  action_keys={self.action_keys}\n  state_keys={self.state_keys}",
            flush=True,
        )

    # ── 观测 ──────────────────────────────────────────────────────────────
    def update_obs(self, obs):
        self._obs = obs

    def update_obs_batch(self, obs_list):
        # 多 env: 存全部, 与 env_idx_list 同序同长 (deploy.py batch 循环保证)。
        self._obs_batch = list(obs_list) if obs_list else []
        self._obs = obs_list[-1] if obs_list else None  # 兼容单 env 兜底

    def _extract_camera(self, obs: dict, candidates: list[str]) -> np.ndarray:
        vision = obs.get("vision", {})
        for name in candidates:
            if name in vision:
                cam = vision[name]
                if isinstance(cam, dict):
                    for key in ("color", "rgb", "colors"):
                        if key in cam:
                            return np.asarray(cam[key])
                return np.asarray(cam)
        raise KeyError(f"missing camera from candidates {candidates} in obs['vision']")

    def _forward_state(self, raw_topics: dict[str, np.ndarray]) -> np.ndarray:
        """raw state (by topic, quat wxyz) → transform.apply → pack 20D (供 forward)。"""
        state_dict: dict[str, np.ndarray] = {}
        for sub, v in raw_topics.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim < 2:
                arr = arr[None, :]
            state_dict[f"state.{sub}"] = arr
        state_norm = self._transform.apply(state_dict)
        parts = []
        for full_key in self.state_keys:
            v = state_norm[full_key]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)  # (20,)

    # ── 推理 ──────────────────────────────────────────────────────────────
    def _prepare_example(self, obs: dict) -> tuple[dict, dict[str, np.ndarray]]:
        """单 obs → (example dict, raw_topics)。单 env / batch 共用。"""
        if obs is None:
            raise RuntimeError("update_obs(batch) must be called before get_action(batch)")
        # 3 图 (顺序对齐 image_keys: base_0_rgb=cam_head, left_wrist=cam_left_wrist, right_wrist=cam_right_wrist)
        images = [
            self._extract_camera(obs, ["cam_head", "head_camera", "cam_high"]),
            self._extract_camera(obs, ["cam_left_wrist", "left_camera"]),
            self._extract_camera(obs, ["cam_right_wrist", "right_camera"]),
        ]
        pils = [_to_train_pil(im, self.image_resolution) for im in images]

        # lang
        lang = obs.get("instruction") or obs.get("instructions")
        if isinstance(lang, (list, tuple)):
            lang = lang[0] if lang else ""
        if not lang:
            lang = self.model_cfg.get("task_name", "")
        lang = str(lang or "").strip()

        # state: forward 用 transformed; unapply 用 raw (做 SE(3) base)
        if "state" not in obs or not obs["state"]:
            raise KeyError("obs['state'] missing (need left/right_ee_pose + _ee_joint_state)")
        raw_topics = _xpolicy_state_to_topics(obs["state"])
        state_fwd = self._forward_state(raw_topics)
        return {"image": pils, "lang": lang, "state": state_fwd}, raw_topics

    def _infer_chunk(self, obs: dict) -> dict[str, np.ndarray]:
        """单 env → 绝对 action by topic (每 topic (chunk_size, dim_k), quat wxyz)。"""
        example, raw_topics = self._prepare_example(obs)
        torch = self._torch
        with torch.inference_mode():
            out = self.model.predict_action(examples=[example])
        norm = np.asarray(out["normalized_actions"], dtype=np.float32)
        if norm.ndim == 3 and norm.shape[0] == 1:
            norm = norm[0]
        if norm.shape[-1] < self.action_dim_actual:
            raise ValueError(f"model output dim {norm.shape[-1]} < action_dim {self.action_dim_actual}")
        norm = norm[..., : self.action_dim_actual]  # (chunk_size, 20)
        # unapply (传 raw state 做 SE(3) 相对→绝对合成)
        result = self.norm_proc.unapply_actions({"org_action": norm, "state": raw_topics})
        return result["action"]

    def _infer_chunk_batch(self, obs_list: list[dict]) -> list[dict[str, np.ndarray]]:
        """N env → 一次 batched forward + per-env unapply → list[{topic: (chunk_size, dim_k)}]。

        PI0/PI05 ``predict_action`` 原生支持 B 维 (``_prepareExamples`` 把 examples list
        ``torch.stack`` 成 batched 张量), 故 N 个 env 共一次 forward (GPU 大头已 batch)。
        ``unapply_actions`` 强制 ``org_action.ndim==2`` (单 env (T,D)), 不支持 batch,
        故按 env 循环 —— 它是 numpy 后处理, N 次 cheap。
        """
        if not obs_list:
            return []
        prepared = [self._prepare_example(obs) for obs in obs_list]
        examples = [p[0] for p in prepared]
        raw_topics_list = [p[1] for p in prepared]

        torch = self._torch
        with torch.inference_mode():
            out = self.model.predict_action(examples=examples)  # (N, 50, 32)
        norm = np.asarray(out["normalized_actions"], dtype=np.float32)
        if norm.ndim == 2:  # 个别实现对 N=1 会挤掉 B 维, 补回
            norm = norm[None]
        if norm.ndim != 3:
            raise ValueError(f"predict_action expected (N, chunk_size, dim); got {norm.shape}")
        if norm.shape[-1] < self.action_dim_actual:
            raise ValueError(f"model output dim {norm.shape[-1]} < action_dim {self.action_dim_actual}")
        norm = norm[..., : self.action_dim_actual]  # (N, chunk_size, 20)

        # unapply per-env (unapply_actions 强制 (T,D) 2D)
        return [
            self.norm_proc.unapply_actions({"org_action": norm[i], "state": raw_topics_list[i]})["action"]
            for i in range(norm.shape[0])
        ]

    def _action_dict_at(self, action_by_topic: dict, t: int) -> dict[str, np.ndarray]:
        """第 t 步的绝对 action → XPolicyLab 双臂 ee action dict。"""
        lp = action_by_topic["end_left_position"][t]
        lr = action_by_topic["end_left_rotation"][t]   # quat wxyz
        rp = action_by_topic["end_right_position"][t]
        rr = action_by_topic["end_right_rotation"][t]
        return {
            "left_ee_pose": np.concatenate([lp, lr]).astype(np.float32),         # (7,) [x,y,z,qw,qx,qy,qz]
            "left_ee_joint_state": np.asarray(action_by_topic["left_gripper"][t], dtype=np.float32).reshape(-1),
            "right_ee_pose": np.concatenate([rp, rr]).astype(np.float32),
            "right_ee_joint_state": np.asarray(action_by_topic["right_gripper"][t], dtype=np.float32).reshape(-1),
        }

    def get_action(self):
        # 单 env (eval_batch=false): update_obs → _infer_chunk (examples=[ex], B=1)
        action_by_topic = self._infer_chunk(self._obs)
        H = self.execute_horizon
        return [self._action_dict_at(action_by_topic, t) for t in range(H)]

    def get_action_batch(self, env_idx_list=None):
        # 多 env (eval_batch=true): update_obs_batch → _infer_chunk_batch (一次 batched forward)
        env_idx_list = env_idx_list or [0]
        obs_batch = self._obs_batch if self._obs_batch else ([self._obs] if self._obs is not None else [])
        if len(obs_batch) != len(env_idx_list):
            raise RuntimeError(
                f"obs_batch len {len(obs_batch)} != env_idx_list len {len(env_idx_list)}; "
                "update_obs_batch 与 get_action_batch 的 env_idx_list 应同序同长 (deploy.py batch 循环保证)"
            )
        actions_by_topic_list = self._infer_chunk_batch(obs_batch)
        H = self.execute_horizon
        return [[self._action_dict_at(at, t) for t in range(H)] for at in actions_by_topic_list]

    def reset(self):
        self._obs = None
        self._obs_batch = []
