# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""UMI relative-6d PI05 扁平 checkpoint: Zenoh 服务 (对接 z-rio)。

起 Zenoh 服务, 常驻模型, 对接 z-rio ``rio/policies/starvla.py`` 客户端。
dict 进 dict 出: 收 ``{video, language, state}``, 返回**绝对** action dict
(按 topic 拆分; 旋转 = quaternion xyzw; 夹爪绝对, 米) + ``org_action`` (模型原始输出)。
相对→绝对的 SE(3) 合成在服务端 post 完成 (用 state 当基准) —— 客户端不做任何坐标变换。
**需要 zenoh** (starVLA conda env 里有)。

════════════════════════════════════════════════════════════════════════
几个踩过的坑 (都已处理)
════════════════════════════════════════════════════════════════════════
  - **PI05 扁平 config**: ``framework.action_dim``/``action_horizon`` 在顶层, 无
    ``framework.action_model`` (读 ``framework.action_model`` 的代码会 KeyError)。
  - **逐时间步 q99 归一化**: 统计量在 ckpt 同目录 ``dataset_statistics.json`` 的
    ``_per_key`` (形状 (H, D))。反归一化复用 ``PolicyNormProcessor`` (与 websocket
    server 同源): 它重建训练 transform 链 + 用 ``_per_key`` 绑定逐时间步统计,
    ``unapply_actions`` 走**完整 transform 链** ``unapply`` (denorm + 相对→绝对 SE(3) 合成
    + topic 拆分), 输出**绝对** action dict (旋转 quaternion xyzw), 与 open-eval 同源。
  - **tokenizer 路径**: ckpt config 指训练机 ``/nix/...`` (本机缺失)。从 merged config
    读 ``framework.tokenizer.model_path`` (deploy overlay 可覆盖到本地副本); 缺失回退
    env ``STARVLA_PALIGEMMA_TOKENIZER``; 都没有则报错。详见 ``resolve_tokenizer_path``。
  - **🔥 图像必须按训练方式拉伸到 224x224**: 训练 ``datasets.py::_pack_sample`` 用
    ``Image.resize((224,224))`` 拉伸; PI0 推理 ``_pil_to_tensor`` 用 ``resize_with_pad``
    (letterbox)。不一致 → 真机 16:9 图被加 ~43% 黑边 → OOD → 小动作 + 不夹。fix:
    推理时先把图拉伸到 image_resolution, 让 PI0 resize_with_pad 变 no-op (见 ``_to_train_pil``)。
  - **state 是 post 输入 (相对→绝对的 SE(3) 基准), 不是 forward 模型输入**
    (forward 时 PI0 内部按 0 填充 state; decode 时 state 进 ``unapply_actions`` 合绝对)。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════
  # serve (starVLA conda env 有 zenoh): --deploy-yaml 承载 tokenizer/device/zenoh 等参数
  PYTHONPATH=/home/zerith/ws/orion/starvla \
  /home/zerith/miniforge3/envs/starVLA/bin/python -m deployment.model_server.server_umi_pi05_rio serve \
      --checkpoint /home/zerith/.models/run_umi_shenzhen_relative_6d_20000step_0727_100151 \
      --deploy-yaml examples/realRobots/z_umi/deploy_files/deploy_umi_pi05_relative_6d.yaml

  # 不用 deploy-yaml 时, 各 flag 仍可直接给 (CLI > deploy-yaml > ckpt > 默认); tokenizer
  # 路径缺失时 export STARVLA_PALIGEMMA_TOKENIZER=<本地路径> 兜底。
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

# zenoh 懒加载: 只有 serve 子命令需要; 缺失时模块仍可 import, serve 启动时报清晰错误。
try:
    import zenoh  # type: ignore
except ImportError:  # pragma: no cover
    zenoh = None  # type: ignore

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.tools import msgpack_numpy
from starVLA.eval.core.checkpoint_utils import resolve_checkpoint, torch_dtype

PROTOCOL_VERSION = 3

# ───────────────────────── 部署 config (overlay) ─────────────────────────
#
# 镜像 open-eval 的 --eval-yaml: --deploy-yaml overlay 跟 ckpt config OmegaConf.merge,
# 承载部署参数 (tokenizer 路径 + device/dtype/listen_addr/key_prefix/unnorm_key)。
# 优先级: CLI > deploy-yaml > ckpt config > 代码默认。


def merge_deploy_overlay(ckpt_cfg: dict, deploy_yaml: Path | None):
    """``OmegaConf.merge(ckpt_cfg, deploy_overlay)`` —— 语义同 ``eval_starvla.load_merged_config``。

    overlay 覆盖 ckpt config 同名字段 (如 ``framework.tokenizer.model_path``), 并可加新段
    ``deploy:`` (device/dtype/listen_addr/key_prefix/unnorm_key)。返回 OmegaConf 节点。
    """
    from omegaconf import OmegaConf

    base = OmegaConf.create(ckpt_cfg)
    overlay = OmegaConf.load(str(deploy_yaml)) if deploy_yaml is not None else OmegaConf.create({})
    return OmegaConf.merge(base, overlay)


def resolve_tokenizer_path(merged_cfg) -> str:
    """从 merged config 取 ``framework.tokenizer.model_path``: 文件存在就用; 否则回退 env;
    都没有则抛带可操作提示的 FileNotFoundError。

    取代旧的硬编码候选路径列表 + ``rglob ~/.cache`` —— 路径来源单一 (config > env)。
    """
    from omegaconf import OmegaConf

    configured = OmegaConf.select(merged_cfg, "framework.tokenizer.model_path", default=None)
    if configured and Path(str(configured)).is_file():
        return str(configured)
    env_path = os.environ.get("STARVLA_PALIGEMMA_TOKENIZER")
    if env_path and Path(env_path).is_file():
        print(
            f"[load] framework.tokenizer.model_path={configured!r} 本机缺失, "
            f"回退 env STARVLA_PALIGEMMA_TOKENIZER={env_path}",
            flush=True,
        )
        return env_path
    raise FileNotFoundError(
        f"tokenizer.model_path={configured!r} 本机不存在。在 deploy overlay 里设 "
        f"framework.tokenizer.model_path=<本地路径>, 或 export STARVLA_PALIGEMMA_TOKENIZER=<path>。"
    )


def build_tokenizer_override(merged_cfg, ckpt_cfg: dict) -> list[str]:
    """若 ckpt 原 tokenizer 路径本机缺失, 给 ``from_pretrained`` 生成 ``framework.tokenizer.model_path=`` 覆盖。

    ``from_pretrained`` 读 ckpt 原始 config.yaml: 若其 tokenizer 路径本机就存在, 返回空 (不覆盖);
    否则用 :func:`resolve_tokenizer_path` (merged config) 解析本地副本并覆盖。其余模型配置覆盖
    请写进 deploy-yaml 的 ``framework.*`` 段 (OmegaConf.merge 即生效, 无需 CLI)。
    """
    ckpt_tok = (ckpt_cfg.get("framework", {}) or {}).get("tokenizer", {}).get("model_path")
    if ckpt_tok and Path(str(ckpt_tok)).is_file():
        return []  # ckpt 原路径本机就有, from_pretrained 直接能用
    eff = resolve_tokenizer_path(merged_cfg)
    print(f"[load] ckpt tokenizer {ckpt_tok!r} 本机缺失 → override -> {eff}", flush=True)
    return [f"framework.tokenizer.model_path={eff}"]


# ───────────────────── policy: 模型加载 + z-rio 服务面 ──────────────────


class UMIPI05Policy:
    """加载一个 UMI relative-6d PI05 ckpt, 常驻提供 z-rio **绝对** action dict + health 元数据。

    合并自原 Inferer + Policy: 模型加载 / forward / 反归一化是内部实现面, ``metadata`` /
    :meth:`predict` / :meth:`reset` 是对外服务面。``predict`` dict 进 dict 出: 收观测
    (含 state, 旋转 quaternion xyzw), 经完整 transform ``unapply`` 解出绝对 action (state 做
    SE(3) 基准); forward 时 state 仍按 0 填充 (PI0 不把 state 作连续输入)。
    """

    def __init__(
        self,
        ckpt_path: str | Path,
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

        # 训练图像分辨率 (merged config = ckpt + deploy overlay)。
        ir = OmegaConf.select(merged_cfg, "framework.image_resolution", default=[224, 224])
        self.image_resolution = (int(ir[0]), int(ir[1]))

        # 反归一化器: PolicyNormProcessor 重建训练 transform 链 + 用 _per_key 绑定逐时间步
        # 统计 (与 websocket PolicyServerWrapper 同源)。复用调用方已加载的 ckpt_cfg /
        # norm_stats, 不再重复 read_mode_config。unapply_actions 走完整 transform 链
        # (denorm + 相对→绝对 SE(3) + topic 拆分) 输出绝对 action dict (旋转 quat xyzw)。
        self._norm_proc = PolicyNormProcessor(
            str(self.ckpt_path),
            unnorm_key=unnorm_key,
            cfg=ckpt_cfg,
            norm_stats=norm_stats,
        )
        self.action_keys = self._norm_proc.action_keys

        # chunk_size (H) / action_dim 由 norm_proc 暴露 (它已持有逐 key 维度 + 逐时间步统计)。
        self.chunk_size = self._norm_proc.chunk_size
        self.action_dim_actual = self._norm_proc.action_dim

        print(
            f"[load] ckpt={self.ckpt_path.name}\n"
            f"       action_keys={self.action_keys} action_dim={self.action_dim_actual} "
            f"chunk={self.chunk_size} image_resolution={self.image_resolution} "
            f"device={device} dtype={dtype}",
            flush=True,
        )

        # tokenizer 覆盖由调用方经 build_tokenizer_override 算好, 透传进 config_overrides。
        print("[load] loading framework (torch.load state_dict) ...", flush=True)
        model = baseframework.from_pretrained(str(self.ckpt_path), config_overrides=config_overrides or None)
        self.model = model.to(device=self.device, dtype=self._torch_dtype).eval()
        self._torch = torch
        # zenoh queryable 可能多线程并发回调 → forward 必须串行化 (CUDA + 多线程 forward 不安全)。
        self._lock = threading.Lock()
        self._infer_count = 0  # 推理计数 (延迟日志用)

    def _to_train_pil(self, image: np.ndarray):
        """uint8 HWC RGB → 拉伸到 image_resolution 的 PIL (复刻训练 _pack_sample)。

        训练用 ``Image.fromarray(img).resize((H,W))`` 拉伸; PI0 推理用 ``resize_with_pad``
        (letterbox)。不一致 → 真机 16:9 图被加黑边 → OOD。这里先按训练拉伸到目标尺寸,
        PI0 的 resize_with_pad 见已是目标尺寸直接 no-op。
        """
        from PIL import Image

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"image must be HWC RGB, got {arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got {arr.dtype}")
        return Image.fromarray(np.ascontiguousarray(arr)).resize(self.image_resolution)

    @property
    def metadata(self) -> dict[str, Any]:
        """z-rio ``StarVla._validate_server_metadata`` 校验的那组契约字段。"""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "checkpoint_path": str(self.ckpt_path),
            "framework": "PI05",
            "normalized_actions": False,
            "action_chunk_size": int(self.chunk_size),
            "action_mode": "absolute",
            "action_representation": "position_3+quaternion_xyzw_4+gripper_1",
            "state_representation": "position_3+quaternion_xyzw_4+gripper_1",
            "rotation_convention": "quaternion_xyzw",
            "gripper_unit": "meter",
            "obs_image_size": list(self.image_resolution),
            "image_count": 1,
            "default_unnorm_key": "new_embodiment",
            "available_unnorm_keys": ["new_embodiment"],
            "action_keys": self.action_keys,
            "state_keys": self._norm_proc.state_keys,
        }

    def predict(self, obs: dict) -> dict:
        """dict 进 dict 出: 观测 → 绝对动作 dict。

        ``obs`` = ``{video: {cam_wrist_0: HWC uint8}, language: str, state: {topic: 1D,
        旋转用 quaternion xyzw}}``。pre (图像拉伸) + forward + post (denorm + 相对→绝对
        SE(3) 合成 + topic 拆分) 全在 ``_lock`` 下串行; post 由
        :meth:`PolicyNormProcessor.unapply_actions` 完成 (与 open-eval 同源)。返回 dict
        含 ``action`` (按 topic 的**绝对**动作, 旋转 quaternion xyzw) 与 ``org_action``
        (模型原始 normalized 输出)。
        """
        video = obs.get("video") or {}
        image = video.get("cam_wrist_0")
        language = obs.get("language")
        state = obs.get("state") or {}
        if image is None:
            raise KeyError("obs['video']['cam_wrist_0'] missing")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("obs['language'] must be a non-empty string")

        torch = self._torch
        with self._lock:
            t0 = time.perf_counter()
            pil = self._to_train_pil(np.ascontiguousarray(np.asarray(image)))
            example: dict[str, Any] = {"image": [pil], "lang": language}
            t_pre = time.perf_counter()
            with torch.inference_mode():
                out = self.model.predict_action(examples=[example])
            t_fwd = time.perf_counter()
            norm = np.asarray(out["normalized_actions"], dtype=np.float32)
            if norm.ndim == 3 and norm.shape[0] == 1:
                norm = norm[0]
            if norm.shape[-1] < self.action_dim_actual:
                raise ValueError(f"model output dim {norm.shape[-1]} < action_dim {self.action_dim_actual}")
            result = self._norm_proc.unapply_actions({"org_action": norm[..., : self.action_dim_actual], "state": state})
            t_post = time.perf_counter()
        # 延迟日志: pre (图拉伸) / fwd (GPU forward) / post (denorm + SE(3)) 三段 + 总时。
        dt_pre = (t_pre - t0) * 1e3
        dt_fwd = (t_fwd - t_pre) * 1e3
        dt_post = (t_post - t_fwd) * 1e3
        dt_total = dt_pre + dt_fwd + dt_post
        self._infer_count += 1
        print(
            f"[infer #{self._infer_count}] total={dt_total:.1f}ms "
            f"(pre={dt_pre:.1f} fwd={dt_fwd:.1f} post={dt_post:.1f}) lang={language!r}",
            flush=True,
        )
        return result

    def reset(self) -> None:
        """PI05 推理无状态; 保留接口对齐 z-rio episode 切换协议。"""


# ──────────────────────── Zenoh 服务 (对接 z-rio) ────────────────────────
#
# z-rio ``rio/policies/starvla.py`` 客户端向 ``<key_prefix>/infer`` 发
# ``{video, language, state}`` (state 旋转 quaternion xyzw), 收**绝对** action dict
# (按 topic)。线协议 = msgpack-numpy (与 z-rio ``_pack_array`` 完全一致)。


def _pack(value: Any) -> bytes:
    from msgpack import _cmsgpack

    return _cmsgpack.Packer(default=msgpack_numpy.pack_array, use_bin_type=True).pack(value)


def _unpack(value: bytes) -> Any:
    from msgpack import _cmsgpack

    unpacker = _cmsgpack.Unpacker(object_hook=msgpack_numpy.unpack_array, raw=False)
    unpacker.feed(value)
    return unpacker.unpack()


class StarVlaZenohServer:
    """通过 Zenoh queryable 暴露 :class:`UMIPI05Policy` (health / infer / reset)。"""

    def __init__(
        self,
        policy: UMIPI05Policy,
        *,
        listen_addr: str,
        key_prefix: str,
        shutdown_timeout: float = 5.0,
    ) -> None:
        self.policy = policy
        self.listen_addr = listen_addr
        self.key_prefix = key_prefix.rstrip("/")
        self.shutdown_timeout = float(shutdown_timeout)
        self._stop = threading.Event()
        self._shutdown_complete = threading.Event()
        self._watchdog_started = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def _start_shutdown_watchdog(self) -> None:
        """Zenoh 清理卡住时兜底退出, 避免模型长期占显存。"""
        if self.shutdown_timeout <= 0 or self._watchdog_started.is_set():
            return
        self._watchdog_started.set()

        def force_exit_if_stuck() -> None:
            if self._shutdown_complete.wait(self.shutdown_timeout):
                return
            print(f"[server] cleanup exceeded {self.shutdown_timeout}s; forcing exit", flush=True)
            os._exit(0)

        threading.Thread(target=force_exit_if_stuck, name="starvla-shutdown-watchdog", daemon=True).start()

    def serve_forever(self, *, install_signal_handlers: bool = True) -> None:
        config = zenoh.Config()
        config.insert_json5("mode", "'peer'")
        config.insert_json5("listen/endpoints", f'["tcp/{self.listen_addr}"]')
        session_cm = zenoh.open(config)
        session = session_cm.__enter__()

        queryables = [
            session.declare_queryable(f"{self.key_prefix}/health", self._handle_health),
            session.declare_queryable(f"{self.key_prefix}/infer", self._handle_infer),
            session.declare_queryable(f"{self.key_prefix}/reset", self._handle_reset),
        ]

        def stop(signum: int, frame: Any) -> None:
            del frame
            print(f"[server] received signal {signum}; stopping", flush=True)
            self.request_stop()
            self._start_shutdown_watchdog()

        if install_signal_handlers:
            import signal

            signal.signal(signal.SIGINT, stop)
            signal.signal(signal.SIGTERM, stop)
        print(
            f"[server] StarVLA PI05 Zenoh listening at tcp/{self.listen_addr} (key prefix {self.key_prefix})",
            flush=True,
        )
        try:
            self._stop.wait()
        finally:
            try:
                for q in queryables:
                    q.undeclare()
                session_cm.__exit__(None, None, None)
            finally:
                self._shutdown_complete.set()

    def _handle_health(self, query: Any) -> None:
        self._reply(query, lambda _: self.policy.metadata)

    def _handle_infer(self, query: Any) -> None:
        self._reply(
            query,
            lambda payload: self.policy.predict(
                {
                    "video": payload["video"],
                    "language": payload["language"],
                    "state": payload.get("state", {}),
                }
            ),
        )

    def _handle_reset(self, query: Any) -> None:
        self.policy.reset()
        self._reply(query, lambda _: {"reset": True})

    def _reply(self, query: Any, handler: Any) -> None:
        """共用的解包 / 协议校验 / 错误封装 / 回复。"""
        with query:
            try:
                envelope = _unpack(query.payload.to_bytes())
                if envelope.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError(
                        f"protocol mismatch: request={envelope.get('protocol_version')} server={PROTOCOL_VERSION}"
                    )
                payload = envelope.get("payload", {})
                if not isinstance(payload, dict):
                    raise TypeError("request payload must be a dict")
                response = {"ok": True, "data": handler(payload)}
            except Exception as error:
                import logging

                logging.exception("StarVLA RPC request failed")
                response = {"ok": False, "error": str(error), "data": {}}
            query.reply(
                query.key_expr,
                zenoh.ZBytes(_pack(response)),
                encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
            )


# ────────────────────────── config 加载 (yaml > ckpt > 默认) ─────────────


@dataclass
class DeployConfig:
    """部署运行参数 (deploy-yaml 的 ``deploy:`` 段), 默认值集中在此 (单一真相源)。

    由 :meth:`from_merged` 从 merged config 解析一次; 优先级 deploy-yaml > ckpt config > 默认。
    """

    device: str = "cuda"
    dtype: str = "bf16"
    unnorm_key: str | None = None
    listen_addr: str = "127.0.0.1:5570"
    key_prefix: str = "z-rio/starvla"
    shutdown_timeout: float = 5.0

    @classmethod
    def from_merged(cls, merged_cfg) -> "DeployConfig":
        from omegaconf import OmegaConf

        section = dict(OmegaConf.select(merged_cfg, "deploy", default={}) or {})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in section.items() if k in known})


@dataclass(frozen=True)
class Deployment:
    """一次加载的全部部署上下文: ckpt + overlay + norm_stats + 部署参数 + tokenizer 覆盖。

    ``read_mode_config`` 只在 :meth:`load` 里跑一次, ``PolicyNormProcessor`` 复用其结果,
    避免 ckpt ``config.yaml`` / ``dataset_statistics.json`` 被读两遍。
    """

    ckpt_path: Path
    ckpt_cfg: dict
    merged_cfg: Any
    norm_stats: dict
    deploy: DeployConfig
    tokenizer_override: list[str]

    @classmethod
    def load(cls, ckpt_arg, deploy_yaml: Path | None) -> "Deployment":
        """``resolve_checkpoint`` → ``read_mode_config`` → ``merge_deploy_overlay``。

        ckpt_cfg 是原始 (未 overlay) config, 用于判断 tokenizer 原路径是否本机存在;
        norm_stats 透传给 ``PolicyNormProcessor`` 复用。
        """
        from starVLA.model.framework.share_tools import read_mode_config

        ckpt_path = resolve_checkpoint(ckpt_arg)
        ckpt_cfg, norm_stats = read_mode_config(str(ckpt_path))
        merged_cfg = merge_deploy_overlay(ckpt_cfg, deploy_yaml)
        return cls(
            ckpt_path=ckpt_path,
            ckpt_cfg=ckpt_cfg,
            merged_cfg=merged_cfg,
            norm_stats=norm_stats,
            deploy=DeployConfig.from_merged(merged_cfg),
            tokenizer_override=build_tokenizer_override(merged_cfg, ckpt_cfg),
        )


def _serve_main(args: argparse.Namespace) -> None:
    if zenoh is None:
        raise ImportError(
            "zenoh 未安装 —— serve 需要 zenoh。用装有 zenoh 的环境运行 "
            "(如 /home/zerith/miniforge3/envs/starVLA/bin/python)。"
        )
    dep = Deployment.load(args.checkpoint, args.deploy_yaml)
    # 部署参数全部来自 deploy-yaml 的 deploy: 段 (deploy > ckpt config > 代码默认); CLI 不再逐项覆盖。
    policy = UMIPI05Policy(
        dep.ckpt_path,
        dep.ckpt_cfg,
        dep.merged_cfg,
        dep.norm_stats,
        device=dep.deploy.device,
        dtype=dep.deploy.dtype,
        unnorm_key=dep.deploy.unnorm_key,
        config_overrides=dep.tokenizer_override,
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
        help="部署 overlay YAML: 跟 ckpt config OmegaConf.merge, 承载 tokenizer 路径 + deploy: 段 "
        "(device/dtype/listen_addr/key_prefix/unnorm_key/shutdown_timeout)。",
    )
    ps.set_defaults(_fn=_serve_main)

    args = parser.parse_args(argv)
    args._fn(args)


if __name__ == "__main__":
    main()
