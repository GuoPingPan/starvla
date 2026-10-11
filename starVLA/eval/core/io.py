"""Stage 1 产物的 schema-stable 读写。

每次 eval run 写出两个 artifact:
  - ``predictions.npz``: per-anchor 的 raw 数组 (模型预测 + 真值)。
  - ``metrics.json``: 聚合指标 + run metadata + classification。

两个文件都内嵌 ``__schema_version``,Stage 2 (可视化) 读到不匹配版本时
应当显式拒绝,而不是静默画出错误结果。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Schema 版本。每次改 predictions.npz 或 metrics.json 的字段结构就 bump。
# 读方按这两个常量校验,不匹配就报错。
PREDICTIONS_SCHEMA_VERSION = 4
METRICS_SCHEMA_VERSION = 4

# obs_images.npz sidecar (可视化用的观测图像) — 跟 predictions.npz 解耦, 自带独立版本。
# Stage 2 缺该文件时安静退化成无图 rrd, 所以它不进 REQUIRED_PREDICTION_KEYS。
OBS_IMAGES_SCHEMA_VERSION = 1
OBS_IMAGES_FILENAME = "obs_images.npz"

# predictions.npz 必须包含的数组 key。多写的字段允许但跨版本不保证存在。
REQUIRED_PREDICTION_KEYS = (
    "anchors",                 # (N,) int64 — anchor frame 索引
    "state_at_anchor",         # (N, 8) raw — 每个 anchor 的输入 state
    "action_gt",               # (N, 8) raw — 每个 anchor 在 horizon_step 处的 GT (单值, 跟 state 同 shape)
    "action_pred_chunks",      # (N, H, 8) raw — 每个 anchor 的完整预测 chunk (16-step)
    "latency_seconds",         # (N,) float64 — per-sample 推理延迟
)


def write_predictions(output_dir: Path, results: dict[str, np.ndarray]) -> Path:
    """写 predictions.npz, 自动附上 schema 版本。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"__schema_version": PREDICTIONS_SCHEMA_VERSION, **results}
    path = output_dir / "predictions.npz"
    np.savez_compressed(path, **payload)
    return path


def read_predictions(output_dir: Path) -> dict[str, np.ndarray]:
    """读 predictions.npz, 校验 schema 版本 + 必填字段。

    读到的 dict 不包含 ``__schema_version`` (内部使用后剥离)。
    """
    path = Path(output_dir) / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions.npz at {path}")
    with np.load(path, allow_pickle=False) as data:
        # 旧文件可能没写版本号,按 v0 处理;不匹配就报错让用户重跑或迁移
        version = int(data["__schema_version"]) if "__schema_version" in data.files else 0
        if version != PREDICTIONS_SCHEMA_VERSION:
            raise ValueError(
                f"predictions.npz schema mismatch: file has v{version}, "
                f"reader expects v{PREDICTIONS_SCHEMA_VERSION}. "
                f"Re-run Stage 1 or migrate the file."
            )
        results = {key: np.asarray(data[key]) for key in data.files if key != "__schema_version"}
    missing = set(REQUIRED_PREDICTION_KEYS) - set(results)
    if missing:
        raise ValueError(f"predictions.npz at {path} missing keys: {sorted(missing)}")
    return results


def write_metrics(output_dir: Path, report: dict[str, Any]) -> Path:
    """写 metrics.json, 自动附上 schema 版本。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"__schema_version": METRICS_SCHEMA_VERSION, **report}
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)
    return path


def read_metrics(output_dir: Path) -> dict[str, Any]:
    """读 metrics.json, 校验 schema 版本。"""
    path = Path(output_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {path}")
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    version = int(report.get("__schema_version", 0))
    if version != METRICS_SCHEMA_VERSION:
        raise ValueError(
            f"metrics.json schema mismatch: file has v{version}, "
            f"reader expects v{METRICS_SCHEMA_VERSION}."
        )
    return report


def _sanitize_cam_key(key: str) -> str:
    """把 cam key (e.g. 'video.cam_wrist_0') 转成合法 npz array 名 (npz key 不能有点)。"""
    return key.replace(".", "_")


def write_obs_images(
    output_dir: Path,
    cam_keys: list[str],
    images_per_cam: dict[str, np.ndarray],
    anchors: np.ndarray,
) -> Path:
    """写 ``obs_images.npz`` sidecar (schema v1), 自带版本号。

    每 cam 一个 ``images_<sanitized>`` uint8 ``(N,H,W,C)`` 数组; ``cam_keys`` 存成 unicode
    str ndarray (无 pickle, 兼容 ``allow_pickle=False`` 读取); ``anchors`` ``(N,)`` int64 跟
    predictions.npz 同序, 供 viz 自描述。

    跟 predictions.npz 解耦: Stage 2 缺该文件时安静退化成无图 rrd。
    """
    output_dir = Path(output_dir)
    payload: dict[str, Any] = {
        "__schema_version": OBS_IMAGES_SCHEMA_VERSION,
        "cam_keys": np.asarray(cam_keys),  # unicode str ndarray, 不触发 pickle
        "anchors": np.asarray(anchors, dtype=np.int64),
    }
    for key in cam_keys:
        arr = np.asarray(images_per_cam[key])
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        payload[f"images_{_sanitize_cam_key(key)}"] = arr
    path = output_dir / OBS_IMAGES_FILENAME
    np.savez_compressed(path, **payload)
    return path


def read_obs_images(output_dir: Path) -> dict[str, np.ndarray] | None:
    """读 ``obs_images.npz`` sidecar。文件不存在 → 返回 ``None`` (优雅退化)。

    存在则校验 schema 版本 (不匹配报错, 同 ``read_predictions`` 逻辑), 返回
    ``{cam_key: (N,H,W,C) uint8}`` (key 还原回原 cam key)。
    """
    path = Path(output_dir) / OBS_IMAGES_FILENAME
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        version = int(data["__schema_version"]) if "__schema_version" in data.files else 0
        if version != OBS_IMAGES_SCHEMA_VERSION:
            raise ValueError(
                f"obs_images.npz schema mismatch: file has v{version}, "
                f"reader expects v{OBS_IMAGES_SCHEMA_VERSION}."
            )
        cam_keys = [str(k) for k in np.asarray(data["cam_keys"]).tolist()]
        out: dict[str, np.ndarray] = {}
        for key in cam_keys:
            out[key] = np.asarray(data[f"images_{_sanitize_cam_key(key)}"])
    return out


def _json_ready(value: Any) -> Any:
    """递归把 numpy / Path 等非 JSON 原生类型转成原生。

    保留 ``__schema_version`` 等普通字段不动。
    """
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


# 公开别名: ``replay_umi_train_episode.py`` 等外部脚本需要把含 numpy / Path 的
# report dict 序列化成 JSON, 复用同一个 helper 避免重复实现。
json_ready = _json_ready
