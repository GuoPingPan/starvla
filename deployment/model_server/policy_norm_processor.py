# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""PolicyNormProcessor — reuse the training-time ComposedModalityTransform.

This class replaces the hand-rolled un-normalization math that previously
lived in every eval client. It rebuilds the *exact* ``ComposedModalityTransform``
used at training time from a checkpoint:

  1. Read ``config.yaml`` next to the ckpt → resolve ``data_mix`` →
     look up ``robot_type`` from ``DATASET_NAMED_MIXTURES`` →
     fetch the ``DataConfig`` from ``ROBOT_TYPE_CONFIG_MAP``.
  2. Build the transform pipeline via ``data_config.transform()``.
  3. Reconstruct a ``DatasetMetadata`` from the saved
     ``dataset_statistics.json`` (which stores **combined** per-modality
     arrays of length ``D``) by splitting it into per-key entries that match
     ``data_config.action_keys`` / ``state_keys``.
  4. ``set_metadata(...)`` binds the metadata into every transform.

After construction the caller simply invokes :meth:`unapply_actions` (dict in /
dict out) to decode model output + raw state into absolute env-space actions
using the same code path as training — there is no second source of truth for
normalization math.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    StateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.model.framework.share_tools import read_mode_config

logger = logging.getLogger(__name__)


def _resolve_robot_type(
    model_cfg: dict,
    unnorm_key: Optional[str] = None,
) -> str:
    """Look up the training robot_type from the saved cfg.

    Convention used by starVLA train scripts:
      ``cfg.datasets.vla_data.data_mix`` is a key in
      ``DATASET_NAMED_MIXTURES`` whose value is a list of
      ``(dataset_name, weight, robot_type)`` tuples.

    When a data_mix contains entries from multiple robot types (e.g.
    ``bridge_rt_1`` covers ``oxe_bridge`` + ``oxe_rt1``), ``unnorm_key``
    is used to identify which embodiment is requested.  In those mixtures
    the ``robot_type`` field of each entry **matches** the top-level key in
    ``dataset_statistics.json``, so ``unnorm_key`` serves as the selector.
    """
    try:
        data_mix = model_cfg["datasets"]["vla_data"]["data_mix"]
    except (KeyError, TypeError) as e:
        raise KeyError(
            "ckpt config.yaml is missing `datasets.vla_data.data_mix`; cannot resolve training-time robot_type."
        ) from e

    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"data_mix={data_mix!r} not in DATASET_NAMED_MIXTURES "
            f"(available: {sorted(DATASET_NAMED_MIXTURES.keys())[:20]} ...). "
            "Did you forget to register the example under examples/<bench>/train_files/data_registry/?"
        )

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    robot_types = sorted({entry[2] for entry in mixture})

    if len(robot_types) == 1:
        return robot_types[0]

    # Multiple robot types in the mixture.
    # Use unnorm_key as a direct selector: for multi-robot mixtures the
    # dataset_statistics.json top-level keys equal the robot_type values.
    if unnorm_key is not None and unnorm_key in robot_types:
        return unnorm_key

    raise ValueError(
        f"data_mix={data_mix!r} contains multiple robot_types {robot_types}. "
        "Pass `unnorm_key` matching one of them to disambiguate "
        f"(e.g. unnorm_key={robot_types[0]!r})."
    )


def _infer_key_dims(
    data_config: Any,
    combined_stats: Dict[str, Any],
    modality_keys: Sequence[str],
    modality: str,
) -> Dict[str, int]:
    """Compute per-key dimensions for splitting combined stats arrays.

    Lookup priority:
      1. ``data_config.<modality>_key_dims`` — explicit dict classvar on the DataConfig
         (required for DataConfigs with non-uniform or multi-dim keys).
      2. Infer uniformly from stats array length: if ``D_total / n_keys`` is an
         integer, use that as the uniform per-key dim.
      3. Fall back to dim=1 when no stats are available (empty combined dict).

    Args:
        data_config: The DataConfig instance for the current robot type.
        combined_stats: The raw stats dict for the chosen unnorm_key (the value
            of ``norm_stats[unnorm_key]`` from ``dataset_statistics.json``).
        modality_keys: Ordered list of full keys for this modality
            (e.g. ``["action.left_joints", ...]``).
        modality: ``"action"`` or ``"state"``.

    Returns:
        Dict mapping each full key to its integer dimension.
    """
    attr = f"{modality}_key_dims"
    if hasattr(data_config, attr):
        return dict(getattr(data_config, attr))

    # 逐 key 统计 (_per_key): 每 subkey 已独立存放。action 是逐时间步 (H, D) 二维,
    # state 是 per-dim (D,) 一维; 末轴长度都是 per-key 维度, np.asarray 统一处理。
    per_key = combined_stats.get("_per_key", {}).get(modality)
    if per_key:
        dims: Dict[str, int] = {}
        for full_key in modality_keys:
            sub = full_key.split(".", 1)[1]
            arr = per_key[sub].get("q01") or per_key[sub].get("min")
            if not arr:
                arr = next(iter(per_key[sub].values()))
            a = np.asarray(arr)
            dims[full_key] = int(a.shape[-1]) if a.ndim >= 1 else 1
        return dims

    combined = combined_stats.get(modality, {})
    stat_arr = next((v for k, v in combined.items() if k != "mask"), None)
    n_keys = len(modality_keys)
    if stat_arr is not None and n_keys > 0:
        D_total = len(stat_arr)
        if D_total == n_keys:
            return {k: 1 for k in modality_keys}
        elif n_keys > 0 and D_total % n_keys == 0:
            dim = D_total // n_keys
            return {k: dim for k in modality_keys}
        else:
            raise ValueError(
                f"Cannot infer per-key dims for modality={modality!r}: "
                f"D_total={D_total} is not evenly divisible by n_keys={n_keys}. "
                f"Add `{attr} = {{key: dim, ...}}` to the DataConfig "
                f"(keys={list(modality_keys)})."
            )
    return {k: 1 for k in modality_keys}


def _build_dataset_metadata(
    stats_for_key: Dict[str, Any],
    embodiment_tag: Any,
    action_keys: Sequence[str],
    state_keys: Sequence[str],
    action_key_dims: Optional[Dict[str, int]] = None,
    state_key_dims: Optional[Dict[str, int]] = None,
) -> DatasetMetadata:
    """Convert the *combined* stats arrays from ``dataset_statistics.json``
    back into a per-subkey :class:`DatasetMetadata` matching what the training
    pipeline produced.

    The saved ``dataset_statistics.json`` stores stats as flat arrays of
    length ``D = sum(per_key_dims)``.  For each ``"action.<sub>"`` key we
    slice out ``dim_k`` elements starting at the current cursor and store them
    under ``statistics.action.<sub> = {"min": [v0..v_{k-1}], ...}``.

    Args:
        stats_for_key: ``norm_stats[unnorm_key]`` from ``dataset_statistics.json``.
        embodiment_tag: Embodiment tag from the DataConfig.
        action_keys: Ordered list of full action keys.
        state_keys: Ordered list of full state keys.
        action_key_dims: Per-key dimension dict (``{full_key: dim_k}``).
            Defaults to dim=1 for every key.
        state_key_dims: Per-key dimension dict for state keys.
            Defaults to dim=1 for every key.
    """
    if action_key_dims is None:
        action_key_dims = {k: 1 for k in action_keys}
    if state_key_dims is None:
        state_key_dims = {k: 1 for k in state_keys}

    def _split_combined(
        combined: Dict[str, Sequence[float]],
        keys: Sequence[str],
        key_dims: Dict[str, int],
    ):
        """Split combined arrays into per-subkey dicts using per-key dims.

        ``combined`` looks like ``{"min": [..D..], "max": [..D..], "mask": [..D..], ...}``.
        ``keys`` is the ordered list of full keys.
        ``key_dims`` maps each full key to its integer dimension.
        Returns ``(stats_per_subkey, meta_per_subkey)``.
        """
        stats_per_subkey: Dict[str, Dict[str, List[float]]] = {}
        meta_per_subkey: Dict[str, StateActionMetadata] = {}
        cursor = 0
        for full_key in keys:
            subkey = full_key.split(".", 1)[1]
            dim_k = key_dims.get(full_key, 1)
            per_key: Dict[str, List[float]] = {}
            for stat_name, arr in combined.items():
                if stat_name == "mask":
                    continue
                end = cursor + dim_k
                if end > len(arr):
                    # Saved combined array shorter than expected (truncated
                    # pad channels etc.). Skip this stat field silently.
                    continue
                per_key[stat_name] = [float(v) for v in arr[cursor:end]]
            stats_per_subkey[subkey] = per_key
            meta_per_subkey[subkey] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(dim_k,),
                continuous=True,
            )
            cursor += dim_k
        return stats_per_subkey, meta_per_subkey

    state_combined = stats_for_key.get("state", {})

    per_key_action = stats_for_key.get("_per_key", {}).get("action")
    if per_key_action:
        # 逐时间步 (H, D) 统计: 已按 subkey 分好 (UMI 相对位姿 q99 归一化), 无需切 combined 数组。
        action_stats: Dict[str, Dict[str, List[float]]] = {}
        action_meta: Dict[str, StateActionMetadata] = {}
        for full_key in action_keys:
            sub = full_key.split(".", 1)[1]
            d_sub = action_key_dims.get(full_key, 1)
            action_stats[sub] = {k: list(v) for k, v in per_key_action[sub].items()}
            action_meta[sub] = StateActionMetadata(absolute=True, rotation_type=None, shape=(d_sub,), continuous=True)
    else:
        action_combined = stats_for_key.get("action", {})
        action_stats, action_meta = _split_combined(action_combined, action_keys, action_key_dims)

    statistics: Dict[str, Any] = {"action": action_stats}
    modalities: Dict[str, Any] = {"video": {}, "action": action_meta}
    if state_combined:
        state_stats, state_meta = _split_combined(state_combined, state_keys, state_key_dims)
        statistics["state"] = state_stats
        modalities["state"] = state_meta

    return DatasetMetadata.model_validate(
        {
            "statistics": statistics,
            "modalities": modalities,
            "embodiment_tag": embodiment_tag.value if hasattr(embodiment_tag, "value") else embodiment_tag,
        }
    )


class PolicyNormProcessor:
    """Server-side normalization helper backed by training-time transforms.

    Construct once per checkpoint; call :meth:`unapply_actions` (dict in / dict
    out) to decode a normalized action chunk + raw state into absolute env-space
    actions.

    Args:
        ckpt_path: Path to the ``*.pt`` checkpoint (the loader looks for
            ``config.yaml`` and ``dataset_statistics.json`` two dirs up).
        unnorm_key: Which top-level key in ``dataset_statistics.json`` to use.
            ``None`` → auto-pick the only key.
        cfg: Optionally pre-loaded ckpt config (dict from ``read_mode_config``).
            Pass together with ``norm_stats`` to skip re-reading the checkpoint;
            either one ``None`` triggers a fresh ``read_mode_config``.
        norm_stats: Optionally pre-loaded ``dataset_statistics.json`` (same shape
            as ``read_mode_config`` returns). See ``cfg``.
    """

    def __init__(
        self,
        ckpt_path: str,
        unnorm_key: Optional[str] = None,
        cfg: Optional[dict] = None,
        norm_stats: Optional[dict] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)
        # 复用调用方已加载的 config / norm_stats, 避免重复 read_mode_config (config.yaml +
        # dataset_statistics.json 各读一遍); 二者任一缺失则回退自读。
        if cfg is None or norm_stats is None:
            cfg_loaded, norm_stats_loaded = read_mode_config(self._ckpt_path)
            cfg = cfg if cfg is not None else cfg_loaded
            norm_stats = norm_stats if norm_stats is not None else norm_stats_loaded
        self._model_cfg = cfg
        self._norm_stats = norm_stats

        # 3-early) Pick the requested unnorm_key (or auto-select) BEFORE
        # resolving robot_type so we can use it as a hint for multi-robot mixtures.
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            # else: defer error to step 3 below after robot_type resolution attempt
        elif unnorm_key not in norm_stats:
            raise KeyError(f"unnorm_key={unnorm_key!r} not in {list(norm_stats.keys())}")
        self._unnorm_key = unnorm_key  # may still be None for multi-key case

        # 1) Resolve which DataConfig was used at training.
        robot_type = _resolve_robot_type(cfg, unnorm_key=unnorm_key)
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(
                f"robot_type={robot_type!r} not in ROBOT_TYPE_CONFIG_MAP "
                f"(available: {sorted(ROBOT_TYPE_CONFIG_MAP.keys())}). "
                "Make sure the example's data_registry/data_config.py is importable."
            )
        self._data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        self._action_keys: List[str] = list(self._data_config.action_keys)
        self._state_keys: List[str] = list(getattr(self._data_config, "state_keys", []))

        # 2) Build training-time transform pipeline.
        transform = self._data_config.transform()
        if not isinstance(transform, ComposedModalityTransform):
            transform = ComposedModalityTransform(transforms=[transform])
        self._transform = transform

        # 3) Pick the requested unnorm_key (finalize; error if still None here).
        if self._unnorm_key is None:
            raise ValueError(
                f"Multiple unnorm_keys in dataset_statistics.json: "
                f"{list(norm_stats.keys())}. Pass unnorm_key explicitly."
            )
        unnorm_key = self._unnorm_key

        # 4) Resolve per-key dims (handles multi-d action/state keys).
        stats_for_unnorm = norm_stats[unnorm_key]
        self._action_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._action_keys, "action"
        )
        self._state_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._state_keys, "state"
        )

        # 5) Build & bind metadata.
        ds_meta = _build_dataset_metadata(
            stats_for_key=stats_for_unnorm,
            embodiment_tag=self._data_config.embodiment_tag,
            action_keys=self._action_keys,
            state_keys=self._state_keys,
            action_key_dims=self._action_key_dims,
            state_key_dims=self._state_key_dims,
        )
        self._transform.set_metadata(ds_meta)
        self._transform.eval()  # mark transforms as eval-mode

        logger.info(
            "PolicyNormProcessor ready: robot_type=%s, unnorm_key=%s, action_keys=%s (dims=%s), state_keys=%s",
            robot_type,
            unnorm_key,
            self._action_keys,
            [self._action_key_dims[k] for k in self._action_keys],
            self._state_keys,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def action_keys(self) -> List[str]:
        return list(self._action_keys)

    @property
    def state_keys(self) -> List[str]:
        return list(self._state_keys)

    @property
    def unnorm_key(self) -> str:
        return self._unnorm_key

    @property
    def available_unnorm_keys(self) -> List[str]:
        return list(self._norm_stats.keys())

    @property
    def transform(self) -> ComposedModalityTransform:
        return self._transform

    @property
    def chunk_size(self) -> int:
        """Action horizon ``H``: 逐时间步 ``_per_key`` action 统计的第一维。

        非逐时间步归一化 (无 ``_per_key``) 时返回 ``0`` —— 调用方按需处理。
        """
        per_key = self._norm_stats.get(self._unnorm_key, {}).get("_per_key", {}).get("action")
        if not per_key:
            return 0
        stats = next(iter(per_key.values()))
        arr = np.asarray(stats.get("q01") or stats.get("min") or next(iter(stats.values())))
        return int(arr.shape[0]) if arr.ndim >= 1 else 0

    @property
    def action_dim(self) -> int:
        """模型 action 输出维度 = 各 action key 维度之和 (``_action_key_dims``)。"""
        return sum(self._action_key_dims.values())

    # ------------------------------------------------------------------
    # Inverse path (model output → env action)
    # ------------------------------------------------------------------
    def unapply_actions(self, data: dict) -> dict:
        """dict 进 dict 出: 模型 normalized 输出 + raw state 经训练 transform 完整 ``unapply``,
        解出**绝对** action (denorm + 相对→绝对 SE(3) 合成 + topic 拆分)。

        与 open-eval ``eval_starvla._decode_to_raw`` 同源 (同一条 transform 链, 单一真值源):
        state 先 forward 归一化, 再与 action 合并做一次 ``unapply``; 链中的
        ``RelativePoseTransform.unapply`` 用 state 当 SE(3) 基准把相对动作合成绝对
        (缺 state 时静默留在相对态, 与旧 ndarray 行为一致)。

        Args (data):
            org_action: ``(T, D)`` 模型 normalized 输出 (rotation_6d 等),
                ``D == sum(action_key_dims.values())``。
            state: 按 topic 的 **raw** state (``subkey -> 1D ndarray``, 旋转用 quaternion xyzw);
                不可得时给 ``{}``。

        Returns:
            同一个 dict, 加上 ``action``: ``{subkey: (T, dim_k) ndarray}`` **绝对** action
            (旋转 quaternion xyzw)。``org_action`` / ``state`` 原样保留供下游选用。
        """
        normalized_actions = np.asarray(data["org_action"])
        assert normalized_actions.ndim == 2, f"Expected org_action (T, D); got shape {normalized_actions.shape}"
        state_raw = data.get("state") or {}

        # 1) raw state -> forward 归一化 (镜像 _decode_to_raw); 无 state 则跳过。
        #    旋转 JIT 需要状态 ≥2D —— 把单帧 1D topic 升成 (1, D) (action 本就是 (H, D))。
        if state_raw:
            state_dict = {}
            for sub, v in state_raw.items():
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim < 2:
                    arr = arr[None, :]
                state_dict[f"state.{sub}"] = arr
            state_norm = self._transform.apply(state_dict)
        else:
            state_norm = {}

        # 2) org_action (T, D) -> per-key {action.<sub>: tensor (T, dim_k)}。
        pred_t = torch.as_tensor(normalized_actions, dtype=torch.float32)
        action_norm: Dict[str, torch.Tensor] = {}
        cursor = 0
        for full_key in self._action_keys:
            dim_k = self._action_key_dims.get(full_key, 1)
            action_norm[full_key] = pred_t[..., cursor : cursor + dim_k]
            cursor += dim_k
        if cursor != normalized_actions.shape[-1]:
            raise ValueError(
                f"Sum of per-key dims ({cursor}) != action_dim ({normalized_actions.shape[-1]}). "
                f"action_keys={self._action_keys}, action_key_dims={self._action_key_dims}"
            )

        # 3) 合并 unapply: denorm + rotation_6d→quat + RelativePose 相对→绝对。
        restored = self._transform.unapply({**state_norm, **action_norm})

        # 4) 取 action.<sub> -> 嵌套 topic; 有限性检查。
        action: Dict[str, np.ndarray] = {}
        for full_key in self._action_keys:
            sub = full_key.split(".", 1)[1]
            v = restored[full_key]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            action[sub] = np.asarray(v, dtype=np.float32)
        if not all(np.isfinite(v).all() for v in action.values()):
            raise ValueError("Decoded action contains NaN or Inf")

        return {**data, "action": action}

    def concat_action(self, result: dict) -> np.ndarray:
        """把 :meth:`unapply_actions` 返回 dict 里的 ``action`` 各 topic 按 ``action_keys``
        顺序拼回 ``(T, D)`` —— 给需要扁平数组的调用方 (如 websocket ``PolicyServerWrapper``) 用。"""
        action = result["action"]
        parts = [np.asarray(action[full_key.split(".", 1)[1]], dtype=np.float32) for full_key in self._action_keys]
        return np.concatenate(parts, axis=-1)
