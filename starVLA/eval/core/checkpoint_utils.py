"""开环评测共享的薄工具函数。

从 ``examples/Z-umi/eval_files/eval_utils.py`` 迁来, 供新的多卡评测入口
``starVLA/training/eval_starvla.py`` 使用。所有函数都是 stateless:

  - ``torch_dtype``         : 字符串名 → torch.dtype (CLI ``--dtype`` 解析用)
  - ``to_numpy``            : Tensor / list / ndarray → numpy
  - ``discover_checkpoint_steps`` : 扫 ``run_dir/checkpoints/`` 自动发现所有 step
  - ``checkpoint_path``     : 根据 step 号找 .pt / .safetensors 文件
  - ``resolve_checkpoint``  : 文件 / run 目录 → step 号最大的 ckpt (部署 CLI 入口)
  - ``pick_mixture_entry``  : 从命名 mixture 里挑一个 dataset entry

复杂业务逻辑 (per-sample decode / episode range / 推理循环) 在
``starVLA/training/eval_starvla.py`` 的 ``VLAEvaluator`` 类里。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

# ─────────────────────────── 小工具 ────────────────────────────────────


def torch_dtype(name: str) -> torch.dtype:
    """字符串名 → torch.dtype。CLI ``--dtype`` 参数解析用。"""
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def to_numpy(value: Any) -> np.ndarray:
    """把 torch.Tensor / list / ndarray 统一成 numpy, Tensor 会 detach+cpu。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ─────────────────────────── checkpoint 查找 ──────────────────────────


_CKPT_NAME_PATTERN = re.compile(r"^steps_(\d+)_pytorch_model\.pt$")


def _checkpoint_search_dirs(run_dir: Path) -> list[Path]:
    """候选 ckpt 所在目录: 先标准 ``run_dir/checkpoints/``, 再扁平 ``run_dir/``。

    扁平布局 (ckpt 直接平铺在 run_dir, 无 ``checkpoints/`` 子目录) 见于部分手动导出
    或旧导出的 run_dir; 两种都支持, 保证都能被 eval 发现。
    """
    dirs: list[Path] = []
    std = run_dir / "checkpoints"
    if std.is_dir():
        dirs.append(std)
    if run_dir.is_dir() and any(_CKPT_NAME_PATTERN.match(p.name) for p in run_dir.iterdir()):
        dirs.append(run_dir)
    return dirs


def discover_checkpoint_steps(run_dir: Path) -> list[int]:
    """扫描 ``run_dir/checkpoints/`` (标准) 或 ``run_dir/`` (扁平) 下所有
    ``steps_<N>_pytorch_model.pt`` 文件, 返回排序后的 step 列表。

    给 ``--checkpoint-steps`` 不指定时用, 自动覆盖所有可用 ckpt。
    """
    dirs = _checkpoint_search_dirs(run_dir)
    if not dirs:
        raise FileNotFoundError(f"No checkpoints directory at {run_dir / 'checkpoints'} (nor flat layout in {run_dir})")
    steps: list[int] = []
    for d in dirs:
        for path in d.iterdir():
            match = _CKPT_NAME_PATTERN.match(path.name)
            if match:
                steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No steps_*_pytorch_model.pt checkpoints found under {dirs}")
    return sorted(set(steps))


def checkpoint_path(run_dir: Path, step: int) -> Path:
    """根据 step 号找到 ckpt 文件实际路径。

    优先匹配 ``steps_<N>_pytorch_model.pt``, 然后 ``steps_<N>_model.safetensors``;
    每个名字先在标准 ``run_dir/checkpoints/`` 找, 再回退扁平 ``run_dir/``。
    两个都不在就抛 FileNotFoundError。
    """
    candidates: list[Path] = []
    for d in _checkpoint_search_dirs(run_dir):
        candidates.append(d / f"steps_{step}_pytorch_model.pt")
        candidates.append(d / f"steps_{step}_model.safetensors")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint for step {step} not found; checked {candidates}")


# 部署 / CLI 入口用的 step+格式通配 pattern (与上面只匹配 ``.pt`` 的
# ``_CKPT_NAME_PATTERN`` 区别: 这里同时接受 ``.safetensors``, 见 resolve_checkpoint)。
_RESOLVE_STEP_PATTERN = re.compile(r"^steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")


def resolve_checkpoint(path: str | Path) -> Path:
    """文件路径或 run 目录 → step 号最大的 ckpt 文件路径。

    部署 / 离线 CLI 入口 (``server_umi_pi05_rio`` 等) 用: 接受直接给
    ``.pt`` / ``.safetensors`` 文件, 或给 run 目录 (标准 ``run_dir/checkpoints/``
    与扁平 ``run_dir/`` 两种布局都扫), 自动取 step 号最大的那个; 同 step 同时存在
    ``.pt`` 与 ``.safetensors`` 时返回 ``.pt`` (与 :func:`checkpoint_path` 优先级一致)。

    与 :func:`discover_checkpoint_steps` 的区别: 后者只扫 ``.pt`` 并返回 step 列表
    (eval 既有契约, 这里不改); 本函数额外兼容 ``.safetensors``, 直接返回文件路径,
    供部署侧 "给个 run 目录就要能跑" 的场景使用。
    """
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.suffix not in {".pt", ".safetensors"}:
            raise ValueError(f"Unsupported checkpoint suffix: {candidate}")
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    search_dirs = [d for d in (candidate / "checkpoints", candidate) if d.is_dir()]
    numbered: list[tuple[int, Path]] = []
    for d in search_dirs:
        for f in d.iterdir():
            match = _RESOLVE_STEP_PATTERN.match(f.name)
            if match:
                numbered.append((int(match.group(1)), f))
    if not numbered:
        raise FileNotFoundError(
            f"No steps_*_pytorch_model.pt / steps_*_model.safetensors under {candidate} " f"(checked {search_dirs})"
        )
    max_step = max(step for step, _ in numbered)
    best = [p for step, p in numbered if step == max_step]
    for p in best:  # .pt 优先 (与 checkpoint_path 的尝试顺序一致)
        if p.suffix == ".pt":
            return p
    return best[0]


# ─────────────────────────── mixture 选择 ──────────────────────────────


def pick_mixture_entry(
    data_mix: str,
    requested: str | None,
) -> tuple[str, str]:
    """从命名 mixture 里挑一个 entry, 返回 ``(dataset_name, robot_type)``。

    mixture 长度为 1 时 ``requested`` 可省; 长度 >1 时必须显式指定 ``requested``,
    避免"训练用 A+B, 评测时默认拿 A"这种隐式选择。
    """
    mixture = DATASET_NAMED_MIXTURES.get(str(data_mix))
    if not mixture:
        raise ValueError(f"Unknown data_mix={data_mix!r}")
    names = [entry[0] for entry in mixture]
    if requested is None:
        if len(mixture) == 1:
            dataset_name, _, robot_type, *_ = mixture[0]  # *_ 容忍可选第 4 元素 (BOS 路径)
            return dataset_name, robot_type
        raise ValueError(
            f"data_mix={data_mix!r} has {len(mixture)} datasets {names}; " f"pass --dataset-name to pick one."
        )
    for dataset_name, _, robot_type, *_ in mixture:  # *_ 容忍可选第 4 元素 (BOS 路径)
        if dataset_name == requested:
            return dataset_name, robot_type
    raise ValueError(f"--dataset-name={requested!r} not in mixture {data_mix!r}; available: {names}")
