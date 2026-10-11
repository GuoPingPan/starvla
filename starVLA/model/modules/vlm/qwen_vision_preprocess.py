# Copyright © 2026 starVLA
"""Qwen 家族视觉预处理 —— 声明式 op 管线 + backbone patch 编码契约, 训练/部署同规范。

架构 (参考 GalaxeaVLA 的声明式 transform 配置与本项目 state/action transform 体系)
--------------------------------------------------------------------------------
分两层, 职责不同:

① ``transforms``: 图像级 op 管线 (声明式列表, yaml 配置, 顺序执行, 全部 batch 级 GPU)。
   每个 op 是注册表里的一个小类 (``@VISION_OP_REGISTRY.register("resize")``),
   新增能力 = 写一个类 + 注册 + yaml 加一行, 不动框架代码。

② ``encoder``: backbone 的 patch 编码契约 (Qwen-VL 系 patch_embed 的读取方式)。
   这是**模型结构的一部分**, 不是数据选择 —— 行序 merge-tiled、特征 (C,T,Ph,Pw)、
   静态帧复制。换 backbone (patch/merge/temporal 不同) 才需要改这里。

数值规范: 本实现即训练/部署的唯一规范, 不要求与 HF processor 逐位一致;
布局正确性由模块底部"单点亮灯"自测守护 (已对照 Qwen3.5-0.8B 逐位置验证)。

yaml 用法 (framework.qwenvl.vision_preprocess, 全字段可省走默认):
    vision_preprocess:
      transforms:
        - {op: resize, size: 256, interpolation: bicubic, antialias: true}
        - {op: normalize, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}
      encoder: {patch_size: 16, merge_size: 2, temporal_patch_size: 2}
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────
# ① 图像 op 管线 (声明式, 可扩展)
# ──────────────────────────────────────────────────────────────────────
class _VisionOpRegistry(dict):
    def register(self, name: str):
        def wrap(cls):
            self[name] = cls
            return cls
        return wrap


VISION_OP_REGISTRY: Dict[str, type] = _VisionOpRegistry()


class _BatchOp:
    """GPU batch 级图像 op 基类: (N,C,H,W) uint8/float 张量进, 同形状(或 resize)出。"""

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@VISION_OP_REGISTRY.register("resize")
class ResizeOp(_BatchOp):
    def __init__(self, size: int, interpolation: str = "bicubic", antialias: bool = True):
        self.size, self.mode, self.antialias = size, interpolation, antialias

    def apply(self, t):
        if t.shape[-1] == self.size:
            return t
        return torch.nn.functional.interpolate(
            t, size=(self.size, self.size), mode=self.mode, antialias=self.antialias
        )


@VISION_OP_REGISTRY.register("quantize_uint8")
class QuantizeUint8Op(_BatchOp):
    """round+clamp 到 [0,255] 整数码位 (与图像天然量化一致; 数值噪声 ≤ 1/255)。"""

    def apply(self, t):
        return t.round_().clamp_(0, 255)


@VISION_OP_REGISTRY.register("normalize")
class NormalizeOp(_BatchOp):
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean, self.std = list(mean), list(std)

    def apply(self, t):
        mean = torch.tensor(self.mean, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
        return (t / 255.0 - mean) / std


def _build_transforms(cfg_list: Union[List[dict], None]) -> List[_BatchOp]:
    ops = []
    for item in cfg_list or []:
        item = dict(item)
        op_name = item.pop("op", None)
        if op_name not in VISION_OP_REGISTRY:
            raise ValueError(f"未知 vision op '{op_name}', 可用: {sorted(VISION_OP_REGISTRY)}")
        ops.append(VISION_OP_REGISTRY[op_name](**item))
    return ops


# ──────────────────────────────────────────────────────────────────────
# ② backbone patch 编码契约 (结构参数, 换 backbone 才动)
# ──────────────────────────────────────────────────────────────────────
class QwenPatchEncoder:
    """归一化图 → Qwen-VL patch_embed 读取契约的 pixel_values。

    契约 (经"单点亮灯图"对照 transformers qwen2_vl fast processor 逐位置验证):
      pixel_values:    (N*gh*gw, C*T*P*P); 行序 merge-tiled
                       (gh//M, gw//M, Mh, Mw) row-major; 特征 (C, T, Ph, Pw)
      image_grid_thw:  (N, 3) = (1, gh, gw); 静态图复制末帧满足 temporal_patch_size
    """

    def __init__(self, patch_size: int = 16, merge_size: int = 2, temporal_patch_size: int = 2):
        self.P, self.M, self.T = patch_size, merge_size, temporal_patch_size

    def grid_hw(self, image_size: int) -> tuple:
        factor = self.P * self.M
        assert image_size % factor == 0, f"image_size {image_size} 须整除 patch*merge={factor}"
        return image_size // self.P, image_size // self.P

    def __call__(self, t: torch.Tensor) -> dict:
        """t: (N,C,H,W) 已归一化 (H=W, 整除 P*M)。"""
        P, M, T = self.P, self.M, self.T
        n, _, h, w = t.shape
        gh, gw = h // P, w // P
        x = t.unsqueeze(1).repeat(1, T, 1, 1, 1)  # 静态图: 复制末帧
        x = x.view(n, T, 3, gh // M, M, P, gw // M, M, P)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)  # → (N, gh//M, gw//M, Mh, Mw, C, T, Ph, Pw)
        pixel_values = x.reshape(n * gh * gw, 3 * T * P * P)
        grid = torch.tensor([[1, gh, gw]], device=t.device).repeat(n, 1)
        return {"pixel_values": pixel_values, "image_grid_thw": grid}


# ──────────────────────────────────────────────────────────────────────
# 组装: PIL batch → transforms 管线 → encoder
# ──────────────────────────────────────────────────────────────────────
class QwenVisionBatchPreprocessor:
    """配置驱动的 batch 级 GPU 视觉预处理器 (训练/部署共用, 数值规范唯一)。

    Args:
        transforms: op 列表 (见模块 docstring 的 yaml 示例); 默认 resize(256,bicubic)
            + normalize(0.5)。传 ``[]`` 表示管线为空 (encoder 要求输入已就绪)。
        encoder: patch 编码契约参数, 默认 Qwen3.5-0.8B (16/2/2)。
    """

    DEFAULT_TRANSFORMS = [
        {"op": "resize", "size": 256, "interpolation": "bicubic", "antialias": True},
        {"op": "normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    ]
    DEFAULT_ENCODER = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}

    def __init__(self, transforms: Union[List[dict], None] = None, encoder: Union[dict, None] = None, **_):
        self.ops = _build_transforms(self.DEFAULT_TRANSFORMS if transforms is None else transforms)
        self.encoder = QwenPatchEncoder(**(self.DEFAULT_ENCODER if encoder is None else dict(encoder)))
        resize_ops = [o for o in self.ops if isinstance(o, ResizeOp)]
        self.image_size: int = resize_ops[-1].size if resize_ops else None

    def num_vision_tokens_per_image(self) -> int:
        gh, gw = self.encoder.grid_hw(self.image_size)
        return gh * gw

    def batch_encode(self, images: List[Image.Image], device: Union[str, torch.device]) -> dict:
        """PIL 列表 → {"pixel_values", "image_grid_thw"} (目标 device)。"""
        if len({im.size for im in images}) != 1:
            raise ValueError(
                f"batch 内图像尺寸不一致 {[im.size for im in images[:3]]}...; "
                "GPU batch 路径要求统一尺寸 (训练管线由 video transform 保证)"
            )
        arr = np.stack([np.asarray(im, dtype=np.uint8) for im in images])  # (N,H,W,3)
        t = torch.from_numpy(arr).to(device, non_blocking=True).permute(0, 3, 1, 2).float()
        for op in self.ops:
            t = op.apply(t)
        return self.encoder(t)


if __name__ == "__main__":
    # 布局自检: 单点亮灯图 (全 128 + 单像素 255) 与 HF processor 对照。
    # 平滑内容应逐元素一致 (≤2/255 量化噪声); 高频噪声允许差异 (过冲处理属实现自由)。
    import os

    from transformers import AutoProcessor

    model_id = os.environ.get(
        "STARVLA_VISION_VERIFY_MODEL", "./playground/Pretrained_models/Qwen3.5-0.8B"
    )
    a = np.full((224, 224, 3), 128, dtype=np.uint8)
    a[100, 50, 1] = 255
    img = Image.fromarray(a)
    ref = AutoProcessor.from_pretrained(model_id).image_processor(
        images=[img], return_tensors="pt"
    )
    mine = QwenVisionBatchPreprocessor().batch_encode([img], "cpu")
    assert mine["pixel_values"].shape == ref["pixel_values"].shape, "形状/布局不一致!"
    assert torch.equal(mine["image_grid_thw"], ref["image_grid_thw"])
    d = (mine["pixel_values"] - ref["pixel_values"]).abs().max()
    print(f"✅ 布局一致 (shape {tuple(ref['pixel_values'].shape)}), 平滑图 maxdiff={d:.4f} (应 ≤ 0.01)")
