"""
PI0 模型 (Physical Intelligence π0 协议的 StarVLA 实现)。

设计概览
--------
PI0 由两条 Gemma 流组成:

    ┌─────────────────────────────────────────────────────┐
    │ Prefix 流 (VLM,PaliGemma 3B)                       │
    │   image_tokens + language_tokens → KV cache         │
    └─────────────────┬───────────────────────────────────┘
                      │ 共享注意力 (forward_shared_gemma_layer)
    ┌─────────────────▼───────────────────────────────────┐
    │ Suffix 流 (Action Expert,小型 Gemma ~1B)            │
    │   state_token + (noisy_action + time)_tokens        │
    │   → v_t (flow matching 速度场)                      │
    └─────────────────────────────────────────────────────┘

两条流在前向时**逐层共享 self-attention** (Q/K/V 拼起来一起算),但每个流
有自己的 FFN 和 layernorm。这是 π0 的核心架构创新。

训练:Flow Matching
------------------
给定 GT action `a`,采样时间 `t ∈ [0,1]` 和噪声 `ε`,构造:
    x_t = t·ε + (1-t)·a
    u_t = ε - a        # 目标速度场
模型预测 v_t,loss = MSE(u_t, v_t)。

推理:从 x_0 = ε (纯噪声) 出发,做 N 步 Euler 积分:
    x_{t+dt} = x_t + dt·v_t,  dt = -1/N

PI05 子类如何复用这个文件
-------------------------
PI05 直接 `class PI05(PI0)`,只覆盖 3 个类属性 (见 PI05.py)。本文件里的
关键 helper `_norm` 同时支持普通 RMSNorm (PI0) 和 adaRMS (PI05),所以
forward 主干不需要分叉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from starVLA.model.modules.vlm.openpi_transformers.gemma import modeling_gemma

from deployment.model_server.tools.image_tools import resize_with_pad, to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.action_model.OpenPI_ActionHead import (
    GemmaDims,
    OpenPI0ActionHead,
    OpenPIActionHeadBase,
    make_att_2d_masks,
    make_att_4d_masks,
)
from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIPaliGemma
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# 默认的图像键名顺序:第三视角 + 左/右手腕视角。
# 顺序很重要,与 checkpoint 的位置编码对齐。
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclass
class Observation:
    """单条观测样本的标准化容器。

    把原始 example dict (image / lang / state) 预处理后封装成这个 dataclass,
    方便在 forward / sample_actions / denoise_step 之间传递。

    - images / image_masks: dict 按 IMAGE_KEYS 顺序,值是 (B, C, H, W) 张量
    - tokenized_prompt: (B, max_token_len) int64,已经 pad/截断
    - tokenized_prompt_mask: (B, max_token_len) bool,True 表示有效 token
    - state: (B, 1, max_state_dim) float32,已经 pad
    """
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    state: torch.Tensor


@dataclass
class PI0DefaultConfig:
    """PI0 默认配置,作为 PI05DefaultConfig 的父类。

    所有字段都可以被 yaml / CLI 覆盖。关键字段说明:
    - action_dim:动作维度 (LIBERO = 7,OFT/DiT 可能不同)
    - action_horizon:单次预测的动作 chunk 长度 (典型 10-50)
    - max_state_dim:state 的固定维度 (用 pad_state_value 填充)
    - discrete_state_input:True 时 state 走语言 tokenizer;False 时走 state_proj
    - num_inference_steps:flow matching 推理步数 (训练时单步,推理时 N 步)
    """

    name: str = "PI0"
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    action_dim: int = 32
    action_horizon: int = 50
    max_state_dim: int = 32
    discrete_state_input: bool = False
    max_token_len: int = 200
    num_inference_steps: int = 10

    image_resolution: list[int] = field(default_factory=lambda: [224, 224])
    image_keys: list[str] = field(default_factory=lambda: list(IMAGE_KEYS))

    tokenizer: dict = field(default_factory=lambda: {"model_path": "", "pad_state_value": -2.0})
    paligemma: dict = field(default_factory=lambda: {"model_name_or_path": ""})
    # PaliGemma 3B 主干维度。这些数字对应官方 PALI-3B checkpoint,改任何一个都
    # 会导致权重加载失败 (shape mismatch)。
    paligemma_config: dict = field(
        default_factory=lambda: {
            "width": 2048,
            "depth": 18,
            "mlp_dim": 16384,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )
    # Action expert (小型 Gemma) 配置。注意 num_kv_heads=1 (MQA),与 PaliGemma
    # 主干一致,这样两者可以做共享 attention。
    action_expert_config: dict = field(
        default_factory=lambda: {
            "width": 1024,
            "depth": 18,
            "mlp_dim": 4096,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )


def _as_dims(config) -> GemmaDims:
    """把 dict 形式的配置转成 GemmaDims dataclass。"""
    return GemmaDims(
        width=int(config.width),
        depth=int(config.depth),
        mlp_dim=int(config.mlp_dim),
        num_heads=int(config.num_heads),
        num_kv_heads=int(config.num_kv_heads),
        head_dim=int(config.head_dim),
    )


class LazyPaliGemmaTokenizer:
    """懒加载的 PaliGemma SentencePiece tokenizer。

    为什么懒加载?
    --------------
    - 多进程训练时,每个 worker 都会构造 framework 对象,如果 tokenizer
      在 __init__ 里就加载文件,会触发重复 IO 和 SentencePiece 内部状态
      冲突。
    - 真正用到 tokenize() 时才加载,避免无用开销。

    state 离散化逻辑
    ----------------
    当传入 state 时,会把每个 state 维度用 256 桶离散化 (linspace(-1, 1, 257)),
    然后拼成 "Task: ..., State: 12 145 0 ...;" 的字符串再 tokenize。
    这是 PI0/PI05 把连续 state 喂进语言模型的标准方式。
    """

    def __init__(self, max_len: int, model_path: str = ""):
        self.max_len = int(max_len)
        self.model_path = str(model_path or "")
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if not self.model_path:
            raise FileNotFoundError(
                "PI0/PI05 tokenizer.model_path is empty. Set framework.tokenizer.model_path "
                "to a local paligemma_tokenizer.model path for raw-text examples."
            )
        import sentencepiece

        with open(self.model_path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        return self._tokenizer

    def tokenize(self, prompt: str, state: np.ndarray | torch.Tensor | None = None):
        """把 prompt + 可选 state 转成 (tokens, mask),已 pad/truncate 到 max_len。"""
        tokenizer = self._load()
        cleaned_text = str(prompt).strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # state 离散化:每个维度用 256 桶 digitize,得到 [0, 255] 的整数 id。
            state_np = np.asarray(state, dtype=np.float32).reshape(-1)
            discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state.tolist()))
            prompt_text = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = tokenizer.encode(prompt_text, add_bos=True)
        else:
            tokens = tokenizer.encode(cleaned_text, add_bos=True) + tokenizer.encode("\n")

        # Pad / truncate 到固定长度,保证 batch 内可堆叠。
        token_len = len(tokens)
        if token_len < self.max_len:
            padding = [0] * (self.max_len - token_len)
            mask = [True] * token_len + [False] * len(padding)
            tokens = tokens + padding
        else:
            if token_len > self.max_len:
                logger.warning("PI tokenizer length %s exceeds max_token_len=%s; truncating", token_len, self.max_len)
            tokens = tokens[: self.max_len]
            mask = [True] * self.max_len
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=bool)


def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None):
    """带可选门控的残差连接。

    PI0 (普通 RMSNorm):gate=None,直接 x + y。
    PI05 (adaRMS):gate 来自 layernorm.dense 输出的第三份,用于缩放残差。
    """
    return x + y if gate is None else x + y * gate


def _norm(layernorm, hidden_states: torch.Tensor, cond: torch.Tensor | None):
    """(核心 helper) 双模式 layernorm:同时支持 PI0 和 PI05。

    为什么这个函数是 PI0/PI05 共享 forward 的关键
    ---------------------------------------------
    - PI0 的 layernorm 是普通 RMSNorm,forward() 不接受 cond 参数。
    - PI05 的 layernorm 是 adaRMS,forward() 接受 cond=timestep embedding。
    如果在 PI0.forward_shared_gemma_layer 里写两套分支,代码会很冗长。
    这里用 try/except 探测 cond 参数,根据情况走不同路径:

    1. cond is None → 走普通 RMSNorm (PI0)
    2. cond is not None 且 layernorm 支持 cond 参数 → 走 adaRMS (PI05)
    3. cond is not None 但 layernorm 不支持 → 手动实现 adaRMS 公式:
       out = rmsnorm(x) * (1 + scale) + shift
       (scale, shift, gate) = layernorm.dense(cond).chunk(3)

    返回值统一为 (normalized_hidden, gate),gate 可能为 None (PI0)。
    """
    try:
        out = layernorm(hidden_states, cond=cond)
    except TypeError as exc:
        if cond is not None:
            dense = getattr(layernorm, "dense", None)
            if dense is None:
                raise RuntimeError(
                    "PI05 requires Gemma adaRMS support, but this layernorm has no dense module."
                ) from exc
            # 手动实现 adaRMS:RMSNorm 后用 (scale, shift, gate) 调制。
            dtype = hidden_states.dtype
            eps = getattr(layernorm, "variance_epsilon", getattr(layernorm, "eps", 1e-6))
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            out = hidden_states * torch.rsqrt(variance + eps)
            scale, shift, gate = dense(cond.to(dense.weight.dtype))[:, None, :].chunk(3, dim=-1)
            return (out * (1 + scale) + shift).to(dtype), gate.to(dtype)
        out = layernorm(hidden_states)
    if isinstance(out, tuple):
        return out
    return out, None


@FRAMEWORK_REGISTRY.register("PI0")
@FRAMEWORK_REGISTRY.register("Pi0")
class PI0(baseframework):
    """PI0 模型主体。

    生命周期
    --------
    __init__ 调用 4 步:
      1. _init_runtime_config:合并 default + yaml + CLI 配置
      2. _build_tokenizer:懒加载 tokenizer 句柄
      3. _build_model_modules:构造 PaliGemma (VLM) + ActionHead (Expert)
      4. to_precision:转 bf16,但保留对精度敏感的层在 fp32

    被注册到 FRAMEWORK_REGISTRY,name="PI0" / "Pi0",所以 yaml 里写
    `framework.name: PI0` 就能通过 build_framework() 实例化。

    PI05 是子类,只覆盖三个类属性,所有方法沿用本类实现。
    """

    default_config_cls = PI0DefaultConfig
    action_head_cls: type[OpenPIActionHeadBase] = OpenPI0ActionHead
    model_name = "PI0"
    discrete_state_input = False

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self._init_runtime_config(config)
        self._build_tokenizer()
        self._build_model_modules()
        self.to_precision(self.config.framework.precision)
        # 允许 bf16 matmul 使用 TF32 (在 Ampere+ 上更快,精度足够)。
        torch.set_float32_matmul_precision("highest")
        self._maybe_compile_hotpath()

    def _init_runtime_config(self, config):
        """合并 default config + 用户 config,把关键字段提到 self 上方便后续访问。"""
        self.config = merge_framework_config(self.default_config_cls, config)
        self.config.framework.name = self.model_name
        self.discrete_state_input = bool(self.config.framework.discrete_state_input)

        self.action_dim = int(self.config.framework.action_dim)
        self.action_horizon = int(self.config.framework.action_horizon)
        self.max_state_dim = int(self.config.framework.max_state_dim)
        self.num_inference_steps = int(self.config.framework.num_inference_steps)
        self.image_resolution = tuple(int(x) for x in self.config.framework.image_resolution)
        self.image_keys = tuple(self.config.framework.image_keys)
        self.state_pad_value = float(self.config.framework.tokenizer.pad_state_value)

        # joint attention 后端。实测 attention 只占 ~1.4% kernel (见 perf 分析),
        # 换 SDPA 对"功率跳动"无实质帮助、吞吐还略低 → 默认 eager,SDPA 仅作可选实验。
        # env STARVLA_USE_SDPA 优先于 yaml framework.use_sdpa。
        _yaml_sdpa = bool(getattr(getattr(self.config, "framework", None), "use_sdpa", False))
        _env_sdpa = os.environ.get("STARVLA_USE_SDPA")
        self.use_sdpa = (
            _yaml_sdpa
            if _env_sdpa is None
            else _env_sdpa.strip().lower() in ("1", "true", "yes", "on")
        )

    def _build_tokenizer(self):
        self.tokenizer = LazyPaliGemmaTokenizer(
            max_len=int(self.config.framework.max_token_len),
            model_path=str(self.config.framework.tokenizer.model_path),
        )

    def _build_model_modules(self):
        """构造 VLM (PaliGemma) 和 Action Expert。

        - vlm:OpenPIPaliGemma 3B,负责 prefix 流 (image + language)
        - action_head:由子类的 action_head_cls 决定 (PI0 或 PI05)
        两者维度独立配置,但共享 attention 时按 num_heads 对齐。
        """
        vlm_config = _as_dims(self.config.framework.paligemma_config)
        expert_config = _as_dims(self.config.framework.action_expert_config)
        self.vlm = OpenPIPaliGemma(
            vlm_config,
            use_adarms=False,
            model_name_or_path=str(self.config.framework.paligemma.model_name_or_path),
        )
        self.action_head = self.action_head_cls(
            expert_config,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_state_dim=self.max_state_dim,
        )

    def _maybe_compile_hotpath(self):
        """torch.compile 纯 GPU 热路径, 融合海量 elementwise/reduce 小 kernel (实测占
        63% + 12%, 是"功率跳动"真凶) → 大幅减少 kernel-launch-bound。

        关键: 只 compile **纯 tensor** 的方法 (forward_shared_gemma_layer 每步 18×、
        embed_prefix 视觉编码), 避开 framework.forward 里的 PIL/numpy 数据预处理 ——
        否则 dynamo 会撞 read-only numpy 的 'cannot set WRITEABLE flag' bug 而崩。

        env STARVLA_COMPILE: 空/0/off → 关; default/1 → mode=default; reduce-overhead /
        max-autotune → 对应模式 (reduce-overhead 走 CUDA graphs, 需静态 shape, 慎用)。
        """
        raw = os.environ.get("STARVLA_COMPILE", "").strip().lower()
        if raw in ("", "0", "off", "false", "no"):
            return
        mode = raw if raw in ("default", "reduce-overhead", "max-autotune") else "default"
        kw = {"mode": mode}
        if mode != "reduce-overhead":
            kw["dynamic"] = True  # 序列长度可能逐 batch 变, 动态 shape 避免反复重编译
        print(
            f"[PI0] 🔧 torch.compile 热路径 (mode={mode}, dynamic={'dynamic' in kw}): "
            f"forward_shared_gemma_layer + embed_prefix; 首步会慢 (编译中)。",
            flush=True,
        )
        self.forward_shared_gemma_layer = torch.compile(self.forward_shared_gemma_layer, **kw)
        self.embed_prefix = torch.compile(self.embed_prefix, **kw)

    def to_precision(self, precision: Literal["bfloat16", "float32"]):
        """切换模型精度,保留对精度敏感的层在 fp32。

        为什么不全 bf16?
        ---------------
        bf16 的尾数只有 7 bit,layernorm 这种"求方差再做 rsqrt"的操作
        在 bf16 下数值不稳,容易让训练 loss spike。所以这些层强制 keep fp32:
          - patch_embedding (vision_tower 输入,影响所有 image token)
          - input_layernorm / post_attention_layernorm (每层归一化)
          - model.norm (最后一层)

        PaliGemma 官方权重就是这种混合精度布局,直接对齐才能正确加载。
        """
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Unsupported PI precision: {precision}")
        self.to(dtype=torch.bfloat16)
        keep_fp32 = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, param in self.named_parameters():
            if any(selector in name for selector in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def _pad_array_2d(self, arr: np.ndarray, target_time: int, target_dim: int, pad_value: float) -> np.ndarray:
        """把 (T, D) 或 (D,) 数组 pad 到固定 (target_time, target_dim)。

        超出会报错 (避免静默截断丢数据)。用 pad_value 填充,PI0 action 用 0.0,
        state 用 -2.0 (PI05 离散模式) 或 0.0 (PI0 连续模式)。
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > target_time or arr.shape[1] > target_dim:
            raise ValueError(f"Input shape {arr.shape} exceeds target {(target_time, target_dim)}")
        out = np.full((target_time, target_dim), pad_value, dtype=np.float32)
        out[: arr.shape[0], : arr.shape[1]] = arr
        return out

    def _pil_to_tensor(self, image) -> torch.Tensor:
        """PIL → (C, H, W) float tensor,数值范围 [-1, 1]。

        resize_with_pad 会保持长宽比,空白处填 0 (黑色)。
        归一化用 (x/255)*2-1 而不是 x/255-0.5,PaliGemma 风格。
        """
        image_np = np.asarray(image.convert("RGB")).copy()
        image_np = resize_with_pad(image_np[None], self.image_resolution[0], self.image_resolution[1])[0]
        return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float() / 255.0 * 2.0 - 1.0

    def _prepare_examples(self, examples: list[dict], include_actions: bool):
        """把原始 example dict 列表转成 batched 张量字典。

        输入 example 格式 (典型):
            {"image": [PIL, PIL, PIL] 或 PIL,
             "image_mask": [True, False, True] 或 None,
             "lang": "pick up the cup",
             "state": np.array([0.1, ...]) 或 None,
             "action": np.array([(horizon, dim)]) 或 None}

        输出 batch:
            {"image": (B, num_images, C, H, W),
             "image_mask": (B, num_images) bool,
             "state": (B, 1, max_state_dim),
             "lang": list[str],
             "action": (B, horizon, dim)  # 仅 include_actions=True}

        图像缺失时的处理
        ----------------
        如果 image_keys 有 3 个但 example 只提供 1 张图,会用第一张图的拷贝
        (涂黑) 填充,并把对应 image_mask 设为 False,让 attention 忽略。
        """
        images, masks, states, actions, prompts = [], [], [], [], []
        action_pad_list = []
        for example in examples:
            raw_images = to_pil_preserve(example["image"])
            if isinstance(raw_images, tuple):
                raw_images = list(raw_images)
            elif not isinstance(raw_images, list):
                raw_images = [raw_images]
            sample_images, sample_masks = [], []
            provided_mask = example.get("image_mask")
            provided_mask = list(provided_mask) if provided_mask is not None else None
            for idx, _key in enumerate(self.image_keys):
                has_image = idx < len(raw_images)
                if has_image:
                    image = raw_images[idx]
                else:
                    # 缺失视角:用第一张图的拷贝涂黑,attention 会通过 mask 忽略。
                    image = raw_images[0].copy()
                    image.paste(0, (0, 0, image.size[0], image.size[1]))
                active = bool(provided_mask[idx]) if provided_mask is not None and idx < len(provided_mask) else has_image
                sample_images.append(self._pil_to_tensor(image))
                sample_masks.append(active)
            images.append(torch.stack(sample_images))
            masks.append(sample_masks)
            prompts.append(str(example.get("lang", example.get("language", ""))))

            # State 处理:discrete_state_input=True 时填 pad_state_value (后续走
            # tokenizer 离散化);False 时填 0 (后续走 state_proj 连续投影)。
            state_value = example.get("state")
            if state_value is None:
                fill = self.state_pad_value if self.config.framework.discrete_state_input else 0.0
                state_np = np.full((1, self.max_state_dim), fill, dtype=np.float32)
            else:
                state_np = np.asarray(state_value, dtype=np.float32)
                if state_np.ndim == 2 and state_np.shape[0] > 1:
                    # 取最后一帧 state (有的数据集提供历史轨迹,只用最新的)
                    state_np = state_np[-1:]
                state_np = self._pad_array_2d(
                    state_np,
                    1 if state_np.ndim == 1 else state_np.shape[0],
                    self.max_state_dim,
                    self.state_pad_value if self.config.framework.discrete_state_input else 0.0,
                )
            states.append(torch.from_numpy(state_np))
            if include_actions:
                action_pad_list.append(example.get("action_pad"))

                action_np = self._pad_array_2d(np.asarray(example["action"], dtype=np.float32), self.action_horizon, self.action_dim, 0.0)
                actions.append(torch.from_numpy(action_np))

        batch = {
            "image": torch.stack(images),
            "image_mask": torch.as_tensor(np.asarray(masks), dtype=torch.bool),
            "state": torch.stack(states),
            "lang": prompts,
        }
        if include_actions:
            batch["action"] = torch.stack(actions)
            # action_dim_actual 由 dataset 提供（来自 yaml 配置，不指定则 fallback 到 action.shape[-1]）
            dim_actual = None
            for ex in examples:
                if "action_dim_actual" in ex:
                    dim_actual = int(ex["action_dim_actual"])
                    break
            batch["action_dim_actual"] = dim_actual if dim_actual is not None else actions[0].shape[-1]
            if any(p is not None for p in action_pad_list):
                batch["action_pad"] = action_pad_list
        return batch

    def _build_observation_from_batch(self, batch):
        """把 batched 张量封装成 Observation dataclass,并 tokenize prompt。

        discrete_state_input=True 时把 state 拼进 prompt (走 tokenizer),
        否则 state 只走 Observation.state (后续 action_head 用)。
        """
        device = self.action_head.action_in_proj.weight.device
        images = {key: batch["image"][:, idx].to(device=device, dtype=torch.float32) for idx, key in enumerate(self.image_keys)}
        masks = {key: batch["image_mask"][:, idx].to(device=device, dtype=torch.bool) for idx, key in enumerate(self.image_keys)}
        tokens, token_masks = [], []
        for idx, prompt in enumerate(batch["lang"]):
            token_state = batch["state"][idx].squeeze(0) if self.config.framework.discrete_state_input else None
            token, mask = self.tokenizer.tokenize(prompt, token_state)
            tokens.append(token)
            token_masks.append(mask)
        return Observation(
            images=images,
            image_masks=masks,
            tokenized_prompt=torch.as_tensor(np.asarray(tokens), device=device),
            tokenized_prompt_mask=torch.as_tensor(np.asarray(token_masks), device=device),
            state=batch["state"].to(device=device, dtype=torch.float32),
        )

    def _preprocess_observation(self, observation: Observation):
        """把 Observation 拆解成 forward 主干方便消费的 5 元组。"""
        return (
            [observation.images[key] for key in self.image_keys],
            [observation.image_masks[key] for key in self.image_keys],
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """构造 prefix 流的 embedding (VLM 主干输入)。

        Prefix 结构 (按 token 顺序):
            [img_0 tokens] [img_1 tokens] [img_2 tokens] [lang tokens]

        返回三个东西:
        - embs: (B, total_len, width) 拼接好的 embedding
        - pad_masks: (B, total_len) bool,True 表示有效位置
        - att_masks: (B, total_len) 0/1,1 表示"新一段开始"
          (用于构造 block-causal attention:每段内部互相看不到,但都能看到前面段)

        lang_emb 乘 sqrt(width):Gemma 风格的 embedding 缩放,跟权重初始化对齐。
        """
        embs, pad_masks, att_masks = [], [], []
        for image, image_mask in zip(images, img_masks, strict=True):
            image_emb = self.vlm.embed_image(image)
            bsize, num_img_embs = image_emb.shape[:2]
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
        lang_emb = self.vlm.embed_language_tokens(lang_tokens)
        embs.append(lang_emb * math.sqrt(lang_emb.shape[-1]))
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]
        return embs, pad_masks, att_masks.expand(pad_masks.shape[0], -1)

    def forward_with_action_expert(
        self,
        *,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        adarms_cond: torch.Tensor | None,
    ):
        """(核心) 跑一遍"VLM + Action Expert 共享 attention"的主干。

        两条流的协作
        ------------
        - prefix 流:hidden_states[0],走 vlm.language_model,cond=None (普通 RMSNorm)
        - suffix 流:hidden_states[1],走 action_head.model,cond=adarms_cond (PI05)
          或 None (PI0)

        每一层都通过 forward_shared_gemma_layer 做:
          1. 两个流各自 layernorm
          2. 两个流的 Q/K/V 拼起来做 joint attention (prefix tokens 看自己,
             suffix tokens 看 prefix + 自己)
          3. attention 输出按流切回去,各自做 FFN + residual

        这样 action expert 可以"读到"VLM 对图像/语言的理解,但 VLM 不会被
        action 信息污染 (block-causal attention 保证)。

        Returns:
            suffix_out: (B, suffix_len, expert_width),最后会被 decode_action_velocity
                        转成 v_t。
        """
        streams = [prefix_embs, suffix_embs]
        conds = [None, adarms_cond]
        models = [self.vlm.language_model, self.action_head.model]
        # 逐层共享 attention,层数由 VLM 主干决定 (depth=18)。
        # Action expert 的层数应该与 VLM 一致,否则这里会越界。
        for layer_idx in range(self.vlm.config.text_config.num_hidden_layers):
            streams = self.forward_shared_gemma_layer(layer_idx, models, streams, conds, attention_mask, position_ids)
        suffix_out, _ = _norm(models[1].norm, streams[1], conds[1])
        return suffix_out

    def forward_shared_gemma_layer(self, layer_idx, models, streams, conds, attention_mask, position_ids):
        """(核心) 单层 Gemma 的双流前向,共享 self-attention。

        步骤
        ----
        1. **Pre-attention**:
           - 两个流各自做 input_layernorm (PI05 用 adaRMS,PI0 普通 RMSNorm)
           - 各自算 Q/K/V,然后**拼起来**沿 sequence 维度

        2. **Joint attention**:
           - 用 VLM 流的 rotary_emb 算旋转位置编码 (两个流共享同一份 RoPE)
           - 用 SDPA (默认) / eager 做完整 attention (由 self.use_sdpa 控制)
           - attention_mask 是 block-causal:prefix tokens 看自己,suffix tokens
             看 prefix + 自己,但 prefix tokens 看不到 suffix tokens

        3. **Post-attention**:
           - 把 attention 输出按原来的流切回去 (用 hidden_states.shape[1] 记录边界)
           - 各自做 o_proj + residual
           - 各自做 post_attention_layernorm + MLP + residual

        为什么这样设计?
        ----------------
        π0 论文的核心创新就是这个"双流共享 attention":action expert 不需要
        重新编码图像/语言,直接"借用"VLM 的注意力分布。这样 action expert
        可以做得很小 (~1B 参数),却继承 VLM 的全部语义理解能力。
        """
        query_states, key_states, value_states, gates = [], [], [], []
        for idx, hidden_states in enumerate(streams):
            layer = models[idx].layers[layer_idx]
            hidden_states, gate = _norm(layer.input_layernorm, hidden_states, conds[idx])
            gates.append(gate)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

        # 拼接两个流的 Q/K/V,一起做 attention。
        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)
        # dummy 张量只为拿到正确 shape/device,用于 rotary_emb 计算。
        dummy = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = self.vlm.language_model.rotary_emb(dummy, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )
        # joint attention。self.use_sdpa=True 时走 SDPA (F.scaled_dot_product_attention),
        # 把 eager 的 repeat_kv→QKᵀ→+mask→softmax→@V 一串小 kernel 融合成 1-2 个。实测
        # attention 仅占 ~1.4% kernel, 对"功率跳动"帮助有限, 故默认 eager (use_sdpa=False)。
        # additive float 4D mask 走 mem-efficient 后端 (实测与原 4D block mask 数值等价)。
        self_attn = self.vlm.language_model.layers[layer_idx].self_attn
        scaling = self_attn.scaling
        if self.use_sdpa:
            # GQA: 复现 eager 的 repeat_kv (num_key_value_groups=1 时 no-op; 两流 groups 一致)。
            kv_groups = getattr(self_attn, "num_key_value_groups", 1) or 1
            k_s, v_s = key_states, value_states
            if kv_groups != 1:
                k_s = modeling_gemma.repeat_kv(key_states, kv_groups)
                v_s = modeling_gemma.repeat_kv(value_states, kv_groups)
            # additive mask 切到 key 长度 (与 eager 的 attention_mask[:,:,:,:Sk] 一致)。
            m = attention_mask[:, :, :, : k_s.shape[-2]] if attention_mask is not None else None
            att_output = F.scaled_dot_product_attention(
                query_states,
                k_s,
                v_s,
                attn_mask=m,
                dropout_p=0.0,
                is_causal=False,
                scale=scaling,
            )
            # SDPA 返回 [B,H,S,d]; eager 返回 transpose(1,2)=[B,S,H,d]。对齐后者, 后续 reshape 不变。
            att_output = att_output.transpose(1, 2)
        else:
            att_output, _ = modeling_gemma.eager_attention_forward(
                self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling,
            )
        head_dim = self.vlm.language_model.layers[layer_idx].self_attn.head_dim
        num_heads = self.vlm.language_model.config.num_attention_heads
        att_output = att_output.reshape(query_states.shape[0], -1, num_heads * head_dim)

        # 切回两个流各自处理 FFN。
        next_streams = []
        start = 0
        for idx, hidden_states in enumerate(streams):
            layer = models[idx].layers[layer_idx]
            end = start + hidden_states.shape[1]
            layer_att = att_output[:, start:end]
            if layer_att.dtype != layer.self_attn.o_proj.weight.dtype:
                layer_att = layer_att.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(layer_att)
            out = _gated_residual(hidden_states, out, gates[idx])
            residual = out.clone()
            out, gate = _norm(layer.post_attention_layernorm, out, conds[idx])
            if out.dtype != layer.mlp.up_proj.weight.dtype:
                out = out.to(layer.mlp.up_proj.weight.dtype)
            out = layer.mlp(out)
            next_streams.append(_gated_residual(residual, out, gate))
            start = end
        return next_streams

    def forward_from_observation(self, observation: Observation, actions: torch.Tensor, noise=None, time=None, return_debug=False, action_dim_actual=None, action_pad=None):
        """训练侧核心前向:从观测+GT action 算 flow matching loss。

        流程:
          1. embed_prefix:images + lang → prefix embeddings
          2. action_head.prepare_flow_targets:从 GT action 采样 (x_t, u_t, t)
             x_t = t·ε + (1-t)·a       # 加噪后的 action
             u_t = ε - a               # 目标速度场
          3. action_head.embed_action_suffix:state + x_t + t → suffix embeddings
          4. forward_with_action_expert:跑双流共享 attention 主干
          5. action_head.decode_action_velocity:把 suffix 输出投影成 v_t
          6. loss = MSE(u_t, v_t)
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        x_t, u_t, noise, time = self.action_head.prepare_flow_targets(actions, noise=noise, time=time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.action_head.embed_action_suffix(state.squeeze(1), x_t, time)
        # VLM 是 bf16 时,suffix embedding 也转 bf16 才能共享 matmul。
        if self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention_mask = make_att_4d_masks(make_att_2d_masks(pad_masks, att_masks))
        # position_ids 按 pad_masks 的 cumsum 算,pad 位置不计入位置编号。
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        suffix_out = self.forward_with_action_expert(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
        )
        v_t = self.action_head.decode_action_velocity(suffix_out)
        loss = F.mse_loss(u_t, v_t, reduction="none")  # (B, H, D)
        mask = torch.ones_like(loss, dtype=torch.bool)

        # 时间维 mask: 轨迹末尾凑不齐 action_horizon 的 pad timestep 不参与 loss
        if action_pad is not None:
            for i, pad in enumerate(action_pad):
                if pad is not None:
                    pad_t = torch.as_tensor(pad, device=loss.device, dtype=torch.bool)
                    mask[i, ~pad_t, :] = False

        # 特征维 mask: pad 到 action_dim 的多余维度不参与 loss
        if action_dim_actual is not None and action_dim_actual < loss.shape[-1]:
            mask[..., action_dim_actual:] = False

        out = {"v_t": v_t, "action_loss": (loss * mask).sum() / mask.sum()}
        if return_debug:
            out.update({"u_t": u_t, "x_t": x_t, "time": time, "noise": noise})
        return out

    def forward(self, examples: list[dict] = None, **kwargs):
        """训练入口:接收原始 example dict,返回含 'action_loss' 的 dict。"""
        batch = self._prepare_examples(examples, include_actions=True)
        observation = self._build_observation_from_batch(batch)
        actions = batch["action"].to(device=self.action_head.action_in_proj.weight.device, dtype=torch.float32)
        return self.forward_from_observation(
            observation,
            actions,
            action_dim_actual=batch.get("action_dim_actual", actions.shape[-1]),
            action_pad=batch.get("action_pad"),
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
            return_debug=bool(kwargs.get("return_debug", False)),
        )

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] | dict = None, **kwargs):
        """推理入口:接收 example,返回 {'normalized_actions': np.ndarray}。

        与 forward 区别:
        - include_actions=False (没有 GT action)
        - 走 sample_actions 做 N 步 Euler 积分
        - 输出在 cpu + numpy,方便部署
        """
        if not isinstance(examples, list):
            examples = [examples]
        batch = self._prepare_examples(examples, include_actions=False)
        observation = self._build_observation_from_batch(batch)
        actions = self.sample_actions(
            observation,
            noise=kwargs.get("noise"),
            num_steps=int(kwargs.get("num_steps", kwargs.get("num_inference_steps", self.num_inference_steps))),
        )
        return {"normalized_actions": actions.cpu().numpy()}

    def sample_actions(self, observation: Observation, noise=None, num_steps: int = 10):
        """(核心) Flow matching 推理:从纯噪声开始,N 步 Euler 积分到 action。

        关键优化
        --------
        - prefix 只 forward 一次 (forward_prefix),拿到 KV cache。
        - 后续 denoise_step 复用这个 KV cache,只重算 suffix 部分。
          这是因为 prefix (图像+语言) 在 N 步里不变,只有 noisy action 在变。
        - 这样推理开销 ≈ N × suffix_forward + 1 × prefix_forward,而不是
          N × (prefix + suffix)_forward。

        积分公式
        --------
        dt = -1/N (负号:t 从 1 走到 0,从噪声走到数据)
        x_{t+dt} = x_t + dt * v_t   (Euler 一步)
        """
        device = self.action_head.action_in_proj.weight.device
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks_4d = make_att_4d_masks(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # forward_prefix 走 Gemma 自带的 attention dispatch。4D additive mask 其实 SDPA 也
        # 支持 (见 forward_shared_gemma_layer 的 use_sdpa 分支, 已实测等价), 但这条推理路径
        # (前缀 KV cache 编码) 频率低, 暂留 eager; 训练热路径已换 SDPA。
        self.vlm.language_model.config._attn_implementation = "eager"
        prefix_output = self.vlm.forward_prefix(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = prefix_output.past_key_values
        bsize = state.shape[0]
        shape = (bsize, self.action_horizon, self.action_dim)
        # 起点 x_0 = ε (纯噪声)。如果用户传了固定 noise,就用它 (用于复现实验)。
        x_t = self.action_head.sample_action_noise(shape, device) if noise is None else torch.as_tensor(noise, device=device, dtype=torch.float32)
        if x_t.ndim == 2:
            x_t = x_t[None]
        dt = torch.tensor(-1.0 / max(num_steps, 1), dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state.squeeze(1), prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """单步去噪:用当前 x_t 和 timestep 预测速度场 v_t。

        复用 sample_actions 里 prefix 的 KV cache,只重算 suffix 部分。
        构造的 attention mask 让 suffix tokens 能 attend 到 prefix 的所有
        token (但反向不行,prefix 看不到 suffix)。
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, _adarms_cond = self.action_head.embed_action_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape
        # prefix_pad_2d_masks:suffix 的每个 token 都能看 prefix 的所有有效 token。
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        # suffix_att_2d_masks:suffix 内部的 block-causal mask (第一个 token 看自己,
        # 后续 token 看前面所有)。
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention_mask = make_att_4d_masks(torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2))
        # position_ids:prefix 部分的位置编号 + suffix 部分的 cumsum。
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        suffix_embs = suffix_embs.to(torch.bfloat16) if self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16 else suffix_embs
        return self.action_head(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            suffix_embs=suffix_embs,
            adarms_cond=_adarms_cond,
        )

    def _is_expected_openpi_missing_key(self, key: str) -> bool:
        """识别"加载 OpenPI 官方权重时预期会缺失"的 key。

        OpenPI 官方权重里,action expert 的 adaRMS layernorm 的 weight 是
        存的,但 dense (PI05 注入的) 模块不存 (因为官方 PI0 没有 dense)。
        这些 key 缺失是正常的,不应触发警告。
        """
        if not getattr(self.action_head, "use_adarms", False):
            return False
        prefix = "action_head.gemma_expert.model."
        if key == prefix + "norm.weight":
            return True
        if not key.startswith(prefix + "layers."):
            return False
        return key.endswith(".input_layernorm.weight") or key.endswith(".post_attention_layernorm.weight")

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
        """支持传目录或文件路径,目录时自动找 model.safetensors / pytorch_model.bin 等。"""
        path = Path(checkpoint_path)
        if path.is_dir():
            for name in ("model.safetensors", "pytorch_model.bin", "model.pt", "checkpoint.pt", "ckpt.pt"):
                candidate = path / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No supported PI checkpoint file found under {path}")
        return path

    @classmethod
    def _load_checkpoint_state_dict(cls, checkpoint_path: str | Path):
        """加载 checkpoint 文件为 state_dict (自动识别 safetensors / pt / bin)。"""
        checkpoint_path = cls._resolve_checkpoint_path(checkpoint_path)
        if str(checkpoint_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # 有些 checkpoint 是 {"state_dict": ...} / {"model": ...} 这种 wrapper。
            if isinstance(state_dict, dict):
                for key in ("state_dict", "model_state_dict", "module", "model"):
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break
        return state_dict

    def _normalize_state_dict(self, state_dict: dict):
        """(核心) 把 OpenPI 官方权重的 key 命名映射到 StarVLA 模型的命名。

        OpenPI 官方仓库的命名:
            paligemma_with_expert.paligemma.*      → VLM 部分
            paligemma_with_expert.gemma_expert.*   → Action Expert 部分
            action_in_proj.* / state_proj.* / ...  → Action Head 投影层

        StarVLA 的命名:
            vlm.model.model.*                      → VLM 部分
            action_head.gemma_expert.*             → Action Expert 部分
            action_head.action_in_proj.* / ...     → Action Head 投影层

        另外处理三种常见的 key 前缀污染:
            model.* / module.* / _orig_mod.* (DataParallel / torch.compile 痕迹)

        embed_tokens 共享 lm_head:PaliGemma 的 embed 和 lm_head 是同一份权重,
        官方权重只存了 lm_head,这里如果 shape 匹配就复制过去。
        """
        current = self.state_dict()
        normalized = {}
        for key, value in state_dict.items():
            clean_key = key
            # 循环剥离 model. / module. / _orig_mod. 前缀 (可能有多层)。
            changed = True
            while changed:
                changed = False
                for prefix in ("model.", "module.", "_orig_mod."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        changed = True
            # OpenPI 命名 → StarVLA 命名的核心映射。
            if clean_key.startswith("paligemma_with_expert.paligemma.model."):
                clean_key = clean_key.replace(
                    "paligemma_with_expert.paligemma.model.",
                    "vlm.model.model.",
                    1,
                )
            elif clean_key.startswith("paligemma_with_expert.paligemma."):
                clean_key = clean_key.replace("paligemma_with_expert.paligemma.", "vlm.model.", 1)
            elif clean_key.startswith("paligemma_with_expert.gemma_expert."):
                clean_key = clean_key.replace("paligemma_with_expert.gemma_expert.", "action_head.gemma_expert.", 1)
            elif clean_key.startswith("action_in_proj."):
                clean_key = clean_key.replace("action_in_proj.", "action_head.action_in_proj.", 1)
            elif clean_key.startswith("action_out_proj."):
                clean_key = clean_key.replace("action_out_proj.", "action_head.action_out_proj.", 1)
            elif clean_key.startswith("state_proj."):
                clean_key = clean_key.replace("state_proj.", "action_head.state_proj.", 1)
            elif clean_key.startswith("action_time_mlp_in."):
                clean_key = clean_key.replace("action_time_mlp_in.", "action_head.action_time_mlp_in.", 1)
            elif clean_key.startswith("action_time_mlp_out."):
                clean_key = clean_key.replace("action_time_mlp_out.", "action_head.action_time_mlp_out.", 1)
            elif clean_key.startswith("time_mlp_in."):
                clean_key = clean_key.replace("time_mlp_in.", "action_head.time_mlp_in.", 1)
            elif clean_key.startswith("time_mlp_out."):
                clean_key = clean_key.replace("time_mlp_out.", "action_head.time_mlp_out.", 1)
            normalized[clean_key] = value
        # PaliGemma 的 embed_tokens 与 lm_head 共享权重。
        embed_key = "vlm.model.model.language_model.embed_tokens.weight"
        lm_head_key = "vlm.model.lm_head.weight"
        if embed_key not in normalized and lm_head_key in normalized and embed_key in current:
            if normalized[lm_head_key].shape == current[embed_key].shape:
                normalized[embed_key] = normalized[lm_head_key]
        return normalized

    def load_pretrained_checkpoint(self, checkpoint_path: str | Path, reload_modules: str | None = None):
        """加载 OpenPI 官方权重,支持全量加载和按子模块加载。

        与 TrainerUtils.load_pretrained_backbones 的区别
        -----------------------------------------------
        那个是通用的 state_dict 加载,这个是 PI0/PI05 专用的,会:
          1. 调用 _normalize_state_dict 做 key 命名映射
          2. 跳过 shape mismatch 的 key (只 warning 不 raise)
          3. 区分"预期缺失" (adaRMS dense 模块) 和"真正缺失" (其他)
        """
        state_dict = self._normalize_state_dict(self._load_checkpoint_state_dict(checkpoint_path))
        if reload_modules:
            # 按子模块加载:用于 cotrain 场景,只重载 vlm 或 action_head。
            for path in [p.strip() for p in reload_modules.split(",") if p.strip()]:
                module = self
                for part in path.split("."):
                    module = getattr(module, part)
                prefix = path + "."
                sub_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                module.load_state_dict(sub_state, strict=False)
            return self

        # 全量加载:跳过 shape mismatch,记录缺失/多余 key 用于诊断。
        current = self.state_dict()
        filtered, shape_mismatch = {}, []
        for key, value in state_dict.items():
            if key in current and getattr(value, "shape", None) == current[key].shape:
                filtered[key] = value
            elif key in current:
                shape_mismatch.append((key, tuple(value.shape), tuple(current[key].shape)))
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        # 把"预期缺失"的 key 从 missing 里挪出来,避免污染告警。
        expected_missing = [key for key in missing if self._is_expected_openpi_missing_key(key)]
        real_missing = [key for key in missing if key not in expected_missing]
        logger.info(
            "%s checkpoint load: checkpoint=%s loaded=%s model=%s missing=%s expected_missing=%s unexpected=%s shape_mismatch=%s",
            type(self).__name__,
            len(state_dict),
            len(filtered),
            len(current),
            len(real_missing),
            len(expected_missing),
            len(unexpected),
            len(shape_mismatch),
        )
        if real_missing:
            logger.info("%s first missing keys: %s", type(self).__name__, real_missing[:10])
        if shape_mismatch:
            logger.info("%s first shape mismatches: %s", type(self).__name__, shape_mismatch[:10])
        return self
