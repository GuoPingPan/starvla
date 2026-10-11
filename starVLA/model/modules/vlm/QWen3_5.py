# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

import os
from typing import Optional
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # 重要: 不要在这里强制 sdpa! transformers 5.3 的 Qwen3.5 视觉注意力在
        # 非 flash 路径下是"逐图 python 循环" (split → sdpa → concat, 144 图/步,
        # py-spy 实测占主线程 ~22% + dynamo graph break); flash 路径走 cu_seqlens
        # 变长内核无循环。yaml 配 flash_attention_2 即生效, 装不上的环境回退 sdpa。
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # ---- 视觉预处理 (配置驱动, 训练/部署同规范) ----
        # yaml framework.qwenvl.vision_preprocess: {transforms: [...], encoder: {...}}
        # null/false/off → 回退 HF image_processor (CPU, 兼容口径)。参考架构见
        # starVLA/model/modules/vlm/qwen_vision_preprocess.py。
        vp_cfg = qwenvl_config.get("vision_preprocess", None)
        if vp_cfg is None or str(vp_cfg).lower() in ("false", "off", "none", ""):
            self.vision_preproc = None
        else:
            from starVLA.model.modules.vlm.qwen_vision_preprocess import QwenVisionBatchPreprocessor

            self.vision_preproc = QwenVisionBatchPreprocessor(**dict(vp_cfg))
            print(
                f"[QWen3_5] 🖼️ vision_preprocess: {len(self.vision_preproc.ops)} ops "
                f"(image_size={self.vision_preproc.image_size}, "
                f"{self.vision_preproc.num_vision_tokens_per_image()} tokens/图, GPU batch)"
            )

        # ---- torch.compile (STARVLA_COMPILE=1): 逐层区域编译 + hook 捕获层输出 ----
        # evt 归因: 稳态 cpu==gpu (CPU 是节奏方), 计算 kernel 仅 ~350ms/步, 其余
        # 为 python/派发开销。compile 把每个 decoder 层的逐算子派发换成编译图 +
        # 融合 elementwise 汤。
        #
        # 坑 (隔离实验复现): transformers 5.3 的 Qwen3_5TextModel.forward 不在
        # 自身循环里收集 hidden_states (eager 的多层输出由 HF 内部分发机制补上),
        # 对 language_model 子树做任何 compile 都会让 hidden_states 塌缩成 1 层,
        # 接口需要 24 层逐层特征 → ValueError。解法: "自有契约"——forward hook
        # 捕获每个 decoder layer 的输出 (数值与 eager hidden_states 逐层等价,
        # 即 layer_i 的返回值), 调用时不再依赖 output_hidden_states; 编译后的
        # layer 照常被 hook (OptimizedModule 保留 nn.Module 钩子机制)。
        self._capture_layers = False
        self._captured_layer_outputs = None
        # 正式开关在配置: framework.qwenvl.torch_compile: true (生产 yaml);
        # STARVLA_COMPILE 环境变量仅调试期强制开/关 (优先级更高)。
        _env_compile = str(os.environ.get("STARVLA_COMPILE", "")).strip().lower()
        _cfg_compile = bool(qwenvl_config.get("torch_compile", False))
        _compile_on = _cfg_compile if _env_compile == "" else _env_compile in ("1", "true", "yes")
        if _compile_on:
            _vlm = getattr(self.model, "model", None)
            _lm = getattr(_vlm, "language_model", None) if _vlm is not None else None
            _layers = getattr(_lm, "layers", None) if _lm is not None else None
            if _layers is not None:

                def _make_hook():
                    def _hook(_module, _inputs, output):
                        if self._capture_layers:
                            self._captured_layer_outputs.append(
                                output[0] if isinstance(output, tuple) else output
                            )
                    return _hook

                _compiled = []
                for _lay in _layers:
                    _c = torch.compile(_lay, dynamic=False)
                    _c.register_forward_hook(_make_hook())
                    _compiled.append(_c)
                _lm.layers = torch.nn.ModuleList(_compiled)
                self._capture_layers = True
                _src = "env" if _env_compile != "" else "yaml torch_compile"
                print(
                    f"[QWen3_5] 🔧 torch.compile 逐层启用: {len(_compiled)} 个 decoder layers "
                    f"+ hook 层输出捕获 (来源: {_src})"
                )
                # vision tower 逐块编译 (无 hidden_states 契约问题)。注意: 仅
                # sdpa 路径需要 (逐图 python 循环); flash_attention_2 走 cu_seqlens
                # 变长内核, 且 varlen 的 .max() 取 int 会让 dynamo fake-tensor
                # 追踪崩 (get_fake_value TorchRuntimeError), 故 flash 下跳过。
                _visual = getattr(_vlm, "visual", None)
                _vblocks = getattr(_visual, "blocks", None) if _visual is not None else None
                if _vblocks is not None and attn_implementation != "flash_attention_2":
                    _visual.blocks = torch.nn.ModuleList(
                        [torch.compile(_vb, dynamic=False) for _vb in _vblocks]
                    )
                    print(f"[QWen3_5] 🔧 torch.compile vision tower: {len(_vblocks)} 个 ViT blocks")
            else:
                print("[QWen3_5] STARVLA_COMPILE=1 但未找到 language_model.layers, 跳过")

        # ---- instruction token 缓存 + 静态 padding (scripts/data-analyze/precompute.py 产物) ----
        # tok_cache: "auto"(默认) → {data_root}/precompute/tok_cache_qwen35.pt 存在即启用;
        # 显式路径 / ""(关)。命中缓存时 build_qwenvl_inputs 跳过 chat template + tokenizer,
        # 未命中的 batch 自动回退在线慢路径 (正确性优先)。
        # pad_seq_len: >0 时把 input_ids/attention_mask 左 pad 到固定长度 —— 序列 shape
        # 静态化是 torch.compile(reduce-overhead)/CUDA Graphs 的前提 (STARVLA_COMPILE)。
        tok_cache_cfg = str(qwenvl_config.get("tok_cache", "auto"))
        self.pad_seq_len = int(qwenvl_config.get("pad_seq_len", 0))
        self._tok_cache = None
        self._n_vision_tokens_per_image = None
        self._fast_text_unverified = True
        self._tok_hits = 0
        self._tok_misses = 0
        # 运行时 memo (默认开): 慢路径算过的 instruction 回填缓存。state-in-text
        # 数据集 (H1) 指令每样本唯一但 epoch 间重复 —— 首 epoch 后全命中,
        # 免去离线重建 instruction 的归一化链路风险。STARVLA_TOK_MEMO=0 关闭。
        _env_memo = os.environ.get("STARVLA_TOK_MEMO", "")
        _cfg_memo = qwenvl_config.get("tok_memo", True)
        self._tok_memo = (
            bool(_cfg_memo) if _env_memo == "" else _env_memo not in ("0", "false", "no")
        )
        if tok_cache_cfg:
            path = None
            if tok_cache_cfg not in ("auto", "1", "true"):
                path = Path(tok_cache_cfg)
            else:
                try:  # 普通属性访问 (trainer 侧 config 是 AccessTrackedConfig 代理, OmegaConf.select 不兼容)
                    root = config.datasets.vla_data.data_root_dir
                except Exception:
                    root = None
                if root:
                    p = Path(root) / "precompute" / "tok_cache_qwen35.parquet"
                    path = p if p.exists() else None
            if path is not None and Path(path).exists():
                import pandas as _pd

                df = _pd.read_parquet(path)
                self._tok_cache = {
                    row["task"]: torch.tensor(row["input_ids"], dtype=torch.long)
                    for _, row in df.iterrows()
                }
                self._tok_cache_tokenizer = str(df["tokenizer"].iloc[0]) if "tokenizer" in df else ""
                model_id_short = str(getattr(getattr(self.model.config, "text_config", None), "name_or_path", "") or model_id)
                if self._tok_cache_tokenizer and self._tok_cache_tokenizer not in model_id_short and model_id_short not in self._tok_cache_tokenizer:
                    print(
                        f"[QWen3_5] ⚠️ tok cache 的 tokenizer ({self._tok_cache_tokenizer}) "
                        f"与当前模型 ({model_id_short}) 不一致, 建议重跑 precompute"
                    )
                print(f"[QWen3_5] 📦 tok cache 已加载: {path} ({len(self._tok_cache)} 条)")
            elif tok_cache_cfg not in ("auto", "1", "true"):
                print(f"[QWen3_5] ⚠️ tok_cache 指定的文件不存在: {tok_cache_cfg}, 走在线路径")
            if self._tok_cache is None and self._tok_memo:
                # memo 模式: 无缓存文件也从空字典起步, 慢路径结果回填
                self._tok_cache = {}
                print("[QWen3_5] 📝 tok memo 已启用 (空缓存起步, 慢路径结果回填)")

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.

        STARVLA_COMPILE 模式下由 hook 捕获逐层输出并补挂到 outputs.hidden_states
        (数值与 eager 的逐层 hidden_states 等价), 绕开 compile 破坏的多层契约。
        """
        use_capture = self._capture_layers
        if use_capture:
            self._captured_layer_outputs = []
            kwargs = {k: v for k, v in kwargs.items() if k != "output_hidden_states"}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        if use_capture:
            outputs.hidden_states = tuple(self._captured_layer_outputs)
            self._captured_layer_outputs = None
        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def _pad_to_static_len(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """左 pad 到 self.pad_seq_len (shape 静态化, compile/CUDA Graphs 前提)。"""
        if self.pad_seq_len <= 0 or input_ids.shape[1] >= self.pad_seq_len:
            return input_ids, attention_mask
        L, cur = self.pad_seq_len, input_ids.shape[1]
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = torch.nn.functional.pad(input_ids, (L - cur, 0), value=pad_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (L - cur, 0), value=0)
        return input_ids, attention_mask

    def _encode_images(self, img_flat):
        """图像侧统一入口: 配置化的 GPU batch 预处理器 (训练/部署同规范)。

        vision_preprocess 配置存在时走 ``QwenVisionBatchPreprocessor`` (GPU, ~10ms/128张,
        替代 CPU image_processor 的 604ms); 配置为 null/off 时回退 HF processor
        (兼容口径)。见 starVLA/model/modules/vlm/qwen_vision_preprocess.py。
        """
        if self.vision_preproc is not None:
            return self.vision_preproc.batch_encode(img_flat, self.model.device)
        return self.processor.image_processor(images=img_flat, return_tensors="pt")

    def _assemble_inputs(self, ids_list, img_flat):
        """文本 ids (各长度) + 图像特征 → 左 pad 静态 batch (模型输入契约)。

        向量化实现: H200 profile 显示逐行 python 循环 (128 次 slice 赋值) 造成
        每步 ~50ms 的 host 停顿 (GPU 队列空转)。改为 advanced-indexing scatter,
        常数次 kernel 完成, 与 batch 大小无关。
        """
        from transformers import BatchFeature

        target = self.pad_seq_len if self.pad_seq_len > 0 else max(t.numel() for t in ids_list)
        lens = torch.tensor([t.numel() for t in ids_list])
        if bool((lens > target).any()):
            raise ValueError(f"序列长 {int(lens.max())} > pad_seq_len {target}, 请增大 pad_seq_len 或重算 tok 缓存")
        b = len(ids_list)
        rows = torch.repeat_interleave(torch.arange(b), lens)
        cols = (target - lens)[rows] + torch.cat([torch.arange(n) for n in lens])
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = torch.full((b, target), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(b, target, dtype=torch.long)
        input_ids[rows, cols] = torch.cat(ids_list)
        attention_mask[rows, cols] = 1  # 左对齐 (与 tokenizer padding_side=left 一致)
        feats = self._encode_images(img_flat)
        return BatchFeature(data={**feats, "input_ids": input_ids, "attention_mask": attention_mask})

    def _slow_text_ids(self, dummy, instruction, n_img):
        """参考实现: 完整 apply_chat_template(tokenize=True) 的单条 ids (对拍/回退用)。"""
        msg = [{"role": "user", "content": [{"type": "image", "image": dummy}] * n_img
                + [{"type": "text", "text": instruction}]}]
        enc = self.processor.apply_chat_template(
            msg, tokenize=True, padding=False, add_generation_prompt=True
        )
        ids = enc["input_ids"] if isinstance(enc, dict) else enc
        return ids[0]

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """Build model inputs from raw data (images + instructions + optional solutions).

        Follows Official Qwen3.5-VL Instruct format.

        架构 (文本/图像解耦, 两条路各自可加速):
        - 图像: 恒走 ``_encode_images`` (GPU batch 预处理, 配置见 vision_preprocess);
        - 文本: tok cache 全命中 → 查表; 否则 chat template + tokenizer (占位图用
          vision_preproc 的目标尺寸 dummy, 展开的 vision token 数与真实网格一致,
          因此慢路径与快路径 token 级同构)。solutions 路径 (训练不用) 走完整慢路径。
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        img_flat = [img for imgs in images for img in imgs]

        if solutions is None:
            # ---- 文本快路径: tok cache 命中 ----
            if self._tok_cache is not None:
                miss_idx = [i for i, k in enumerate(instructions) if k not in self._tok_cache]
                if not miss_idx:
                    self._tok_hits += len(instructions)
                    ids_list = [self._tok_cache[k] for k in instructions]
                    return self._assemble_inputs(ids_list, img_flat).to(self.model.device)
                self._tok_misses += len(miss_idx)

            # ---- 文本慢路径: chat template (dummy 占位图, 尺寸=预处理目标) ----
            # 逐条处理: H1 的 state 进文本后每条长度不同, 批量 padding=False 会让
            # processor 内部 np.array(ragged) 直接 ValueError; 且占位图数量必须
            # 与该样本真实视图数一致 (vision token 展开数取决于图片数)。
            if self.vision_preproc is not None:
                size = self.vision_preproc.image_size
                dummy = Image.new("RGB", (size, size))
            else:
                dummy = img_flat[0]
            # 快路径: tokenize=True 会把 dummy 图送进 processor 像素管线
            # (~20ms/条, 结果被 GPU 预处理丢弃)。改为渲染串 (无图对象, 占位符
            # 不展开) → 按 processor 的真实展开数手动扩展 image_pad → 纯
            # tokenizer。展开数首次用一条真慢调用自校准, 并对拍首条结果,
            # 不一致自动回退慢路径 (正确性优先)。H1 实测 850ms → 89ms / 32 条。
            if self._n_vision_tokens_per_image is None:
                msg1 = [{"role": "user", "content": [
                    {"type": "image", "image": dummy}, {"type": "text", "text": "calib"}]}]
                enc1 = self.processor.apply_chat_template(
                    msg1, tokenize=True, padding=False, add_generation_prompt=True
                )
                ids1 = enc1["input_ids"][0] if isinstance(enc1, dict) else enc1[0]
                _pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                self._n_vision_tokens_per_image = max(1, sum(1 for t in ids1 if t == _pad_id))
            n_v = self._n_vision_tokens_per_image
            ids_list = []
            for i, instruction in enumerate(instructions):
                k = len(images[i]) if i < len(images) else 1
                msg = [{"role": "user", "content": [{"type": "image"}] * k
                        + [{"type": "text", "text": instruction}]}]
                s = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                if isinstance(s, list):
                    s = s[0]
                s = s.replace("<|image_pad|>", "<|image_pad|>" * n_v)
                ids_list.append(
                    self.processor.tokenizer(s, add_special_tokens=False)["input_ids"]
                )
            if self._fast_text_unverified:
                ref = self._slow_text_ids(dummy, instructions[0],
                                          len(images[0]) if images else 1)
                if ref is not None and list(ids_list[0]) != list(ref):
                    print("[QWen3_5] 文本快路径与慢路径 token 不一致, 回退慢路径")
                    ids_list = None
                self._fast_text_unverified = False
            if ids_list is None:
                ids_list = [
                    self._slow_text_ids(dummy, instruction,
                                        len(images[i]) if i < len(images) else 1)
                    for i, instruction in enumerate(instructions)
                ]
            # 运行时 memo: 慢路径算出的 ids 回填缓存 —— state-in-text 数据集
            # (如 H1) instruction 每样本唯一, 但 epoch 间必然重复; 首个 epoch
            # 后全命中。字符串来自真实训练管线, 零重建风险。
            if self._tok_memo and self._tok_cache is not None:
                for i in miss_idx:
                    self._tok_cache[instructions[i]] = torch.as_tensor(
                        ids_list[i], dtype=torch.long
                    )
            out = self._assemble_inputs(
                [torch.as_tensor(ids, dtype=torch.long) for ids in ids_list], img_flat
            )
            return out.to(self.model.device)

        # ---- solutions 完整慢路径 (含 labels, 训练不用; 保持原行为) ----
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        batch_inputs["input_ids"], batch_inputs["attention_mask"] = self._pad_to_static_len(
            batch_inputs["input_ids"], batch_inputs["attention_mask"]
        )
        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
