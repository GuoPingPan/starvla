"""
PI05 模型 (Physical Intelligence π0.5 协议的 StarVLA 实现)。

与 PI0 的关系 (核心阅读点)
--------------------------
本文件只有 ~60 行,**99% 的逻辑继承自 PI0**。这不是偷懒,而是真实反映了
π0.5 论文的设计哲学:π0.5 是在 π0 基础上的"小改 + 大效果"。

π0 → π0.5 的三个本质差异:

1. **Action Head 不同**:`action_head_cls` 从 `OpenPI0ActionHead` 换成
   `OpenPI05ActionHead`。后者把时间嵌入从"拼到 action token 序列里"
   改为"通过 adaRMS 条件化每一层 layernorm"。
   详见 `starVLA/model/modules/action_model/OpenPI_ActionHead.py:268`。

2. **默认离散化 state 输入**:`discrete_state_input = True`,把 robot state
   离散化为 256 桶后塞进语言 prompt ("Task: ..., State: 0.3 0.5 ...;")。
   不再单独走 `state_proj` 投连续 token。

3. **adaRMS 注入机制**:每个 Gemma 层的 RMSNorm 多了一个 `dense` 子模块
   (Linear(cond_dim, width*3)),输出 (scale, shift, gate) 用 timestep
   调制归一化输出。`dense.weight` 初始化为 0,所以训练初期是恒等映射。

starVLA 未实现的 π0.5 部分
--------------------------
- 论文里的 web-scale teacher-student 蒸馏流程 (本实现只有 student 架构)。
- 论文里更长的 hierarchical action chunking (本实现默认 horizon=50)。

所以这个文件本质是"π0.5 架构 + π0 训练流程"的混合体。

混合配置说明
------------
PI05DefaultConfig 默认 `discrete_state_input=True`,但 yaml 里经常
override 成 `false` (例如 `pi05_libero_8gpu.yaml:14`)。这种"PI05 架构
+ PI0 风格的连续 state 输入"完全合法,因为 PI0 父类的 `_prepare_examples`
里同时处理两种模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from starVLA.model.framework.VLM4A.PI0 import PI0, PI0DefaultConfig, IMAGE_KEYS
from starVLA.model.modules.action_model.OpenPI_ActionHead import OpenPI05ActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class PI05DefaultConfig(PI0DefaultConfig):
    """PI05 默认配置。

    与 `PI0DefaultConfig` 字段几乎完全相同,只覆盖以下三项:
    - `name`: "PI05" (用于日志、registry、checkpoint 元信息)
    - `discrete_state_input`: True (PI0 是 False)
    - 其他字段保留 PI0 默认 (action_dim=32 / horizon=50 / num_inference_steps=10)

    PI05 与 PI0 的真正差异在 action_head_cls,不在配置字段。
    """

    name: str = "PI05"
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    action_dim: int = 32
    action_horizon: int = 50
    max_state_dim: int = 32
    # True 时 state 走语言 tokenizer 离散化;False 时走 action_head.state_proj。
    # 这是 PI0/PI05 默认行为的主要差异点。
    discrete_state_input: bool = True
    max_token_len: int = 200
    num_inference_steps: int = 10

    image_resolution: list[int] = field(default_factory=lambda: [224, 224])
    image_keys: list[str] = field(default_factory=lambda: list(IMAGE_KEYS))

    tokenizer: dict = field(default_factory=lambda: {"model_path": "", "pad_state_value": -2.0})
    paligemma: dict = field(default_factory=lambda: {"model_name_or_path": ""})
    # PaliGemma 3B 主干配置 (与 PI0 完全一致,沿用 OpenPI 官方权重布局)。
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
    # Action expert 配置:小型 Gemma (~1B 参数),与 PI0 一致。
    # PI0/PI05 的差异在 layernorm 行为 (adaRMS),不在 expert 维度。
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


@FRAMEWORK_REGISTRY.register("PI05")
@FRAMEWORK_REGISTRY.register("Pi05")
class PI05(PI0):
    """OpenPI PI05 模型,通过 StarVLA framework API 暴露。

    只覆盖三个类属性,所有方法 (forward / sample_actions / load_pretrained_checkpoint)
    全部继承自 PI0:
      - default_config_cls:指向 PI05DefaultConfig (discrete_state_input=True)
      - action_head_cls:换成 OpenPI05ActionHead (adaRMS 条件化)
      - model_name:用于日志和 registry

    为什么不重写 forward?
    --------------------
    PI0 的 forward 主干里用 `_norm()` 这个双模式 helper:
      - layernorm 没有 cond 入参 → 走普通 RMSNorm (PI0 行为)
      - layernorm 有 cond 入参   → 走 adaRMS (PI05 行为)
    模型构造时,OpenPIGemma 在 use_adarms=True 下会给每层注入 dense 模块,
    所以后续 forward 自动走 PI05 路径,不需要子类重写。

    类属性 vs 实例属性的注意点
    --------------------------
    `discrete_state_input = True` 在这里是**类属性**。PI0.__init__ 里的
    `_init_runtime_config` 会读取 `self.config.framework.discrete_state_input`
    覆盖到 `self.discrete_state_input` 实例属性上。所以最终生效值取决于 yaml
    配置,类属性只是 fallback。
    """

    default_config_cls = PI05DefaultConfig
    action_head_cls = OpenPI05ActionHead
    model_name = "PI05"
    discrete_state_input = True
