# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA 训练器 (VLATrainer)
基于原生 PyTorch + Accelerate + DeepSpeed 实现,刻意保持训练循环的显式与可读,
方便研究者直接 hack。

设计约定
--------
1. 运行时状态优先存到 dict 里 (data info / process info / config 等),便于
   跨框架/数据集扩展时灵活传递。
2. 多个 dataloader 适配异构数据/混合数据集 (VLA / VLM / cotrain 各一路)。
3. 每种训练策略独占一个 `trainer_*.py` 文件,避免大型 if-else 分支。

核心流程 (main 函数)
--------------------
  1. wrap_config             —— 把 OmegaConf cfg 包装为 AccessTrackedConfig,
                                  跟踪字段访问,后续只把"真正用到的字段"写回 yaml。
  2. warn_unused_yaml_fields —— 早一些暴露"配了但代码不读"的死字段 (如 action_type)。
  3. setup_directories       —— 创建 output_dir / checkpoints 子目录。
  4. build_framework         —— 根据 cfg.framework.name 从 FRAMEWORK_REGISTRY
                                  实例化 VLA 模型 (PI0 / PI05 / QwenPI / ...)。
  5. prepare_data            —— 构建 VLA 训练 dataloader。
  6. setup_optimizer_and_scheduler —— 构造 AdamW + LR scheduler。
  7. VLATrainer(...).prepare_training() → .train() —— 真正跑训练循环。
"""

# 标准库
import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

# 第三方库
import numpy as np
import torch
import torch.distributed as dist

# NPU 支持:如果环境装了 torch_npu,自动把代码里的 "cuda" 调用映射到 NPU;
# 纯 GPU 环境下 import 失败会被静默忽略 (no-op)。
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# 本地模块
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args
from starVLA.training.trainer_utils.logger import build_tracker
from starVLA.training.trainer_utils.perf_analyzer import PerformanceAnalyzer

# 模块级初始化 Accelerator + DeepSpeed 插件。放在模块顶层是为了让其他文件
# (dataloader、model 模块等) 通过 import 拿到同一个 accelerator 实例,
# 保证分布式状态全局唯一。
# 仅在 DS launch 下挂 plugin: accelerate launch --config_file <deepspeed.yaml>
# 会在子进程注入 ACCELERATE_USE_DEEPSPEED=true; 否则 (裸 python / 默认
# accelerate launch) 强制挂 plugin 会把单进程也拖进 DeepSpeed 初始化
# (MPI 探测报 mpi4py 缺失), 无法做无 DS 的调试/单卡复放。
_use_ds_launch = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
deepspeed_plugin = DeepSpeedPlugin() if _use_ds_launch else None
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

logging.getLogger("DeepSpeed").setLevel(logging.WARNING)

accelerator.print(accelerator.state)

# 可选 bit 级确定性 (验证 resume 数值零偏差用; 有非确定性实现的算子只告警
# 不报错): CUBLAS_WORKSPACE_CONFIG 必须在首个 cuBLAS 句柄创建前设置。
if os.environ.get("STARVLA_DETERMINISTIC", "0").lower() in ("1", "true", "yes", "on"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    accelerator.print("🔒 STARVLA_DETERMINISTIC=1: deterministic algorithms enabled (warn_only)")

# PaliGemma / fast tokenizer 在多进程下会触发死锁,显式关掉并行。
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


def load_fast_tokenizer():
    """加载 physical-intelligence/fast tokenizer (OpenPI 协议使用)。"""
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def warn_unused_yaml_fields(cfg) -> None:
    """检测并 warning 提示"配了但代码不读"的 YAML 字段。

    背景:某些字段 (典型如 `datasets.vla_data.action_type`) 在 20+ 份 yaml 和
    数据集集成文档里都被推荐填写,但 starVLA 实际没有任何 Python 代码读取它。
    如果用户依赖它来切换归一化方式,会得到与预期完全不同的训练效果且毫无报错。

    真正生效的是另外两条独立路径:
      1. `action_mode` (abs/delta/rel) → 决定统计量计算路径,见
         `starVLA/dataloader/gr00t_lerobot/datasets.py` 的
         `_init_action_mode` / `_apply_action_mode`。
      2. 每个 robot_type 对应的 DataConfig.transform() → 决定每个 action/state
         key 的归一化方式 (q99 / min_max / ...)。配置位置在
         `examples/simBenchmarks/<bench>/train_files/data_registry/data_config.py`。
    """
    # 只在 rank 0 打 warning,避免多卡重复刷屏。
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    try:
        action_type = cfg.datasets.vla_data.action_type
    except Exception:
        # cfg 可能是 AccessTrackedConfig、DictConfig 或已被 unwrap 的 dataclass,
        # 字段缺失时的报错方式不一,统一吞掉。
        return
    if not action_type:
        return
    logger.warning(
        f"`datasets.vla_data.action_type={action_type!r}` is set but NOT consumed by any starVLA code. "
        "Action representation is controlled by `action_mode` + the robot_type DataConfig transform. "
        "This field is deprecated and will be removed in a future release."
    )


def setup_directories(cfg) -> Path:
    """创建 output_dir 与 checkpoints/ 子目录,并把 cfg.output_dir 字段填上。

    目录结构:
      {run_root_dir}/{run_id}/{MMDD_HHMMSS}/   ← 新训,自动带时间避免覆盖
      {run_root_dir}/{run_id}/                  ← resume,直接复用
    新训时间戳由 rank 0 广播给所有 rank，避免多机时各节点跨秒生成不同目录。
    只有 rank 0 创建目录，其他 rank 在 barrier 后继续。
    """
    is_resume = getattr(getattr(cfg, "trainer", None), "is_resume", False)
    if is_resume:
        cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    else:
        if dist.is_initialized():
            timestamp = [datetime.now().strftime("%m%d_%H%M%S") if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(timestamp, src=0)
            ts = timestamp[0]
        else:
            ts = datetime.now().strftime("%m%d_%H%M%S")
        cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id, ts)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """构建 VLA 训练 + 在线评测两路 DataLoader。

    训练侧走 train 划分 (随机采样); 在线评测侧 (eval_action_model) 走 eval 划分
    (确定性采样), 反映泛化能力而非记忆。若未启用 ``datasets.vla_data.split``,两路
    都是全量数据 (评测侧只是改成确定性采样),行为与改动前等价。

    实际的 dataset / sampler / collate 构造在 `build_dataloader` 里完成,这里
    只是入口。`dispatch_batches=False` 关掉 accelerate 的自动 batch 分发 ——
    我们用 DistributedSampler 自己控制每个 rank 拿哪些样本,避免 accelerate
    重复切片。
    """
    if os.environ.get("STARVLA_SYNTHETIC", "0") in ("1", "true", "yes"):
        logger.info("🧪 STARVLA_SYNTHETIC=1 — synthetic H1 data (不落地/不解码/不读 CFS)")
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, mode="train")
    vla_eval_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, mode="eval")

    accelerator.dataloader_config.dispatch_batches = False
    # barrier 保证所有 rank 都把 dataloader 构造完再返回,防止快 rank 跑到下一步
    # 时慢 rank 的数据集统计/缓存还没生成。
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader, vla_eval_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """构造 AdamW 优化器 + LR scheduler。

    - `build_param_lr_groups` 把模型参数按 backbone / action_head / projector
      等模块拆成多组,每组配独立 LR (在 yaml 的 trainer.learning_rate 下配)。
    - `fused=True` 使用 CUDA fused AdamW 实现,显著降低 kernel launch 开销。
    - `scheduler_specific_kwargs` 里可能含有 `get_scheduler` 不认识的自定义键
      (例如 transformer 版本升级后多出来的字段),先剥离再传。
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    # fused=True 降 kernel launch 开销; 但与 DeepSpeed checkpoint 恢复存在
    # state 布局错位 (见 trainer_tools.setup_optimizer_and_scheduler 注释),
    # env STARVLA_FUSED_ADAMW=0 可关。CUDA 不可用时自动 fallback。
    fused_available = torch.cuda.is_available() and os.environ.get(
        "STARVLA_FUSED_ADAMW", "1"
    ).lower() not in ("0", "false", "no", "off")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=fused_available,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    """VLA 训练器主体。

    继承 `TrainerUtils` 是为了复用 checkpoint 加载、参数冻结、模块替换等通用
    工具;本类只负责训练循环本身。
    """

    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator, vla_eval_dataloader=None):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        # 在线评测 dataloader (eval 划分, 确定性采样)。None 时回退到训练 dataloader。
        self.vla_eval_dataloader = vla_eval_dataloader if vla_eval_dataloader is not None else vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        # total_batch_size 用于日志展示和 LR scheduler 的 total_steps 校验:
        # = per_device_bs × num_processes × gradient_accumulation_steps
        self.total_batch_size = self._calculate_total_batch_size()

        # GPU 性能分析器:默认关闭零开销。用 STARVLA_PROFILE=1 或
        # trainer.profile.enabled 开启,采集一小段 step 诊断"通信 vs 计算"瓶颈。
        self.perf = PerformanceAnalyzer(
            self.config, self.accelerator, output_dir=Path(self.config.output_dir)
        )
        # 零扰动 CUDA event 分段计时 (STARVLA_EVT=1): 稳态归因用, 与
        # perf 的带 sync 剖面互补。见 trainer_utils/event_timer.py。
        from starVLA.training.trainer_utils.event_timer import EventTimer

        self.evt = EventTimer()

    def prepare_training(self):
        """训练开始前的准备工作 (核心)。顺序很重要,不能随意调换:

            1. 设种子 (每个 rank 不同)
            2. 先存初始配置           —— 任何后续步骤失败,配置都已有迹可循
            3. checkpoint 加载         —— 区分 pretrained init 和 resume
            4. LR scheduler 重放       —— resume 时把内部 step 快进到断点
            5. 冻结模块                —— 在 accelerate.prepare 之前,避免
                                          被冻结参数也被 DeepSpeed 包装进优化器
            6. accelerator.prepare     —— 包装 model / optimizer / dataloader
            7. tracker (SwanLab / WandB) —— last, to avoid early network overhead
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 给每个 rank 不同 seed,保证数据增强/dropout 等随机性在多卡间产生差异;
        # 否则所有 rank 看到同样的 dropout mask 会变相降低有效 batch size。
        # 3047 是无 seed 字段时的兜底偏移,无特殊含义。
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # 提前保存 cfg 快照:即便后面任意一步 (ckpt load / DeepSpeed init / 数据集
        # 构建) 崩溃,output_dir 也已经有了可追溯的 config.full.yaml,方便调试或
        # from_pretrained 加载。
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # accelerator.prepare 会把 model/optimizer/dataloader 包装成分布式版本:
        #   - model     → DeepSpeed engine (含 ZeRO 切片)
        #   - optimizer → DeepSpeed 包装的优化器 (实际反向/step 都走 DS)
        #   - dataloader→ DistributedSampler + gradient accumulation 调度
        # 在线评测 dataloader 一并 prepare (注入 DistributedSampler 切分 anchor)。
        self.model, self.optimizer, self.vla_train_dataloader, self.vla_eval_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.vla_eval_dataloader,
        )

        # ── 全量训练状态恢复 (真 resume): 必须在 engine 创建之后 ──────────────
        # fp32 master 权重 + AdamW m/v 只存在于 DeepSpeed 包装里, prepare 前
        # 的 _init_checkpointing 只做了 weights-only 热重启兜底。这里若发现
        # steps_N_ds/ 全量状态则恢复 (优化器/RNG/步数), 失败时保留兜底路径。
        if getattr(self, "_full_state_tag", None):
            self._restore_full_state()

        # ── 调试开关: 静态复读 dataloader (data_time ≡ 0) ─────────────────────
        # 必须在 accelerate.prepare 之后替换: DeepSpeed 要求 prepare 的 args
        # 里有真 DataLoader (train_micro_batch_size_per_gpu: auto)。详见
        # starVLA/dataloader/static_repeat.py 模块注释。
        # env STARVLA_STATIC_DATA 优先于 yaml datasets.vla_data.static_repeat。
        _vla_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        _yaml_static = bool(getattr(_vla_cfg, "static_repeat", False))
        _env_static = os.environ.get("STARVLA_STATIC_DATA")
        _static_on = (
            _env_static.strip().lower() in ("1", "true", "yes", "on")
            if _env_static is not None
            else _yaml_static
        )
        if _static_on:
            from starVLA.dataloader.static_repeat import StaticRepeatDataLoader

            _env_nb = os.environ.get("STARVLA_STATIC_BATCHES")
            _yaml_nb = getattr(_vla_cfg, "static_repeat_batches", None)
            _num_batches = int(_env_nb) if _env_nb else int(_yaml_nb or 8)
            if self.accelerator.is_main_process:
                logger.info(
                    f">> [static-repeat] 启用: 预取 {_num_batches} 个 batch 后无限复读 "
                    f"(timing/data 将恒为 ~0; 只用于性能/数值调试, 勿评估收敛)"
                )
            self.vla_train_dataloader = StaticRepeatDataLoader(
                self.vla_train_dataloader, num_batches=_num_batches, tag="train", log_every_fetch=True
            )
            self.vla_eval_dataloader = StaticRepeatDataLoader(
                self.vla_eval_dataloader, num_batches=_num_batches, tag="eval"
            )

        self._init_tracker()

    def _calculate_total_batch_size(self):
        """计算全局 batch size (per_device × num_processes × grad_accum)。"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_tracker(self):
        """Initialize experiment tracker (SwanLab / WandB)."""
        self.tracker = build_tracker(self.config)
        self.tracker.init(self.config, self.config.output_dir, self.accelerator)

    def _save_initial_configs(self):
        """训练开始时保存两份配置快照。

        1. `config.full.yaml`:OmegaConf 合并后的完整 cfg,**所有**字段都写进去
           (含默认值),用于复现和调试。
        2. `config.yaml`:只包含代码**实际访问过**的字段,通过 AccessTrackedConfig
           追踪得到。后续每次 checkpoint 都会刷新一次,目的是让 yaml 配置文件
           精简到"真正影响训练结果"的最小集合,方便分享与 diff。
        """
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """处理 checkpoint 加载逻辑,区分两种场景:

        - `is_resume=True`:从上次中断处继续训练。两级恢复:
          1. 这里先做 weights-only 热重启兜底 (找最新 steps_X 权重 + 记步数,
             scheduler 由 _adjust_lr_scheduler_for_resume 快进);
          2. 若存在 steps_N_ds/ 全量状态, prepare 后由 _restore_full_state 恢复
             fp32 master + AdamW m/v + RNG (真 resume), 覆盖兜底加载的权重。
        - `pretrained_checkpoint` (无 is_resume):加载预训练权重作为初始化,
          但 step 数从 0 开始 (典型如用 OpenPI 权重继续训 LIBERO)。
        """
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        # 全量状态 (steps_N_ds/) 由 _save_full_state 写入, prepare 后才能恢复;
        # 这里只做发现 + 记步数, 并让 completed_steps 取两者较新者。
        self._full_state_tag, self._full_state_steps = self._find_latest_full_state()

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if self._full_state_tag:
                self.completed_steps = max(self.completed_steps, self._full_state_steps)
                logger.info(
                    f"Full training state found: {self._full_state_tag} "
                    f"(steps={self._full_state_steps}, restore after DeepSpeed engine init)"
                )
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            # reload_modules 允许只重载部分子模块 (如只重载 VLM,不重载 action_head),
            # 用于 cotrain 场景下分阶段加载。
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """断点续训时把 LR scheduler 的内部 step 数快进到对应位置。

        LR scheduler 内部维护一个 last_lr 计数器,新建时是 0。如果不快进,warmup
        会从 0 开始重新跑一遍,导致续训前几步 LR 不连续 (甚至降到 base_lr 的
        1/100),把已训好的权重打飞。

        全量状态恢复路径也依赖这里: 我们的 ds_config 没给 DeepSpeed 配 scheduler
        (engine.load_checkpoint 的 load_lr_scheduler_states 无从恢复), torch
        scheduler 的位置统一由本函数按 completed_steps 快进, 两条路径行为一致。
        """
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _find_latest_full_state(self):
        """在 checkpoint_dir 里找步数最大的 steps_N_ds/ 全量状态目录。

        Returns:
            (tag, steps): tag 如 "steps_30000_ds"; 没有则 (None, 0)。
        """
        if not os.path.isdir(self.checkpoint_dir):
            return None, 0
        best_steps, best_tag = 0, None
        for name in os.listdir(self.checkpoint_dir):
            m = re.match(r"steps_(\d+)_ds$", name)
            if m and os.path.isdir(os.path.join(self.checkpoint_dir, name)):
                n = int(m.group(1))
                if n > best_steps:
                    best_steps, best_tag = n, name
        return best_tag, best_steps

    def _save_full_state(self, emergency: bool = False, steps_override: int | None = None):
        """保存全量训练状态 (真 resume 点): 权重 + fp32 master + AdamW m/v + RNG。

        - DeepSpeed engine: save_checkpoint 写 per-rank 分片 + RNG + (有 scheduler
          时) scheduler 状态。集合操作, 所有 rank 必须同时调用 —— 因此本方法
          不能放进 is_main_process guard。
        - 非 DS 后端 (DDP/FSDP): accelerator.save_state (model+optimizer+
          scheduler+RNG), 步数写 sidecar full_state.json。
        - completed_steps 放 client_state, 恢复时以此为准 (engine.global_steps
          与它一致)。steps_override: 看门狗紧急快照在 optimizer step 之后调用,
          此时 train loop 尚未 +=1, 用它修正成"已完成 N 步"语义。
        - rolling 清理: 只保留最新 STARVLA_FULLSTATE_KEEP (默认 2, 最小 1) 份
          steps_*_ds/ —— ZeRO-2 下 AdamW 状态 ~50GB 量级, 不清理会涨满盘。
        - emergency=True: 首爆后快照, 与周期保存同一命名空间, 天然成为最新
          resume 点, 续训即跳过污染区段。
        """
        steps = self.completed_steps if steps_override is None else int(steps_override)
        tag = f"steps_{steps}_ds"
        ckpt_dir = os.path.join(self.checkpoint_dir, tag)
        if hasattr(self.model, "save_checkpoint"):
            self.model.save_checkpoint(
                self.checkpoint_dir,
                tag=tag,
                client_state={"completed_steps": steps, "emergency": emergency},
            )
            # DS 0.16.9 的 engine.save_checkpoint 不落 RNG (model_states.pt 里无
            # rng_states 键) —— load 后 CUDA RNG 停留在 set_seed 起点, 恢复后的
            # flow t/ε 流从头开始而非断点续流; 相位错位 + 硬 batch 在 bf16 反向
            # 溢出 → 恢复后 2-3 步 NaN (两阶段复现实证)。这里按 rank 落 sidecar,
            # _restore_full_state 里手动应用。accelerate 的 save_state (非 DS 路径)
            # 自带 RNG 保存, 无需此补丁。
            rank = dist.get_rank() if dist.is_initialized() else 0
            rng_payload = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            }
            torch.save(rng_payload, os.path.join(ckpt_dir, f"rng_state_rank{rank}.pt"))
            # Adam moments sidecar: DS 0.16.9 的 load_optimizer_states=True 恢复
            # 有毒 —— checkpoint 文件本身干净、fp32 master 恢复逐位正确, 但灌回的
            # Adam state 会让第 2 次更新把部分 master (实测 patch_embedding) 写成
            # NaN (梯度有限, 更新量 ~1e-6, 不可能合法溢出)。改为从引擎内部
            # optimizer 原样导出 state_dict、恢复时原样灌回 —— 同一对象同一布局,
            # 零转换; _restore_full_state 里 load_optimizer_states=False 跳过 DS 路径。
            inner_opt = getattr(self.model.optimizer, "optimizer", None)
            if inner_opt is not None and getattr(inner_opt, "state", None):
                torch.save(
                    inner_opt.state_dict(),
                    os.path.join(ckpt_dir, f"adam_state_rank{rank}.pt"),
                )
        else:
            self.accelerator.save_state(ckpt_dir)
            if self.accelerator.is_main_process:
                with open(os.path.join(ckpt_dir, "full_state.json"), "w") as f:
                    json.dump({"completed_steps": steps, "emergency": emergency}, f)
        if self.accelerator.is_main_process:
            logger.info(
                f"💾 full training state saved → {ckpt_dir} (steps={steps})"
                + (" (emergency: post-first-spike snapshot)" if emergency else "")
            )
            self._prune_full_states()
        self.accelerator.wait_for_everyone()

    def _prune_full_states(self):
        """rolling 删除旧的 steps_*_ds/, 只留最新 keep 份 (rank0 调用)。"""
        keep = max(1, int(os.environ.get("STARVLA_FULLSTATE_KEEP", "2")))
        entries = []
        for name in os.listdir(self.checkpoint_dir):
            m = re.match(r"steps_(\d+)_ds$", name)
            if m and os.path.isdir(os.path.join(self.checkpoint_dir, name)):
                entries.append((int(m.group(1)), name))
        entries.sort()
        for _, name in entries[:-keep]:
            shutil.rmtree(os.path.join(self.checkpoint_dir, name), ignore_errors=True)
            logger.info(f"🗑 pruned old full state: {name}")

    def _restore_full_state(self):
        """恢复全量训练状态 (fp32 master + AdamW m/v + RNG + 步数)。

        只在 _full_state_tag 存在时由 prepare_training 调用 (engine 已创建)。
        失败不抛 —— 保留 _init_checkpointing 已加载的权重走热重启兜底, 只丢
        优化器/RNG 状态; 典型失败原因: world size 与保存时不同 (ZeRO 分片不可
        跨卡数重分片)、分片文件损坏。
        """
        tag = self._full_state_tag
        path = os.path.join(self.checkpoint_dir, tag)
        logger.info(f"Restoring full training state from {path}")
        try:
            if hasattr(self.model, "load_checkpoint"):
                load_path, client_state = self.model.load_checkpoint(
                    self.checkpoint_dir,
                    tag=tag,
                    # 见 _save_full_state 的 adam_state sidecar 注释: DS 0.16.9 的
                    # optimizer state 恢复路径有毒 (恢复后第 2 次更新写 NaN 进
                    # master), Adam moments 改走自管 sidecar, DS 路径显式关闭。
                    load_optimizer_states=False,
                    load_lr_scheduler_states=True,
                    load_module_strict=False,
                )
                if not load_path:
                    # DS 某些失败路径只告警不抛错, 返回 (None, None) —— 显式转失败,
                    # 走下面的热重启回退分支。
                    raise RuntimeError("DeepSpeed load_checkpoint returned no load path")
                steps = int((client_state or {}).get("completed_steps", self._full_state_steps))
                # Adam moments: DS 0.16.9 的两条恢复路径都不可用 ——
                # a) load_optimizer_states=True: 恢复后第 2 次更新把部分 master
                #    (实测 patch_embedding, fp32 分区 flat[2:]) 写成 NaN —— 梯度
                #    有限、master 恢复逐位正确, 是 DS 恢复灌回的 state 与引擎
                #    lazy-init 布局错位;
                # b) 自管 sidecar 灌回 (STARVLA_ADAM_SIDECAR=1): 跨进程首次
                #    engine.step() 的 lazy_init_hp_params_optimizer_state 仍会
                #    重置/搬移灌入的 state, 同样在第 2 次更新写 NaN (算术解剖
                #    实证: 更新量非 0 的元素 NaN, 更新量 0 的元素幸存)。
                # 默认 warm (丢弃 moments, 只保权重/master/RNG/scheduler/数据流
                # 对齐): 实测 resume 后全程健康; 代价是 β2 记忆重建期 (~几十步)
                # 更新幅度有轻微畸变。sidecar 保留为后续深挖入口。
                rank = dist.get_rank() if dist.is_initialized() else 0
                if os.environ.get("STARVLA_ADAM_SIDECAR", "0").lower() in ("1", "true", "yes", "on"):
                    adam_path = os.path.join(path, f"adam_state_rank{rank}.pt")
                    if os.path.isfile(adam_path):
                        inner_opt = getattr(self.model.optimizer, "optimizer", None)
                        if inner_opt is not None:
                            inner_opt.load_state_dict(
                                torch.load(adam_path, map_location="cuda", weights_only=False)
                            )
                            logger.info(f"Adam moments restored from sidecar: {adam_path}")
                    else:
                        logger.warning(f"⚠️ {tag} 缺 adam_state_rank{rank}.pt, moments 按 warm 处理")
                else:
                    logger.warning(
                        "⚠️ Adam moments 按 warm 重启 (丢弃): DS 0.16.9 的两条恢复路径都会在"
                        "恢复后第 2 次更新写 NaN 进 fp32 master (详见 _restore_full_state 注释);"
                        "权重/master/RNG/scheduler/数据流均已精确恢复, 仅 β2 记忆重建期有轻微畸变"
                    )
                # DS checkpoint 不含 RNG (见 _save_full_state 的 sidecar 注释):
                # 手动恢复, 让 flow t/ε 采样流从断点续流 —— 这是数值零偏差
                # resume 的必要条件, 缺失时 (旧 checkpoint) 大声告警。
                rng_path = os.path.join(path, f"rng_state_rank{rank}.pt")
                if os.path.isfile(rng_path):
                    payload = torch.load(rng_path, map_location="cpu", weights_only=False)
                    torch.set_rng_state(payload["cpu"])
                    if payload.get("cuda") is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(payload["cuda"])
                    random.setstate(payload["python"])
                    np.random.set_state(payload["numpy"])
                else:
                    logger.warning(
                        f"⚠️ {tag} 缺 rng_state_rank{rank}.pt (旧版 checkpoint?): "
                        "flow 采样流将从 set_seed 起点重放, resume 数值会有相位差"
                    )
            else:
                self.accelerator.load_state(path)
                with open(os.path.join(path, "full_state.json")) as f:
                    steps = int(json.load(f)["completed_steps"])
            self.completed_steps = steps
            logger.info(
                f"✅ full state restored: steps={steps} "
                "(fp32 master + AdamW moments + RNG; scheduler 位置已由 "
                "_adjust_lr_scheduler_for_resume 快进对齐)"
            )
        except Exception as exc:
            logger.warning(
                f"⚠️ full state restore FAILED ({type(exc).__name__}: {exc}); "
                "回退 weights-only 热重启 (丢优化器/RNG 状态)。全量状态恢复要求与"
                "保存时相同的 NUM_GPUS (ZeRO 分片按卡数切)。"
            )


    def _load_checkpoint(self, checkpoint_path):
        """加载完整训练状态 (model + optimizer + scheduler),用于断点续训。"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """保存当前训练状态的 checkpoint (两层)。

        1. 轻量权重: state_dict 存 steps_N_pytorch_model.pt / .safetensors
           (rank0), 供 eval / 部署 / 离线取证直接加载。
        2. 全量状态: steps_N_ds/ (所有 rank, DeepSpeed 分片) —— fp32 master +
           AdamW m/v + RNG + client_state{completed_steps}, 是 is_resume 时的
           首选恢复点; rolling 只留最新 STARVLA_FULLSTATE_KEEP (默认 2) 份。
        3. 配置:AccessTrackedConfig 把"训练过程中被新访问到的字段"追加写入
           config.yaml,保证后续可以从 yaml 复现训练。
        """
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

        if isinstance(self.config, AccessTrackedConfig):
            logger.info("📊 Saving accessed configuration...")
            output_dir = Path(self.config.output_dir)
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info("✅ Configuration files saved")

        # 全量训练状态 (真 resume 点): 所有 rank 集合调用, 不能放进上面的
        # is_main_process guard —— DeepSpeed 分片写是集合操作。
        self._save_full_state()

        self.accelerator.wait_for_everyone()

    def _tok_cache_stats(self):
        """tok cache/memo 命中率 (0-1); 接口无计数时返回 None 不上报。"""
        _iface = getattr(getattr(self.model, "module", self.model), "qwen_vl_interface", None)
        h, m = getattr(_iface, "_tok_hits", 0), getattr(_iface, "_tok_misses", 0)
        if h + m == 0:
            return None
        return h / (h + m)

    def _log_metrics(self, metrics):
        """Record metrics to logger + experiment tracker (SwanLab / WandB)."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            # tensor → float 只在日志步做 (这一步才允许触发设备同步)
            for k, v in metrics.items():
                if torch.is_tensor(v):
                    metrics[k] = v.item()
            self._maybe_audit_sample(metrics)
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            _tok = self._tok_cache_stats()
            if _tok is not None:
                metrics["tok_cache/hit_rate"] = _tok
            self.tracker.log(metrics, step=self.completed_steps)
            _metrics_line = " ".join(
                f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            logger.info(f"Step {self.completed_steps} | {_metrics_line}")

    def _maybe_audit_sample(self, metrics):
        """grad norm 异常时把"嫌疑样本"落盘 (数据溯源审计, run 目录 sample_audit.jsonl)。

        判定: gn > STARVLA_AUDIT_FACTOR(默认5) × 近 50 个日志步中位数 且 gn > 2.0
        (绝对下限过滤训练早期的正常大梯度)。触发时从 self._audit_last_psl
        (最新一步逐样本 loss, train loop 存的引用) + batch 内 provenance 取证,
        按损失降序列出全部样本。事后统计哪个 dataset/episode/task 反复作案。
        """
        import statistics
        from collections import deque
        from pathlib import Path

        gn = metrics.get("grad_norm")
        if gn is None:
            return
        gn = float(gn)
        if not hasattr(self, "_audit_recent"):
            self._audit_recent = deque(maxlen=50)
        self._audit_recent.append(gn)
        if len(self._audit_recent) < 20:
            return
        med = statistics.median(self._audit_recent)
        factor = float(os.environ.get("STARVLA_AUDIT_FACTOR", "5"))
        if gn <= factor * med or gn < 2.0:
            return

        psl = getattr(self, "_audit_last_psl", None)
        batch = getattr(self, "_audit_last_batch", None) or []
        samples = []
        if psl is not None:
            vals = psl.detach().float().cpu().tolist()
            provs = [ex.get("provenance", {}) if isinstance(ex, dict) else {} for ex in batch]
            order = sorted(range(len(vals)), key=lambda i: -vals[i])
            samples = [
                {"loss": vals[i], **(provs[i] if i < len(provs) else {})}
                for i in order[: len(vals)]
            ]
        rec = {
            "step": int(self.completed_steps),
            "grad_norm": gn,
            "recent_median": med,
            "factor": factor,
            "batch_size": len(batch),
            "samples": samples,
        }
        audit_path = Path(self.config.output_dir) / "sample_audit.jsonl"
        try:
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.warning(
                f"🚨 [audit] grad_norm={gn:.2f} > {factor}×median({med:.2f}); "
                f"{len(samples)} 个嫌疑样本已写入 {audit_path}"
            )
        except Exception as e:  # 审计自身绝不打断训练
            logger.warning(f"[audit] 落盘失败 (不影响训练): {e}")

    def _create_data_iterators(self):
        """把 dataloader 转成 iterator,供 _get_next_batch 持续拉取。"""
        # resume 数据流对齐: build_dataloader 已按 checkpoint 步数把 dataset 推进到
        # 对应 epoch (set_epoch), 这里把 trainer 侧的 epoch 计数同步成同值 ——
        # 否则首个 StopIteration 会把 epoch 从 0 重数, 与连续训练错位一个 epoch。
        ds = getattr(self.vla_train_dataloader, "dataset", None)
        self.vla_epoch_count = int(getattr(ds, "epoch", 0) or 0)
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vla_eval_iter = iter(self.vla_eval_dataloader)
        self.vla_eval_epoch_count = 0

    def _get_next_batch(self):
        """拉下一个 batch,dataset 跑完一轮自动 reset。

        - 用 StopIteration 检测一轮结束。
        - `_reset_dataloader` 会重建 iterator 并维护 epoch 计数,触发 dataset
          里基于 epoch 的 resample / shuffle 行为。
        """
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def _get_next_eval_batch(self):
        """从在线评测 dataloader (eval 划分) 拉一个 batch, 耗尽自动 reset。

        与 ``_get_next_batch`` 同构, 但走 ``self.vla_eval_dataloader``, 不消耗
        训练 batch —— 在线 sanity eval 跑在模型从未训练过的评测轨迹上。
        """
        try:
            return next(self.vla_eval_iter)
        except StopIteration:
            self.vla_eval_iter, self.vla_eval_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_eval_dataloader, self.vla_eval_epoch_count
            )
            return next(self.vla_eval_iter)

    def train(self):
        """训练主循环 (核心)。

        每个 iteration 做的事:
          1. 计时拉一个 batch (data_time)
          2. 跑一次 forward + backward + step (model_time,详见 _train_step)
          3. 只在 sync_gradients=True 时推进 completed_steps
             —— 因为开启 gradient accumulation 后,N 个 micro-step 才是 1 个
                真实 step,避免 step 数被高估 N 倍。
          4. eval_interval 到了就跑在线 eval (mse_score)
          5. logging_frequency 到了就打日志
          6. save_interval 到了就存 checkpoint
          7. 达到 max_train_steps 退出

        data_time 与 model_time 分别计时,可以直接看出训练瓶颈是在 IO 还是在
        计算,调 num_workers / prefetch_factor 时很关键。
        """
        self._log_training_config()
        self._create_data_iterators()
        # tracker 在 prepare_training 里已建好;注入给分析器用于推 profile/* 标量。
        self.perf.tracker = self.tracker
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # STARVLA_FIXED_BATCH=1: 固定复用首个 batch (隔离实验) —— 数据路径 (dataloader/
        # 视频解码/网络存储抖动) 完全出局, 只测模型路径的稳态吞吐与稳定性。
        # 用于区分 "慢在数据" vs "慢在模型/宿主"。仅诊断用, 不要用于正式训练。
        fixed_batch = os.environ.get("STARVLA_FIXED_BATCH", "0") in ("1", "true", "yes")
        # STARVLA_GC_FREEZE (默认开): warmup 后 gc.collect()+gc.freeze() —— 把常驻
        # 对象移入永久代, 之后 gen2 回收不再遍历它们。evt 归因 + 对照探针实测:
        # bs32 1130→864ms (-24%) 且 bwd 分段方差消失; bs48 组合 +59% 吞吐。
        # gc.freeze 不禁用 GC (新垃圾仍正常回收), 无泄漏风险, 故转正默认开;
        # 排查问题时 STARVLA_GC_FREEZE=0 关闭。
        # 正式开关在配置: trainer.gc_freeze (默认 true, 生产 yaml 显式声明);
        # STARVLA_GC_FREEZE=0 仅调试期强制关闭。
        _env_gcf = os.environ.get("STARVLA_GC_FREEZE", "")
        gc_freeze = (
            bool(getattr(self.config.trainer, "gc_freeze", True))
            if _env_gcf not in ("0", "false", "no")
            else False
        )
        gc_freeze_at = int(os.environ.get("STARVLA_GC_FREEZE_AT", "15"))
        gc_frozen = False
        fixed_vla = None
        while self.completed_steps < self.config.trainer.max_train_steps:
            self.perf.on_step_begin(self.completed_steps)
            t_start_data = time.perf_counter()
            if fixed_batch:
                if fixed_vla is None:
                    fixed_vla = self._get_next_batch()
                batch_vla = fixed_vla
            else:
                self.evt.on_data_start(self.completed_steps)
                batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            self.evt.on_model_start(self.completed_steps)
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()
            self.evt.on_model_end(self.completed_steps)

            # grad norm 异常审计的素材: 最新一步的逐样本 loss + batch 溯源
            # (provenance)。只存引用, 零同步零拷贝; 异常判定发生在 _log_metrics
            # 的日志步 (那里本来就有 float 转换)。
            self._audit_last_psl = step_metrics.pop("_per_sample_loss", None)
            self._audit_last_batch = batch_vla

            # DeepSpeed 后端下 accelerator.sync_gradients 不再由 accumulate() 维护
            # (DS engine 自己管梯度累积边界), _train_step 用 _optimizer_step 标志回报
            # "本次 micro-step 是否触发了真实优化器更新"; 非 DS 后端回退到 sync_gradients。
            optimizer_stepped = step_metrics.pop("_optimizer_step", self.accelerator.sync_gradients)
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if (
                self.config.trainer.eval_interval > 0
                and self.completed_steps % self.config.trainer.eval_interval == 0
            ):
                # eval_interval: 0 = 禁用在线 eval。predict_action 有副作用 (切 eager
                # attention + empty_cache), 怀疑其引入周期性扰动时先关掉隔离。
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)
            self.evt.maybe_flush(self.completed_steps)

            # 分析器:窗口内采集 data/model 分段;窗口结束自动出诊断报告。
            self.perf.on_step_end(
                self.completed_steps,
                data_time=t_end_data - t_start_data,
                model_time=t_end_model - t_start_model,
            )

            if gc_freeze and not gc_frozen and self.completed_steps >= gc_freeze_at:
                import gc

                gc.collect()
                gc.freeze()
                gc_frozen = True
                logger.info(f">> gc.freeze() at step {self.completed_steps} (STARVLA_GC_FREEZE)")

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """在线 action eval:从 eval 划分取 1 个 batch,跑 predict_action 算 MSE。

        单 batch sanity check,不替代离线开环评测 (eval_starvla.py)。MSE 在
        normalized 空间,跟训练 loss 同空间,可直接对比趋势。

        三个要点:
        - RNG 隔离:predict_action 内部的 flow-matching 噪声 (sample_action_noise,
          PI0.py:868) 会消费 GPU RNG。不隔离会扰动后续训练的 dropout / 数据增强随机流。
          这里在进函数前快照 CPU+CUDA state,无论是否异常都在 finally 里恢复。
        - NaN-safe:bf16 + eager attention 下训练可能发散 (见
          docs/pi05_training_nan_debug.md),权重变 NaN 后 score 也是 NaN。这里检测到
          NaN 时**不写 eval/mse key**(而不是写 nan),因为 _SwanLabTracker.log 的
          try/except 遇 nan 会 disable 整个 tracker (logger.py:96)。曲线缺点本身就是
          发散预警 —— 比 grad_norm 可信,后者被 DeepSpeed 剪枝掩盖成 1.0。
        - predict_action 不再传 use_ddim / num_ddim_steps (PI0 遗留参数,PI05 的
          predict_action 只认 num_steps, 见 PI0.py:828),用模型自身的
          num_inference_steps (UMI PI05 = 10)。
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        # 快照 RNG + attention config: eval 后全部恢复, 不污染后续训练。
        # predict_action 的 sample_actions (PI0.py:856) 会把 vlm._attn_implementation
        # 改成 "eager" 且不恢复 —— 保存原值, finally 里还原。
        # 注: ``vlm`` 是 PI0/PI05 框架特有属性; QwenPI_v3 / QwenOFT 等框架没有
        # (且其 predict_action 不改 attn impl), 快照/恢复按属性存在性条件执行。
        cpu_rng = torch.get_rng_state()
        cuda_rngs = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        vlm_cfg = getattr(getattr(unwrapped, "vlm", None), "language_model", None)
        vlm_cfg = getattr(vlm_cfg, "config", None)
        orig_attn = getattr(vlm_cfg, "_attn_implementation", None) if vlm_cfg is not None else None

        try:
            examples = self._get_next_eval_batch()
            actions = [example["action"] for example in examples]
            # predict_action 必须和训练 forward 走同样的精度上下文 (autocast bf16)。
            # 训练环境里模型权重 dtype 状态 (DeepSpeed bf16 包装 + to_precision) 和纯推理
            # 不同, 不加 autocast 会在 gemma linear 上撞 BFloat16 != float 的 dtype mismatch
            # (predict_action 内 sample_actions → denoise_step → gemma expert forward)。
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = unwrapped.predict_action(examples=examples)

            if self.accelerator.is_main_process:
                normalized_actions = output_dict["normalized_actions"]
                actions = np.array(actions)
                # predict_action 输出 padding 到 framework.action_dim (如 32), 只取前
                # action_dim_actual 维 (GT action 的最后一维) 再与 GT 比较 —— 与
                # eval_starvla.py 的 [..., :action_dim_actual] 切片一致, 否则 (B,H,32)
                # vs (B,H,10) 广播报错。
                normalized_actions = normalized_actions[..., : actions.shape[-1]]
                num_pots = np.prod(actions.shape)
                score = TrainerUtils.euclidean_distance(normalized_actions, actions)
                mse = score / num_pots
                # NaN 时不记 key (曲线断点 = 发散预警), 避免 nan 进 tracker 把它 disable。
                if not np.isnan(mse):
                    step_metrics["eval/mse"] = float(mse)

            del examples, output_dict
        finally:
            # 全部恢复: attention config + RNG, 并强制释放 predict_action 在 CUDA cache
            # 里的残留 (forward_prefix use_cache=True 的 KV cache 等), 确保下个 train step
            # 干净分配 —— 否则残留叠加训练常驻会在后续 backward 顶穿显存。
            if vlm_cfg is not None and orig_attn is not None:
                vlm_cfg._attn_implementation = orig_attn
            torch.set_rng_state(cpu_rng)
            if cuda_rngs is not None:
                torch.cuda.set_rng_state_all(cuda_rngs)
            torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """训练正式开始前打印关键训练参数,方便排查配置错误。"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _weights_finite(self) -> bool:
        """更新后权重守卫: 一旦 master→bit16 同步把 NaN 写进权重, 立即 fail-fast。

        resume 路径若踩中 DS 恢复 bug (见 _restore_full_state 注释) 或任何
        未知的更新异常, 权重在毒化后继续训练只会烧卡空转 (实测 nan loss
        无限循环)。这里在 optimizer step 后扫 bit16 flat buffer (~8GB 读,
        ~10ms 量级), 发现非有限立即抛错 —— 训练者拿到的错误信息直接指向
        恢复问题, 而不是几小时后才发现 loss=nan。env STARVLA_WEIGHT_GUARD=0 可关。
        """
        if os.environ.get("STARVLA_WEIGHT_GUARD", "1").lower() in ("0", "false", "no", "off"):
            return True
        try:
            with torch.no_grad():
                opt = getattr(self.model, "optimizer", None)
                if opt is not None and hasattr(opt, "parallel_partitioned_bit16_groups"):
                    for buf in opt.parallel_partitioned_bit16_groups:
                        for t in buf:
                            if not torch.isfinite(t).all():
                                return False
                    return True
                base = self.model.module if hasattr(self.model, "module") else self.model
                for p in base.parameters():
                    if not torch.isfinite(p).all():
                        return False
                return True
        except Exception:
            return True

    def _first_nonfinite_weight_buffer(self) -> dict | None:
        """Locate the first bad weight buffer for opt-in numerical diagnostics.

        This deliberately inspects the same bit16 partitions as
        ``_weights_finite``. It is only called when STARVLA_NUMERICS_DIAG=1,
        because counting bad elements adds a GPU synchronization.
        """
        try:
            with torch.no_grad():
                opt = getattr(self.model, "optimizer", None)
                if opt is not None and hasattr(opt, "parallel_partitioned_bit16_groups"):
                    for group_idx, group in enumerate(opt.parallel_partitioned_bit16_groups):
                        for buffer_idx, buf in enumerate(group):
                            finite = torch.isfinite(buf)
                            if not bool(finite.all()):
                                return {
                                    "group": group_idx,
                                    "buffer": buffer_idx,
                                    "nonfinite": int((~finite).sum().item()),
                                    "nan": int(torch.isnan(buf).sum().item()),
                                    "inf": int(torch.isinf(buf).sum().item()),
                                    "numel": buf.numel(),
                                    "dtype": str(buf.dtype),
                                }
                    return None
                base = self.model.module if hasattr(self.model, "module") else self.model
                for name, param in base.named_parameters():
                    finite = torch.isfinite(param)
                    if not bool(finite.all()):
                        return {
                            "parameter": name,
                            "nonfinite": int((~finite).sum().item()),
                            "nan": int(torch.isnan(param).sum().item()),
                            "inf": int(torch.isinf(param).sum().item()),
                            "numel": param.numel(),
                            "dtype": str(param.dtype),
                        }
                return None
        except Exception as exc:
            return {"inspection_error": repr(exc)}

    @staticmethod
    def _numeric_diag_scalar(value):
        if value is None:
            return None
        try:
            if torch.is_tensor(value):
                return float(value.detach().float().item())
            return float(value)
        except Exception as exc:
            return f"unavailable ({exc!r})"

    def _local_gradient_diagnostics(self) -> dict:
        """Inspect this rank's gradients without DeepSpeed collectives (opt-in)."""
        try:
            opt = getattr(self.model, "optimizer", None)
            source = "parameters"
            gradients = []
            if opt is not None:
                averaged = getattr(opt, "averaged_gradients", None)
                if averaged:
                    source = "averaged_gradients"
                    groups = averaged.values() if isinstance(averaged, dict) else averaged
                    for group_idx, group in enumerate(groups):
                        if group is None:
                            continue
                        tensors = group.values() if isinstance(group, dict) else (
                            [group] if torch.is_tensor(group) else group
                        )
                        gradients.extend((group_idx, idx, tensor) for idx, tensor in enumerate(tensors)
                                         if torch.is_tensor(tensor))
                if not gradients:
                    flat = getattr(opt, "grads_partition_flat_buffer", None)
                    if flat is not None:
                        source = "grads_partition_flat_buffer"
                        tensors = [flat] if torch.is_tensor(flat) else flat
                        gradients = [(0, idx, tensor) for idx, tensor in enumerate(tensors)
                                     if torch.is_tensor(tensor)]
                if not gradients:
                    source = "fp32_master_grad"
                    gradients = [(idx, 0, part.grad) for idx, part in enumerate(
                        getattr(opt, "single_partition_of_fp32_groups", []))
                        if part.grad is not None]
            if not gradients:
                base = self.model.module if hasattr(self.model, "module") else self.model
                source = "parameters"
                gradients = [(idx, 0, param.grad) for idx, param in enumerate(base.parameters())
                             if param.grad is not None]

            sum_sq = 0.0
            count = 0
            first_bad = None
            with torch.no_grad():
                for group_idx, tensor_idx, grad in gradients:
                    finite = torch.isfinite(grad)
                    if first_bad is None and not bool(finite.all()):
                        first_bad = {
                            "group": group_idx, "tensor": tensor_idx,
                            "nonfinite": int((~finite).sum().item()),
                        }
                    norm = float(torch.linalg.vector_norm(grad.detach().float()).item())
                    sum_sq += norm * norm
                    count += grad.numel()
            return {
                "source": source,
                "tensors": len(gradients),
                "numel": count,
                "local_l2": math.sqrt(sum_sq),
                "first_nonfinite": first_bad,
            }
        except Exception as exc:
            return {"inspection_error": repr(exc)}

    def _grads_finite(self) -> bool:
        """更新前守卫: 检查梯度是否全部有限, 拦截 clip 的 inf×0=NaN 毒化。

        机制 (0803 文档 + 本次 L20X 实证): bf16 反向在某些硬 batch 上溢出到
        inf; DeepSpeed 的 clip 计算 clip_coef = max_norm/total_norm, total_norm
        =inf → coef=0 → 梯度 ×0 时 inf×0=**NaN** 被无声写进 fp32 master
        (实测 patch_embedding.weight, 一个参数即毒化整网)。clip 挡不住,
        必须在 update 前显式判断并跳过本步。

        ZeRO-2 的梯度真实存放位置随配置组合而变 (contiguous_gradients →
        grads_partition_flat_buffer; 否则 averaged_gradients / 参数 .grad /
        master.grad), 全部扫一遍。注意 bit16 flat buffer 本体是**权重**,
        检查它没有预警价值 (坏 = 已经晚了)。
        检测失败 (属性/时序变化) 时返回 True —— 守卫自身不能成为故障点。
        """
        try:
            with torch.no_grad():
                opt = getattr(self.model, "optimizer", None)
                if opt is not None and hasattr(opt, "single_partition_of_fp32_groups"):
                    buf = getattr(opt, "grads_partition_flat_buffer", None)
                    if buf is not None:
                        if torch.is_tensor(buf):
                            if not torch.isfinite(buf).all():
                                return False
                        else:
                            for t in buf or []:
                                if t is not None and not torch.isfinite(t).all():
                                    return False
                    for group_grads in getattr(opt, "averaged_gradients", {}).values():
                        for g in (group_grads or {}).values() if isinstance(group_grads, dict) else (group_grads or []):
                            if g is not None and not torch.isfinite(g).all():
                                return False
                    for part in opt.single_partition_of_fp32_groups:
                        if part.grad is not None and not torch.isfinite(part.grad).all():
                            return False
                    return True
                base = self.model.module if hasattr(self.model, "module") else self.model
                for p in base.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        return False
                return True
        except Exception as exc:
            logger.warning("Gradient finite check could not inspect DeepSpeed buffers: %r", exc)
            return True

    def _grad_watch(self, grad_norm, action_loss, batch_vla):
        """梯度爆炸看门狗 · 前置段 (零通信, backward 后 / optimizer 前)。

        只在原地读 grad_norm 并记录"本地是否触发"到 _grad_watch_pending ——
        **不做任何集合通信**: DeepSpeed overlap_comm 的异步梯度规约在
        backward 返回后仍可能挂起, 此处插入外来 NCCL allreduce 会与之竞态
        毒化梯度 (L20X 单卡 step-3 NaN 实证)。对齐/紧急快照/现场 dump 全部
        延后到 _grad_watch_finalize (engine.step() 返回后, DS 通信已 settle)。

        env 开关 (默认开):
          STARVLA_GRAD_WATCH=0                     # 关闭
          STARVLA_GRAD_WATCH_THRESH=100            # 触发阈值 (默认 100)
          STARVLA_GRAD_WATCH_MAXDUMP=20            # dump 次数上限
        """
        if os.environ.get("STARVLA_GRAD_WATCH", "1").lower() in ("0", "false", "no", "off"):
            self._grad_watch_pending = None
            return
        if grad_norm is None:
            return
        gn = float(grad_norm)
        thresh = float(os.environ.get("STARVLA_GRAD_WATCH_THRESH", "100"))
        self._grad_watch_pending = (
            (gn, thresh, float(action_loss.item()), batch_vla)
            if (not math.isfinite(gn)) or gn > thresh
            else None
        )

    def _grad_watch_finalize(self, optimizer_stepped: bool):
        """梯度爆炸看门狗 · 后置段 (optimizer step 之后, 通信安全点)。

        跨 rank allreduce 对齐触发 (紧急保存是集合操作, 各 rank 必须同时进),
        首次触发时保存"本步更新后"的全量训练状态 —— 爆炸刚起步的快照, 既是
        取证基线也是 resume 点。client_state 记 completed_steps+stepped,
        与 _save_checkpoint 的"已完成 N 步"语义对齐。之后 dump 案发现场
        (batch / per-parameter 梯度排行 / meta) 到 {output_dir}/grad_watch/。

        注意:
        - DeepSpeed ZeRO 下 p.grad 已被搬进分区 buffer, 排行可能为空
          (batch/meta 仍有效); 完整排行在离线单卡复放里做。
        - 只抓现场不跳步 —— 雪崩完整演化; 取证见
          scripts/replay_grad_explosion.py。
        """
        if os.environ.get("STARVLA_GRAD_WATCH", "1").lower() in ("0", "false", "no", "off"):
            return
        pending = getattr(self, "_grad_watch_pending", None)
        self._grad_watch_pending = None
        local = pending is not None

        if dist.is_initialized():
            flag = torch.tensor([int(local)], device="cuda" if torch.cuda.is_available() else "cpu")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            any_triggered = bool(flag.item())
        else:
            any_triggered = local

        if any_triggered and not getattr(self, "_grad_emergency_saved", False):
            self._grad_emergency_saved = True
            try:
                completed = self.completed_steps + int(bool(optimizer_stepped))
                self._save_full_state(emergency=True, steps_override=completed)
            except Exception as exc:
                logger.warning(f"⚠️ 紧急全量快照失败 (训练继续): {type(exc).__name__}: {exc}")

        if not local:
            return
        gn, thresh, loss_val, batch_vla = pending
        count = getattr(self, "_grad_watch_count", 0)
        max_dump = int(os.environ.get("STARVLA_GRAD_WATCH_MAXDUMP", "20"))
        if count >= max_dump:
            return
        self._grad_watch_count = count + 1

        rank = dist.get_rank() if dist.is_initialized() else 0
        dump_dir = Path(self.config.output_dir) / "grad_watch" / f"step_{self.completed_steps}_rank{rank}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        def _to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu()
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_cpu(v) for v in obj)
            return obj

        torch.save(_to_cpu(batch_vla), dump_dir / "batch.pt")

        lines = []
        base = self.model.module if hasattr(self.model, "module") else self.model
        for name, p in base.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            lines.append((name, float("nan") if not torch.isfinite(g).all() else float(g.norm())))
        lines.sort(key=lambda kv: (kv[1] == kv[1], -(kv[1] if kv[1] == kv[1] else 0.0)))
        with open(dump_dir / "grad_ranking.txt", "w") as f:
            for name, n in lines[:500]:
                f.write(f"{n:.6e}  {name}\n")

        meta = {
            "step": self.completed_steps,
            "rank": rank,
            "grad_norm": gn,
            "loss": loss_val,
            "thresh": thresh,
            "dump_index": count,
            "grad_tensors_readable": len(lines),
        }
        with open(dump_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.warning(
            f"🚨 GRAD WATCH: step={self.completed_steps} rank={rank} "
            f"grad_norm={gn:.3e} loss={loss_val:.4f} → {dump_dir}"
        )

    def _train_step(self, batch_vla, batch_vlm=None):
        """单个 micro-step 的前向 + 反向 + 优化 (核心)。

        DeepSpeed 与 Accelerate 的协作 (梯度累积)
        -----------------------------------------
        DeepSpeed (ZeRO stage 2/3) 与 Accelerate 的 ``accumulate()`` context
        manager 不兼容: ``accumulate()`` 在非同步 micro-step 会进入 ``no_sync()``,
        而 ZeRO-2 在 backward 阶段就分区梯度, ``no_sync`` 与其互斥, 抛
        "no_sync context manager is incompatible with gradient partitioning
        logic of ZeRO stage 2"。``gradient_accumulation_steps=1`` 时不炸
        (accumulate 永远走 nullcontext, 不进 no_sync), 一旦开梯度累积即崩。

        因此当 prepare 后的 model 是 DeepSpeed engine 时, 直接通过 engine 驱动
        梯度累积: ``engine.backward()`` 累积、``engine.step()`` 只在 engine 自己
        判定的累积边界上更新 (梯度清零 / allreduce / clip 都由 DS 内部按
        ds_config 的 gradient_accumulation_steps / gradient_clipping 完成)。
        本函数通过 ``_optimizer_step`` 标志回报"本次是否触发了真实优化器更新",
        供 train() 主循环推进 completed_steps。非 DeepSpeed 后端仍走
        ``accelerate.accumulate()``。

        混合精度
        --------
        ``torch.autocast("cuda", dtype=torch.bfloat16)`` 把 forward 中的 matmul
        等运算跑成 bf16,显存占用减半且比 fp16 不需要 grad scaler。部分对精度
        敏感的层 (norm、patch_embedding) 已在 framework.to_precision 里单独
        keep 在 fp32,这里 autocast 不会覆盖它们。

        LR scheduler 时机
        -----------------
        LR scheduler 必须只在真实优化器 step 完成时 (DeepSpeed 的累积边界 /
        Accelerate 的 sync_gradients=True) step 一次。如果每个 micro-step 都
        step,warmup 会被加速 N 倍 (N=grad_accum_steps),cosine decay 也会过早
        触底,与 max_train_steps 错位。
        """
        is_ds_engine = hasattr(self.model, "is_gradient_accumulation_boundary") and hasattr(
            self.model, "backward"
        )

        if is_ds_engine:
            # DeepSpeed engine 自己管梯度累积: 不要用 accelerate.accumulate/no_sync
            # (与 ZeRO-2 梯度分区冲突), 也不要手动 zero_grad (engine.step 内部在
            # 边界上清零)。backward 在非边界 micro-step 累积, step 只在边界更新。
            with self.perf.region("forward"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.evt.on_forward_start(self.completed_steps)
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    per_sample_loss = output_dict.get("per_sample_loss")
                self.evt.on_forward_end(self.completed_steps)

            with self.perf.region("backward"):
                self.model.backward(action_loss)
                self.evt.on_backward_end(self.completed_steps)

            # 梯度裁剪由 DS engine 在 step() 内部按 ds_config.gradient_clipping
            # 完成; 这里只取 grad_norm 用于记录。get_global_grad_norm 在首步或
            # 不支持该 API 时为 None, 后续记录处会跳过。
            optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
            if hasattr(self.model, "get_global_grad_norm"):
                grad_norm = self.model.get_global_grad_norm()
            else:
                params = self.model.module.parameters() if hasattr(self.model, "module") else self.model.parameters()
                total_norm = 0.0
                for p in params:
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5

            self._grad_watch(grad_norm, action_loss, batch_vla)

            grads_finite = self._grads_finite()
            numerics_diag = os.environ.get("STARVLA_NUMERICS_DIAG", "0").lower() in (
                "1", "true", "yes", "on"
            )
            pre_weight_diag = None
            grad_diag = None
            if numerics_diag and optimizer_stepped:
                pre_weight_diag = self._first_nonfinite_weight_buffer()
                grad_diag = self._local_gradient_diagnostics()
                logger.warning(
                    "Numerics before step %s: action_loss=%s grad_norm=%s "
                    "grads_finite=%s local_grad=%s first_nonfinite_weight=%s",
                    self.completed_steps,
                    self._numeric_diag_scalar(action_loss),
                    self._numeric_diag_scalar(grad_norm),
                    grads_finite,
                    grad_diag,
                    pre_weight_diag,
                )
            if grads_finite:
                with self.perf.region("optimizer"):
                    self.model.step()
                if optimizer_stepped:
                    self.lr_scheduler.step()
                if not self._weights_finite():
                    detail = ""
                    if numerics_diag:
                        detail = (
                            f"; action_loss={self._numeric_diag_scalar(action_loss)}"
                            f" grad_norm={self._numeric_diag_scalar(grad_norm)}"
                            f" local_grad={grad_diag}"
                            f" pre_weight={pre_weight_diag}"
                            f" post_weight={self._first_nonfinite_weight_buffer()}"
                        )
                    raise RuntimeError(
                        f"🛑 step {self.completed_steps}: optimizer 更新后权重出现非有限值 "
                        f"(resume 恢复异常或数值溢出){detail}。last known good: 最近一次 "
                        "steps_N_ds 全量状态; 建议用更早的全量状态重试或上报。"
                    )
            else:
                try:
                    self.model.optimizer.zero_grad()
                except Exception:
                    pass
                logger.warning(
                    f"🛡 non-finite grads @ step {self.completed_steps}: 跳过本次更新 "
                    "(拦截 clip 的 inf×0=NaN; 本步 batch 已被 grad_watch 记录)"
                )
            self._grad_watch_finalize(optimizer_stepped and grads_finite)

            metrics = {"action_dit_loss": action_loss, "_per_sample_loss": per_sample_loss, "_optimizer_step": optimizer_stepped and grads_finite}
            if not grads_finite:
                metrics["grad_skipped"] = 1.0
            if grad_norm is not None:
                # DS 的 grad_norm 可能是 GPU tensor — float() 会触发 D2H 同步
                # (trace 实测 ~12.5ms/步)。原样传递, 由 _log_metrics 在日志步统一转换。
                metrics["grad_norm"] = grad_norm
            return metrics

        # 非 DeepSpeed 后端 (DDP/FSDP 等): 用 Accelerate 的 accumulate() 管理梯度累积。
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with self.perf.region("forward"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.evt.on_forward_start(self.completed_steps)
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    per_sample_loss = output_dict.get("per_sample_loss")
                    total_loss = action_loss
                self.evt.on_forward_end(self.completed_steps)

            with self.perf.region("backward"):
                self.accelerator.backward(total_loss)
                self.evt.on_backward_end(self.completed_steps)

            # 梯度裁剪:flow-matching loss 在训练早期容易出现大梯度 (state/action
            # 还没归一化稳定时),1.0 是 VLA 训练的常用保守值。
            if self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
            else:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** 0.5

            self._grad_watch(grad_norm, action_loss, batch_vla)

            # 同 DS 分支: 非有限梯度直接跳步 (clip_grad_norm_ 对 nan 同样无效)。
            grads_finite = self._grads_finite()
            if grads_finite:
                with self.perf.region("optimizer"):
                    self.optimizer.step()
            else:
                self.optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    f"🛡 non-finite grads @ step {self.completed_steps}: 跳过本次更新"
                )
            self._grad_watch_finalize(bool(self.accelerator.sync_gradients) and grads_finite)
            if self.accelerator.sync_gradients and grads_finite:
                self.lr_scheduler.step()

        metrics = {"action_dit_loss": action_loss, "_per_sample_loss": per_sample_loss, "_optimizer_step": self.accelerator.sync_gradients and grads_finite}
        if not grads_finite:
            metrics["grad_skipped"] = 1.0
        # grad_norm 在 DeepSpeed 首步 (get_global_grad_norm 尚未就绪) 或不支持该
        # API 的后端上可能为 None;此时跳过该指标,避免向 tracker 传 None 触发
        # "Failed to extract scalar value from NoneType"。
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm)
        return metrics

    def _dump_tok_cache(self):
        """memo 累积的 tok cache 落盘 (rank0), 重启时经 tok_cache: auto 免重算。

        H1 类 state-in-text 数据集指令每样本唯一, 一个 epoch 的 memo 即覆盖
        全量样本; 生产长跑中断重启后从 parquet 直读, 跳过首 epoch 慢路径。
        """
        if not self.accelerator.is_main_process:
            return
        try:
            _iface = getattr(getattr(self.model, "module", self.model), "qwen_vl_interface", None)
            cache = getattr(_iface, "_tok_cache", None)
            if not cache or len(cache) <= getattr(_iface, "_tok_cache_initial_len", 0):
                return
            import pandas as _pd

            root = self.config.datasets.vla_data.data_root_dir
            out = Path(root) / "precompute" / "tok_cache_qwen35.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            model_id = str(
                getattr(getattr(_iface, "model", None), "name_or_path", "")
                or self.config.framework.qwenvl.base_vlm
            )
            _pd.DataFrame(
                {
                    "task": list(cache.keys()),
                    "input_ids": [v.tolist() for v in cache.values()],
                    "tokenizer": [model_id] * len(cache),
                }
            ).to_parquet(out)
            logger.info(f"📝 tok cache 已落盘: {out} ({len(cache)} 条)")
        except Exception as e:  # 落盘失败不影响训练收尾
            logger.warning(f"tok cache 落盘失败 (忽略): {e}")

    def _finalize_training(self):
        """Training end: save final model + finish tracker + cleanup."""
        # 兜底:若分析窗口未自然结束 (如 max_steps 太小),尽力出一份报告。
        self.perf.finalize()
        self._dump_tok_cache()
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            self.tracker.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    """训练入口:把各组件拼装起来执行训练。

    装配顺序不能随意调换:
      rank_logging → wrap_config → warn → setup_directories → build_framework →
      prepare_data → setup_optimizer_and_scheduler → VLATrainer →
      prepare_training → train
    """
    # 最早期: 多卡下只有 rank0 说话 (logging filter + stdout 整流 + tqdm disable),
    # WARNING+/traceback 全 rank 保留。逃生门 STARVLA_LOG_ALL_RANKS=1。
    from starVLA.training.trainer_utils.rank_logging import setup_rank_logging

    setup_rank_logging()
    logger.info("VLA Training :: Warming Up")

    # STARVLA_COMPILED_AUTOGRAD=1 (实验): 编译 autograd 引擎本身。py-spy 归因:
    # 主线程 23% 时间在 _engine_run_backward 的逐节点 python 驱动上 (即使
    # layer 已逐层编译, 反向仍由 python autograd 引擎逐节点调度)。torchtitan
    # 同款方案。要求反向路径整体可编译, graph break 会回退。
    if os.environ.get("STARVLA_COMPILED_AUTOGRAD", "0") in ("1", "true", "yes"):
        import torch._dynamo as _dynamo

        _dynamo.config.compiled_autograd = True
        logger.info("🔧 compiled_autograd 已启用 (STARVLA_COMPILED_AUTOGRAD=1)")

    # AccessTrackedConfig 是 OmegaConf 的包装,拦截所有字段访问并记录,训练结束
    # 后能产出"只含真正生效字段"的精简 yaml,便于复现和 diff。
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # 在 cfg 合并完成、尚未进入业务逻辑前先暴露问题 (action_type 等)。
    warn_unused_yaml_fields(cfg)

    output_dir = setup_directories(cfg=cfg)
    # build_framework 会根据 cfg.framework.name 从 FRAMEWORK_REGISTRY 实例化
    # 对应模型类 (PI0/PI05/QwenPI/Gemma4PI/...),并加载预训练权重到子模块。
    vla = build_framework(cfg)
    vla_train_dataloader, vla_eval_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        vla_eval_dataloader=vla_eval_dataloader,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    # 加载 yaml,然后用 CLI 上的 --foo.bar=baz 形式 override yaml 同名字段。
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # 把旧版本 yaml 字段名升级到当前 schema (version_id="0.21"),幂等且不改类
    # 签名。例如旧字段 `lr` 会映射到 `trainer.learning_rate.base`。详见
    # bar/config_tighten.md。
    cfg = apply_config_compat(cfg)

    # 记下原始 yaml 路径,后面 setup_directories 会把它复制到 output_dir 留档。
    cfg.config_yaml = args.config_yaml

    # debug 模式:rank 0 启动 debugpy listener,等待 IDE 连上再继续。
    # 用法:在 yaml 里设 is_debug: true,然后 IDE 连 10092 端口。
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
