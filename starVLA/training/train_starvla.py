# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA 训练器 (VLATrainer)
============================
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
import os
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
deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

logging.getLogger("DeepSpeed").setLevel(logging.WARNING)

accelerator.print(accelerator.state)

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
    只有 rank 0 创建,其他 rank 直接跳过,避免并发 mkdir 报错。
    """
    is_resume = getattr(getattr(cfg, "trainer", None), "is_resume", False)
    if is_resume:
        cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    else:
        ts = datetime.now().strftime("%m%d_%H%M%S")
        cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id, ts)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

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
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
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

        - `is_resume=True`:从上次中断处继续训练,在 output_dir/checkpoints/ 找
          最新的 steps_X 目录,恢复 model + optimizer + scheduler + step 数。
        - `pretrained_checkpoint` (无 is_resume):加载预训练权重作为初始化,
          但 step 数从 0 开始 (典型如用 OpenPI 权重继续训 LIBERO)。
        """
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
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
        """
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """加载完整训练状态 (model + optimizer + scheduler),用于断点续训。"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """保存当前训练状态的 checkpoint。

        - model:state_dict 保存为 .pt 或 .safetensors (由 save_format 决定)。
        - 配置:AccessTrackedConfig 此时会把"训练过程中被新访问到的字段"追加
          写入 config.yaml,保证后续可以从 yaml 还原训练。
        - optimizer/scheduler:由 DeepSpeed 自身管理 (在 checkpoint_dir 下),
          这里只显式保存 model。
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

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record metrics to logger + experiment tracker (SwanLab / WandB)."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            self.tracker.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """把 dataloader 转成 iterator,供 _get_next_batch 持续拉取。"""
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

        while self.completed_steps < self.config.trainer.max_train_steps:
            self.perf.on_step_begin(self.completed_steps)
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # accumulate context 在退出时会自动判断"是不是这一步的最后一个
            # micro-step",sync_gradients=True 表示当前是最后一个 micro-step,
            # 真实 step 完成了。
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # 分析器:窗口内采集 data/model 分段;窗口结束自动出诊断报告。
            self.perf.on_step_end(
                self.completed_steps,
                data_time=t_end_data - t_start_data,
                model_time=t_end_model - t_start_model,
            )

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """简单的在线 action eval:在当前 batch 上跑一次 predict_action,对比
        GT action 算 MSE。仅用于训练过程中的 sanity check,不替代
        LIBERO/real-robot 等正式 benchmark。

        注意:这一步会从**在线评测 dataloader** (eval 划分) 消耗一个 batch
        (self._get_next_eval_batch),不再吃训练 batch,所以训练步数不受影响;
        指标反映模型在未见过的评测轨迹上的泛化能力。
        """
        examples = self._get_next_eval_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

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
            step_metrics["mse_score"] = score / num_pots

        del examples
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

    def _train_step(self, batch_vla, batch_vlm=None):
        """单个 micro-step 的前向 + 反向 + 优化 (核心)。

        与 Accelerate 的协作
        --------------------
        `self.accelerator.accumulate(self.model)` 是关键 context manager:
          - 进入时:如果是 gradient accumulation 的第 1 个 micro-step,会同步
            所有 rank (等上一个真实 step 的 allreduce 结束)。
          - 内部:forward / backward 正常执行,但只有最后 1 个 micro-step 才
            真正触发跨 rank 的梯度 allreduce。
          - 退出时:根据 accumulation 计数判断 sync_gradients。

        混合精度
        --------
        `torch.autocast("cuda", dtype=torch.bfloat16)` 把 forward 中的 matmul
        等运算跑成 bf16,显存占用减半且比 fp16 不需要 grad scaler。部分对精度
        敏感的层 (norm、patch_embedding) 已在 framework.to_precision 里单独
        keep 在 fp32,这里 autocast 不会覆盖它们。

        LR scheduler 时机
        -----------------
        LR scheduler 必须只在 sync_gradients=True 时 step 一次。如果每个
        micro-step 都 step,warmup 会被加速 N 倍 (N=grad_accum_steps),cosine
        decay 也会过早触底,与 max_train_steps 错位。

        关于 zero_grad
        --------------
        每个 micro-step 都 zero_grad:配合 DeepSpeed ZeRO 是正确的,因为 DS
        内部会管理梯度切片的累积语义。如果改为 N 个 micro-step 才 zero 一次,
        反而会和 DS 的 grad accumulation 实现冲突。
        """
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with self.perf.region("forward"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    total_loss = action_loss

            with self.perf.region("backward"):
                self.accelerator.backward(total_loss)

            # 梯度裁剪:flow-matching loss 在训练早期容易出现大梯度 (state/action
            # 还没归一化稳定时),1.0 是 VLA 训练的常用保守值。
            if self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
                # DeepSpeed 下 accelerator.clip_grad_norm_ 恒返回 None (DeepSpeed
                # 在 backward/step 内部自行裁剪,accelerate 无法回传范数)。此时改
                # 用 DeepSpeed 引擎缓存的 get_global_grad_norm() 拿到真实范数用于
                # 记录;非 DeepSpeed 后端仍用 clip_grad_norm_ 的返回值。
                if grad_norm is None and hasattr(self.model, "get_global_grad_norm"):
                    grad_norm = self.model.get_global_grad_norm()
            else:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** 0.5

            with self.perf.region("optimizer"):
                self.optimizer.step()
            # 见上方 docstring 中"LR scheduler 时机":scheduler 必须只在真实
            # step 完成时 (sync_gradients=True) 才前进一步。
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        metrics = {"action_dit_loss": action_loss.item()}
        # grad_norm 在 DeepSpeed 首步 (get_global_grad_norm 尚未就绪) 或不支持该
        # API 的后端上可能为 None;此时跳过该指标,避免向 tracker 传 None 触发
        # "Failed to extract scalar value from NoneType"。
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm)
        return metrics

    def _finalize_training(self):
        """Training end: save final model + finish tracker + cleanup."""
        # 兜底:若分析窗口未自然结束 (如 max_steps 太小),尽力出一份报告。
        self.perf.finalize()
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
      wrap_config → warn → setup_directories → build_framework →
      prepare_data → setup_optimizer_and_scheduler → VLATrainer →
      prepare_training → train
    """
    logger.info("VLA Training :: Warming Up")

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
