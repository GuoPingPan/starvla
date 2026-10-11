# Copyright © 2026 starVLA
"""Rank 感知日志: 多卡训练只有 rank0 说话, 错误全员可见。

设计 (三层拦截 + 逃生门)
------------------------
1. logging Filter: 非 rank0 的 INFO/DEBUG 丢弃 (Using mixture/vision_preprocess/
   DiT params 等 8 份重复的来源); WARNING 及以上**全 rank 放行** —— 崩溃定位
   必须知道是哪个 rank 挂了 (H1 启动实录: 8 份 traceback 虽吵但有用), 且
   非 rank0 的 WARNING/ERROR 自动加 [rank N] 前缀。
2. stdout 重定向: 非 rank0 的 stdout → devnull。datasets.py 等处 60+ 个裸
   print 不逐个改造 (改不胜改), 整流最省; traceback 走 stderr 不受影响。
   dataloader worker (fork 自非主 rank) 一并安静。
3. tqdm disable: datasets.py 的 6 处进度条在非 rank0 关闭 (见 tqdm_disabled)。
4. 逃生门: STARVLA_LOG_ALL_RANKS=1 强制全开 (debug 用)。

用法: train_starvla.main() 第一行 ``setup_rank_logging()``; 进度条处用
``tqdm_disabled()``。单进程 (无 LOCAL_RANK) 行为完全不变。
"""

from __future__ import annotations

import logging
import os
import sys


def _env_rank() -> int:
    for k in ("LOCAL_RANK", "RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(k)
        if v not in (None, ""):
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def is_rank_zero() -> bool:
    """rank 判定 (env 优先, 兼容 accelerate/torchrun/singularity 各Launcher)。"""
    if os.environ.get("STARVLA_LOG_ALL_RANKS", "0") in ("1", "true", "yes"):
        return True
    return _env_rank() == 0


def tqdm_disabled() -> bool:
    """进度条在非 rank0 关闭 (tqdm 的 disable 参数)。"""
    return not is_rank_zero()


class _RankFilter(logging.Filter):
    """非 rank0: INFO/DEBUG 丢弃; WARNING+ 放行并打 [rank N] 标。"""

    def __init__(self, rank: int, main: bool):
        super().__init__()
        self.rank, self.main = rank, main

    def filter(self, record: logging.LogRecord) -> bool:
        if self.main or record.levelno >= logging.WARNING:
            if not self.main:
                record.msg = f"[rank {self.rank}] {record.msg}"
            return True
        return False


def setup_rank_logging() -> bool:
    """在训练入口最早期调用 (任何数据集/模型 import 之后、训练开始之前)。返回 is_rank_zero。

    实现要点 (踩过的坑):
    - 仓库在 import 期 (trainer_utils/__init__ 的 overwatch) 就对 stdout 建好了
      StreamHandler —— 该 handler 绑定的是**旧的 sys.stdout 对象**, 事后重定向
      sys.stdout 管不到它。因此:
      a) filter 必须挂在 **handler** 上 (logger 级 filter 对子 logger 传播来的
         记录不生效, 这是 python logging 的经典坑);
      b) 非 rank0 把 root handlers 的流改指 stderr (setStream), WARNING+ 由此
         可见; INFO/DEBUG 被 filter 丢弃。
    - 裸 print 仍靠 sys.stdout 重定向消灭 (它们不走 handler)。
    """
    rank = _env_rank()
    main = is_rank_zero()
    filt = _RankFilter(rank, main)

    root = logging.getLogger()
    root.addFilter(filt)  # 直接在 root 上发的记录

    if main:
        for h in list(root.handlers):
            h.addFilter(filt)  # 传播来的记录 (logger 级 filter 管不到, 挂 handler 上)
    else:
        # 仓库 import 期的 RichHandler 内部 Rich Console 绑死旧 stdout (无
        # setStream 可用), 干脆整体替换为 stderr 的朴素 StreamHandler ——
        # WARNING+ (已通过 filter) 由 stderr 存活, INFO/DEBUG 被 filter 丢弃。
        plain = logging.StreamHandler(sys.stderr)
        plain.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        plain.addFilter(filt)
        plain.setLevel(logging.WARNING)
        root.handlers = [plain]

    if not main:
        try:
            sys.stdout = open(os.devnull, "w")
        except Exception:
            pass
        for noisy in ("albumentations", "transformers", "datasets", "filelock"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
    else:
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        logging.getLogger("albumentations").setLevel(logging.ERROR)

    return main
