# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, 1.0 (the "License");
# you may not use this file except in compliance with the License.

"""静态复读 DataLoader (调试工具)
================================

目的
----
把「数据管线」从训练耗时里**完全剥离**, 纯测模型侧 (forward/backward/optimizer +
GPU 进程内的 CPU 预处理) 的性能与数值行为:

- 启动时从真实 dataloader 拉取 ``num_batches`` 个 batch 缓存在内存;
- 之后 ``__iter__`` 无限循环吐出同一批数据 → ``timing/data`` 恒为 ~0;
- 底层 worker 尽力关停 (persistent_workers 场景), 消除后台解码对主进程的 CPU 抢占。

典型用法 (配合内建 profiler)
----------------------------
.. code-block:: bash

    # 方式 A: yaml 开关 (datasets.vla_data.static_repeat: true)
    NUM_GPUS=1 BATCH=6 STARVLA_PROFILE=1 STARVLA_PROFILE_GAP=1 \
      bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh \
           examples/realRobots/zerith_pi05/train_files/debug_static_data.yaml

    # 方式 B: 环境变量开 (不用改 yaml, 任意现有配置上加)
    STARVLA_STATIC_DATA=1 STARVLA_STATIC_BATCHES=8 ...

解读
----
- ``timing/data ≈ 0`` 是本工具的**设计行为**, 不代表数据管线没问题;
  真实数据侧吞吐请看本工具预取阶段的逐 batch 耗时日志 (顺便量化了 C 类瓶颈);
- ``timing/model`` 此时的构成 = GPU 进程内 CPU 预处理 (A 类) + 纯 GPU 计算
  (B 类), 与正常训练的差值即数据管线 (C 类) + 多卡 straggler 的贡献;
- 同一批数据无限复读 → 训练 loss 会在几个 batch 间震荡, **只用于性能/数值
  调试, 不能用于收敛性评估**; 但对 NaN 复现非常有用 (确定性输入)。

实现约束
--------
- 必须在 ``accelerator.prepare`` **之后**替换 dataloader: DeepSpeed 的
  ``train_micro_batch_size_per_gpu: auto`` 要求 args 里存在真 DataLoader
  (accelerate 1.x ``_prepare_deepspeed``), 自定义类在 prepare 前传入会报错。
- 不继承 ``torch.utils.data.DataLoader``: 继承会被 accelerate 包成
  DataLoaderShard 并按真 DataLoader 语义迭代, 与复读行为冲突。
"""

import time


def _log(msg: str) -> None:
    """调试工具直接 print (accelerate logger 需先初始化状态, 独立使用会崩)。"""
    print(f"[static-repeat] {msg}", flush=True)


class StaticRepeatDataLoader:
    """拉取固定数量 batch 后无限复读的伪 DataLoader。

    对 trainer 暴露的最小接口: ``iter()`` / ``next()`` / ``len()``。
    iterator 永不 StopIteration (无限 epoch), 因此 trainer 的
    ``_reset_dataloader`` 逻辑不会被触发 —— 正是我们要的: 数据源完全冻结。
    """

    def __init__(self, source, num_batches: int = 8, tag: str = "train", log_every_fetch: bool = False):
        """
        :param source: 真实 (通常已 accelerate.prepare 过的) DataLoader
        :param num_batches: 预取并缓存的 batch 数
        :param tag: 日志标签 ("train" / "eval")
        :param log_every_fetch: 预取阶段逐 batch 打耗时 (量化冷启/热读差异)
        """
        self._source = source
        self._num_batches = max(1, int(num_batches))
        self._batches = None
        self._tag = tag
        self._log_every_fetch = log_every_fetch

    # ---- trainer 消费的接口 -------------------------------------------------
    def __len__(self) -> int:
        return len(self._batches) if self._batches is not None else self._num_batches

    def __iter__(self):
        # generator 体到首次 next() 才执行 → eval loader 只有真被用到才会预取。
        if self._batches is None:
            self._load()
        assert self._batches is not None
        while True:  # 无限 epoch: 永不 StopIteration, 数据完全冻结
            for batch in self._batches:
                yield batch

    # ---- 内部 ----------------------------------------------------------------
    def _load(self):
        """一次性拉满 num_batches 个 batch, 计时并尽力关停底层 worker。"""
        t_all = time.perf_counter()
        batches = []
        it = iter(self._source)
        per_batch_ms = []
        for i in range(self._num_batches):
            t0 = time.perf_counter()
            batches.append(next(it))
            dt_ms = (time.perf_counter() - t0) * 1e3
            per_batch_ms.append(dt_ms)
            if self._log_every_fetch:
                _log(f"{self._tag}: 预取 batch {i + 1}/{self._num_batches}: {dt_ms:.1f} ms")
        self._batches = batches
        self._shutdown_source_workers()
        warm = per_batch_ms[1:] or per_batch_ms
        _log(
            f"{self._tag}: 预取完成 {self._num_batches} batch, "
            f"首个(冷启) {per_batch_ms[0]:.1f} ms, 后续均值 {sum(warm) / len(warm):.1f} ms, "
            f"总 {1e3 * (time.perf_counter() - t_all):.1f} ms; 底层 worker 已尝试关停。"
        )

    def _shutdown_source_workers(self):
        """尽力关掉底层 DataLoader 的 persistent workers, 释放 CPU。

        失败静默: DataLoaderShard 同样有 _iterator._shutdown_workers, 但任何
        内部差异都不应让调试工具本身挂掉。
        """
        dl = self._source
        iterator = getattr(dl, "_iterator", None)
        if iterator is None:
            return
        try:
            iterator._shutdown_workers()
        except Exception:
            pass
        try:
            dl._iterator = None  # type: ignore[attr-defined]
        except Exception:
            pass
