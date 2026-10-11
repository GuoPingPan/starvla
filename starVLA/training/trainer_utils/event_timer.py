# Copyright 2025 starVLA community. All rights reserved.
#
# CUDA event 区域计时器: 用 stream 上的事件对把每步的 GPU 时间线切成段,
# 不做任何 synchronize (每 flush_every 步才同步一次取数), 因此对 overlap
# 流水线和 launch 提交节奏零扰动 —— 这是它与 perf_analyzer (窗口内带 sync
# 的诊断剖面) 的本质区别, 用于稳态归因:
#
#   gpu_total   = ev(model_start) -> ev(model_end)   GPU 时间线上一步的时长
#   fwd/bwd/opt = 内部子段 (同一 stream 上的事件对差值)
#   cpu_total   = 同段的墙钟; gpu_total < cpu_total 说明 GPU 先于 CPU 跑完
#   submit_gap  = ev(model_end)[i] -> ev(model_start)[i+1]: GPU 等下一次提交
#                 的空窗, 大 => CPU/launch bound; 小 => GPU bound
#   data_wait   = CPU 侧取 batch 的等待 (loader 供给)
#
# 开关: STARVLA_EVT=1 (默认关)。仅 rank0 记录, 避免多 rank 间计时噪音。
# 数值精度: elapsed_time 单位 ms, 事件对在 command stream 上严格有序。

import os
import time
from collections import deque

import torch

from starVLA.training.trainer_utils.logger import logger


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


_evt_singleton = None
_NOOP = None


def evt_global():
    """进程内共享实例 (QwenPI_v3 等模型侧子段打点用; 未启用时返回 no-op 安全对象)。"""
    if _evt_singleton is not None:
        return _evt_singleton
    global _NOOP
    if _NOOP is None:
        _NOOP = _NoopEvt()
    return _NOOP


class _NoopEvt:
    enabled = False

    def mark(self, *a, **k):
        pass


class _StepEvents:
    __slots__ = ("marks", "cpu_marks")

    def __init__(self):
        self.marks = {}
        self.cpu_marks = {}


class EventTimer:
    """零扰动 (不 sync) 的 CUDA event 分段计时。用法见模块 docstring。"""

    def __init__(self, flush_every=10):
        self.enabled = _env_bool("STARVLA_EVT")
        self.flush_every = max(1, int(flush_every))
        self._pending = deque()
        self._steps = 0
        global _evt_singleton
        if self.enabled:
            _evt_singleton = self

    def mark(self, step, name, cpu_name=None):
        """在当前 stream 上打事件点。step 为当前 completed_steps (None=沿用上一步)。"""
        if not self.enabled:
            return
        if step is None and self._pending:
            step = self._pending[-1][0]
        if not self._pending or self._pending[-1][0] != step:
            self._pending.append((step, _StepEvents()))
        se = self._pending[-1][1]
        if name not in se.marks:
            se.marks[name] = torch.cuda.Event(enable_timing=True)
            se.cpu_marks[name] = time.perf_counter()
        se.marks[name].record()

    # ---- 训练循环固定接线点 ----
    def on_data_start(self, step):
        self.mark(step, "ds")

    def on_model_start(self, step):
        self.mark(step, "ms")

    def on_model_end(self, step):
        self.mark(step, "oe")  # oe 兼作 model_end

    def on_forward_start(self, step):
        self.mark(step, "fs")

    def on_forward_end(self, step):
        self.mark(step, "fe")

    def on_backward_end(self, step):
        self.mark(step, "be")

    def maybe_flush(self, step):
        """每 flush_every 步同步一次并输出窗口内均值 (在日志步调用, 该处本来
        就要取标量, 同步成本被摊掉)。"""
        if not self.enabled:
            return
        self._steps += 1
        if self._steps % self.flush_every != 0:
            return
        if not self._pending:
            return
        torch.cuda.synchronize()
        rows = []
        while self._pending:
            rows.append(self._pending.popleft())
        agg = {}
        for (_step, se) in rows:
            m = se.marks
            if not {"ms", "oe"} <= m.keys():
                continue
            def el(a, b):
                return m[a].elapsed_time(m[b]) if a in m and b in m else None
            for key, pair in (
                ("gpu_total", ("ms", "oe")),
                ("fwd", ("fs", "fe")),
                ("bwd", ("fe", "be")),
                ("opt", ("be", "oe")),
                ("prep", ("fs", "vs")),
                ("inbuild", ("vs", "bs")),
                ("bkb", ("bs", "ae")),
                ("proj", ("ae", "ve")),
                ("encode", ("vs", "ve")),
                ("dit", ("ds", "fe")),
            ):
                v = el(*pair)
                if v is not None:
                    agg.setdefault(key, []).append(v)
            cpu = se.cpu_marks.get("oe", 0) - se.cpu_marks.get("ms", 0)
            if cpu > 0:
                agg.setdefault("cpu_total", []).append(cpu * 1000)
        # 相邻步的 submit_gap: 上一步 oe -> 下一步 ms
        for (_s1, se1), (_s2, se2) in zip(rows, rows[1:]):
            if "oe" in se1.marks and "ms" in se2.marks:
                agg.setdefault("submit_gap", []).append(se1.marks["oe"].elapsed_time(se2.marks["ms"]))
        if not agg:
            return

        def stat(v):
            return f"{sum(v) / len(v):7.1f}" if v else "    n/a"

        logger.info(
            "evt | " + " ".join(
                f"{k}={stat(v)}ms" for k, v in (
                    ("gpu_total", agg.get("gpu_total", [])),
                    ("fwd", agg.get("fwd", [])),
                    ("prep", agg.get("prep", [])),
                    ("inbuild", agg.get("inbuild", [])),
                    ("bkb", agg.get("bkb", [])),
                    ("proj", agg.get("proj", [])),
                    ("encode", agg.get("encode", [])),
                    ("dit", agg.get("dit", [])),
                    ("bwd", agg.get("bwd", [])),
                    ("opt", agg.get("opt", [])),
                    ("cpu_total", agg.get("cpu_total", [])),
                    ("submit_gap", agg.get("submit_gap", [])),
                )
            )
        )
