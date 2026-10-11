# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
GPU 性能分析器 (PerformanceAnalyzer)
=====================================
回答一个核心问题:**训练慢,到底是通信瓶颈还是计算瓶颈?**

背景
----
StarVLA 训练循环 (train_starvla.py) 原本只把每步切成 `data_time` 和
`model_time`,且全程没有 `torch.cuda.synchronize()`。于是 `model_time` 把
**计算** (forward/backward 的 matmul) 和 **通信** (DeepSpeed ZeRO 的
allreduce / reduce_scatter) 混在一起,无法判断 GPU 没拉满是哪一类原因。

DeepSpeed ZeRO-2 默认开 `overlap_comm=true`,梯度 reduce-scatter 与反向计算
是**重叠**在不同 CUDA stream 上的。所以光看"墙钟时间"或"内核时间求和"
会被重叠骗到——必须同时看 **GPU SM 实际占用率** (GPU 到底忙不忙) 才能下结论。

三层信号
--------
1. **torch.profiler 内核级拆分** (主,不扰动 overlap):
   把 CUDA 内核按 `nccl_*` (通信) vs 其余 (计算) 聚合,得到 comm/compute 时间
   与 `comm_ratio`。
2. **GPU SM 利用率采样** (主,区分"重叠" vs "GPU 真闲着"):
   rank0 后台线程周期采样 `torch.cuda.utilization()`。占用率低 = GPU 在等
   (通信或数据);高 = 计算打满。
3. **同步式 step 分段计时** (次,给人看的可读拆分):
   仅在采样窗口内给 forward/backward/optimizer 加 `cuda.synchronize()`,
   得到干净墙钟拆分。**注意**:加 sync 会破坏 overlap_comm 的重叠,所以这部分
   只看相对量级,且在报告里明确标注"含同步扰动"。

最后给出**结论性诊断**:通信瓶颈 / 计算瓶颈 / 数据IO瓶颈 / 混合 + 建议动作。

零侵入
------
默认关闭。关闭时 `region()` 返回 `contextlib.nullcontext()`,`on_step_*` 直接
return,训练循环零额外开销。只在 rank0 跑,不影响其他 rank。

启用方式
--------
环境变量 (零编辑,贴合"先大致跑一下"):
    STARVLA_PROFILE=1           开关 (默认关)
    STARVLA_PROFILE_SKIP=3      跳过前 N 步 (等 cudnn autotune / lazy init 稳定)
    STARVLA_PROFILE_STEPS=8     采集 N 步
    STARVLA_PROFILE_TRACE=1     导出 chrome trace (默认开)
或 yaml (可选): trainer.profile.{enabled, skip_steps, num_steps, export_trace}
env 优先于 yaml 优先于默认。
"""

import contextlib
import json
import os
import statistics
import threading
import time
from pathlib import Path

import torch
from accelerate.logging import get_logger

logger = get_logger(__name__)


def _env_bool(name, default=False):
    """读环境变量布尔值 (1/true/yes/on → True),未设返回 default。"""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    """读环境变量整数,未设或非法返回 default。"""
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


class PerformanceAnalyzer:
    """可开关的 GPU 性能分析器。默认关闭,关闭时所有方法为 no-op。

    生命周期由 trainer 的 `on_step_begin` / `on_step_end` 驱动:
      - 前 `skip_steps` 步:不采集 (让 autotune/lazy-init 稳定)
      - 接下来 `num_steps` 步:profiler + 采样线程 + 分段计时全部开启
      - 窗口结束:自动 stop + 打印诊断 + 写 JSON + 导 trace + 自关闭
    """

    def __init__(self, cfg, accelerator, output_dir):
        self.accelerator = accelerator
        self.output_dir = Path(output_dir) if output_dir else None
        # is_main_process 在 accelerator 创建后任意时刻都安全可调。
        self.is_rank0 = bool(accelerator.is_main_process)
        # tracker 由 trainer 在 train() 里注入 (tracker 在 prepare_training 才建好)。
        self.tracker = None

        # ---- 解析配置: env > yaml > 默认 ----
        # defensive 读取: trainer.profile 可能整个不存在 (AccessTrackedConfig 对
        # 缺失字段会抛 AttributeError,getattr(...,None) 兜住)。
        prof_cfg = getattr(getattr(cfg, "trainer", None), "profile", None)
        yaml_enabled = bool(getattr(prof_cfg, "enabled", False)) if prof_cfg else False
        self.enabled = _env_bool("STARVLA_PROFILE", yaml_enabled)
        self.skip_steps = _env_int(
            "STARVLA_PROFILE_SKIP", int(getattr(prof_cfg, "skip_steps", 3)) if prof_cfg else 3
        )
        self.num_steps = _env_int(
            "STARVLA_PROFILE_STEPS", int(getattr(prof_cfg, "num_steps", 8)) if prof_cfg else 8
        )
        self.export_trace = _env_bool(
            "STARVLA_PROFILE_TRACE", bool(getattr(prof_cfg, "export_trace", True)) if prof_cfg else True
        )
        # 空闲间隔分析: 解析 chrome trace 算 GPU idle 比例 + 间隔大小分布,
        # 用于诊断"功率跳动"(kernel-launch-bound vs 大气泡)。默认关 (解析大 trace 较重)。
        self.analyze_gaps = _env_bool("STARVLA_PROFILE_GAP", False)
        if self.analyze_gaps:
            self.export_trace = True  # 空闲分析依赖 trace

        # 全局 batch size,用于算吞吐 (per_device × num_processes × grad_accum)。
        try:
            self._total_batch_size = (
                cfg.datasets.vla_data.per_device_batch_size
                * accelerator.num_processes
                * accelerator.gradient_accumulation_steps
            )
        except Exception:
            self._total_batch_size = None

        # ---- 运行时状态 ----
        self._iter_count = 0  # 训练开始以来的迭代数
        self._in_window = False  # 当前这一步是否在采集窗口内
        self._last_step = 0
        self._prof_started = False
        self._profiling_done = False  # 窗口已结束并报告过,后续永远 no-op
        self._prof = None
        self._sampler_thread = None
        self._sampler_stop = None
        self._util_samples = []
        self._power_samples = []  # GPU 功率 (W) — 量化"功率跳动"幅度
        self._mem_samples = []
        self._region_times = {}  # name -> [per-step wall seconds]
        self._window_data_times = []
        self._window_model_times = []

        if self.enabled and self.is_rank0:
            logger.info(
                f"📊 PerformanceAnalyzer 启用 (rank0): skip_steps={self.skip_steps}, "
                f"num_steps={self.num_steps}, export_trace={self.export_trace}"
            )
        elif self.enabled and not self.is_rank0:
            # 非 rank0 不采集 (避免 8× 磁盘 + 扰动),但仍走 on_step_* 计数,
            # 保证与 rank0 步进一致 (region 在非 rank0 恒返回 nullcontext)。
            pass

    # ===================================================================
    # 对外钩子 (trainer 调用)
    # ===================================================================
    def on_step_begin(self, step):
        """每个迭代开头调。负责在窗口起点启动 profiler + 采样线程。"""
        self._last_step = step
        if not self.enabled or self._profiling_done:
            self._in_window = False
            return
        # 当前迭代索引 = self._iter_count (在 on_step_end 里自增)
        self._in_window = self.skip_steps <= self._iter_count < self.skip_steps + self.num_steps
        if self.is_rank0 and self._iter_count == self.skip_steps:
            self._start_profiling()

    def on_step_end(self, step, data_time=None, model_time=None):
        """每个迭代结尾调。负责在窗口终点 stop + 生成报告。"""
        if not self.enabled or self._profiling_done:
            return
        if self._in_window:
            if data_time is not None:
                self._window_data_times.append(data_time)
            if model_time is not None:
                self._window_model_times.append(model_time)
        is_last_profiled = self._iter_count == self.skip_steps + self.num_steps - 1
        self._iter_count += 1
        self._in_window = False
        if is_last_profiled:
            if self.is_rank0:
                self._stop_and_report()
            self._profiling_done = True

    def region(self, name):
        """包裹 forward/backward/optimizer 段。窗口内 (rank0) 起同步计时+nvtx;
        否则返回 nullcontext 零开销。

        窗口外加 sync 会破坏 overlap_comm,所以仅在 _in_window 内启用。
        """
        if not (self.enabled and self.is_rank0 and self._in_window):
            return contextlib.nullcontext()
        return self._timed_region(name)

    def finalize(self):
        """训练结束兜底:若窗口未自然结束 (如 max_steps 太小),尽力出一份报告。"""
        if not self.enabled or self._profiling_done:
            return
        if self.is_rank0 and self._prof_started:
            self._stop_and_report()
        self._profiling_done = True

    # ===================================================================
    # 内部实现
    # ===================================================================
    @contextlib.contextmanager
    def _timed_region(self, name):
        """同步式分段计时 context manager (仅窗口内 rank0 用)。

        前后各一次 cuda.synchronize 拿到干净墙钟;并 push 一个 nvtx range,
        这样导出的 chrome trace / nsys 里也能看到分段标注。
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            self._region_times.setdefault(name, []).append(time.perf_counter() - t0)

    def _start_profiling(self):
        """启动 torch.profiler + GPU 利用率采样线程。"""
        import torch.profiler as tp

        # 窗口峰值显存:重置峰值计数器,stop 后读到的就是窗口内峰值。
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        self._prof = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA],
            record_shapes=False,  # 关掉省开销
            with_stack=False,
            with_modules=False,
        )
        self._prof.start()
        self._prof_started = True
        self._util_samples = []
        self._power_samples = []
        self._mem_samples = []
        self._region_times = {}
        self._window_data_times = []
        self._window_model_times = []
        self._sampler_stop = threading.Event()
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()
        logger.info(
            f"📊 [Profiler] 开始采集: 跳过前 {self.skip_steps} 步, 采集 {self.num_steps} 步..."
        )

    def _read_gpu_stats(self):
        """一次读 GPU (SM 利用率 %, 功率 W, 已分配显存 GB)。优先一次 nvidia-smi
        拿全 (pynvml 缺失时这是唯一能拿到 power/util 的途径); 失败则 torch 兜底
        util/mem (此时无功率)。任一字段失败返回 None。

        nvidia-smi 的 power.draw 直接反映"功率跳动":min/max 跨度大 = GPU 在
        busy/idle 间反复横跳。
        """
        try:
            import subprocess

            dev = torch.cuda.current_device()
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,power.draw,memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(dev),
                ],
                text=True,
                timeout=2,
            ).strip().splitlines()[0].split(",")
            util = int(out[0].strip())
            power = float(out[1].strip())  # Watts
            mem = float(out[2].strip()) / 1024  # MiB -> GB
            return util, power, mem
        except Exception:
            pass
        # 兜底: torch (无 power)
        util = mem = None
        try:
            util = int(torch.cuda.utilization())
        except Exception:
            pass
        try:
            mem = torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            pass
        return util, None, mem

    def _sample_loop(self):
        """后台线程:周期采样 GPU SM 利用率 + 功率 + 已分配显存。

        - 子线程先 set_device 建立 CUDA primary context:某些 setup 下后台线程
          直接调 cuda API 会因没有 current context 而失败。
        - 采样失败被吞掉,对应 stats 降级为 None,不影响主诊断。
        """
        try:
            torch.cuda.set_device(torch.cuda.current_device())
        except Exception:
            pass
        while not self._sampler_stop.is_set():
            util, power, mem = self._read_gpu_stats()
            if util is not None:
                self._util_samples.append(util)
            if power is not None:
                self._power_samples.append(power)
            if mem is not None:
                self._mem_samples.append(mem)
            self._sampler_stop.wait(0.1)

    def _stop_and_report(self):
        """停止 profiler + 采样线程,生成诊断报告。"""
        if not self._prof_started:
            return
        try:
            self._prof.stop()
        except Exception:
            pass
        self._prof_started = False
        if self._sampler_stop is not None:
            self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
        try:
            self._report()
        except Exception as e:
            logger.warning(f"📊 [Profiler] 生成报告失败 (不影响训练): {e}")

    @staticmethod
    def _classify_kernel(key):
        """内核名 → 'comm' (NCCL 通信) 或 'compute'。"""
        k = key.lower()
        if "nccl" in k or "communicator" in k or k.startswith("c10d"):
            return "comm"
        return "compute"

    def _collect_kernel_times(self):
        """从 profiler 聚合通信/计算的设备时间 (微秒)。"""
        comm_us, compute_us = 0, 0
        comm_keys = {}
        try:
            events = self._prof.key_averages()
        except Exception:
            return comm_us, compute_us, comm_keys
        for ev in events:
            dt = getattr(ev, "self_device_time_total", 0) or 0
            if dt <= 0:
                continue
            if self._classify_kernel(ev.key) == "comm":
                comm_us += dt
                short = ev.key.split("\n")[0][:42]
                comm_keys[short] = comm_keys.get(short, 0) + dt
            else:
                compute_us += dt
        return comm_us, compute_us, comm_keys

    @staticmethod
    def _stats(samples):
        if not samples:
            return None
        s = sorted(samples)

        def pct(p):
            return s[min(len(s) - 1, int(len(s) * p))]

        return {
            "avg": statistics.fmean(samples),
            "p50": pct(0.5),
            "p95": pct(0.95),
            "max": max(samples),
            "min": min(samples),
            "n": len(samples),
        }

    def _diagnose(self, comm_ratio, util_stats, data_avg, model_avg):
        """根据三层信号给出诊断结论 + 依据 + 建议。

        判定优先级: 数据IO > 通信 > 计算 > 混合。
        """
        cr = comm_ratio if comm_ratio is not None else 0.0
        util_avg = util_stats["avg"] if util_stats else None
        reasons, suggestions = [], []

        if data_avg is not None and model_avg is not None and data_avg > model_avg * 1.1:
            verdict = "DATA_IO"
            reasons.append(
                f"data_time({data_avg*1000:.0f}ms) > model_time({model_avg*1000:.0f}ms): 取数据比计算还慢"
            )
            suggestions = [
                "增大 datasets.vla_data.num_workers / prefetch_factor",
                "检查 video_backend (torchvision_av) 视频解码是否 CPU 瓶颈",
                "确认 load_all_data_for_training 已开 (减少重复 IO)",
                "考虑把解码/resize 移到 GPU,或离线预编码成更快的格式",
            ]
        elif cr > 0.35 and (util_avg is None or util_avg < 60):
            verdict = "COMMUNICATION"
            reasons.append(
                f"通信占比 {cr*100:.0f}% 且 GPU SM 利用率均值 "
                f"{util_avg if util_avg is not None else '?'}% (<60%): GPU 在等通信"
            )
            suggestions = [
                "ZeRO stage 2 → 3 (参数也切片,减少通信量)",
                "调大 reduce_bucket_size / allgather_bucket_size (5e8 → 1e9 试)",
                "export NCCL_DEBUG=INFO 确认是否走 NVLink/P2P",
                "增大 gradient_accumulation_steps 减少同步边界频率",
            ]
        elif util_avg is not None and util_avg > 80 and cr < 0.20:
            verdict = "COMPUTE"
            reasons.append(
                f"GPU SM 利用率均值 {util_avg:.0f}% (>80%) 且通信占比仅 {cr*100:.0f}%: 计算已打满"
            )
            suggestions = [
                "对 3B 模型这通常属正常——加大 per_device_batch_size 提高吞吐",
                "确认用了 bf16 + flash/triton attention",
                "想再榨干:考虑 tensor-parallel,或用激活重计算换更大 batch",
            ]
        else:
            verdict = "MIXED"
            reasons.append(
                f"通信占比 {cr*100:.0f}%, SM 利用率 {util_avg}, 无明显单一瓶颈"
            )
            suggestions = [
                "打开 chrome trace 找时间线上的 '气泡' (gap)",
                "先消最大的那项 (data_time 或 comm_ratio)",
            ]
        return verdict, reasons, suggestions

    def _print_gap_report(self, gap):
        """打印 GPU 空闲间隔分析 (诊断"功率跳动"根因)。"""
        L = []
        L.append("")
        L.append("╔═══════════════ GPU 空闲间隔分析 (功率跳动根因) ═══════════════╗")
        L.append(f"║ 内核数: {gap['n_kernels']}, 时间跨度 {gap['span_ms']:.0f} ms")
        L.append(
            f"║ GPU busy {gap['busy_ms']:.0f} ms / idle {gap['idle_ms']:.0f} ms  "
            f"→ idle 占比 {gap['idle_ratio']*100:.1f}%  (共 {gap['gap_count']} 个空闲段)"
        )
        L.append("║ 【空闲段大小分布】(看大头在哪一档 → 定性病因)")
        for label, st in gap["gap_buckets"].items():
            if st["count"] > 0:
                L.append(f"║   {label:11s}: {st['count']:6d} 段, 共 {st['total_ms']:8.1f} ms")
        L.append("║ 最大的几段空闲 (ms): " + ", ".join(f"{g:.1f}" for g in gap["top_gaps_ms"]))
        # 定性: 极小空隙多 = kernel-launch-bound; 大空隙 = 气泡
        tiny = sum(gap["gap_buckets"][k]["total_ms"] for k in ("<100µs", "100µs-1ms"))
        big = sum(gap["gap_buckets"][k]["total_ms"] for k in ("1-10ms", "10-100ms", ">100ms"))
        if gap["idle_ratio"] < 0.05:
            verdict = f"idle 仅 {gap['idle_ratio']*100:.0f}% — GPU 基本连轴转,功率跳动是正常 step 结构"
        elif tiny > big:
            verdict = "kernel-launch-bound: 大量极小空隙, CPU 发 kernel 跟不上 (eager attention 典型) → 试 flash/SDPA attention 或 torch.compile"
        elif big > tiny:
            verdict = "大气泡: 少数大空隙, 数据/通信/CPU 同步等待 → 结合 comm_ratio / data_time / 最大空隙位置定位"
        else:
            verdict = "混合: 大小空隙都不少 → 看 chrome trace 时间线找具体气泡"
        L.append(f"║ ▶ 定性: {verdict}")
        L.append("╚══════════════════════════════════════════════════════════════╝")
        self.accelerator.print("\n".join(L))

    def _report(self):
        """聚合三层信号 → 打印诊断块 + 写 JSON + 导 trace + 推 tracker 标量。"""
        comm_us, compute_us, comm_keys = self._collect_kernel_times()
        comm_s = comm_us / 1e6
        compute_s = compute_us / 1e6
        total_s = comm_s + compute_s
        comm_ratio = (comm_s / total_s) if total_s > 0 else None
        cr_disp = comm_ratio if comm_ratio is not None else 0.0

        util_stats = self._stats(self._util_samples)
        power_stats = self._stats(self._power_samples)
        mem_stats = self._stats(self._mem_samples)
        data_avg = statistics.fmean(self._window_data_times) if self._window_data_times else None
        model_avg = statistics.fmean(self._window_model_times) if self._window_model_times else None
        region_avg = {k: (statistics.fmean(v) if v else 0.0) for k, v in self._region_times.items()}

        # 吞吐 = 全局 batch / 每步墙钟 (data + model)
        throughput = None
        if self._total_batch_size and data_avg is not None and model_avg is not None:
            step_wall = data_avg + model_avg
            if step_wall > 0:
                throughput = self._total_batch_size / step_wall

        try:
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        except Exception:
            peak_mem_gb = None

        verdict, reasons, suggestions = self._diagnose(comm_ratio, util_stats, data_avg, model_avg)

        # ---- rank0 控制台诊断块 ----
        L = []
        L.append("")
        L.append("╔════════════════════ GPU 性能诊断 (rank0) ═══════════════════╗")
        L.append(f"║ 采样窗口: skip={self.skip_steps} 步, profile={self.num_steps} 步, rank0")
        if self._total_batch_size:
            L.append(
                f"║ 全局 batch: {self._total_batch_size} "
                f"(per_dev×{self.accelerator.num_processes}卡×"
                f"accum={self.accelerator.gradient_accumulation_steps})"
            )
        L.append("║")
        L.append("║ 【每步墙钟】")
        if data_avg is not None:
            L.append(f"║   data  (取数据) : {data_avg*1000:8.1f} ms")
        if model_avg is not None:
            L.append(f"║   model (前反优化): {model_avg*1000:8.1f} ms")
        if throughput is not None:
            L.append(f"║   吞吐           : {throughput:8.2f} samples/s")
        L.append("║")
        L.append("║ 【GPU 内核时间】(设备时间求和; overlap_comm 下可能 > 墙钟)")
        if total_s > 0:
            L.append(f"║   计算 compute   : {compute_s*1000:8.1f} ms  ({(1-cr_disp)*100:5.1f}%)")
            L.append(f"║   通信 comm(nccl): {comm_s*1000:8.1f} ms  ({cr_disp*100:5.1f}%)")
            if comm_keys:
                top = sorted(comm_keys.items(), key=lambda x: -x[1])[:3]
                names = ", ".join(f"{k}:{v/1e3:.0f}ms" for k, v in top)
                L.append(f"║   通信 top 内核  : {names}")
        else:
            L.append("║   (未采集到 CUDA 内核时间)")
        L.append("║")
        L.append("║ 【GPU SM 利用率】(后台采样)")
        if util_stats:
            L.append(
                f"║   avg={util_stats['avg']:4.0f}%  p50={util_stats['p50']:4.0f}%  "
                f"p95={util_stats['p95']:4.0f}%  peak={util_stats['max']:4.0f}%  (n={util_stats['n']})"
            )
        else:
            L.append("║   (未能采样 SM 利用率 — 仅参考内核时间比例)")
        if power_stats:
            swing = power_stats["max"] - power_stats["min"]
            L.append(
                f"║ 【GPU 功率】 avg={power_stats['avg']:4.0f}W  min={power_stats['min']:4.0f}W  "
                f"max={power_stats['max']:4.0f}W  (跳动 {swing:.0f}W{' ←明显跳动!' if swing > 150 else ''})"
            )
        if mem_stats:
            L.append(f"║ 【显存】 allocated avg={mem_stats['avg']:.1f} GB, sampled peak={mem_stats['max']:.1f} GB")
        if peak_mem_gb is not None:
            L.append(f"║         max_memory_allocated (窗口峰值) = {peak_mem_gb:.1f} GB")
        L.append("║")
        L.append("║ 【分段计时】(含同步扰动, 仅看相对量级, ms/step)")
        for k in ("forward", "backward", "optimizer"):
            if k in region_avg:
                L.append(f"║   {k:9s}: {region_avg[k]*1000:8.1f}")
        L.append("║")
        L.append(f"║ ▶ 诊断结论: {verdict}")
        for r in reasons:
            L.append(f"║     - {r}")
        L.append("║   建议:")
        for sg in suggestions:
            L.append(f"║     • {sg}")
        L.append("╚══════════════════════════════════════════════════════════════╝")
        self.accelerator.print("\n".join(L))

        # ---- 导出 chrome trace (拖进 chrome://tracing 可视化) ----
        trace_path = str(self.output_dir / "profile_trace_rank0.json") if self.output_dir is not None else None
        if self.export_trace and trace_path is not None:
            try:
                self._prof.export_chrome_trace(trace_path)
                size_mb = os.path.getsize(trace_path) / (1024**2)
                if size_mb > 50:
                    # 大模型 + 多步会产生很大的 trace,浏览器加载会卡。诊断结论其实
                    # 已经在 summary.json 里,trace 只是可选的深挖工具。
                    logger.warning(
                        f"📊 [Profiler] Chrome trace 高达 {size_mb:.0f} MB,浏览器打开可能卡死。"
                        f"如不需要可视化,设 STARVLA_PROFILE_TRACE=0 跳过;或减小 STARVLA_PROFILE_STEPS。"
                        f"(诊断结论见 profile_summary.json)"
                    )
                else:
                    logger.info(f"📊 [Profiler] Chrome trace 已写入 {trace_path} ({size_mb:.1f} MB, chrome://tracing 打开)")
            except Exception as e:
                logger.warning(f"📊 [Profiler] 导出 trace 失败: {e}")

        # ---- 空闲间隔分析 (诊断"功率跳动"根因, 需 STARVLA_PROFILE_GAP=1) ----
        gap_info = None
        if self.analyze_gaps and trace_path is not None and os.path.exists(trace_path):
            try:
                gap_info = analyze_trace_idle(trace_path)
                if gap_info is not None:
                    self._print_gap_report(gap_info)
                else:
                    logger.warning("📊 [Profiler] 空闲分析: trace 内未找到 GPU 内核事件 (cat=kernel)")
            except Exception as e:
                logger.warning(f"📊 [Profiler] 空闲分析失败 (不影响训练): {e}")

        # ---- 写 JSON 摘要 (机读) ----
        summary = {
            "verdict": verdict,
            "reasons": reasons,
            "skip_steps": self.skip_steps,
            "num_steps": self.num_steps,
            "total_batch_size": self._total_batch_size,
            "step_data_ms": data_avg * 1000 if data_avg is not None else None,
            "step_model_ms": model_avg * 1000 if model_avg is not None else None,
            "throughput_samples_per_s": throughput,
            "compute_ms": compute_s * 1000,
            "comm_ms": comm_s * 1000,
            "comm_ratio": comm_ratio,
            "gpu_util": util_stats,
            "gpu_power_w": power_stats,
            "mem_gb": mem_stats,
            "peak_mem_gb": peak_mem_gb,
            "region_ms": {k: v * 1000 for k, v in region_avg.items()},
            "idle_analysis": gap_info,
        }
        if self.output_dir is not None:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with open(self.output_dir / "profile_summary.json", "w") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"📊 [Profiler] 摘要已写入 {self.output_dir / 'profile_summary.json'}")
            except Exception as e:
                logger.warning(f"📊 [Profiler] 写 JSON 失败: {e}")

        # ---- tracker 标量 (走现有 self.tracker.log, 仅标量) ----
        if self.tracker is not None:
            try:
                step_wall_ms = ((data_avg + model_avg) * 1000) if (data_avg and model_avg) else 0.0
                self.tracker.log(
                    {
                        "profile/comm_ratio": cr_disp,
                        "profile/gpu_util_avg": util_stats["avg"] if util_stats else 0.0,
                        "profile/gpu_power_avg": power_stats["avg"] if power_stats else 0.0,
                        "profile/step_time_ms": step_wall_ms,
                        "profile/peak_mem_gb": peak_mem_gb if peak_mem_gb is not None else 0.0,
                        "profile/idle_ratio": gap_info["idle_ratio"] if gap_info else 0.0,
                    },
                    step=self._last_step,
                )
            except Exception:
                pass


def analyze_trace_idle(trace_path, top_n=8):
    """解析 chrome trace, 算 GPU 空闲间隔分布 (诊断"功率跳动"根因)。

    把所有 GPU 内核事件 (cat='kernel', 含 nccl 内核; 以及 'gpu_memcpy') 的
    [ts, ts+dur] 区间按时间排序后合并 (跨 stream 取并集 —— 这样 overlap_comm
    的通信流与计算流重叠也算 busy), 补集就是 GPU 真正空闲的时段。

    返回 dict(idle 比例 + 空闲段大小直方图 + 最大几段); trace 内无内核事件返回 None。
    极小空隙多 → kernel-launch-bound (CPU 发 kernel 跟不上);
    少数大空隙 → 数据/通信/CPU 同步气泡。
    """
    import json as _json

    with open(trace_path) as f:
        data = _json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    intervals = []
    for e in events:
        cat = e.get("cat", "") or ""
        # cat='kernel' = PyTorch 标记的 GPU 内核执行 (含 nccl); 'gpu_memcpy' = 拷贝。
        # 排除 cuda_runtime/cuda_driver (那是 CPU 侧 API 调用,不是 GPU 执行)。
        if ("kernel" in cat or "gpu_memcpy" in cat) and "ts" in e and "dur" in e and e["dur"] > 0:
            intervals.append((e["ts"], e["ts"] + e["dur"]))
    if not intervals:
        return None
    intervals.sort()
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    total_busy = sum(e - s for s, e in merged)
    span = merged[-1][1] - merged[0][0]
    if span <= 0:
        return None
    total_idle = span - total_busy
    gaps = [merged[i][0] - merged[i - 1][1] for i in range(1, len(merged))]
    buckets = [("<100µs", 1e2), ("100µs-1ms", 1e3), ("1-10ms", 1e4), ("10-100ms", 1e5), (">100ms", 1e18)]
    hist = {label: {"count": 0, "total_us": 0.0} for label, _ in buckets}
    for g in gaps:
        for label, bound in buckets:
            if g < bound:
                hist[label]["count"] += 1
                hist[label]["total_us"] += g
                break
    top = sorted(gaps, reverse=True)[:top_n]
    return {
        "n_kernels": len(intervals),
        "busy_ms": total_busy / 1e3,
        "idle_ms": total_idle / 1e3,
        "span_ms": span / 1e3,
        "idle_ratio": total_idle / span,
        "gap_count": len(gaps),
        "gap_buckets": {k: {"count": v["count"], "total_ms": v["total_us"] / 1e3} for k, v in hist.items()},
        "top_gaps_ms": [g / 1e3 for g in top],
    }
