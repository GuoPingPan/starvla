# QwenPI_v3 训练效率 Debug 实录（H100 8 卡 · umi → H1）

> 承接 `docs/qwenpi_v3_speedup_case_study.md`（第一轮：fp32/repeat16 双 bug，A800 4×）。
> 本篇记录第二轮：H100 生产化过程中"利用率跑不满"的完整排查链——五层根因、
> 九轮实验、修复矩阵、遗留方案。**所有结论都有 trace/实验数据支撑，可复现。**

## 一、问题现象（最初 complaint）

- H100 8×80GB，QwenPI_v3 (1.2B) bs128：SM 平均 9.6%（p50=3%），功率 187-291W（TDP 700W）
- `fwd/bwd = 2.2 倒挂`（2495ms/1128ms）
- 周期性 `timing/data` 7-25s 停顿（每 ~40 步）
- `timing/model` 在 0.65-1.9s 间大幅波动

## 二、五层根因（按发现顺序）

| # | 根因 | 证据 | 修复 | commit |
|---|---|---|---|---|
| 1 | **图像预处理在主进程 CPU**：128 张图 HF image_processor 604ms/步，藏在 model region 里（timing/data 测不到） | fwd/bwd 倒挂 + 本地计时复现 | GPU batch 化预处理（配置化 op 管线 + patch 编码契约） | `5f0c162` |
| 2 | **每步隐藏同步**：`action_loss.item()` 等 GPU 排空 ~30ms + `float(grad_norm)` D2H 12.5ms | trace 归因 cudaDeviceSynchronize 落在 python 层/copy_ | tensor 传递，日志步才转换 | `e2ab912` 后续 |
| 3 | **assemble 逐行 python 循环**：128 次 slice 赋值 → 50ms host 停顿 | trace: 每步一个 ~50ms gap 前后皆 elementwise | advanced-indexing scatter（0.05ms） | `0cd351f` |
| 4 | **视频解码供给赤字 + ffmpeg 线程失控** | 见下方实验矩阵 | STAGE_SHM（部分）+ 帧缓存（待做） | `802b0a4` |
| 5 | **steps_index 多进程竞态**：缓存命中路径跳过 barrier，外部预计算进程写入时机导致 barrier 错位 | H1 启动 FileNotFoundError | flock read-or-build | `478bcdf` |

## 三、数据侧九轮实验矩阵（本轮主战场）

| 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|
| 固定 batch | STARVLA_FIXED_BATCH=1 | model 全程 0.6-1.2s 零尖刺 | **模型无罪，数据实锤** |
| w16 | 16 workers | data ✓ 但 model ×2（1.8-1.9s） | CPU 争用（**OMP=1 救不了 ffmpeg 线程**） |
| w8 + pf8 | 缓冲 64 batch | step 40 一次 28.7s 停顿 | 启动积压算术：pf8×8w×160样本×60ms=77s 积压，step40 时欠 17-21s 吻合 |
| w8 + pf8 + **SHM** | 数据进 /dev/shm | step 40 仍停 21.5s | **证伪 I/O 假设：解码是 CPU 计算密集**（shm 只省 25%） |
| w16 + OMP=1 | | model 仍 1.8-1.9 | ffmpeg 解码线程不受 OMP 约束，数千线程挤 128 核 |

**数据侧结论**：解码 ~60-80ms/样本是 CPU 计算（torchvision/PyAV 解码+PIL 变换），
不是 I/O。供给需要 ≥12 worker，CPU 安静需要 ≤8 → 调 worker 是死局。
**终解 = 帧级缓存**（decode-once → /dev/shm memmap → dataloader 零解码），待实现。

## 四、模型侧收尾（重计算）

- gradient checkpointing（backbone HF API + DiT forward 接线）：bs64 model 持平（**launch-bound 下 +33% 计算藏进空闲，显存白赚**），bs128 单卡 62GB 跑通（原必 OOM）→ bs160 8 卡可行，bs192 超 80GB（fla autotune 峰值压垮）→ **H100 落点 bs160**
- DiT CUDA Graphs（reduce-overhead）：busy 59→64%，但调研发现多卡+DS 下有分布式 NaN/内存池风险（PyTorch 论坛 #224248、DS #6718/9）→ **默认关，留钩子**

## 五、最终生产配置（umi · H100 8 卡）

```bash
NUM_GPUS=8 BATCH=160 NUM_WORKERS=8 PREFETCH=8 \
  STAGE_SHM=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 稳态: model 0.95-1.15s → ~1250 samples/s (数据停顿待帧缓存彻底归零)
```

## 六、H1 (d18480a6) 接入要点

- 22 数据集 / ~200 任务 / 3 视图 / 30D 双臂 / **889GB**
- 序列长度实测：含 state 357 token（3 视图×64 + state 110 + 文本）→ `pad_seq_len: 384`
- state 进文本 → tok cache 无法命中（慢文本路径，bs32 时 ~100ms/步，可接受）
- 起步 `BATCH=32`（序列当量 = umi 的 4×）
- 路径归并：`examples/data-analyze/train_files/run_h1_d18480a6_qwenpi_v3.yaml` + `data_registry/h1_dataset.py`（一模型一路径）
- **数据体检待做**：≥3 个数据集有 'huangchangyun' 占位任务串（grad 异常高危）
- 889GB 超 STAGE_SHM 上限 → 帧缓存需按 RAM 预算专门设计

## 七、方法论增量（相对第一轮）

1. **固定 batch 是万能隔离器**：数据/模型归因一步到位（STARVLA_FIXED_BATCH）
2. **假设要设计"证伪实验"**：SHM 花半小时证伪了自己的 I/O 假设，避免在错误方向上继续投入
3. **周期性停顿先算"积压算术"**：pf × workers × 样本/批 × 单样本耗时 = 初始积压；与停顿出现步数对不上就是别的病
4. **OMP=1 ≠ CPU 安静**：ffmpeg/PIL 有自己的线程体系，进程数预算要按"线程总数"算
5. **外部进程 + 多 rank 共享缓存文件**：任何"存在性检查→读"都要 flock，barrier 对齐假设一戳就破
6. **profiler 本身的开销会污染测量**（87k kernel 记录 > 被测对象），关键数字用无 profiler 的稳态日志交叉验证

## 八、遗留清单

| 项 | 状态 |
|---|---|
| 帧级缓存（umi + H1） | 设计完待实现（本轮最大遗留） |
| H1 stats 预计算 22/22 | 后台进行中（flock 修复后可与训练并发） |
| H1 数据体检（垃圾任务剔除） | 待跑 |
| grad norm 异常数据审计（provenance + per-sample loss + 落盘） | provenance 已注入，后两件进行中 |
| backbone compile/CUDA Graphs | 调研结论：多卡+DS 风险高，暂缓 |
