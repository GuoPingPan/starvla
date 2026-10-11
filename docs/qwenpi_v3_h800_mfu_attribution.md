# QwenPI_v3 单节点 H800 MFU 归因与调优实录

> 第三轮性能优化完整记录（前两轮见 `qwenpi_v3_speedup_case_study.md` / `qwenpi_v3_h100_efficiency_debug.md`）。
> 本轮主题：**归因驱动的单节点 MFU 优化**——所有结论都有独立测量证据，所有修复都单独验证。
>
> 最终成果：H1 数据集 8×H800 稳态 1130ms/步 → ~870ms/步（bs32 口径），bs48 吞吐 **+59%~+80%**，
> 全部零数值风险或经对拍验证。

---

## 0. 背景与起点

| 项 | 值 |
|---|---|
| 模型 | QwenPI_v3 = Qwen3.5-0.8B backbone（混合线性注意力）+ LayerwiseFM DiT（356M），共 1.215B 全可训 |
| 数据 | H1 双臂 22 数据集（889GB，3 视图，state 进文本，seq 384 = 3×64 视觉 + ~165 文本） |
| 硬件 | 单节点 8×H800 80GB（NV18 全互联），DeepSpeed ZeRO-2，bf16，激活重计算开 |
| 基线 | bs32/卡，稳态 ~1130ms/步（波动 900-1550），SM 利用率 47-54%，**MFU ≈ 10%** |
| 理论 | 6ND 估算理想 ~90-120ms/步 → 差距近 10× |

---

## 1. 工具建设（先磨刀）

### 1.1 百舸 AIHC 任务自动化

**问题**：调试循环依赖人在控制台点任务 + 复制报错，一轮 20 分钟。

**调查过程**（值得复用的思路）：
1. dev box 是 `aibox` pod → 确认平台为百度百舸 AIHC 系
2. 官方 AIHC CLI（v1.3.2）安装配置后，`.bosenv` 的 BOS 专用 AK 查不到资源池
3. 拿到正确凭证后，从历史任务名猜出资源池 `aihc-rdw-bj`（job list 报错信息里泄露了归属账号）
4. 用户提供队列名"训练-H卡" → `aihc queue get` 的 AccessDenied 报错里带出队列 ID → 打通

**踩坑**（export 格式 ≠ create 格式）：
- serverless 池拒绝 `hostNetwork=true`
- datasource type `emptydir` 创建侧非法（devshm 由平台注入）
- `hang-detection-stack-timeout-minutes` 必须 >0
- cfs datasource 必须带 `cfsInstanceId`（从本机 `mount` 表挖到 `cfs-pVlRwhavQO`）
- **cpu/memory 必须按节点粒度（175 核/2767GB）**，自己缩放出的 176/2768 调度器无法匹配

产物：`scripts/aihc/launch_train.sh`——canary 全链路验证后投入使用。

### 1.2 零扰动 CUDA event 计时器（归因的关键武器）

**问题**：已有的 `STARVLA_PROFILE` 剖面器在窗口内逐区域 `cuda.synchronize`，
**打点本身改变了被测系统**（sync 打断 overlap 流水线，制造出假的"通信瓶颈"）。

**方案**：`starVLA/training/trainer_utils/event_timer.py`（`STARVLA_EVT=1`）
- 在 stream 上打事件对（fwd/bwd/opt/步边界），**不 sync**，每 10 步才同步一次取窗口均值
- 输出六元组：`gpu_total / fwd / bwd / opt / cpu_total / submit_gap`
- 判读方法：
  - `gpu_total == cpu_total`（精确相等）→ GPU 跟着 CPU 节奏走，**CPU-bound**
  - `submit_gap` 大 → 步间 CPU 没喂上；小 + 相等 → 步内 CPU 是节奏方
- 本地 sanity：单卡 GPU-bound 场景 submit_gap=3.4ms ✓

---

## 2. 归因：三层独立证据交叉验证

> 方法论核心：**不要用一个测量工具的输出给另一个工具的疑问下结论。**

| 测量手段 | 结果 | 结论 |
|---|---|---|
| CUDA event 计时（稳态，无扰动） | `gpu_total==cpu_total==900.6ms`，`submit_gap=2.2ms`，每个窗口都精确相等 | CPU 是节奏方 |
| Trace kernel 按 stream 求和 | 计算 stream 忙时仅 **347ms/步**（169,344 kernel / 12 步），占用率 22% | GPU 算力大量闲置 |
| NCCL 带宽基准（torchrun 8 进程，CUDA event 计时） | 2GB allreduce = **10.3ms（340GB/s）**，default 与 SIMPLE 无差 | 通信无罪 |

**归因结论**：
```
每步 ~900ms = ~350ms GPU 计算 kernel（本身也碎） + ~30-60ms 真实通信
            + ~500ms CPU 节奏开销 ★主瓶颈★
              （autograd 反向 python 节点遍历 + 每步 5.6k 次 kernel launch 派发
               + autocast/DS hook/python 胶水 + GC 暂停）
```

### 2.1 曾经的三个错误假设（都被证据推翻）

1. **"通信占 48%"** —— 剖面器 sync 制造的假象：sync 后首次集合通信吸收全队 rank 偏斜，
   表现为 1.0-1.6s 的 `RING_LL` 巨型 kernel（小张量 spin 等人，时长=等待）。
   真实梯度 allreduce 只要 3-10ms/次。
2. **"cgpu 虚拟化砍了 P2P"** —— `nvidia-smi topo -m` 显示 NV18 全互联；带宽基准满血。
3. **"数据供给抖动"** —— 固定 batch（`STARVLA_FIXED_BATCH=1`）后波动依旧 → 不是数据方差。

### 2.2 教训：日志分页与时序

- 百舸 job logs API 的 `-m maxLines` 返回**前 N 行**而非尾部；用 `-s <unix时间戳>` 过滤
- tqdm 的 `\r` 不产生新日志行，"8 小时无日志"≠挂了（配合 CFS 产物 mtime / exec 容器核实）
- 测带宽要 `torchrun --nproc_per_node=8`：单进程 `init_process_group` 是 world_size=1 的空操作
  （我们第一次测出 "43TB/s" 的荒谬数据就是这么来的）

---

## 3. 逐项修复

### 3.1 文本慢路径：850ms → 89ms（commit `105051f`）

**现象**：`apply_chat_template(tokenize=True)` 逐条 27ms，32 条/步 ≈ 850ms。

**根因**：tokenize=True 会把 dummy 占位图送进 processor **完整像素管线**（~20ms/条），
而其结果被我们的 GPU 预处理（vision_preproc）**完全丢弃**——纯浪费。

**修法**：渲染串（无图对象）→ 按 processor 真实展开数手动扩展 `<|image_pad|>` → 纯 tokenizer。
- 展开数自校准：首条真慢调用数出每图 token 数（256×256 → patch16+merge2 → **64/图**，不是想当然的 256）
- 保险丝：首条与慢路径逐 token 对拍，不一致自动永久回退
- **验证**：32 条 H1 变长文本 token 级一致

### 3.2 GC 暂停：-24% 且方差消失（commit `dec6668`）

**现象**：固定 batch 下 bwd 分段在 479↔1128ms 波动——数据相同却有周期性长尾。

**假设→验证**：warmup 后 `gc.collect()+gc.freeze()`（常驻对象移入永久代）→
bs32 均值 1130→864ms，**波动带收紧到 858-894**。

**关键细节**：`gc.freeze()` 不禁用 GC（新垃圾仍正常回收），无泄漏风险 → 转正默认开。
排查时 `STARVLA_GC_FREEZE=0`。

### 3.3 batch 摊薄：bs32 → bs48（+59% 组合收益）

**逻辑**：CPU python/派发开销大部分**不随 batch 增长**（算子数不变，张量变大），
GPU 又有 3 倍余量 → 加大 per-GPU batch 直接摊薄固定开销。

**实测矩阵**（固定 batch 稳态）：

| 配置 | 步时 | 吞吐 | 相对基线 |
|---|---|---|---|
| bs32 基线 | 1130ms（抖） | 28.3 | — |
| bs32 + gcfreeze | 864ms（稳） | 37.0 | +31% |
| bs48 | 1425ms（抖） | 33.7 | +19% |
| **bs48 + gcfreeze** | **1070ms（稳）** | **44.9** | **+59%** |
| bs64 | OOM（74+11GB > 80GB） | — | 显存上限 |

注意 bs48 的 bwd 随 batch 放大（620→913ms）——大张量 kernel 变长 + DiT 未编译的 python 仍在。

### 3.4 骨干网逐层编译（commit `dec6668`）

**大坑：transformers 5.3 + Qwen3.5 的多层 hidden_states 契约**。
接口需要 24 层逐层特征（LayerwiseFM 的层条件注入）。
隔离实验复现：eager 25 层 → 对 language_model 子树**任何形式的 compile**（整体或逐层）都塌缩成 1 层。
根因：`Qwen3_5TextModel.forward` 自身不收集 hidden_states（eager 的多层输出靠 HF 内部分发机制补上）。

**解法："自有契约"**：
1. 逐层 `torch.compile(layer, dynamic=False)`（24 个 decoder layers）
2. **forward hook 捕获每个 layer 的输出**（数值 = eager hidden_states 逐层等价）
3. `forward()` 里不再传 `output_hidden_states`，捕获结果补挂到 `outputs.hidden_states`
4. OptimizedModule 保留 nn.Module 钩子机制 → hook 照常触发

效果：fwd 405→390、bwd 620→580，整体 +6%（有限——因为大头在编译区之外）。

### 3.5 vision tower 编译（commit `4d64453`）

同款逐块配方用于 `visual.blocks`（12 个 ViT blocks）。
bs48 每步 144 张图过 ViT。无 hidden_states 契约问题，直接成功。

### 3.6 tok 运行时 memo（commit `6699681`）

**问题**：H1 指令含离散化 state，**每样本唯一**（task + `[STATE] 95 133 203... [ACTION]`），
umi 式"任务集 tok cache"不适用。

**否决的方案**：离线表格重建 instruction——需精确复刻数据管线的 state 归一化链
（逐 key min_max/q99 统计量 + fp16 转换），任何不一致都是静默错误，风险过高。

**采用**：运行时 memo——慢路径算过的 ids **回填内存缓存**；epoch 间样本必然重复，
首 epoch 后全命中；`_finalize_training` 时 rank0 落盘 parquet，重启经 `tok_cache: auto` 直读。
- 字符串来自真实训练管线，**零重建风险**
- Step 日志新增 `tok_cache/hit_rate` 指标可观测
- 收益：首 epoch 后每步省 ~130ms（bs48）

### 3.7 DiT 编译四连坑（封存，commit `b566a3f`）

356M DiT 是 kernel 汤，理论收益大，但 torch 2.6 × diffusers Attention × 非重入 checkpoint 组合连踩四坑：

| # | 方案 | 坑 |
|---|---|---|
| 1 | 整体编译 forward | inductor 后端 bug：`Node mm_347 was invalid, but is output` |
| 2 | 逐块编译 | 非重入 checkpoint 元数据校验误报 → `determinism_check="none"` 修复（保留） |
| 3 | 逐块 + 修复 | AOT 反向：`Only floating point ... can require gradients`（块级 bool mask 入 saved ctx） |
| 4 | 块内 attn/ff 子模块 | 同 3 错（diffusers Attention 内部整型 position_ids 被 AOT 保存） |

**结论**：需 torch 升级或改造 diffusers Attention 后再战。开关 `STARVLA_COMPILE_DIT` 默认关。

---

## 4. 配置管理规范（用户要求，已落实）

- **正式参数全部进 yaml**：`per_device_batch_size: 48`、`framework.qwenvl.torch_compile: true`、
  `trainer.gc_freeze: true`、`tok_cache: auto`
- **环境变量只做调试期强制开关**（优先级高于配置）：
  `STARVLA_COMPILE / STARVLA_GC_FREEZE / STARVLA_TOK_MEMO / STARVLA_COMPILE_DIT / STARVLA_EVT /
   STARVLA_PROFILE / STARVLA_FIXED_BATCH`
- 生产任务零环境变量即吃到全部调优

## 5. 最终配方与数字

```bash
bash scripts/aihc/launch_train.sh --run-id <run> \
  --yaml examples/data-analyze/train_files/run_h1_d18480a6_qwenpi_v3.yaml \
  --replicas 1 --gpus 8 --batch 48 --max-steps 50000 --save-every 5000
```

| 优化项 | 累计效果 | 风险 |
|---|---|---|
| 文本快路径 850→89ms | 已含在所有后续测量中 | 对拍验证，零 |
| gc.freeze | -24% 步时 + 方差消失 | 零（不禁用 GC） |
| bs48 | +59% 吞吐 | 训练语义（全局 batch 变化）已确认接受 |
| decoder+vision 编译 | 再 +6% | bf16 融合级数值微差（audit 把关） |
| tok memo | 首 epoch 后再 -13% | 零（真实管线字符串） |

## 6. 方法论沉淀（复用价值最高的部分）

1. **先归因，后开药**：每个"瓶颈"都要至少两个独立手段交叉验证。
   本轮最大的弯路（"通信占 48%"）就是单一工具 + 未察觉的测量扰动。
2. **警惕 profiler 的观察者效应**：sync 打点改变 overlap 行为；
   CUDA event 对是流水线友好的替代品。
3. **消变量实验设计**：固定 batch（消除数据方差）、自校准 + 对拍保险丝（正确性自证）、
   canary 探针（链路验证与业务解耦）。
4. **数值不变的改动永远优先于数值会变的**：graphs/memo/算子替换 < compile。
5. **坑要封存成地图**：四连坑每一层都有记录和出路，下次直接从第 4 层继续。
6. **优化栈分层记账**：每项优化单独 probe 实测，防止叠加后归因失效。

## 8. 第二轮：py-spy 真凶定位 → flash_attention_2（-40%，生产 v2 采用）

**武器**：py-spy 二进制放 CFS 共享，给训练 pod exec 采样（注意采的是 `train_starvla.py`
子进程而非 accelerate 父进程；用 `--nonblocking` 避开 ptrace 崩溃）。

- 主线程分解：23% `_engine_run_backward`（autograd python 引擎逐节点调度）+
  4.3% OpOverload 派发；compiled_autograd 实验**负结果**（1300 vs 966ms），封存。
- evt 子段再分解：fwd 中 **bkb（QwenVL 模型 forward）≈500ms 占 86%**，
  DiT 前向仅 45ms——推翻"DiT 是前向大头"预设。
- 真凶：transformers 5.3 的 `Qwen3_5VisionAttention` 在 sdpa 路径下是**逐图
  python 循环**（split→sdpa→concat ×144 图/步，`listcomp@1041` 占主线程 11.4% +
  dynamo graph break 恢复点）。flash 路径走 `cu_seqlens` 变长内核无循环。
- **修法**：删掉 `QWen3_5.py` 硬编码 `attn_implementation="sdpa"` 的旧 bug
  （覆盖 yaml 配置），yaml 切 `flash_attention_2`；flash-attn 2.7.4.post1
  wheel 入 CFS（torch2.6 cu124 cp310 abiFALSE）装入共享 venv。
  flash 下跳过 ViT 块编译（varlen `.max()` 取 int 让 dynamo fake-tensor 追踪崩）。
- **效果**：bkb 500→230ms（**-55%**），fwd 590→300ms（-49%），整机 ~1330→**~800ms（-40%）**。
  生产 v2（`orion-h1-prod-v2`）重启后 model=0.87s、吞吐 436 样本/s（= copy18 的 12.8×）。

### 生产 v2 运行实录（单节点 8×H800，50k 步）

| 时间（相对启动） | 现象 |
|---|---|
| +14min | decoder 24 层编译初始化，bs48（vision 编译被 flash 跳过） |
| +60min | loss 1.09→0.68@step60，model=0.88s，grad_norm 健康 |
| step 8280+ | loss 0.013-0.016，grad_norm ~0.5 |
| 功率 | 8 卡 util 100%（一卡周期性 0% 为预处理线程交错），功耗 ~120W/700W TDP——kernel 仍偏小但利用率拉满，模型侧已无可压榨余量 |
| ckpt | `steps_5000_pytorch_model.pt` 2.8GB 正常落盘，可继续 |

**教训**：初始"epoch 步数约 250k"的估算是错读（把总帧数 96M 当步数；
真实构成是 30 步滑窗采样 + 90% split + mixture 权重）。memo 缓慢回填依然
成立（首轮慢路径的副作用收益），但 hit_rate 绝对值（0.0001）不能作为
epoch 进度的证据——以数据停顿消失为准。

## 9. 遗留与下一步

- DiT 编译续命（torch 升级 / Attention 整型 position_ids 改造）→ bwd 剩余 python 墙的主菜
- VL splice 的 `get_rope_index` python 循环（每步 48 次 itertools.groupby）向量化
- H1 数据卫生：5 个人名占位垃圾任务串待清洗
- 4 机多卡的步时放大问题（GC 彩票 × 跨机 allreduce 等最慢者）未单独验证——
  本轮所有单节点优化理论上同样适用于多机
