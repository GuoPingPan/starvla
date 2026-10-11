# 百舸 (AIHC) 任务队列训练使用指南

> **一句话**：本机用 `scripts/aihc/launch_train.sh` 把训练提交到百度百舸 (AIHC) H800 队列；任务容器内由
> `scripts/train.sh` 执行"拉代码 → 拉数据 → 拉预训练 → 训练 → 上传产物"六段流水线；训练代码不是本地的，
> 而是 GitLab CI 打包上传到 BOS 的 commit zip。
>
> 相关文档：train.sh 内部逐行详解见 [train_sh_run_flow.md](train_sh_run_flow.md)；训练代码来源
> (commit ↔ BOS zip) 见 [ci_and_training_source.md](../workflow/ci_and_training_source.md)；ckpt 取回见
> [bos_ckpt.md](bos_ckpt.md)。

---

## 1. 端到端链路

```
你的本机 (装 aihc CLI)
  │  bash scripts/aihc/launch_train.sh --run-id X --yaml <yaml> ...
  │  参数化渲染 job JSON → aihc job create -f job.json -p aihc-serverless -q <queue>
  ▼
百度百舸 AIHC (H800 队列, RDMA, CFS 挂 /mnt/cfs)
  │  任务容器执行生成的命令:
  │    bash /mnt/cfs/code/train.sh <yaml> <commit>   ← 提交时从仓库同步到 CFS, 与镜像解耦
  ▼
scripts/train.sh 六段流水线 (CFS 版本)
  ├─ 段1  配置环境   双云凭证 / /root/data→/mnt/cfs 软链 / NCCL+RDMA 自愈 / uv·compile 缓存
  ├─ 段2  拉代码     按 commit 从 bos://umi-gripper/starvla_res/ 下 CI zip
  │                 → 解压 /root/code/repo_<sha>/ + uv sync 建 venv (flock 并发安全)
  ├─ 段3  拉数据     download_data.py 按 yaml 的 data_mix 从 BOS 拉数据集 (幂等)
  ├─ 段4  拉预训练   Orion/pretrained/ → /root/data/pretrain_models/ (幂等)
  ├─ 段4.5 (可选)   COMPILE_WARMUP=1 时单卡 3 步预热产出 torch.compile 共享缓存
  ├─ 段5  训练       复用 run_umi_pi05_relative_6d.sh launcher → accelerate + DeepSpeed ZeRO-2
  └─ 段6  上传产物   整个 run 目录 → bos://umi-gripper/Orion/ckpts/<RUN_ID>/ (仅成功后)
  ▼
产物取回 (本机): scripts/bos_ckpt.sh download <RUN_ID> [timestamp]
```

两个容易误解的点：

1. **训练代码 ≠ 提交机上的代码**。容器里跑的 `train.sh` 来自 **CFS**（`launch_train.sh` 提交时把仓库
   当前 `scripts/train.sh + cloud_io.py` 同步到 `/mnt/cfs/code/`，与镜像无关），它段 2 按 `--commit`
   从 BOS 拉指定 commit 的 CI zip，之后数据下载/训练都在那份代码上跑。
   所以改了本地代码必须先 `git push` 等 CI 打包（1-2 分钟）才能训到。
2. **多机时数据只在 rank0 节点下载**，其余节点轮询共享盘 sentinel 等待（产物落 CFS，全节点可见）。

---

## 2. 前置条件

| 项 | 说明 |
| --- | --- |
| aihc CLI | 提交机上已装并 `aihc config` 配好凭证 |
| BOS 凭证 | 容器内 `/root/.bosenv`（`BOS_ACCESS_KEY_ID` / `BOS_SECRET_ACCESS_KEY`）或环境变量 |
| SwanLab (可选) | `--swanlab-key` 注入，或容器内 `/root/.swanlab_env`；不注入则 `--local` 本地模式 |
| CI 产物 | 训练 commit 必须已 push 且 CI 已把 zip 传到 `bos://umi-gripper/starvla_res/`（查法见 [ci_and_training_source.md](../workflow/ci_and_training_source.md) §5） |
| CFS | 任务 JSON 里 datasource 自动挂 `cfs-pVlRwhavQO` → `/mnt/cfs`，无需手动 |

---

## 3. 快速开始

```bash
# ① 正式训练: 4 机 × 8 卡 50k 步 (镜像 base_trian:0819new, H800)
bash scripts/aihc/launch_train.sh \
  --run-id h1_d18480a6_qwenpi_v3_first \
  --yaml examples/data-analyze/train_files/run_h1_d18480a6_qwenpi_v3.yaml \
  --replicas 4 --max-steps 50000 --save-every 5000

# ② 单机短跑/调试
bash scripts/aihc/launch_train.sh --run-id smoke_<名> \
  --yaml <yaml路径> --replicas 1 --gpus 8 --max-steps 200 --save-every 100 \
  --swanlab-key "$SWANLAB_API_KEY"

# ③ canary 探针: 1 机 1 卡 30s 验证队列/镜像/挂盘链路通不通
bash scripts/aihc/launch_train.sh --run-id canary --command "hostname; nvidia-smi -L; sleep 30" \
  --replicas 1 --gpus 1 --name orion-launcher-canary

# ④ 只渲染 job JSON 不提交 (改字段/排错用)
bash scripts/aihc/launch_train.sh --run-id X --yaml <yaml> --dry-run
```

训练提交后：控制台/AIHC CLI 看任务状态与日志；训练曲线看 SwanLab（注入了 key）或任务日志；
产物在 `bos://umi-gripper/Orion/ckpts/<RUN_ID>/<时间戳>/`。

---

## 4. launch_train.sh 参数全表

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--run-id ID` | 必填* | RUN_ID = 模型输出目录名，也是云端 ckpts 目录名 |
| `--yaml PATH` | 必填* | 训练 yaml。**必须存在于 `--commit` 对应代码里的相对路径**（见 §5 坑 1） |
| `--command STR` | — | 完全自定义命令（与 `--yaml` 二选一，canary 用） |
| `--name NAME` | `<run_id前40>-<MMdd_HHmmss>` | AIHC 任务名（小写） |
| `--replicas N` | 1 | 机器数（多机自动走 rank0 下载数据 + sentinel 同步） |
| `--gpus N` | 8 | 每机 GPU 数 |
| `--gpu-type T` | `baidu.com/h800_80g_cgpu` | GPU 型号 |
| `--batch N` | 32 | 每卡 batch（透传 BATCH） |
| `--workers N` / `--prefetch N` | 8 / 8 | dataloader workers / prefetch_factor |
| `--max-steps N` | 200 | 训练步数（**默认 200 是冒烟值，正式训练必传**） |
| `--save-every N` | 100000 | 存 ckpt 间隔（**默认 10 万 = 不中途存，长任务建议显式设小**） |
| `--grad-accum N` | 1 | 梯度累积 |
| `--commit HASH` | `3c768d9` | train.sh 的 COMMIT 参数（CI zip 的 commit 前缀） |
| `--local` | 关 | 追加 `--local`（SwanLab 本地模式） |
| `--swanlab-key KEY` | 读 `$SWANLAB_API_KEY` | 注入 SwanLab 凭证 |
| `--priority P` | normal | low / normal / high |
| `--cpu N` / `--memory N` | 175 / 2767 | 固定值照抄正式任务规格；自行缩放会产生非标准值导致调度匹配不到节点 |
| `--env K=V` | — | 追加环境变量，可多次（如 `--env STARVLA_COMPILE=max-autotune`） |
| `--dry-run` | 关 | 只打印 job JSON |

任务侧固定项（改需编辑脚本）：`POOL=aihc-serverless`、`QUEUE=aihcq-gwzuewk6egoh`、
`IMAGE=ccr-2owfeef4-vpc.cnc.bj.baidubce.com/orion/base_trian:0819new`、RDMA 开启、
CFS 挂 `/mnt/cfs`、hang-detection 容错（`HANG_LOG_TIMEOUT_MIN` 默认 8 分钟，冷 torch.compile
静默超 8 分钟会被误判挂起——首跑/冷缓存用 `--env HANG_LOG_TIMEOUT_MIN=30` 放宽）。

---

## 5. 生成的任务命令 & 已知坑

`--yaml` 模式下任务容器内实际执行的命令（`launch_train.sh` 渲染；提交时脚本先把仓库当前
`scripts/train.sh + cloud_io.py` 原子同步到 `/mnt/cfs/code/`，可用 `TRAIN_SH_DIR` 覆盖目标目录）：

```bash
export <你的 --env 与 SWANLAB_KEY/MAX_STEPS/GRAD_ACCUM>
CLOUD=bos NUM_GPUS=<N> BATCH=<N> ... \
  bash /mnt/cfs/code/train.sh <yaml> <commit> [--local]
```

**坑 1 — yaml 路径相对"COMMIT 拉下来的 repo"解析**。train.sh 段 2 会 `cd repo_<commit解出的sha>`，
段 3 起 `python download_data.py <yaml>` 的相对路径在这份 repo 里找。所以 `--yaml` 传的路径必须
**在 `--commit` 那份代码里存在**；新增的 yaml 要先 push 进那个 commit。

**坑 2 —（已解决，2026-09-21）bootstrap 与镜像解耦**。此前容器跑的是镜像内置旧版 train.sh
（BOS-only、不认 `CLOUD=oss`、默认路径还是 z_umi），新版特性对 bootstrap 段不生效。
现改为 CFS 上的 train.sh（提交时自动同步），镜像只提供运行时（uv / python / GPU / CFS 挂载），
旧镜像也能跑新流水线；曾为旧镜像临时加的旧路径兼容 shim 已随之删除。

**坑 3 — 训练失败不上传产物**。train.sh `set -e`：段 5 训练非零退出 → 段 6 上传不执行，
中间 ckpt 只在共享盘。见 §7 优化建议 ①。

**坑 4 — `--commit` 默认 `3c768d9` 是写死的旧值**。不传就训练旧代码；训练新代码务必显式传
（或先 `latest`——注意 `latest` 取的是 `starvla_res/` 最新 zip，可能是别人的分支）。

**坑 5 — SAVE_EVERY 默认 100000**。launch_train.sh 默认不中途存 ckpt；多机长任务中途被抢占/
挂掉则全部损失。正式训练显式 `--save-every 5000`。

---

## 6. 容器内 train.sh 关键可覆盖变量

提交时用 `--env K=V` 注入（launch_train.sh 已透传的除外）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `BATCH` / `MAX_STEPS` / `SAVE_EVERY` | 10 / 20000 / 5000 | train.sh 默认；launcher 参数 > yaml |
| `NUM_GPUS` | 全部可见卡 | 平台注入 `NPROC_PER_NODE` 时优先 |
| `RUN_ID` | `<yaml名>_<MAX_STEPS>step` | 产物目录名 |
| `UPLOAD_DIR` | `bos://.../ckpts/<RUN_ID>/` | 覆盖上传目标 |
| `COMPILE_WARMUP=1` | 关 | 多卡前先单卡 3 步产出编译缓存，防 max-autotune 与 NCCL 互等死锁 |
| `STARVLA_COMPILE` | — | torch.compile 模式（`max-autotune-no-cudagraphs` 较稳） |
| `HANG_LOG_TIMEOUT_MIN` | 8 | hang-detection 无日志超时（分钟） |
| `STARVLA_SYNTHETIC=1` | 关 | 跳过数据下载用假数据压测链路 |
| `DL_WAIT_MIN` | 60 | 多机 worker 等 rank0 下载数据的超时 |
| `SWANLAB_ENV_FILE` | `/root/.swanlab_env` | SwanLab 凭证文件（可指共享盘共用） |

---

## 7. 优化建议（按优先级）

1. **失败也上传产物（高，小改动）**：train.sh 段 5 改为 `bash launcher ... || TRAIN_RC=$?`，
   `trap` EXIT 时无论成败都跑段 6（失败时上传到 `ckpts/<RUN_ID>_failed/`）。避免几天训练中断后
   只能上节点手工捞共享盘。
2. **打通断点续训（高）**：AIHC 任务被抢占/失败重提后，目前 `is_resume=false` 从头训。可让
   train.sh 检测 `RUN_ROOT_DIR/<RUN_ID>/` 已有 checkpoints 时自动设 `trainer.is_resume=true`，
   配合 `--save-every` 调小，长任务可断点续跑。
3. **~~bootstrap 与 zip 版本不一致~~（已落实 2026-09-21）**：train.sh 改为提交时同步到 CFS 执行
   （见 §5 坑 2），与镜像解耦，一处维护。
4. **launch_train.sh 生成命令清理（部分落实 2026-09-21）**：无效 `cd /root/code/repo_*` 两行已删；
   待办：`--commit` 默认改 `latest` 并在提交时回显解析出的真实 sha；`--save-every` 默认改 5000。
5. **提交前预检（中）**：加 `--preflight`：本地校验 yaml 存在于该 commit 的 zip、data_mix 各数据集
   有 BOS URI、预训练路径存在，避免任务排到队、起容器后才发现配置错（每次浪费排队时间）。
6. **任务状态速查命令（低）**：补 `launch_train.sh status/list/logs` 子命令封装 `aihc job ...`，
   免去记 CLI 语法；顺带把 POOL/QUEUE/IMAGE 提成 `AIHC_*` 环境变量可覆盖。
7. **SwanLab 凭证共享盘化（低）**：容器统一 `SWANLAB_ENV_FILE=/mnt/cfs/.swanlab_env`，
   提交时不再逐任务传 key。

---

## 8. 产物取回

```bash
# 列出某 RUN_ID 的所有时间戳
./scripts/bos_ckpt.sh list <RUN_ID>
# 取最新时间戳的 final 权重到 ~/.models/<RUN_ID>/
./scripts/bos_ckpt.sh download <RUN_ID>
# 取指定时间戳的第 20000 步 ckpt
./scripts/bos_ckpt.sh download <RUN_ID> <MMdd_HHmmss> 20000
```

详见 [bos_ckpt.md](bos_ckpt.md)。

---

## 9. 排错速查

| 症状 | 原因/做法 |
| --- | --- |
| 段 2 报 "没拉到源码 zip" | commit 没 push / CI 没跑完 / CI 凭证只传了另一朵云；查法见 ci_and_training_source.md §5 |
| 起容器不久被杀, 日志停在编译 | hang-detection 误判：`--env HANG_LOG_TIMEOUT_MIN=30` |
| 多机 allreduce 慢 ~2x | 容器缺 libibverbs，train.sh 已自愈；确认日志有 "[RDMA] libibverbs 就绪" |
| data_mix 报不在 DATASET_NAMED_MIXTURES | yaml 在 commit 代码里找不到/太旧 → 确认坑 1 |
| 多机 worker 等数据超时 | rank0 下载慢，`DL_WAIT_MIN` 调大；或数据集缺 BOS URI 需预置共享盘 |
| max-autotune 多卡死锁 | `COMPILE_WARMUP=1 --env STARVLA_COMPILE=...` 先单卡预热 |
