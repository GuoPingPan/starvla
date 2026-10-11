# UMI PI05 冒烟测试记录

> 测试日期：2026-08-05
> 测试目标：验证 UMI · PI05 训练流水线能否端到端跑通（配置环境 → 拉代码 → 拉数据 → 拉预训练 → 训练 → 上传）
> 最终结果：**✅ 两次冒烟均成功**，权重成功上传到 `bos://benchmark-liberoplus/Alex/pi_checkpoints/`

---

## 1. 测试背景与目标

计划跑的流水线命令（train.sh 六段，commit `c06b17c`）：

```bash
BATCH=2 bash scripts/train.sh \
  examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml \
  c06b17c0afaf6cdc800f339b2d26c34802ed7586 \
  --local
```

本次共做**两次冒烟测试**：

| 次数 | 机器 | 方式 | 配置 | 步数 |
| --- | --- | --- | --- | --- |
| 第一次 | 本机（A100×1） | 工作区代码 + 工作区 venv 直跑 launcher | `run_umi_shenzhen_relative_6d_v2_forceclose.yaml` | 3 |
| 第二次 | 新开发机 `aibox-r9f71e8a2473` | train.sh 完整六段 | `run_umi_shenzhen_relative_6d_v2.yaml` | 100 |

---

## 2. 环境与前置检查

### 2.1 预检结果（跑之前先检查）

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| GPU | ✅ | NVIDIA A100-SXM4-80GB ×1 |
| 共享盘 /mnt/cfs | ✅ 已挂载 | 数据/权重/代码/环境都在上面 |
| 数据集 | ✅ | `/mnt/cfs/data/z-umi/data/20260726133821_lerobot_7d6be9ce_noep58_forceclose/` 已就位 |
| 预训练权重 | ✅ | `/mnt/cfs/data/pretrain_models/.../model.safetensors`（16GB）已就位 |
| BOS 凭证 | ❌ 缺失 | 需写入 `/root/.bosenv` 或环境变量 |
| SwanLab 凭证 | ❌ 缺失 | 不影响（`--local` 模式免登录） |
| /root/data、/root/code 软链 | ❌ 不存在 | train.sh 段 1 会自动建 |

### 2.2 关键路径速查

| 资源 | 逻辑路径（代码写死） | 真实位置（共享盘） |
| --- | --- | --- |
| 代码 + 环境 | `/root/code/repo_<sha>/` + `.venv/` | `/mnt/cfs/code/repo_<sha>/` |
| 数据集 | `/root/data/z-umi/data/<id>/` | `/mnt/cfs/data/z-umi/data/<id>/` |
| 预训练权重 | `/root/data/pretrain_models/` | `/mnt/cfs/data/pretrain_models/` |
| 训练产物 | `/root/data/model_outputs/<RUN_ID>/` | `/mnt/cfs/data/model_outputs/<RUN_ID>/` |

---

## 3. 遇到的报错与解决过程

### 3.1 BOS 凭证缺失

**现象**：直接跑 train.sh 在段 2 立刻失败——`bos_io.py make_client()` 检测不到 `BOS_ACCESS_KEY_ID` 直接 `exit(2)`。

**解决**：把 AK/SK 写入 `/root/.bosenv`（train.sh 段 1 会 `set -a; . /root/.bosenv` 自动 source）：

```bash
chmod 600 /root/.bosenv
```

> ⚠️ 安全提醒：凭证在对话中出现过，测试完建议去 BOS 后台轮换。

### 3.2 代理冲突（uv / boto3 连 7897 被拒）——最核心的坑

**现象**（两种形态）：

```
# uv 拉 boto3
error: Request failed after 3 retries
  Caused by: Failed to fetch: `https://mirrors.aliyun.com/pypi/simple/boto3/`
  Caused by: tunnel error: ... tcp connect error → Connection refused (os error 111)

# boto3 连 BOS
botocore.exceptions.ProxyConnectionError: Failed to connect to proxy URL: "http://127.0.0.1:7897"
```

**根因**：本机存在**两套代理变量**，大小写指向不同端口：

| 变量 | 值 | 端口状态 |
| --- | --- | --- |
| `http_proxy` / `https_proxy`（小写） | `127.0.0.1:64500` | ✅ 活着（反向 SSH 隧道） |
| `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `all_proxy` | `127.0.0.1:7897` | ❌ 死（Codex 隧道未启用） |

不同工具读不同套别：curl 读小写（64500 能通）、uv/boto3 读大写（7897 被拒）。且 `~/.profile` 登录时通过 `~/.codex_proxy_env.sh` 还会把**大小写全部**覆盖成 7897。

**解决（临时）**：跑命令时清掉坏的大写代理：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  bash scripts/train.sh ...
```

**解决（永久）**：在 `~/.bashrc` 末尾追加（`.bashrc` 在 `.profile` 之后加载，可覆盖）：

```bash
export http_proxy=http://127.0.0.1:64500
export https_proxy=http://127.0.0.1:64500
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

**验证**：`bash -lc 'curl -sI https://www.baidu.com'` → `HTTP/1.1 200 OK`。

> 完整原理（含 LocalForward vs RemoteForward）见 `docs/zerith_docs/infra/proxy_issue_fix.md`。

### 3.3 共享盘 venv 的 python 解释器断链

**现象**：段 2 复用 venv 后报：

```
error: Failed to inspect Python interpreter from active virtual environment at `.venv/bin/python3`
  Caused by: Broken symlink at `.venv/bin/python3`, was the underlying Python interpreter removed?
```

**根因**：共享盘上 `repo_<sha>/.venv` 是在别的机器创建的，`bin/python` 软链写死指向那台机器的 `/root/.local/share/uv/python/cpython-3.10.19-.../bin/python3.10`；新机器 uv 没装 3.10 → 断链。检查发现共享盘 **12 个 repo 的 venv 全断链**。

**解决**：用 uv 装回**同版本** 3.10.19（uv 目录命名是确定性的，装完软链自动复活，torch/deepspeed/boto3 全复用，无需重建）：

```bash
uv python install 3.10.19
```

> 前提：网络能到 github（走好代理即可）；系统 apt 的 Ubuntu 24.04 只有 python 3.12，没有 3.10。

### 3.4 swanboard 未装（非阻塞）

**现象**：

```
ERROR: ImportError: Please install swanboard to use 'local' mode: pip install 'swanlab[dashboard]'
```

**影响**：`swanlab watch` 本地面板打不开，但**训练数据照常保存**到 `swanlog/`，不阻塞训练。

**修复（可选）**：

```bash
/root/code/repo_c06b17c0/.venv/bin/uv pip install "swanlab[dashboard]"
```

---

## 4. 第一次冒烟：工作区直跑（forceclose，3 步）

> 背景：当时共享盘 venv 断链 + github 被墙，train.sh 完整路径走不通，改用**工作区代码 + 工作区 venv（python 3.11）**直跑 launcher 验证训练代码本身。

### 4.1 命令

```bash
cd /root/workspace/starvla
nohup env ACCEL_BIN=/root/workspace/starvla/.venv/bin/accelerate \
  REPO_ROOT=/root/workspace/starvla \
  BATCH=2 MAX_STEPS=3 SAVE_EVERY=2 SWANLAB_MODE=local WANDB_PROJECT=starvla \
  bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh \
    examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2_forceclose.yaml \
  > /tmp/opencode/train_local.log 2>&1 &
```

> 与原始命令的差异：不下载代码（用工作区 c06b17c）、用工作区 venv（3.11）、1 张 GPU、3 步。

### 4.2 日志时间线（关键行）

```
DeepSpeed TorchBackend(NCCL) → DistributedType.DEEPSPEED, Num processes: 1, Device: cuda:0
Mixed precision type: bf16 | ds_config: ZeRO-2
"VLA Training :: Warming Up"
LR Group base: lr=5e-05, num_params=812
📝 Full config saved at .../forceclose_3step/0805_174820/config.full.yaml
📦 loading checkpoint: /root/data/pretrain_models/pi05_base_starvla/.../model.safetensors
100%|...| 3/3 [00:27~00:46, ~11~15s/it, model_times≈0.6~1.2]
Training complete. Final model saved at .../final_model
"... and that's all, folks!"
```

### 4.3 结果

| 环节 | 结果 |
| --- | --- |
| PI05 模型构建 + 加载 16GB 预训练权重 | ✅ |
| norm 统计在线计算 | ✅ |
| 3 步训练（bf16） | ✅ |
| checkpoint 保存（SAVE_EVERY=2 → steps_2） | ✅ |
| final_model 保存 + SwanLab local | ✅ |
| 进程正常退出 | ✅ |

产物：`/root/data/model_outputs/run_umi_shenzhen_relative_6d_v2_forceclose_3step/0805_174820/`
（`checkpoints/steps_2_pytorch_model.pt` 8.3GB、`final_model/pytorch_model.pt` 8.3GB、config、swanlog）

### 4.4 直跑方式的路径对照（环境 / 预训练 / 数据 / 输出权重）

> 适用范围：**launcher 直跑工作区代码**（绕开 train.sh 段2/4 的本地冒烟方式），以 `run_umi_shenzhen_relative_6d_v2.yaml` 为例。所有路径来自 yaml + launcher 默认值。

| 内容 | 逻辑路径 | 真实位置（共享盘） |
| --- | --- | --- |
| 运行环境（venv） | `/root/workspace/starvla/.venv` | 本机本地盘（工作区，python 3.11） |
| π0.5 预训练权重 | `/root/data/pretrain_models/pi05_base_starvla/pi05_base_starvla/fp32/model.safetensors` | `/mnt/cfs/data/pretrain_models/...` |
| 分词器 | `/root/data/pretrain_models/paligemma-3b-pt-224/tokenizer.model` | `/mnt/cfs/data/pretrain_models/...` |
| 训练数据集 | `/root/data/z-umi/data/20260726133821_lerobot_7d6be9ce_noep58/` | `/mnt/cfs/data/z-umi/data/...` |
| 输出权重（最终） | `/root/data/model_outputs/<RUN_ID>/<时间戳>/final_model/pytorch_model.pt` | `/mnt/cfs/data/model_outputs/...` |
| 中途 checkpoint | `/root/data/model_outputs/<RUN_ID>/<时间戳>/checkpoints/steps_*.pt` | `/mnt/cfs/data/model_outputs/...` |

**来源说明**：

- **运行环境**：`ACCEL_BIN=$PWD/.venv/bin/accelerate` 指定用工作区 venv（区别于 train.sh 用的共享盘 `/root/code/repo_<sha>/.venv`）。
- **预训练权重 / 分词器**：来自 yaml `trainer.pretrained_checkpoint` 和 `framework.tokenizer.model_path`。
- **数据集**：来自 yaml `data_root_dir=/root/data/z-umi/data` + `data_mix=shenzhen_relative_6d_2_nooutlier`（对应目录 `20260726133821_lerobot_7d6be9ce_noep58/`）。
- **输出权重**：RUN_ID = `<yaml文件名>_<MAX_STEPS>step`，`run_root_dir=/root/data/model_outputs/`（launcher 默认，覆盖 yaml 里的 `/nix/data/model_outputs`）。本次 `MAX_STEPS=3` → RUN_ID=`run_umi_shenzhen_relative_6d_v2_3step`。

> 所有 `/root/data`、`/root/code` 都是软链 → 真实位置在 `/mnt/cfs` 共享盘（train.sh 段 1 自动建）。

---

## 5. 第二次冒烟：新开发机 train.sh 全链路（v2，100 步）

> 环境问题修复后（3.2 代理、3.3 venv），在新开发机 `aibox-r9f71e8a2473-5bc7658477-l628v` 上走通 **train.sh 完整六段**。

### 5.1 实际执行的命令

```bash
rm -rf ~/code ~/data
cd /root/workspace/starvla
export BOS_ACCESS_KEY_ID=<AK>
export BOS_SECRET_ACCESS_KEY=<SK>
export SWANLAB_API_KEY=<key>

BATCH=2 MAX_STEPS=100 SAVE_EVERY=10 \
UPLOAD_DIR="bos://benchmark-liberoplus/Alex/pi_checkpoints/" \
bash scripts/train.sh \
    examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml \
    c06b17c0afaf6cdc800f339b2d26c34802ed7586 \
    --local
```

> 注：首次因代理问题失败；修复 3.2 后重跑成功。修复后的终端里**无需** `env -u ...`。

### 5.2 成功运行的日志关键行（分析）

| 阶段 | 日志证据 |
| --- | --- |
| 段2 代码 | `>> [段2] 复用已有环境 /root/code/repo_c06b17c0/.venv` |
| 段3 数据 | `data_mix = shenzhen_relative_6d_2_nooutlier`；本地派生数据跳过（无 BOS 地址，共享盘已有） |
| 段4 预训练 | `download_dir: 0 个文件`（全部已存在，幂等） |
| 分布式 | `>> 单机模式: num_processes=1`；`Mixed precision type: bf16`；ZeRO-2 |
| 数据集 | `[norm-stats] 命中缓存 .../stats_gr00t.json`；train 335/373 轨迹、eval 38/373 |
| 模型 | `3616.758M Total, 3616.758 Trainable`（未冻结） |
| 训练 | `3/3 [00:26] model_times≈0.73~1.19`；`Checkpoint saved at .../steps_2` |
| 完成 | `Training complete. Final model saved at .../final_model` |
| 上传 | `>> [上传] ... -> bos://benchmark-liberoplus/Alex/pi_checkpoints/` |

### 5.3 上传到 BOS 的结果

目标：`bos://benchmark-liberoplus/Alex/pi_checkpoints/`（train.sh 段 6 通过 `UPLOAD_DIR` 覆盖默认的 `Orion/ckpts/<RUN_ID>/`）。

上传布局：
```
bos://benchmark-liberoplus/Alex/pi_checkpoints/<MMDD_HHMMSS>/
├── config.full.yaml / config.yaml / dataset_statistics.json
├── checkpoints/steps_*.pt            （8.3GB）
└── final_model/pytorch_model.pt      （8.3GB，最终权重）
```

验证命令：
```bash
bcecmd bos ls "bos:/benchmark-liberoplus/Alex/pi_checkpoints/"          # bcecmd 单斜杠
# 或 boto3：bos:// 双斜杠
uv run --no-project --with boto3 python scripts/bos_io.py ls "bos://benchmark-liberoplus/Alex/pi_checkpoints/0805_203403/" -r
```

---

## 6. 测试结论与遗留事项

### 6.1 结论

1. **训练代码全链路可跑**——模型/数据/训练循环/checkpoint/SwanLab/BOS 上传全部正常。
2. 第二次冒烟走的是 **train.sh 完整六段**（从 BOS 拉代码→复用 venv→数据/权重幂等→训练→上传自定义 BOS 目录）。
3. 数据集 `shenzhen_relative_6d_2_nooutlier`：train 335/373 轨迹、eval 38/373；norm 统计命中缓存，启动快。

### 6.2 遗留事项

| 事项 | 说明 | 建议 |
| --- | --- | --- |
| 安全 | BOS AK/SK 和 SWANLAB key 曾出现在对话/本地文件/`启动命令.md` | **尽快轮换密钥** |
| swanboard | `--local` 模式下 `swanlab watch` 面板打不开 | venv 里装 `swanlab[dashboard]` |
| 冒烟产物 | `run_umi_shenzhen_relative_6d_v2_3step`（8.3GB×2）还在共享盘 | 确认后可删 |
| `rm -rf ~/code ~/data` | 只删软链，共享盘数据无损 | train.sh 段 1 会自动重建软链 |

---

## 7. 训练启动命令速查（新开发机）

### 7.1 正式版（全量训练）

```bash
BATCH=2 \
UPLOAD_DIR="bos://benchmark-liberoplus/Alex/pi_checkpoints/" \
bash scripts/train.sh \
    examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml \
    c06b17c0afaf6cdc800f339b2d26c34802ed7586
```

（默认 20000 步 / 每 5000 存；`UPLOAD_DIR` 可去掉以传回默认 `Orion/ckpts/<RUN_ID>/`）

### 7.2 冒烟版（小步数快速验证）

```bash
BATCH=2 MAX_STEPS=3 SAVE_EVERY=2 \
UPLOAD_DIR="bos://benchmark-liberoplus/Alex/pi_checkpoints/" \
bash scripts/train.sh \
    examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml \
    c06b17c0afaf6cdc800f339b2d26c34802ed7586 \
    --local
```

看到 `Training complete. Final model saved` 即链路正常，再放行全量。

### 7.3 分布式参数

launcher 已内置多机支持（`run_umi_pi05_relative_6d.sh` 第 93-112 行）：

| 场景 | 需要设置的变量 |
| --- | --- |
| 单机多卡 | 什么都不用设，`NUM_GPUS` 默认取全部 GPU |
| 多机多卡 | `NUM_MACHINES=N`、`MACHINE_RANK=0/1/...`、`MAIN_PROCESS_IP=主节点IP`（每节点跑同命令） |
| SLURM 集群 | `MACHINE_RANK` 自动取 `$SLURM_NODEID`，`MAIN_PROCESS_IP` 自动取 `$MASTER_ADDR` |

多机示例：
```bash
# 主节点（node 0）
NUM_MACHINES=2 MACHINE_RANK=0 MAIN_PROCESS_IP=<主节点IP> BATCH=2 \
  bash scripts/train.sh <yaml> <commit>
# 从节点（node 1）
NUM_MACHINES=2 MACHINE_RANK=1 MAIN_PROCESS_IP=<主节点IP> BATCH=2 \
  bash scripts/train.sh <yaml> <commit>
```

> 说明：训练跑在 **train.sh 段2 下载的 `/root/code/repo_<sha>` 代码 + 其 `.venv`**；`scripts/train.sh` 及 `bos_io.py`/`download_data.py` 从你 `cd` 的目录读取，所以路径要对。

---

## 8. 相关文档

- 流水线详解：`docs/zerith_docs/infra/train_sh_run_flow.md`
- Bash 语法速学：`docs/zerith_docs/guides/bash_basics.md`
- 代理排查与修复：`docs/zerith_docs/infra/proxy_issue_fix.md`
- 项目总结：`docs/zerith_docs/guides/StarVLA_项目总结.md`
