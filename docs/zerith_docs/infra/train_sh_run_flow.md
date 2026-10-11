# train.sh 训练流水线运行流程详解

> **双云适配（2026-08）**：流水线已支持阿里云 OSS / 百度云 BOS。
> - 云读写统一走 `scripts/cloud_io.py`（`bos_io.py` 保留为兼容 shim）：`oss://` URI 走 oss2，`bos://` 走 boto3+bcecmd。
> - 凭证：OSS 用 `OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET`（或 `/root/.ossenv`，或 PAI DSW 凭证服务 localhost:7002）；BOS 沿用 `BOS_*`。
> - 云选择 auto：OSS 凭证可得优先 OSS；`CLOUD=oss|bos` 强制。
> - 共享盘：`SHARED_DIR` 可覆盖，默认 `/mnt/data`（阿里云 CPFS），回退 `/mnt/cfs`（百度云 CFS）。
> - 数据集跨云镜像：`DATA_URI_REMAP` 把 data_config 里的 `bos://` URI 重映射到 OSS 前缀（OSS 环境下 train.sh 注入默认值）。
> - 默认产物桶：OSS `algo-public0/orion/`，BOS `umi-gripper/Orion/`。
>
> 针对命令：
> ```bash
> BATCH=2 bash scripts/train.sh \
>   examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2_forceclose.yaml \
>   2623e668ccdeddc3d4f1b9686d50ad5b470e563f \
>   --local
> ```

本文档按"从你敲下回车到训练结束上传"的时间线，逐步讲解整条流水线做了什么、每个环节调用哪个脚本/函数、产物落在哪里，以及关键参数是如何覆盖的。

---

## 0. 命令逐段拆解

| 片段 | 含义 | 去处 |
| --- | --- | --- |
| `BATCH=2` | 环境变量覆盖，设置**单卡 batch size = 2** | 被 `train.sh` 读到，最终以 `--datasets.vla_data.per_device_batch_size 2` 覆盖 yaml 里的 `per_device_batch_size: 4` |
| `bash scripts/train.sh` | 主入口脚本 | `scripts/train.sh` |
| `...forceclose.yaml` | 位置参数 `$1` = CONFIG_YAML（训练配置） | 训练/数据/预训练配置来源 |
| `2623e66...63f` | 位置参数 `$2` = COMMIT（源码产物 commit sha，全量 40 位） | 用于从 BOS 拉取该 commit 的源码 zip |
| `--local` | 标志参数 → `export SWANLAB_MODE=local` | SwanLab 走本地模式（无需登录，配合 `swanlab watch` 看面板），而非默认的云端 online |

注意：`--local` 可以在命令任意位置出现（`train.sh` 第 52-61 行会把 `--local` 从参数里摘出来，剩下的两个再当位置参数）。本命令等价于 `SWANLAB_MODE=local bash train.sh <yaml> <commit>`。

---

## 1. 整体架构：一条命令 = 6 段流水线

`scripts/train.sh` 注释里自己写了 6 段（文件头第 7 行）：**配置环境 → 拉代码 → 拉数据 → 拉预训练 → 训练 → 上传全部产物**。

```
┌─ 段1 配置环境 ──┐   ┌─ 段2 拉代码 ──┐   ┌─ 段3 拉数据 ──┐   ┌─ 段4 拉预训练 ──┐   ┌─ 段5 训练 ──┐   ┌─ 段6 上传 ──┐
│ 凭证/路径/ENV   │ → │ zip→repo_<sha> │ → │ 按yaml拉数据   │ → │ pretrained     │ → │ accelerate  │ → │ ckpts       │
└─────────────────┘   └───────────────┘   └───────────────┘   └───────────────┘   └─────────────┘   └─────────────┘
   /root/.bosenv        uv run --with boto3   download_data.py    bos_io get-dir      run_umi_pi05_   bos_io put-dir
   /root/.swanlab_env   bos_io src-zip        (读 yaml + data_   Orion/pretrained/    relative_6d.sh  → Orion/ckpts/
                        uv sync --python 3.10  config.py)          → /root/data/       (内部调           <RUN_ID>/
                                              → /root/data/         pretrain_models    train_starvla.py)
                                                z-umi/data
```

关键点：**训练并不是直接在当前仓库跑**。脚本会先按 commit 从 BOS 拉一个"CI 产出的源码 zip"，解压到共享盘 `/root/data/code/repo_<sha>`（软链指向 `/mnt/cfs/code`），用 `uv` 建独立 venv，然后 `cd` 进那个目录训练。所以当前工作区的代码改动不影响本次训练。

下面逐步展开。

---

## 2. 段 1：配置环境（`train.sh` 第 78-114 行）

1. **BOS 凭证**：如果环境变量里没有 `BOS_ACCESS_KEY_ID`，则 source `/root/.bosenv`（凭据不入库）。
2. **SwanLab 凭证**：如果环境变量里没有 `SWANLAB_API_KEY`，则 source `/root/.swanlab_env`。这解决的是"非交互 bash 拉起时 `~/.zshrc` 不会被 source、拿不到 key"的问题。
3. **BOS 默认参数**：`BOS_ENDPOINT=s3.bj.bcebos.com`、`BOS_BUCKET=umi-gripper`、`BOS_NAMESPACE=Orion`。
4. **路径软链**：确保 `/root/data -> /mnt/cfs/data`、`/root/code -> /mnt/cfs/code`（NFS 共享盘）。本段注释的 mount 部分被注释掉了（第 30-36 行），说明共享盘已挂好。
5. **训练环境变量**：
   - `CUDA_HOME=/usr/local/cuda`（DeepSpeed 需要真实 nvcc）
   - `NCCL_DEBUG=WARN`、`NCCL_BLOCKING_WAIT=1`、`NCCL_ASYNC_ERROR_HANDLING=1`、`NCCL_TIMEOUT=10000`
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（减少显存碎片，防大 batch 下 NCCL 缓冲区 OOM）
   - `TOKENIZERS_PARALLELISM=false`、`HF_HOME=~/.cache/huggingface`、`WANDB_MODE=disabled`

### 顶部可调项（第 62-76 行）

| 变量 | 默认值 | 本命令 | 最终值 |
| --- | --- | --- | --- |
| `CONFIG_YAML` | shenzhen 配置 | 位置参数 | `...run_umi_shenzhen_relative_6d_v2_forceclose.yaml` |
| `COMMIT` | `latest` | 位置参数 | `2623e66...63f` |
| `NUM_GPUS` | `nvidia-smi -L \| wc -l`（所有可见卡） | 未设 | 全部 GPU |
| `BATCH` | 10 | **2** | **2** |
| `MAX_STEPS` | 20000 | 未设 | 20000 |
| `SAVE_EVERY` | 5000 | 未设 | 5000 |
| `PORT` | 29501 | 未设 | 29501 |
| `WANDB_PROJECT` | starvla | 未设 | starvla |

路径常量：`DATA_DIR=/root/data`、`CODE_DIR=/root/code`、`RUN_ROOT_DIR=/root/data/model_outputs`、`PRETRAINED_LOCAL=/root/data/pretrain_models`。

---

## 3. 段 2：按 commit 拉代码并建环境（第 116-150 行）

1. `mktemp` 建唯一临时目录 `$CODE_DIR/.srczip.XXXXXX`（并发安全，多个任务不会互相覆盖下载中的 zip）。
2. 此刻还没建好 python 环境，用 `uv run --no-project --with boto3 python scripts/bos_io.py src-zip <commit> --dest-dir $DL_TMP/` 临时带上 boto3 跑一次下载。
   - `bos_io.py fetch_src_zip`（第 285-325 行）：在 `bos://umi-gripper/starvla_res/` 下列出所有 `starvla_*.zip`，zip 命名规则是 `starvla_<ref>_<ts>_<sha>.zip`，按前缀匹配 commit（短/长 sha 都行，本命令传的是全量 40 位）。若有歧义会报错列出候选。
3. 从 zip 文件名末段解析出**真实 commit sha**：`REAL_SHA=2623e66...63f`。
4. `flock` 串行化"解压 + 建 venv"（同 commit 多个任务排队）：
   - 若 `/root/code/repo_$REAL_SHA/.venv` 已存在 → 直接复用（跳过解压 + uv sync）。
   - 否则 `unzip -q` 到 `repo_$REAL_SHA/`，然后 `uv sync --python 3.10` 建环境。
5. `cd /root/code/repo_$REAL_SHA`，激活 `.venv/bin/activate`。
6. `uv pip install boto3`（pyproject 里 boto3 在 sagemaker extra，默认不装）。
7. `export PYTHONPATH=$PWD` —— **之后所有命令都在解压出的这份代码里跑**。

> 产物路径：`/root/code/repo_2623e66...63f/`（代码）＋ `.venv/`（独立 python 环境）。

### 3.1 段 2 逐行详解（train.sh 第 123-150 行）

**总体目标**：从 BOS 拉"CI 编译好的源码 zip" → 解压到共享盘 `/root/code/repo_<真实sha>/` → 用 uv 建独立 python 环境 → cd 进去激活 → 之后训练全在这份代码上跑（与当前工作区无关）。

#### ① 建目录 + 临时目录（124-126 行）

```bash
mkdir -p "$CODE_DIR"                                             # ①
DL_TMP="$(mktemp -d "$CODE_DIR/.srczip.XXXXXX")"                 # ②
trap 'rm -rf "$DL_TMP" 2>/dev/null || true' EXIT INT TERM        # ③
```

- **① `mkdir -p "$CODE_DIR"`**：确保 `/root/code`（→ `/mnt/cfs/code` 共享盘）存在。`-p` = 递归建、已存在不报错（幂等）。
- **② `DL_TMP="$(mktemp -d "$CODE_DIR/.srczip.XXXXXX")"`**：`mktemp -d` 按模板建**唯一**临时目录，`XXXXXX` 被随机字符替换，返回路径存进 `DL_TMP`。为什么用唯一名字？多任务共享 `/mnt/cfs/code`，固定名会互相覆盖。
- **③ `trap '...' EXIT INT TERM`**：注册信号陷阱——脚本退出(EXIT)/被Ctrl+C(INT)/被kill(TERM)时先执行 `rm -rf "$DL_TMP"`，保证临时目录必被清理。`2>/dev/null` 丢错误，`|| true` 防 `set -e` 误杀。

#### ② 拉源码 zip（127-128 行）

```bash
uv run --no-project --with boto3 python "$BOS_IO" src-zip "$COMMIT" --dest-dir "$DL_TMP/"
```

- `uv run --no-project --with boto3`：临时跑命令，不依赖当前项目（`--no-project`），临时带上 boto3（`--with`）。为什么绕？此刻环境还没建好，但 `bos_io.py` 需要 boto3 连 BOS。
- `python "$BOS_IO" src-zip "$COMMIT"`：`bos_io.py fetch_src_zip` 在 `bos://umi-gripper/starvla_res/` 按 commit 前缀匹配下载 zip 到 `$DL_TMP/`。

#### ③ 找最新 zip + 校验（129-130 行）

```bash
ZIP="$(ls -t "$DL_TMP"/starvla_*.zip 2>/dev/null | head -1)"
[ -n "$ZIP" ] || { echo "ERROR: 没拉到源码 zip (先跑 GitLab CI 上传产物到 starvla_res/)" >&2; exit 1; }
```

- `"$DL_TMP"/starvla_*.zip`：glob 通配符展开成所有匹配 zip；`ls -t` 按时间新→旧；`| head -1` 取最新一份；`2>/dev/null` 吞掉"无匹配"的报错（交给下一步统一处理）。
- `[ -n "$ZIP" ]` 判断非空；为空则 `||` 执行 `{ echo ... >&2; exit 1; }`——**校验失败给明确提示并退出**。`>&2` 把错误打到 stderr。

#### ④ 解析真实 commit sha（131-133 行）

```bash
REAL_SHA="$(basename "$ZIP" .zip | awk -F_ '{print $NF}' | tr 'A-Z' 'a-z')"
REPO_DIR="$CODE_DIR/repo_$REAL_SHA"
```

zip 名格式 `starvla_<ref>_<ts>_<sha>.zip`，末段即真实 sha。管道三步（以 `starvla_dev-main_20260804_2623e66.zip` 为例）：

| 步骤 | 命令 | 输出 |
| --- | --- | --- |
| 1 | `basename "$ZIP" .zip` | `starvla_dev-main_20260804_2623e66` |
| 2 | `awk -F_ '{print $NF}'` | `2623e66`（按`_`切分取末段） |
| 3 | `tr 'A-Z' 'a-z'` | `2623e66`（大写转小写） |

**用真实 sha 命名目录**：不同 commit 的任务各自独立目录，互不覆盖。

#### ⑤ flock 加锁：解压 + 建环境（135-144 行）★ 核心

```bash
(
  flock 9
  if [ -d "$REPO_DIR/.venv" ]; then
    echo ">> [段2] 复用已有环境 $REPO_DIR/.venv (跳过解压 + uv sync)"
  else
    rm -rf "$REPO_DIR" && mkdir -p "$REPO_DIR"
    unzip -q "$ZIP" -d "$REPO_DIR"
    ( cd "$REPO_DIR" && uv sync --python 3.10 )
  fi
) 9>"$CODE_DIR/.repo_$REAL_SHA.lock"
```

- **为什么要锁**：多个任务可能同时用同一个 commit，若同时解压/`uv sync` 到同一目录会互相踩坏。
- **`( ... )`**：子 shell，里面 cd 不影响外面。
- **`9>"$CODE_DIR/.repo_$REAL_SHA.lock"`**：把文件描述符 9 绑到锁文件（锁名带 sha，不同 commit 用不同锁）。
- **`flock 9`**：抢锁——同一时刻只有一个进程持有，其它阻塞排队。同 commit 任务串行，不同 commit 并行。
- **双重检查**：拿到锁后**再查一次** `.venv` 是否存在（避免"两个任务都看到不存在后抢着建"），存在则**复用**（第二次跑同一 commit 秒过），否则 `rm -rf` 清半成品 → `unzip -q` 解压 → `( cd ... && uv sync --python 3.10 )` 子 shell 里建 venv。

#### ⑥ 收尾：清理 + 进目录 + 激活（145-150 行）

```bash
rm -rf "$DL_TMP"; trap - EXIT INT TERM     # ①
cd "$REPO_DIR"                             # ②
. .venv/bin/activate                       # ③
uv pip install --quiet boto3               # ④
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # ⑤
```

- **①**：删临时目录（zip 已解压没用了）；`trap - 信号` **清空**陷阱（目录已删，退出时不必再删）。用 `;` 而非 `&&`：rm 失败也要继续清 trap。
- **② `cd "$REPO_DIR"`**：**从这行起当前目录就是 `/root/code/repo_<sha>`**，后面所有相对路径都相对它。
- **③ `. .venv/bin/activate`**：激活虚拟环境，之后 `python`/`pip`/`accelerate` 都指向 `.venv`。
- **④ `uv pip install --quiet boto3`**：补装 boto3（pyproject 里它在 `sagemaker` 可选依赖组，`uv sync` 默认不装；段 3/4/6 的 bos_io 都需要它）。
- **⑤ `export PYTHONPATH="$PWD:${PYTHONPATH:-}"`**：导出环境变量（子进程可见），把解压代码目录加进模块搜索路径 → 后面 `import starVLA` 就是 import 这份代码。

#### 段 2 完整流程图

```
┌─ mkdir -p $CODE_DIR（确保 /root/code 存在）───────────┐
└──────────────────────────────────────────────────────┘
                     ↓
┌─ mktemp 建唯一临时目录 .srczip.XXXXXX + trap 兜底删除 ─┐
└──────────────────────────────────────────────────────┘
                     ↓
┌─ uv 临时带 boto3 跑 bos_io src-zip <commit> ─────────┐
│   按 commit 从 starvla_res/ 下载 starvla_*.zip        │
└──────────────────────────────────────────────────────┘
                     ↓
┌─ 校验 zip + 解析真实 sha → REPO_DIR=repo_2623e66... ──┐
└──────────────────────────────────────────────────────┘
                     ↓
┌─ flock 锁 → 再查 .venv ──────────────────────────────┐
│   存在？ ──是──→ 复用，跳过解压/装依赖                 │
│   不存在 → rm + unzip + uv sync(py3.10)               │
└──────────────────────────────────────────────────────┘
                     ↓
┌─ 删临时目录、清 trap → cd 进 repo、激活 venv、        ┐
│   补装 boto3、设 PYTHONPATH                          │
└──────────────────────────────────────────────────────┘
                     ↓
            → 进入段 3（拉数据）
```

> **一句话总结段 2**：用"下载到临时目录 → 锁内解压建环境 → 复用 venv"三步，把指定 commit 的源码变成一套随时可跑的训练环境，并发安全、重复任务秒过。

---

## 4. 段 3：拉数据（第 152-155 行）

`python scripts/download_data.py <CONFIG_YAML>` 的逻辑（`download_data.py` 全文 135 行）：

1. 从 yaml 解析 `datasets.vla_data.data_root_dir` 和 `data_mix`：
   - `data_root_dir = /root/data/z-umi/data`
   - `data_mix = shenzhen_relative_6d_2_nooutlier_forceclose`
2. 用 **ast（不 import）** 从 `examples/realRobots/zerith_pi05/train_files/data_registry/data_config.py` 抽取 `DATASET_NAMED_MIXTURES`，取出该 mix 的条目：
   ```python
   "shenzhen_relative_6d_2_nooutlier_forceclose": [
       ("20260726133821_lerobot_7d6be9ce_noep58_forceclose", 1.0,
        "umi_2gripper_relative_6d", ""),   # 第4元素 = bos:// URI，此处为空
   ],
   ```
3. 对每个条目看元组第 4 元素（`bos_uri`）：
   - **本命令的 forceclose 数据集第 4 元素是空串** → 打 `⚠ 无 BOS 路径, 跳过 —— 训练会缺数据` 的警告并跳过。所以这个本地派生数据集（forceclose 是 `build_gripper_forceclose.py` 离线构建的，见 data_config.py 第 174 行注释）**必须已经存在于 `/root/data/z-umi/data/20260726133821_lerobot_7d6be9ce_noep58_forceclose/`**，否则训练时 dataloader 会报找不到数据。
   - 有 bos_uri 的（比如 `shenzhen_relative_6d_2` 里的原始数据）才会走"下载 zip → 解压到 `<data_root_dir>/<dataset_id>/`"。
4. 幂等：目标目录下 `meta/info.json` 已存在则跳过。

> 结论：本命令第 3 段实际是"核对+跳过"，数据需预先放在 `/root/data/z-umi/data/` 下（NFS 共享盘）。

### 4.1 段 3 逐行详解

**总体目标**：按 yaml 的 `data_mix` 找到数据集清单，对每个条目看有没有 BOS 地址——有就下载+解压（幂等），没有就跳过。

> 注意：段 3 在段 2 之后执行，此时已 `cd` 进 `/root/code/repo_<sha>` 并激活 venv，所以这里的 `python` 是解压出的代码环境里的 python，跑的是解压代码里的 `scripts/download_data.py`。

#### 第一步：解析 yaml（`load_yaml_fields`，第 34-45 行）

```python
cfg = OmegaConf.load(config_yaml)
return cfg.datasets.vla_data.data_root_dir, cfg.datasets.vla_data.data_mix
```

从 yaml 读出两个关键字段：

| 字段 | 值 | 作用 |
| --- | --- | --- |
| `data_root_dir` | `/root/data/z-umi/data` | 数据落到哪 |
| `data_mix` | `shenzhen_relative_6d_2_nooutlier_forceclose` | 拉哪些数据集 |

（OmegaConf 加载失败会回退 PyYAML。）

#### 第二步：ast 抽配置（`extract_named_mixtures`，第 48-56 行）

```python
tree = ast.parse(Path(data_config_path).read_text())
for node in tree.body:
    if isinstance(node, ast.Assign) and 目标是 DATASET_NAMED_MIXTURES:
        return ast.literal_eval(node.value)
```

用 Python **ast（抽象语法树）** 解析 `data_config.py`，只提取 `DATASET_NAMED_MIXTURES = {...}` 的字面值，**并不真正 import 执行**——好处是不会触发 `import starVLA` 那套重依赖，解析快又安全。

#### 第三步：找本命令的数据集列表（第 105-109 行）

```python
if data_mix not in mixtures:
    raise SystemExit(f"ERROR: data_mix={data_mix!r} 不在 DATASET_NAMED_MIXTURES。可用: {list(mixtures)}")
```

找不到 mix 会列出所有可用名字。找到了，`entries` 就是：

```python
[
    ("20260726133821_lerobot_7d6be9ce_noep58_forceclose", 1.0, "umi_2gripper_relative_6d", ""),
    ("20260803035455_lerobot_5852395f_noep81_forceclose", 1.0, "umi_2gripper_relative_6d", ""),
]
```

**每个条目是 4 元组**：

| 位置 | 含义 | 本例第 1 条 | 本例第 2 条 |
| --- | --- | --- | --- |
| `[0]` | dataset_id（本地目录名） | `20260726133821_lerobot_7d6be9ce_noep58_forceclose` | `20260803035455_lerobot_5852395f_noep81_forceclose` |
| `[1]` | 采样权重 | 1.0 | 1.0 |
| `[2]` | robot_type（决定变换链） | `umi_2gripper_relative_6d` | 同上 |
| `[3]` | **bos_uri**（下载地址，可为空） | `""`（空） | `""`（空） |

#### 第四步：逐个处理（第 117-127 行）

```python
for entry in entries:
    dataset_id = entry[0]
    bos_uri = entry[3] if len(entry) >= 4 else None
    if not bos_uri:
        print(f"  ⚠ {dataset_id}: 无 BOS 路径 (元组第4元素未配), 跳过 —— 训练会缺数据")
        n_skip += 1
        continue
    ok = download_one_dataset(client, dataset_id, bos_uri, data_root_dir, dry_run=args.dry_run)
```

**本命令实际发生的事：两条全部跳过**——第 4 元素全是空串 → `if not bos_uri` 成立 → 打 `⚠` 警告跳过。**段 3 对你这条命令一行数据都不会下。**

为什么没有 BOS 地址？看 data_config.py 第 166-177 行注释：forceclose 是本地派生数据集（`bar/build_gripper_forceclose.py` 改夹爪列生成的），没传上 BOS。训练前必须**手动确保**它已躺在本地：

```
/root/data/z-umi/data/20260726133821_lerobot_7d6be9ce_noep58_forceclose/
├── meta/
│   ├── info.json          ← 判断"数据是否就位"的标志文件
│   ├── modality.json
│   └── stats_gr00t.json   ← norm 统计缓存（forceclose 已删，训练时在线重算）
└── data/...               ← parquet + 视频
```

#### 若有 bos_uri 会怎么走（`download_one_dataset`，第 71-94 行）

```python
target = Path(data_root_dir) / dataset_id          # /root/data/z-umi/data/<id>
if (target / "meta" / "info.json").exists():       # 幂等：已就位 → 跳过
    print(f"  ✓ 已存在, 跳过: {target}")
    return True
bucket, key = bos_io.parse_bos_uri(bos_uri)        # 解析 bos://bucket/key
target.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:         # 先下到临时目录（防半成品）
    zip_path = Path(tmp) / f"{dataset_id}.zip"
    bos_io.download_file(client, bucket, key, zip_path)   # boto3 下载（断点续传+重试）
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
flatten_if_wrapped(target)                         # zip 自带一层包装目录时把内容提上来
ok = (target / "meta" / "info.json").exists()      # 解压后校验
```

关键设计：
- **先下临时目录再解压**：下载中途失败不会在目标目录留半成品。
- **幂等标志 = `meta/info.json`**：存在即视为已就位，重复跑 train.sh 不重复下载。
- **`flatten_if_wrapped`（第 60-68 行）**：zip 解压出来若是 `xxx/xxx/...` 多套一层，会把子目录内容**提升**到目标目录，保证布局统一是 `<data_root>/<dataset_id>/meta/info.json`。
- `bos_io.download_file`：优先 `bcecmd`，不行走 boto3 流式 + 断点续传 + SIGALRM 看门狗 + 指数退避重试。

#### 收尾输出

```python
print(f"\n完成: {n_ok} 拉取/已存在, {n_skip} 跳过(缺路径)")
if n_skip:
    print("提示: 缺路径的数据集需在 data_config.py 的元组第4位补 bos:// URI")
```

本命令预计输出：

```
data_mix      = shenzhen_relative_6d_2_nooutlier_forceclose
data_root_dir = /root/data/z-umi/data
目标布局     = <data_root_dir>/<dataset_id>/ (LeRobot v3.0)
  ⚠ 20260726133821_lerobot_7d6be9ce_noep58_forceclose: 无 BOS 路径 ..., 跳过 —— 训练会缺数据
  ⚠ 20260803035455_lerobot_5852395f_noep81_forceclose: 无 BOS 路径 ..., 跳过 —— 训练会缺数据

完成: 0 拉取/已存在, 2 跳过(缺路径)
提示: 缺路径的数据集需在 data_config.py 的元组第4位补 bos:// URI
```

> ⚠️ **本命令最大的坑**：段 3 一行数据都不会下。如果 `7d6be9ce_noep58_forceclose` 没有预置在本地，段 5 训练时 `build_dataloader` 会直接报"找不到数据集"而崩溃。跑这条命令前务必确认 `/root/data/z-umi/data/20260726133821_lerobot_7d6be9ce_noep58_forceclose/meta/info.json` 存在。

#### 段 3 知识点小结

| 知识点 | 出处 |
| --- | --- |
| 环境变量 `$DL_DATA` / `$CONFIG_YAML` | train.sh 第 23、63 行 |
| python 脚本接收位置参数 | download_data.py 第 100-101 行 argparse |
| OmegaConf 读 yaml + `.datasets.vla_data.xxx` 取字段 | 第 38 行 |
| ast / literal_eval 安全解析（不 import） | 第 48-56 行 |
| 元组解包 `entry[0]` / `entry[3]`，`len(entry) >= 4` 容错 | 第 121 行 |
| 幂等：`(target / "meta" / "info.json").exists()` | 第 75 行 |
| `tempfile.TemporaryDirectory()` 临时目录上下文 | 第 85 行 |
| zipfile 解压 | 第 89-90 行 |
| 目录扁平化（shutil.move） | 第 60-68 行 |

---

## 5. 段 4：拉预训练（第 157-162 行）

```bash
python scripts/bos_io.py get-dir "bos://umi-gripper/Orion/pretrained/" \
    /root/data/pretrain_models --exclude '.hfd/*'
```

递归下载 `Orion/pretrained/` 下全部文件到 `/root/data/pretrain_models/`（跳过 `.hfd/*`）。已存在且大小一致的文件跳过（幂等，16GB 预训练不重复下）。本命令的 yaml 依赖这里的两样东西：

- `framework.tokenizer.model_path = /root/data/pretrain_models/paligemma-3b-pt-224/tokenizer.model`（PaliGemma SentencePiece tokenizer）
- `trainer.pretrained_checkpoint = /root/data/pretrain_models/pi05_base_starvla/pi05_base_starvla/fp32/model.safetensors`（π0.5 基础模型权重）

### 5.1 段 4 逐行详解

**总体目标**：把云端 `Orion/pretrained/` 目录下的预训练模型**整目录递归下载**到本地 `/root/data/pretrain_models/`（→ `/mnt/cfs/data/pretrain_models/` 共享盘），为段 5 的"用预训练权重初始化训练"备好粮草。

#### train.sh 里的完整代码（第 160-162 行）

```bash
mkdir -p "$PRETRAINED_LOCAL"                                        # ①
python "$BOS_IO" get-dir "bos://${BOS_BUCKET}/${BOS_NAMESPACE}/pretrained/" \
    "$PRETRAINED_LOCAL" --exclude '.hfd/*'                          # ②
```

- **① `mkdir -p "$PRETRAINED_LOCAL"`**：确保本地目录存在。`PRETRAINED_LOCAL=/root/data/pretrain_models`（train.sh 第 76 行定义）。
- **② `get-dir` 子命令参数拆解**：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `get-dir` | — | bos_io.py 子命令："递归下载目录" |
| `bos://umi-gripper/Orion/pretrained/` | — | 云端源目录。`umi-gripper`=bucket，`Orion/pretrained/`=key 前缀 |
| `$PRETRAINED_LOCAL` | `/root/data/pretrain_models` | 本地目标目录 |
| `--exclude '.hfd/*'` | — | 跳过匹配该 glob 的文件（HuggingFace 缓存 `.hfd`，训练用不到） |

#### bos_io.py get-dir 内部四步

**第一步：parse_bos_uri 拆 URI（第 76-85 行）**

```python
p = urlparse(uri)
bucket = p.netloc                    # "umi-gripper"
key = p.path.lstrip("/")             # "Orion/pretrained/"
```

**第二步：列目录（`list_objects`，第 91-103 行）**

```python
paginator = client.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=""):
    for o in page.get("Contents", []):
        files.append((o["Key"], o.get("Size", 0)))
```

- `delimiter=""` = **递归列出**所有对象（不只第一层）。
- 拿到 `(key, size)` 清单，例如：
  - `Orion/pretrained/paligemma-3b-pt-224/tokenizer.model`
  - `Orion/pretrained/pi05_base_starvla/pi05_base_starvla/fp32/model.safetensors`

**第三步：逐个下载（`download_dir`，第 225-248 行）**

```python
for key, size in files:
    rel = key[len(prefix):] if key.startswith(prefix) else key
    if not rel or rel.endswith("/"):
        continue                        # 跳过"目录项"
    if _excluded(rel, exclude):
        continue                        # 跳过 .hfd/* 排除项
    dst = local_dir / rel               # 重建相对路径
    if dst.exists() and dst.stat().st_size == size:
        continue                        # 已存在且大小一致 → 幂等跳过
    download_file(client, bucket, key, dst)
```

关键点：
- **重建相对路径**：`rel` = key 去掉 `Orion/pretrained/` 前缀，本地按同样相对结构放 → 远端/本地目录树一一对应。
- **幂等**：本地文件已存在**且大小相同** → 跳过。这是 16GB 大模型不重复下的关键（重复跑 train.sh 秒过）。
- `_excluded`：`fnmatch` 匹配 `.hfd/*`。
- `download_file`：优先 `bcecmd`，不行走 boto3 流式 + 断点续传 + SIGALRM 看门狗 + 指数退避重试。

**第四步：下载完的本地结构（本机实测）**

```
/mnt/cfs/data/pretrain_models/            ==  /root/data/pretrain_models
├── paligemma-3b-pt-224/
│   └── tokenizer.model                  ← yaml: framework.tokenizer.model_path
└── pi05_base_starvla/
    └── pi05_base_starvla/
        └── fp32/
            └── model.safetensors        ← yaml: trainer.pretrained_checkpoint
```

#### 段 4 的意义

段 2/3/4 是"把云端资源拉齐"，段 4 是**训练启动前最后一次资源准备**：
1. **tokenizer** 在本地——多进程训练每个 worker 都要读它（`LazyPaliGemmaTokenizer` 懒加载），放共享盘避免重复下载。
2. **预训练权重**在本地——段 5 训练**不是从零开始**，用 `pi05_base_starvla` 权重初始化（backbone 复用 VLM 语义理解），只有 action head 随机初始化。

#### 段 4 知识点小结

| 知识点 | 出处 |
| --- | --- |
| `mkdir -p` 幂等建目录 | train.sh 第 160 行 |
| 变量代入：`"bos://${BOS_BUCKET}/${BOS_NAMESPACE}/pretrained/"` | train.sh 第 161 行 |
| 命令跨行续接 `\` | train.sh 第 161-162 行 |
| argparse 子命令 + `--exclude` | bos_io.py 第 348-351 行 |
| `urlparse` 解析 URI → (bucket, key) | bos_io.py 第 79-84 行 |
| `list_objects_v2` 分页列出 | bos_io.py 第 91-103 行 |
| 相对路径重建 + 大小校验幂等 | bos_io.py 第 233-241 行 |
| fnmatch 排除 | bos_io.py 第 220-222 行 |

> **一句话总结段 4**：把云端 `Orion/pretrained/` 的 tokenizer + π0.5 基础权重递归拉到共享盘本地目录（幂等、跳过 `.hfd`），为段 5 训练备好初始化权重。

---

## 6. 段 5：训练（第 164-176 行）

### 6.1 train.sh 的准备工作

```bash
yaml_stem="$(basename "$CONFIG_YAML" .yaml)"                       # run_umi_shenzhen_relative_6d_v2_forceclose
RUN_ID="${RUN_ID:-${yaml_stem}_${MAX_STEPS}step}"                  # run_umi_shenzhen_relative_6d_v2_forceclose_20000step
export RUN_ID MAX_STEPS BATCH SAVE_EVERY LOG_EVERY NUM_WORKERS PREFETCH PORT WANDB_PROJECT NUM_GPUS
export REPO_ROOT="$PWD"                                            # /root/code/repo_<sha>
export ACCEL_BIN="$PWD/.venv/bin/accelerate"
bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh "$CONFIG_YAML"
```

要点：
- **RUN_ID 默认 = `<yaml文件名>_<MAX_STEPS>step`**。本命令 = `run_umi_shenzhen_relative_6d_v2_forceclose_20000step`（MAX_STEPS 未覆盖，取默认 20000）。
- `REPO_ROOT` 指向解压目录（该目录无 `.git`，launcher 默认 `git rev-parse` 拿不到根目录，所以显式传入）。
- `ACCEL_BIN` 指向 `uv sync` 建的 venv 里的 accelerate。
- 训练逻辑**不重写**，直接复用 `run_umi_pi05_relative_6d.sh` 启动器。

### 6.2 launcher：`run_umi_pi05_relative_6d.sh`

这是"一个 bash + 任意 yaml"的通用启动脚本，全流程：

1. `$1` 转绝对路径并校验存在；`cd $REPO_ROOT`。
2. CUDA 环境：`CUDA_HOME=/usr/local/cuda` 加进 PATH；`PYTORCH_CUDA_ALLOC_CONF` 若未设则设 `expandable_segments:True`。
3. 训练参数（环境变量可覆盖，注意这些默认值**优先于 yaml**）：

| 参数 | launcher 默认 | 本命令实际 | 对比 yaml |
| --- | --- | --- | --- |
| `NUM_GPUS` | `torch.cuda.device_count()` | 全部 GPU | — |
| `BATCH` | 10 | **2**（env） | yaml `per_device_batch_size: 4` → **被 2 覆盖** |
| `MAX_STEPS` | 20000 | 20000 | yaml `max_train_steps: 10000` → **被 20000 覆盖** |
| `SAVE_EVERY` | 5000 | 5000 | yaml `save_interval: 2000` → **被 5000 覆盖** |
| `LOG_EVERY` | 10 | 10 | yaml `logging_frequency: 10` → 一致 |
| `NUM_WORKERS` | 8 | 8（未设，默认） | yaml `num_workers: 4` → **被 8 覆盖** |
| `PREFETCH` | 4 | 4 | yaml `prefetch_factor: 2` → **被 4 覆盖** |
| `WANDB_PROJECT` | `pi05_z_umi_relative_6d` | `starvla`（train.sh 导出） | yaml `wandb_project: umi_relative_6d_pi05` → **被 starvla 覆盖** |
| `PORT` | 29501 | 29501 | — |

4. `run_id`：`${RUN_ID:-<yaml_stem>_<MAX_STEPS>step}`，本命令已被 train.sh 导出的 RUN_ID 锁定。
5. 输出目录：`output_dir=/root/data/model_outputs/<RUN_ID>/`，把 launcher 脚本本身和 yaml 拷进去留档。
6. `ACCEL_BIN`（被 train.sh 设为 venv 里的 accelerate）。
7. 组装最终命令（第 114-125 行）：

```bash
/root/code/repo_<sha>/.venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29501 \
  --num_processes <NUM_GPUS> \
  starVLA/training/train_starvla.py \
  --config_yaml <CONFIG_YAML绝对路径> \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.num_workers 8 \
  --datasets.vla_data.prefetch_factor 4 \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --run_root_dir /root/data/model_outputs/ \
  --run_id run_umi_shenzhen_relative_6d_v2_forceclose_20000step \
  --wandb_project starvla
```

> 注意：单机模式（`NUM_MACHINES=1` 且无 `MAIN_PROCESS_IP`）只加 `--num_processes`；多机才追加 `--num_machines/--machine_rank/--main_process_ip`。DeepSpeed 配置见 `starVLA/config/deepseeds/ds_config.yaml`：**ZeRO-2**、bf16 开启、fp16 关闭、grad accum=1、`gradient_clipping=1.0`。

`--local` 的作用在这里生效：`SWANLAB_MODE=local` 被传到 `train_starvla.py`，`build_tracker` 构造 `_SwanLabTracker`，`init` 时读 `SWANLAB_MODE` → `local` 分支**不需要登录**，直接 `swanlab.init(mode="local")`，日志写到 `output_dir/swanlog/`，可用 `swanlab watch` 看面板（`logger.py` 第 46-89 行）。

### 6.3 段 5 逐行详解（train.sh → launcher → accelerate → 训练）

**总体目标**：算好 RUN_ID → 组装 accelerate 命令 → fork N 个进程跑 `train_starvla.py`，完成 20000 步 flow-matching 训练。

#### ① train.sh 的准备工作（第 164-176 行）

```bash
yaml_stem="$(basename "$CONFIG_YAML" .yaml)"       # run_umi_shenzhen_relative_6d_v2_forceclose
RUN_ID="${RUN_ID:-${yaml_stem}_${MAX_STEPS}step}"  # run_umi_shenzhen_relative_6d_v2_forceclose_20000step
export RUN_ID MAX_STEPS BATCH SAVE_EVERY LOG_EVERY NUM_WORKERS PREFETCH PORT WANDB_PROJECT NUM_GPUS
export REPO_ROOT="$PWD"                            # /root/code/repo_<sha>
export ACCEL_BIN="$PWD/.venv/bin/accelerate"
bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh "$CONFIG_YAML"
```

- **RUN_ID** = `<yaml文件名>_<MAX_STEPS>step`，是本次训练的唯一标识（产物目录/上传路径都用它）。
- **`export 变量1 变量2 ...`**：一次性导出多个变量——launcher 的"输入接口"，train.sh 靠它控制 launcher 参数而不改 launcher 代码。
- **`REPO_ROOT="$PWD"`**：解压目录没有 `.git`，launcher 默认 `git rev-parse` 拿不到根，必须显式传。
- **`ACCEL_BIN`**：覆盖 launcher 默认的 accelerate 路径，指向 venv 里的。

#### ② launcher 组装最终命令（`run_umi_pi05_relative_6d.sh`）

launcher 五步：

1. **解析参数**：`CONFIG_YAML` 转绝对路径、校验存在；`cd $REPO_ROOT`。
2. **设 CUDA 环境**：`CUDA_HOME=/usr/local/cuda` 加进 PATH。为什么？DeepSpeed 需要**真实的 nvcc**，登录节点的 `~/.local/bin/nvcc` 是 stub 不能用。
3. **训练参数默认值**（第 62-84 行）：`BATCH=10 / MAX_STEPS=20000 / SAVE_EVERY=5000 / NUM_WORKERS=8 / PREFETCH=4`——**优先级：train.sh 导出的 env > launcher 默认 > yaml**。
4. **算 run_id + 建输出目录**：`output_dir=/root/data/model_outputs/<RUN_ID>/`，把 launcher 脚本和 yaml 拷进去留档。
5. **多机开关**：你的命令是单机（`NUM_MACHINES=1`）→ 只加 `--num_processes <GPU数>`；多机才追加 `--num_machines/--machine_rank/--main_process_ip`。

最终命令（BATCH=2 代入后）：

```bash
/root/code/repo_<sha>/.venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29501 --num_processes <GPU数> \
  starVLA/training/train_starvla.py \
  --config_yaml <绝对路径>.yaml \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.num_workers 8 \
  --datasets.vla_data.prefetch_factor 4 \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --run_root_dir /root/data/model_outputs/ \
  --run_id run_umi_shenzhen_relative_6d_v2_forceclose_20000step \
  --wandb_project starvla
```

> ⚠️ **CLI 参数覆盖 yaml**：`per_device_batch_size 2`（覆盖 yaml 的 4）、`max_train_steps 20000`（覆盖 10000）、`save_interval 5000`（覆盖 2000）、`num_workers 8`（覆盖 4）、`prefetch_factor 4`（覆盖 2）。**实际训练 = 2×GPU 全局 batch、20000 步、每 5000 步存一次**。

#### ③ accelerate launch + DeepSpeed

- `accelerate launch` 读 `deepspeed_zero2.yaml`（外层，声明用 DeepSpeed、进程数），然后**按 `--num_processes` fork 出 N 个进程**，每个进程独立跑一份 `train_starvla.py`。
- 内层 `ds_config.yaml` 关键项：
  ```json
  { "bf16": {"enabled": true}, "zero_optimization": {"stage": 2, ...},
    "gradient_accumulation_steps": 1, "gradient_clipping": 1.0 }
  ```
  - **ZeRO-2**：Adam 优化器状态切成 N 份分到各卡，3B+1B 模型才放得下。
  - **bf16**：前向/反向用 bf16（显存减半、无需 grad scaler），敏感层保持 fp32。

#### ④ 接下来发生什么

`train_starvla.py` 的完整流程见第 7 节（读配置→建 PI05 模型→建数据→优化器→prepare_training 加载预训练权重→train 主循环）。这里只给骨架：

```
accelerate launch
  └─ fork N 个进程，各跑一份 train_starvla.py
      ├─ 读 yaml + CLI 覆盖（batch=2, steps=20000...）
      ├─ setup_directories（output_dir 带时间戳）
      ├─ build_framework → PI05(PI0)：OpenPIPaliGemma(3B) + OpenPI05ActionHead(~1B) → bf16
      ├─ prepare_data：UMILeRobotDataset + 变换链 + 在线 norm
      ├─ AdamW 分模块 LR + cosine 调度
      ├─ prepare_training：加载 pi05_base 权重 → accelerate.prepare → swanlab(local)
      └─ train()：20000 步 × (forward 流式匹配 → backward → step)，每 5000 步存 ckpt
```

#### 段 5 知识点小结

| 知识点 | 出处 |
| --- | --- |
| `basename` 去后缀、`${RUN_ID:-默认}` 双回退 | train.sh 第 167-168 行 |
| `export 多个变量` 传给子脚本 | train.sh 第 171 行 |
| 环境变量作为脚本"输入接口" | train.sh → launcher 全程 |
| `readlink -f` 转绝对路径 | launcher 第 42 行 |
| `${VAR:-默认}` 参数默认值 | launcher 第 63-75 行 |
| `ACCEL_FLAGS=(...)` 数组 + `${ACCEL_FLAGS[@]}` 展开 | launcher 第 103-112 行 |
| 数组追加 `+=` | launcher 第 106-110 行 |
| `"${ACCEL_BIN}" launch ...` 执行最终命令 | launcher 第 114-125 行 |
| `set --` 后直接 `"$@"` 执行 | launcher 第 131 行 |

> **一句话总结段 5**：train.sh 算好 RUN_ID 后调 launcher，launcher 组装出 accelerate 命令（DeepSpeed ZeRO-2 + bf16），launch 出 N 个进程跑 `train_starvla.py`——读配置 → 建 PI05 模型（3B VLM + 1B 专家）→ 建数据 → 加载预训练权重 → 跑 20000 步 flow-matching 训练。**`train_starvla.py` 内部的详细过程见第 7 节。**

---

## 7. 训练主程序 `train_starvla.py` 流程

`accelerate` 会按 `--num_processes` 启动 N 个进程，每个进程独立跑一份 `train_starvla.py`（rank 0 是主进程）。单进程内部流程：

```
main(cfg)
 ├─ wrap_config                 # 包成 AccessTrackedConfig，跟踪字段访问 → 产出精简 config.yaml
 ├─ warn_unused_yaml_fields     # 提示配了但没被读的字段（如 action_type）
 ├─ setup_directories           # 建 output_dir + checkpoints/
 ├─ build_framework(cfg)        # 按 framework.name=PI05 实例化模型
 ├─ prepare_data(...)           # 构建 train + eval 两个 dataloader
 ├─ setup_optimizer_and_scheduler  # AdamW 分模块 LR + cosine 调度
 ├─ VLATrainer.prepare_training()  # seed → 存config → 加载ckpt → 冻结 → accelerate.prepare → tracker
 └─ VLATrainer.train()          # 主训练循环
```

### 7.1 入口与配置（第 733-766 行）

1. `OmegaConf.load(yaml)` 读配置。
2. `normalize_dotlist_args` 解析 CLI 的 `--foo.bar=baz` 形式覆盖，`OmegaConf.merge` 进 cfg —— 所以 `--datasets.vla_data.per_device_batch_size 2` 等参数在这里覆盖 yaml。
3. `apply_config_compat` 做字段兼容（老 yaml 字段升级，幂等）。
4. `is_debug=true` 时 rank0 起 debugpy 等 IDE（本命令 yaml `is_debug: false`，跳过）。

### 7.2 目录与配置快照

`setup_directories`（第 125-145 行）：

- `is_resume=false` → 输出目录带时间戳：`/root/data/model_outputs/<RUN_ID>/<MMDD_HHMMSS>/`。
- 建 `checkpoints/` 子目录。
- rank0 在训练第一步就把两份配置写盘：`config.full.yaml`（全量）和 `config.yaml`（只含实际访问过的字段），方便崩溃后追溯/复现。

### 7.3 模型构建：`build_framework` → PI05

`build_framework`（`base_framework.py:85`）会 `_auto_import_framework_modules()` 自动扫描注册所有 framework 文件，然后按 `cfg.framework.name = "PI05"` 从 `FRAMEWORK_REGISTRY` 取到 `PI05` 类并实例化。

`PI05`（`PI05.py`）本身只有约 60 行，`class PI05(PI0)`，只覆盖 3 个类属性：
- `action_head_cls = OpenPI05ActionHead`（adaRMS 条件化，π0→π0.5 的核心差异）
- `discrete_state_input = True`（类属性，但最终以 yaml 的 `discrete_state_input: false` 覆盖生效）
- `model_name = "PI05"`

真正实现全在 `PI0.__init__`（`PI0.py:294-302`）：

1. `_init_runtime_config`：合并 default config（`PI05DefaultConfig`）＋ yaml ＋ CLI，把 `action_dim`(32) / `action_horizon`(50) / `max_state_dim`(32) / `num_inference_steps`(10) / `image_resolution`(224×224) 提到 self。
2. `_build_tokenizer`：懒加载 `LazyPaliGemmaTokenizer`（路径 = yaml `tokenizer.model_path` = `/root/data/pretrain_models/paligemma-3b-pt-224/tokenizer.model`）。state 离散化/拼接 prompt 都靠它。
3. `_build_model_modules`：构造两条 Gemma 流
   - `self.vlm = OpenPIPaliGemma(...)`：**Prefix 流**，PaliGemma 3B（width=2048, depth=18），编码图像 + 语言。
   - `self.action_head = OpenPI05ActionHead(...)`：**Suffix 流**，小型 Gemma ~1B（width=1024, depth=18）＋ Flow Matching 投影层，负责 state + noisy action → 速度场 v_t。PI05 在每层 RMSNorm 注入 `dense` 模块（adaRMS）。
4. `to_precision("bfloat16")`：整体转 bf16，但 `patch_embedding`、`input_layernorm`、`post_attention_layernorm`、`model.norm` 保持 fp32（数值稳定）。
5. `torch.set_float32_matmul_precision("highest")`、按 `STARVLA_COMPILE` 环境变量可选 torch.compile 热路径。

> 预训练权重加载（`trainer.pretrained_checkpoint`）不是在这里做的，而是 VLATrainer 的 `_init_checkpointing`（见 7.5）。

### 7.4 数据准备：`prepare_data` → dataloader → 数据集与变换链

`prepare_data`（第 148-169 行）调用 `build_dataloader(cfg, dataset_py="lerobot_datasets", mode="train"/"eval")`：

1. `lerobot_datasets.py get_vla_dataset`：
   - `DATASET_NAMED_MIXTURES[data_mix]` 从**运行时注册表**取（`registry.py discover_and_merge` 自动扫描 `examples/**/train_files/data_registry/data_config.py` 合并进去，所以训练时能找到 UMI 的 mix 和 robot_type）。
   - 取出 `("20260726133821_lerobot_7d6be9ce_noep58_forceclose", 1.0, "umi_2gripper_relative_6d")`，`make_LeRobotSingleDataset` → data_config 里有 `make_dataset` 钩子 → 实例化 **`UMILeRobotDataset`**。
2. `UMILeRobotDataset`（`umi_dataset.py`）针对 UMI 数据的特殊处理：
   - **没有独立 action 列**：`_get_action_from_state` 从 `state[t+1]` 合成绝对下一帧 action（末帧 hold），这样后面的 `RelativePoseTransform` 才能做几何正确的 SE(3) 相对化。
   - 没有 `meta/modality.json`：从 `umi_modality_template.json` 自动拷贝（原子写）。
   - 在线注入 `action_pad`（时间维 mask，轨迹末尾凑不齐 50 帧的 timestep 标 False）和 `action_dim_actual`(10)。
3. 变换链（`_UMI2GripperRelative6DConfig.transforms`，data_config.py 第 89-121 行），原始 8D → 最终 10D：

```
state(8D) + action(8D, 由 state[t+1] 合成)
  → StateActionToTensor                     # 转 tensor
  → RelativePoseTransform                   # action 位姿相对到当前 state[t0]（SE(3) 群运算）
  → QuaternionTo6DRotation                  # 四元数(4) → rotation_6d(6)，state+action 都是
  → NormalizationTransform(state, min_max)  # state 按 (D',) 逐维 min_max
  → PerStepNormalizationTransform(action, q99)  # action 按 (H=50, D') 每个时间步独立 q01/q99 归一化
```

4. **在线 norm 统计**（`set_transforms_metadata` → `_resolve_norm_statistics` → `_calculate_norm_statistics`）：训练首次启动时按 stride=10 采样跑一遍变换链（归一化 transform 此时 normalizer 为空 → passthrough），算各 subkey 的 min/max/mean/std/q01/q99，缓存到 `meta/stats_gr00t.json`（cache key 含 norm_stride / action_horizon / per_step_action_norm，变了自动失效重算）。多卡时只让 rank0 算，其余 rank barrier 后读缓存。
   - 本数据集是 forceclose 变体，`build_gripper_forceclose.py` 只改了夹爪列（闭合 0 → 张开 ~0.027 → 线性降到纯 0），action 由 state[t+1] 派生自动生效，**stats/steps 缓存已删**，所以首次启动会重算统计。
5. `datasets.split`（train_percent=90）→ 训练走 train 侧（随机采样），在线评测走 eval 侧（每 10 条取 1 条，确定性采样）。
6. `DataLoader` 参数来自 CLI 覆盖：`batch_size=2, num_workers=8, prefetch_factor=4, pin_memory=true, persistent_workers=true`。

### 7.5 `VLATrainer.prepare_training()`（第 235-283 行，顺序敏感）

1. 每个 rank 不同 seed（`cfg.seed + rank`，seed=42）。
2. 先存 config 快照（任何后续步骤崩溃都有迹可循）。
3. `_init_checkpointing`：
   - `is_resume=false` → 不走断点续训。
   - 有 `pretrained_checkpoint` → `load_pretrained_backbones` 加载 `/root/data/pretrain_models/pi05_base_starvla/pi05_base_starvla/fp32/model.safetensors`，`completed_steps=0`。注意 PI0 有自己的 `load_pretrained_checkpoint` 做 OpenPI 权重 key 映射（`paligemma_with_expert.*` → StarVLA 命名，跳过 shape mismatch 的 key，识别 adaRMS dense 缺失是预期）。
4. `_adjust_lr_scheduler_for_resume`：从 0 开始无需快进。
5. 冻结：`freeze_modules: ''`（yaml 为空）→ 不冻结任何模块。
6. `setup_distributed_training` = `accelerator.prepare(model, optimizer, dataloaders)`：model/optimizer 被 DeepSpeed（ZeRO-2）包装，dataloader 注入 DistributedSampler。
7. `_init_tracker`：`build_tracker` → `SWANLAB_MODE=local` → `swanlab.init(mode="local")`。

### 7.6 优化器与调度器（`setup_optimizer_and_scheduler`，第 172-204 行）

- `build_param_lr_groups`：按模块把参数分组，每组独立 LR（yaml `trainer.learning_rate`）：
  - `base: 5e-5`（backbone）
  - `action_model: 2e-4`（action head，随机初始化，LR 更高）
  - `paligemma: 5e-5`（显式指定，避免匹配到 action_model）
- `torch.optim.AdamW(fused=True, betas=[0.9,0.95], weight_decay=0.01, eps=1e-8)`。
- scheduler：`cosine_with_min_lr`，warmup 1000 步，总 20000 步，min_lr=1e-6。

### 7.7 主训练循环 `train()`（第 471-543 行）

每步（真实 step，grad_accum=1 所以 micro-step = step）：

```
while completed_steps < 20000:
  batch = next(train_dataloader)          # data_time 计时
  _train_step(batch)                       # model_time 计时
    with accelerator.accumulate(model):
      optimizer.zero_grad()
      with torch.autocast("cuda", bfloat16):   # 混合精度
        out = model.forward(batch)            # ← PI0.forward
        loss = out["action_loss"]
      accelerator.backward(loss)
      clip_grad_norm_(max_norm=1.0)
      optimizer.step()
      lr_scheduler.step()                    # 仅 sync_gradients 时
  completed_steps += 1
  # eval_interval(1e8, 几乎不触发) → 在线 eval
  # logging_frequency(10) → 记日志 + 推 SwanLab
  # save_interval(5000) → 存 checkpoint
  if completed_steps >= 20000: break
```

**模型前向（flow matching）**：`PI0.forward`（`PI0.py:797-810`）→ `_prepare_examples`（图像 pad 到 224²、state 取最后一帧 pad 到 max_state_dim、action pad 到 (50,32)）→ `_build_observation_from_batch`（discrete_state_input=false → state 走连续 `state_proj`，不拼进 prompt）→ `forward_from_observation`：

```
embed_prefix(images + lang)          # Prefix 流 embedding
action_head.prepare_flow_targets(a)  # 采样 t∈[0,1] 和噪声 ε
                                     #   x_t = t·ε + (1-t)·a
                                     #   u_t = ε - a
action_head.embed_action_suffix      # state + x_t + timestep → Suffix embedding（PI05 用 adaRMS cond）
forward_with_action_expert           # 双流逐层共享 self-attention，18 层
action_head.decode_action_velocity   # suffix 输出 → v_t
loss = MSE(u_t, v_t)                 # 时间维(action_pad) + 特征维(action_dim_actual..31) 做 mask
```

双流共享 attention 是 π0 的核心：prefix（VLM 3B）和 suffix（Expert ~1B）每层的 Q/K/V 拼起来一起算 attention，但各自走自己的 FFN/RMSNorm；attention mask 是 block-causal（suffix 能看到 prefix，prefix 看不到 suffix），action expert 直接"借用"VLM 的语义理解而不污染 VLM。

**checkpoint 保存**（`_save_checkpoint`）：每 5000 步，rank0 把 state_dict 存为 `checkpoints/steps_<N>_pytorch_model.pt`（yaml `save_format: pt`），并追加 `summary.jsonl`；DeepSpeed 自管 optimizer/scheduler 状态。

---

## 8. 段 6：上传全部产物（第 178-183 行）

```bash
output_dir=/root/data/model_outputs/run_umi_shenzhen_relative_6d_v2_forceclose_20000step
python scripts/bos_io.py put-dir "$output_dir" \
    "bos://umi-gripper/Orion/ckpts/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/"
```

递归把整个 run 目录（含时间戳子目录、checkpoints、config、swanlog）上传到 BOS，防止训练中断后无法恢复。

### 8.1 段 6 逐行详解

**总体目标**：把整个 run 目录（含所有 checkpoint、配置、日志）递归上传到云端 `Orion/ckpts/<RUN_ID>/`，**防止训练中断后无法恢复**——即使共享盘炸了，云端还有一份完整产物。

#### train.sh 里的完整代码（第 178-183 行）

```bash
echo ">> [上传] ${output_dir} -> bos://${BOS_BUCKET}/${BOS_NAMESPACE}/ckpts/${RUN_ID}/"   # ①
python "$BOS_IO" put-dir "$output_dir" "bos://${BOS_BUCKET}/${BOS_NAMESPACE}/ckpts/${RUN_ID}/"   # ②
echo "✅ 完成: ${RUN_ID}  ->  bos://${BOS_BUCKET}/${BOS_NAMESPACE}/ckpts/${RUN_ID}/"     # ③
```

- **① 打印**：`${output_dir}` = `/root/data/model_outputs/<RUN_ID>`（train.sh 第 176 行算好的）。
- **② 上传**：调用 `bos_io.py put-dir` 子命令。
- **③ 完成提示**。

变量代入后实际执行：

```bash
python scripts/bos_io.py put-dir \
    /root/data/model_outputs/run_umi_shenzhen_relative_6d_v2_forceclose_20000step \
    "bos://umi-gripper/Orion/ckpts/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/"
```

#### bos_io.py put-dir 内部流程

**第一步：parse_bos_uri 拆 URI（第 76-85 行）**

```python
p = urlparse("bos://umi-gripper/Orion/ckpts/<RUN_ID>/")
bucket = p.netloc          # "umi-gripper"
prefix = p.path.lstrip("/")# "Orion/ckpts/<RUN_ID>/"
```

**第二步：upload_dir 递归上传（第 258-279 行）**

```python
def upload_dir(client, local_dir, bucket, prefix, exclude=None, quiet=False):
    prefix = prefix.rstrip("/") + "/"
    n = 0
    for path in sorted(local_dir.rglob("*")):          # 递归遍历所有文件
        if not path.is_file():
            continue                                   # 跳过目录
        rel = path.relative_to(local_dir).as_posix()   # 相对路径
        if any(fnmatch(rel, p) for p in exclude):
            continue
        key = prefix + rel                             # 云端完整 key
        print(f"  ↑ {path} -> bos://{bucket}/{key}")
        upload_file(client, path, bucket, key)         # boto3 上传
        n += 1
    print(f"upload_dir: {n} 个文件 -> bos://{bucket}/{prefix}")
```

- **`local_dir.rglob("*")`**：递归遍历本地目录下所有条目（`glob` 只看一层，`rglob` 递归全部）。
- **`path.is_file()`**：跳过目录、只传文件。
- **`rel = path.relative_to(local_dir).as_posix()`**：算相对路径并转成 `/` 分隔（跨平台）。
- **`key = prefix + rel`**：拼出云端完整路径，**本地目录树和云端一一对应**。
- **`upload_file`（第 254-256 行）**：`client.upload_file(str(local_path), bucket, key)`——boto3 高层 API，**大文件自动分片并发上传**（multipart）。

#### 上传了哪些文件

`output_dir`（= `<RUN_ID>/`）下的全部内容，实际包括：

```
<RUN_ID>/                                  ← output_dir
├── <MMDD_HHMMSS>/                         ← 训练实际写入的时间戳子目录
│   ├── config.full.yaml                   # 全量配置
│   ├── config.yaml                        # 精简配置
│   ├── run_umi_pi05_relative_6d.sh        # launcher 留档
│   ├── run_umi_..._forceclose.yaml        # 训练 yaml 留档
│   ├── summary.jsonl                      # 步数记录
│   ├── swanlog/                           # SwanLab 本地日志
│   └── checkpoints/
│       ├── steps_5000_pytorch_model.pt
│       ├── steps_10000_pytorch_model.pt
│       ├── steps_15000_pytorch_model.pt
│       └── steps_20000_pytorch_model.pt
```

> 为什么上传整个 RUN_ID 目录而不是时间戳子目录？trainer 每次新训都会生成新的 `MMDD_HHMMSS/` 子目录，而 train.sh 的 `output_dir` 只到 RUN_ID 层——上传整层可以保证"无论哪次训练跑出来的产物都在云端这个 RUN_ID 下"，与本地结构一致，方便后续恢复。

#### 与下载的对比：上传没有幂等跳过

| | 下载（download_dir） | 上传（upload_dir） |
| --- | --- | --- |
| 幂等 | 已存在且大小一致 → 跳过 | **没有**，同名直接覆盖 |
| 用途 | 省流量（16GB 预训练不重下） | 保证云端一定是最新产物 |

> ⚠️ 注意：上传是**覆盖式**的。如果同一个 RUN_ID 重复跑，云端同名文件会被覆盖为最新。

#### 段 6 知识点小结

| 知识点 | 出处 |
| --- | --- |
| `${output_dir}` 变量引用 | train.sh 第 176、181-182 行 |
| argparse 子命令 `put-dir` | bos_io.py 第 353-355 行 |
| `urlparse` 解析 URI → (bucket, prefix) | bos_io.py 第 79-84 行 |
| `local_dir.rglob("*")` 递归遍历 | bos_io.py 第 266 行 |
| `path.is_file()` 过滤目录 | bos_io.py 第 267 行 |
| `relative_to().as_posix()` 相对路径 | bos_io.py 第 269 行 |
| `key = prefix + rel` 路径拼接 | bos_io.py 第 272 行 |
| `client.upload_file` 多部分上传 | bos_io.py 第 254-256 行 |

> **一句话总结段 6**：把整个 run 目录递归上传到 `bos://umi-gripper/Orion/ckpts/<RUN_ID>/`（覆盖式、无幂等），云端留一份完整产物兜底，训练全流程到此结束。

---

## 9. 产物路径汇总

| 内容 | 路径 |
| --- | --- |
| 源码（解压） | `/root/code/repo_2623e66...63f/`（→ `/mnt/cfs/code/`） |
| 运行环境 | `/root/code/repo_2623e66...63f/.venv/` |
| 数据 | `/root/data/z-umi/data/20260726133821_lerobot_7d6be9ce_noep58_forceclose/` |
| 预训练 | `/root/data/pretrain_models/`（tokenizer + pi05_base_starvla） |
| 训练产物根 | `/root/data/model_outputs/<RUN_ID>/` |
| 本次 run 目录 | `/root/data/model_outputs/<RUN_ID>/<MMDD_HHMMSS>/` |
| checkpoint | `<output_dir>/checkpoints/steps_20000_pytorch_model.pt` 等 |
| 配置留档 | `<output_dir>/config.full.yaml`、`config.yaml` |
| SwanLab 本地日志 | `<output_dir>/swanlog/` |
| BOS 上传 | `bos://umi-gripper/Orion/ckpts/<RUN_ID>/` |

---

## 10. 本命令关键事实清单

1. **BATCH=2 是单卡 batch size**（`--datasets.vla_data.per_device_batch_size 2`），覆盖 yaml 的 4。全局 batch = 2 × GPU数。
2. **max_train_steps 实际是 20000**（launcher 默认），覆盖 yaml 的 10000；save_interval 实际 5000，覆盖 yaml 的 2000。
3. **num_workers 实际 8**，覆盖 yaml 的 4；prefetch_factor 实际 4，覆盖 yaml 的 2。
4. **RUN_ID** = `run_umi_shenzhen_relative_6d_v2_forceclose_20000step`。
5. **forceclose 数据不通过 train.sh 下载**（data_config 里第 4 元素为空），必须预置在 `/root/data/z-umi/data/`，且因 stats 缓存已删，首次训练会在线重算 norm 统计（~百秒级）。
6. **权重从 pi05_base_starvla 初始化**，不是从零训练；action head 随机初始化用更高 LR（2e-4）。
7. **`--local` = `SWANLAB_MODE=local`**：SwanLab 本地模式，免登录，日志落 `output_dir/swanlog/`。
8. **训练在解压出的 `repo_<sha>` 代码上跑**，与当前工作区无关；commit 就是 `2623e668ccdeddc3d4f1b9686d50ad5b470e563f`。
9. 训练结束后整个 run 目录自动上传 `bos://umi-gripper/Orion/ckpts/<RUN_ID>/`。

---

## 11. 调试速查

| 想做什么 | 做法 |
| --- | --- |
| 只打印 accelerate 命令不真跑 | `DRY_RUN=1 bash run_umi_pi05_relative_6d.sh <yaml>` |
| 看实际生效的训练参数 | 训练启动后读 `<output_dir>/config.full.yaml` |
| 检查 forceclose 数据是否就位 | `ls /root/data/z-umi/data/20260726133821_lerobot_7d6be9ce_noep58_forceclose/meta/info.json` |
| 单独调试 dataloader | `uv run python starVLA/dataloader/lerobot_datasets.py --config_yaml <yaml>` |
| 单独调试 norm 统计 | `uv run python examples/realRobots/zerith_pi05/train_files/data_registry/umi_dataset.py --dataset_path /root/data/z-umi/data/<dataset_id>` |
| 重新算 norm 统计 | 删除 `<dataset>/meta/stats_gr00t.json` 后重跑 |
| 改 batch 而不改别的 | `BATCH=4 bash scripts/train.sh ...` |
| 从上次断点续训 | yaml `trainer.is_resume: true`，脚本会在 output_dir/checkpoints 找最新 steps_X |

---

## 12. 常见疑问解答（FAQ）

### Q1：bash 脚本是严格顺序执行的吗？段 5 训练很久，段 6 要等它吗？

**是，默认严格顺序执行**。bash 的规则是从第一行到最后一行，一条命令拿到退出码才执行下一条。

- 段 5（train.sh 第 175 行 `bash run_umi_pi05_relative_6d.sh`）**没有加 `&`（后台）**，是前台阻塞调用——`train.sh` 一直等到训练全部结束（内部 accelerate 启动的所有进程退出）才继续段 6。
- 训练 20000 步跑几天，段 6 就等几天。
- **`set -e` 的连带行为**：训练成功（返回 0）→ 段 6 上传；训练中途崩了（返回非 0）→ `train.sh` 当场退出，**段 6 不会执行**，已训的 checkpoint 不上传。

> 想改"即使失败也上传"，需在脚本里做 `bash launcher ... || true` 或 `trap` 处理，当前 train.sh 没这么做。

### Q2：第二次运行同一个命令，权重会重新拉吗？会写到新地方吗？

分两半看：

| 资源 | 第二次运行行为 | 会换地方吗 |
| --- | --- | --- |
| 预训练权重（输入） | `get-dir` 检测"已存在且大小一致" → **跳过** | 不会，还在 `/root/data/pretrain_models/` |
| 数据集 | `meta/info.json` 已存在 → **跳过** | 不会 |
| 代码 + venv | 同 commit → flock 检测 `.venv` 已存在 → **复用** | 不会 |
| 训练产物（输出） | 重新训一遍 → 写到 **`<RUN_ID>/<新时间戳>/`** | **会**，每次新增一个 `MMDD_HHMMSS` 子目录 |

实测证据：`run_umi_shenzhen_relative_6d_v2_forceclose_20000step/` 下已有 5 个时间戳子目录（0804_212924、0804_234305、0805_103512、0805_105158、0805_135956），对应 5 次重复训练。

**设计意图**：输入资源幂等（省流量省时间），输出不覆盖（每次训练是独立实验、带时间戳防冲掉）。

### Q3：数据、预训练权重、代码、环境各自在哪？

全部在 `/mnt/cfs` 共享盘上，经软链暴露为 `/root/...` 路径：

| 资源 | 逻辑路径（代码写死） | 真实位置（共享盘） |
| --- | --- | --- |
| 代码 | `/root/code/repo_<sha>/` | `/mnt/cfs/code/repo_<sha>/` |
| 环境 | `/root/code/repo_<sha>/.venv/` | `/mnt/cfs/code/repo_<sha>/.venv/` |
| 数据集 | `/root/data/z-umi/data/<id>/` | `/mnt/cfs/data/z-umi/data/<id>/` |
| 预训练权重 | `/root/data/pretrain_models/` | `/mnt/cfs/data/pretrain_models/` |
| 训练产物 | `/root/data/model_outputs/<RUN_ID>/` | `/mnt/cfs/data/model_outputs/<RUN_ID>/` |

预训练权重实测内容：
- `paligemma-3b-pt-224/tokenizer.model`（4.1 MB，分词器）
- `pi05_base_starvla/pi05_base_starvla/fp32/model.safetensors`（16 GB，π0.5 权重）

**软链只是"门牌"，建软链不下载任何东西**；数据/权重"是否有"由段 3/段 4 的幂等检查决定——这台机器上它们早已就位，所以段 3/段 4 秒过。

### Q4：建环境是什么过程？换新 commit 会建新环境吗？

**环境 = `repo_<sha>/.venv`，跟 commit 绑定。**

建环境过程（段 2 flock 块内）：`unzip` 解压代码 → `uv sync --python 3.10` 读 `pyproject.toml` 装全依赖生成 `.venv`（首次较慢，几分钟到十几分钟）。

| 场景 | 代码目录 | 环境 |
| --- | --- | --- |
| 同 commit 跑第二次 | 复用 `repo_<sha>/` | **复用** `.venv`（秒过） |
| 换 commit | 新建 `repo_<新sha>/` | **新建** `.venv`（全量 uv sync 一次） |

判断依据是 **commit 是否变了**，不是"跑第几次"。实测共享盘已有 12 个 `repo_*`，11 个带 `.venv`，正是每个 commit 一个独立环境。

### Q5：每个 commit 一个环境，不是很占存储空间吗？

**会占，实测每个 `.venv` 约 7GB**（torch + transformers + deepspeed 全家桶），当前 12 个 repo 环境总计 ~77GB。但这是"按 commit 隔离环境"的刻意取舍：

- **隔离性**：不同 commit 依赖版本可能不同，一人一环境 = 永不互相踩坏；
- **可复现**：`repo_<sha>` = 该 commit 的完整运行环境快照；
- **已有省空间机制**：uv 全局缓存（wheel 不重复下载）+ 同文件系统硬链接 + 同 commit 复用 venv。

缓解办法：

| 做法 | 效果 |
| --- | --- |
| 删除不再用的旧 `repo_<sha>` | 直接释放数 GB~数十 GB |
| 只保留常用 commit | 控制环境数量 |
| 同 commit 反复跑 | 复用，零新增 |

> ⚠️ 删除前确认没有正在用它的任务。
