# bos_ckpt.sh 使用说明

`bos_ckpt.sh` 用于在对象存储与本地之间上传 / 下载 / 列出模型 checkpoint。

> **双云适配（2026-08）**：底层已从 `bcecmd` 迁移到 `scripts/cloud_io.py`（不再依赖 BCE CLI）。
> 阿里云 OSS / 百度云 BOS 均可用；云自动选择（OSS 凭证可得优先，`CLOUD=oss|bos` 可强制，
> 凭证 `/root/.ossenv` 或环境变量，与 train.sh 同一套）。用法/参数完全不变。

- 脚本路径：`scripts/bos_ckpt.sh`
- 底层命令：`python scripts/cloud_io.py get/put/ls`（oss2 / boto3）
- 远端根路径：`${CKPT_USER_ROOT}ckpts/`，默认 OSS 环境 `oss://algo-public0/orion/`、BOS 环境 `bos://umi-gripper/Orion/`（`CKPT_USER_ROOT` 或旧名 `BOS_USER_ROOT` 可覆盖）

---

## 1. 命令示例讲解

```bash
bash scripts/bos_ckpt.sh download run_umi_shenzhen_relative_6d_v2_forceclose_20000step 0806_042239
```

### 1.1 参数解析过程

脚本将 `#` 之后的参数按顺序分类：

| 参数 | 值 | 去向 |
|------|-----|------|
| `download` | 子命令 | `MODE="download"` |
| `run_umi_shenzhen_relative_6d_v2_forceclose_20000step` | 模型名 | `MODEL`（位置参数 `POS[0]`） |
| `0806_042239` | 时间戳 | 匹配正则 `^[0-9]{4}_[0-9]{6}$`，赋给 `TIMESTAMP` |

> 时间戳识别规则：形如 `0806_042239`（8 位数字，前 4 位 `MMDD`，后 6 位 `HHMMSS`）。
> 剩余位置参数若不是时间戳则被当作 `CKPT`（未传时 `download` 默认 `final`）。

### 1.2 最终调用

```bash
do_download "run_umi_shenzhen_relative_6d_v2_forceclose_20000step" "final" "0806_042239"
```

即：**从 BOS 下载模型 `run_umi_shenzhen_relative_6d_v2_forceclose_20000step`、时间戳为 `0806_042239` 的 `final` checkpoint（默认值）。**

### 1.3 执行流程（do_download）

1. **解析模型名**：`remote_name = run_umi_shenzhen_relative_6d_v2_forceclose_20000step`（不是 `bos://` 开头的完整路径，无需进一步解析）。
2. **解析远端 run**：`resolve_remote_run` 在 BOS 上校验时间戳 `0806_042239` 是否存在，存在则 `RES_RUN="0806_042239"`。
3. **列出远端 ckpt**：`list_remote_ckpts` 列出该模型 + 时间戳下所有 `.pt/.ckpt` 文件。
4. **解析 ckpt**：`resolve_ckpt "final"` 选中 `final_model/pytorch_model.pt`（即最终权重）。
5. **执行下载**：
   ```
   bcecmd bos cp \
     bos://umi-gripper/Orion/ckpts/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0806_042239/final_model/pytorch_model.pt \
     ~/.models/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0806_042239/final_model/pytorch_model.pt \
     -y
   ```
6. **附带下载 sidecar 文件**（若远端存在）：`config.yaml`、`dataset_statistics.json`。

### 1.4 下载到本地哪里

本地目标根目录默认 `~/.models/<模型名>/<时间戳>/...`（`MODEL_STORE="${HOME}/.models"`），本地会**保留远端的嵌套目录结构**。

```
~/.models/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/
└── 0806_042239/
    ├── final_model/
    │   └── pytorch_model.pt
    ├── config.yaml                 (若有)
    └── dataset_statistics.json     (若有)
```

---

## 2. 常用用法

### download（下载）

```bash
# 指定模型 + 时间戳，下载 final 权重（与上面的示例相同）
bash scripts/bos_ckpt.sh download <模型名> <时间戳>

# 指定 ckpt 步数
bash scripts/bos_ckpt.sh download <模型名> <时间戳> 20000          # steps_20000_pytorch_model.pt

# 指定其它 ckpt 名称（支持模糊匹配）
bash scripts/bos_ckpt.sh download <模型名> <时间戳> steps_20000

# 不指定时间戳：默认取 BOS 上 mtime 最新的时间戳
bash scripts/bos_ckpt.sh download <模型名>

# 直接传完整 BOS 路径（自动解析出模型名 + 时间戳）
bash scripts/bos_ckpt.sh download bos://umi-gripper/Orion/ckpts/<模型名>/<时间戳>

# 指定下载到其它目录
bash scripts/bos_ckpt.sh download <模型名> <时间戳> --out /path/to/dir
```

### upload（上传）

```bash
# 上传本地训练输出整个目录（含 time stamp 子目录、sidecar 文件）
bash scripts/bos_ckpt.sh upload /root/data/model_outputs/<模型名>/<时间戳>

# 以模型名方式上传（从 LOCAL_ROOT 下查找）
bash scripts/bos_ckpt.sh upload <模型名> <时间戳>

# 只上传指定 ckpt，且不附带 sidecar
bash scripts/bos_ckpt.sh upload <模型名> <时间戳> 20000 --no-sidecar
```

### list（列出）

```bash
bash scripts/bos_ckpt.sh list                          # 列出本地训练输出 / 本地模型库 / BOS 顶层条目
bash scripts/bos_ckpt.sh list <模型名>                  # 列出某模型本地 + 远端的时间戳与 ckpt
```

---

## 3. 参数说明

### 子命令

| 子命令 | 作用 |
|--------|------|
| `upload` | 本地上传 → BOS，默认附带 sidecar 文件 |
| `download` | BOS → 本地，默认 `final` ckpt |
| `list` | 列出本地 / 远端模型与 ckpt |

### download 输入形式（任选其一）

| 输入 | 说明 |
|------|------|
| `<model> <timestamp>` | 指定模型与时间戳 |
| `<model>` | 默认取 BOS 上最新时间戳（按 mtime） |
| `bos://.../Orion/ckpts/<model>/<ts>[/...]` | 完整 BOS 路径，自动解析 model + timestamp |

### ckpt 取值

| ckpt 值 | 含义 |
|---------|------|
| `latest` | 步数最大的 `steps_<N>_pytorch_model.pt` |
| `final`（download 默认） | `final_model/pytorch_model.pt` |
| `<数字>` 如 `20000` | `steps_20000_pytorch_model.pt` |
| `<完整/相对名>` | 精确匹配 |
| `<子串>` 如 `steps_20000` | 模糊匹配 |

### 选项

| 选项 | 说明 |
|------|------|
| `--timestamp TS` | 显式指定时间戳（也可作为位置参数） |
| `--as NAME` | upload 时指定远端模型名（默认 = 本地模型名） |
| `--out DIR` | download 本地目标根目录（默认 `~/.models/<模型名>`） |
| `--store DIR` | 本地模型库根目录（默认 `~/.models`） |
| `--local-root DIR` | 训练输出根目录（默认 `/root/data/model_outputs`） |
| `--no-sidecar` | 不附带 `config.yaml` / `dataset_statistics.json` |
| `-n, --dry-run` | 只打印 bcecmd 命令，不真正传输 |
| `-h, --help` | 显示帮助 |

---

## 4. 远端布局

```
bos://umi-gripper/Orion/ckpts/<模型名>/<时间戳>/final_model/pytorch_model.pt
bos://umi-gripper/Orion/ckpts/<模型名>/<时间戳>/checkpoints/steps_<N>_pytorch_model.pt
bos://umi-gripper/Orion/ckpts/<模型名>/<时间戳>/config.yaml
bos://umi-gripper/Orion/ckpts/<模型名>/<时间戳>/dataset_statistics.json
```

## 5. 常用环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `BOS_USER_ROOT` | `bos://umi-gripper/Orion/` | 用户命名空间根（其下 `ckpts/`、`src/` …） |

## 6. 注意事项

- 脚本依赖 `bcecmd` CLI，未安装时会报错：`未找到 bcecmd, 请先安装/配置 BCE CLI`。
- `download` 默认下载 `final`，如需中间步骤权重请显式传 ckpt 值。
- 时间戳缺失/不存在时，脚本会打印远端可用时间戳列表后退出（`die`）。
- 多时间戳时默认选最新（mtime），可用 `--timestamp` 指定。

