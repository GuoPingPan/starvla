# bos_ckpt.sh（Checkpoint 上传/下载/列出）

## Purpose

`scripts/bos_ckpt.sh` 在 BOS（百度对象存储）与本地之间上传 / 下载 / 列出模型 checkpoint 的行为合同。底层用 `bcecmd bos cp <SRC> <DST> -y`，BOS 根路径为 `${BOS_USER_ROOT}ckpts/`（默认 `bos://umi-gripper/Orion/`）。完整叙述见 `docs/zerith_docs/infra/bos_ckpt.md`。

## Requirements

### Requirement: 三个子命令

脚本 SHALL 提供 `upload`（本地上传 → BOS，默认附带 sidecar）、`download`（BOS → 本地，默认 `final` ckpt）、`list`（列出本地/远端模型与 ckpt）三个子命令。

#### Scenario: 选择子命令
- **WHEN** 传第一参数为 `download`/`upload`/`list`
- **THEN** 分别走下载、上传、列出三条路径

### Requirement: download 输入形式

`download` SHALL 接受三种输入任选其一：`<model> <timestamp>`（指定模型与时间戳）、`<model>`（默认取 BOS 上 mtime 最新的时间戳）、`bos://.../Orion/ckpts/<model>/<ts>[/...]`（完整 BOS 路径，自动解析出 model + timestamp）。

#### Scenario: 不指定时间戳
- **WHEN** 只传模型名
- **THEN** 默认取 BOS 上 mtime 最新的时间戳

### Requirement: 时间戳识别

时间戳识别规则 SHALL 为正则 `^[0-9]{4}_[0-9]{6}$`（8 位数字：前 4 位 `MMDD`，后 6 位 `HHMMSS`，如 `0806_042239`）；剩余位置参数若不匹配时间戳正则则被当作 `CKPT`（download 未传时默认 `final`）。

#### Scenario: 第二个参数是时间戳
- **WHEN** 传 `download <model> 0806_042239`
- **THEN** 匹配时间戳正则，赋给 `TIMESTAMP`，ckpt 取默认 `final`

### Requirement: ckpt 取值

ckpt 值 SHALL 支持：`latest`（步数最大的 `steps_<N>_pytorch_model.pt`）、`final`（download 默认，`final_model/pytorch_model.pt`）、`<数字>`（如 `20000` → `steps_20000_pytorch_model.pt`）、`<完整/相对名>`（精确匹配）、`<子串>`（如 `steps_20000` 模糊匹配）。

#### Scenario: 指定步数 ckpt
- **WHEN** download 传 ckpt 值 `20000`
- **THEN** 选中 `steps_20000_pytorch_model.pt`

### Requirement: 本地保留远端嵌套目录结构

download 本地目标根目录 SHALL 默认 `~/.models/<模型名>/<时间戳>/...`（`MODEL_STORE="${HOME}/.models"`，可用 `--out` 覆盖），本地保留远端的嵌套目录结构。

#### Scenario: 本地结构对齐远端
- **WHEN** 下载 `final_model/pytorch_model.pt`
- **THEN** 本地落到 `~/.models/<model>/<ts>/final_model/pytorch_model.pt`，目录层级与远端一致

### Requirement: 远端布局

远端布局 SHALL 为 `bos://umi-gripper/Orion/ckpts/<模型名>/<时间戳>/final_model/pytorch_model.pt`、`.../checkpoints/steps_<N>_pytorch_model.pt`、`.../config.yaml`、`.../dataset_statistics.json`。

#### Scenario: 远端路径组成
- **WHEN** 上传或下载某 ckpt
- **THEN** 远端 key 由 `BOS_USER_ROOT` + `ckpts/` + `<模型名>/<时间戳>/` + 相对文件路径组成

### Requirement: sidecar 文件

download/upload SHALL 默认附带 sidecar 文件（若远端/本地存在）：`config.yaml`、`dataset_statistics.json`；可用 `--no-sidecar` 关闭。

#### Scenario: download 附带 sidecar
- **WHEN** download final 权重且远端存在 `config.yaml`/`dataset_statistics.json`
- **THEN** 一并下载到本地同结构目录

### Requirement: 时间戳缺失/不存在给清晰报错

时间戳缺失或不存在时，脚本 SHALL 打印远端可用时间戳列表后 `die` 退出；多时间戳时默认选最新（mtime），可用 `--timestamp` 指定。

#### Scenario: 时间戳不存在
- **WHEN** 指定的时间戳在 BOS 上不存在
- **THEN** 打印远端可用时间戳列表后退出

### Requirement: 依赖 bcecmd

脚本 MUST 依赖 `bcecmd` CLI，未安装时报错 `未找到 bcecmd, 请先安装/配置 BCE CLI`。

#### Scenario: 未装 bcecmd
- **WHEN** 环境没有 `bcecmd`
- **THEN** 脚本报错 `未找到 bcecmd, 请先安装/配置 BCE CLI` 并退出

## References

- 完整说明：`docs/zerith_docs/infra/bos_ckpt.md`
- 脚本：`scripts/bos_ckpt.sh`
- BOS IO 底层：`scripts/bos_io.py`
