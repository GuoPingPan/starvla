# CI 与训练代码来源（commit ↔ BOS zip 关系）

> 核心一句话：**训练用的代码不是本地工作区，而是「GitLab CI 打包 → 上传 BOS → train.sh 段2 下载」的 zip。push 是触发 CI 的事件，BOS 是产物存放处。**

---

## 1. 训练代码从哪里来

```
你本地 git push
   ↓
GitLab 服务器收到新 commit
   ↓ 自动读取该 commit 里的 .gitlab-ci.yml
   ↓ 创建流水线（pipeline）
   ↓ Runner 执行任务
   ↓
[package] 把源码打成 zip：starvla_<分支>_<时间戳>_<commit短号>.zip
   ↓
[upload:bos] 用 bos_io.py 上传到 bos://umi-gripper/starvla_res/
   ↓
train.sh 段2 用 commit 号从 BOS 下载 zip
   ↓ 解压 + uv sync 建环境
   ↓
开始训练（train_starvla.py）
```

**关键点**：train.sh 段 2 从 **BOS** 拉 zip，**不直接访问 gitlab、不看本地 git**。所以"本地有代码"≠"能训练"，必须 BOS 上有对应 commit 的 zip。

---

## 2. CI 是什么 + 为什么 push 会触发

**CI = Continuous Integration（持续集成）**：代码每次变更后自动执行的构建/检查流水线。

**触发机制**：不是因为你写了命令，而是 **"push 这个事件本身"**。GitLab 服务器一收到新提交，就自动检查仓库里有没有 `.gitlab-ci.yml`——有就按它执行。相当于"你提交订单，系统自动发货"。

| 角色 | 做什么 |
| --- | --- |
| 你本地 | `git push`（触发事件） |
| GitLab 服务器 | 读 `.gitlab-ci.yml`，创建 pipeline |
| Runner | 实际执行打包/上传任务 |
| BOS | 接收产物 zip |

---

## 3. `.gitlab-ci.yml` 的作用与流水线

文件位置：仓库根目录 `.gitlab-ci.yml`。本项目的流水线有 2 个 job（接力）：

```
stages: package → upload

[package] job：把 starVLA/scripts/examples/docs... 打成 zip（纯标准库，不需装依赖）
[upload:bos] job：用 scripts/bos_io.py put 把 zip 推到 bos://<BOS_BUCKET>/starvla_res/
```

### 关键配置点

- **触发规则**：`rules` 被注释 → **每次 push 都跑**（任何分支）。若想收窄，取消注释并按需改，如 `if: $CI_COMMIT_BRANCH == "dev/Alex"`。
- **zip 命名**（`.gitlab-ci.yml` 第 50-55 行）：
  ```python
  ref = os.environ['CI_COMMIT_REF_NAME'].replace('/', '-')   # dev/Alex → dev-Alex
  zip_name = f"{os.environ['PKG_NAME']}_{ref}_{ts}_{sha}.zip"  # starvla_dev-Alex_20260806102232_22cd18a0.zip
  ```
  分支里的 `/` 会被替换成 `-`（文件名不能含 `/`）。

---

## 4. 没 push 会怎样（三条出路）

**现象**：本地 commit 没 push → GitLab 没有 → CI 不跑 → BOS 无 zip → train.sh 段2 报错：

```
ERROR: 找不到 commit xxx 的 zip。最近可用: ...
```

**三条出路**：

| 出路 | 做法 | 何时用 |
| --- | --- | --- |
| ① 用已打包的旧 commit | `bash scripts/train.sh <yaml> c06b17c...` | 本地改动不需要训练时 |
| ② push 等 CI 打包 | `git push` → 等 1-2 分钟 → 用新 commit 训 | 需要训练新代码 |
| ③ 绕过 train.sh 直跑本地代码 | launcher 直跑工作区（ACCEL_BIN/REPO_ROOT 覆盖） | 立刻训练本地未 push 代码 |

方式 ③ 示例：
```bash
cd /root/workspace/starvla
env ACCEL_BIN=$PWD/.venv/bin/accelerate REPO_ROOT=$PWD \
  BATCH=2 MAX_STEPS=3 SAVE_EVERY=2 SWANLAB_MODE=local \
  bash examples/realRobots/zerith_pi05/train_files/run_umi_pi05_relative_6d.sh \
    examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d_v2.yaml
```

---

## 5. 如何检查某个 commit 在不在云端（OSS / BOS）

> 双云适配后：CI 产物双云各传一份 —— 阿里云 OSS `oss://algo-public0/starvla_res/` 与百度云 BOS
> `bos://umi-gripper/starvla_res/` 都有（配了哪边 CI 凭证传哪边，两边都配则两边都传）。
> 查询命令两者同款，仅 scheme 不同。

### 方法 1：列目录 + grep（推荐）

```bash
# 阿里云 OSS（凭证: 环境变量 或 /root/.ossenv）
set -a; [ -f /root/.ossenv ] && . /root/.ossenv; set +a
uv run --no-project --with oss2 --with boto3 python scripts/cloud_io.py ls "oss://algo-public0/starvla_res/" \
  | grep -i 22cd18

# 百度云 BOS（凭证: 环境变量 或 /root/.bosenv）
set -a; [ -f /root/.bosenv ] && . /root/.bosenv; set +a
uv run --no-project --with oss2 --with boto3 python scripts/cloud_io.py ls "bos://umi-gripper/starvla_res/" 2>/dev/null \
  | grep -i 22cd18
```

判断：**有输出（zip 文件名）→ ✅ 存在；无输出 → ❌ 不存在**。

### 方法 2：直接试下载（train.sh 同款逻辑）

```bash
uv run --no-project --with oss2 --with boto3 python scripts/cloud_io.py src-zip <commit号> --dest-dir /tmp/check_zip
# 可加 --cloud oss|bos 强制某朵云 (默认 auto: OSS 凭证可得优先)
```
能下载 → 存在；报 `找不到 commit ... 的 zip` → 不存在。

### 实测示例

| 查询 | 结果 |
| --- | --- |
| `c06b17c` | ✅ `starvla_dev-train_anylze_..._c06b17c0.zip` |
| `22cd18a` | ✅ `starvla_dev-Alex_..._22cd18a0.zip` |
| `abcdef1`（编的） | ❌ 无输出 |

> 注：`cloud_io.py` 兼容旧 `bos_io.py` 的全部子命令；URI 用 `oss://` 或 `bos://`（双斜杠）。

---

## 6. 相关文档

- 流水线详解（段2 拉代码）：`docs/zerith_docs/infra/train_sh_run_flow.md`
- 冒烟测试记录：`docs/zerith_docs/umi/umi_pi05_smoke_test.md`
- 训练启动命令速查：`启动命令.md`
