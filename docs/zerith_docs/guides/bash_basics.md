# Bash 基础语法速学手册

> 面向 **bash 零基础**读者，以 `scripts/train.sh` 的真实代码为例子，手把手讲清楚写 bash 脚本用到的所有语法。
> 建议配合 `docs/zerith_docs/infra/train_sh_run_flow.md`（流水线讲解）一起看。

---

## 目录

1. [脚本头部：Shebang 与注释](#1-脚本头部shebang-与注释)
2. [set 安全开关](#2-set-安全开关)
3. [变量：赋值、取值、默认值](#3-变量赋值取值默认值)
4. [命令替换与管道](#4-命令替换与管道)
5. [位置参数与特殊变量](#5-位置参数与特殊变量)
6. [if 条件判断](#6-if-条件判断)
7. [for 循环与 case 分支](#7-for-循环与-case-分支)
8. [环境变量、export 与 source](#8-环境变量export-与-source)
9. [重定向与文件描述符](#9-重定向与文件描述符)
10. [数组](#10-数组)
11. [glob 通配符](#11-glob-通配符)
12. [trap 信号陷阱](#12-trap-信号陷阱)
13. [子 shell 与 flock 并发锁](#13-子-shell-与-flock-并发锁)
14. [字符串操作](#14-字符串操作)
15. [常用命令速查表](#15-常用命令速查表)
16. [bash 文件 vs ./ 直接执行](#16-bash-文件-vs--直接执行)
17. [综合练习：翻译一段真实脚本](#17-综合练习翻译一段真实脚本)
18. [实战：run_umi_pi05_open_eval.sh 语法讲解](#18-实战run_umi_pi05_open_evalsh-语法讲解)

---

## 1. 脚本头部：Shebang 与注释

```bash
#!/usr/bin/env bash
```

**Shebang（`#!` 行）**：只有一行，写在文件第一行。它"长得像注释"（`#` 开头），但**不是普通注释**，作用是声明"这个可执行文件用哪个解释器来跑"。

- 你执行 `./train.sh` 时，内核读这行，知道要用 `env bash` 来运行它。
- 你用 `bash train.sh` 时，解释器已被显式指定，这行被忽略。

```bash
# 普通注释：以 # 开头，整行被忽略，纯文档
# train.sh 头部大段 # 就是文档
```

**小结**：

| 行 | 是否被执行 |
| --- | --- |
| `# 注释` | 不执行 |
| `#!/usr/bin/env bash` | 仅 `./script` 直接执行时被读取 |

---

## 2. set 安全开关

```bash
set -euo pipefail
```

一行堵住 bash 三个经典坑：

| 开关 | 作用 |
| --- | --- |
| `-e` | 任何命令失败（非零退出码）→ 脚本**立刻退出** |
| `-u` | 使用**未定义变量** → 报错退出（防止变量名拼错当空串） |
| `o pipefail` | 管道 `a \| b` 中 b 失败就算失败（默认只看最后一条） |

**退出码**：Unix 命令成功返回 `0`，失败返回非 0。`exit 1` 就是"以失败退出"。

---

## 3. 变量：赋值、取值、默认值

### 3.1 基本用法

```bash
VAR=值        # 赋值。注意：= 两边不能有空格！
echo "$VAR"   # 取值
echo "${VAR}" # 取值（推荐带花括号）
```

### 3.2 为什么都要加双引号

```bash
echo "$VAR"    # 安全：路径含空格也不会被拆
echo $VAR      # 危险：值里有空格会被拆成多个参数
```

**凡是 `$变量` 都建议写成 `"$变量"`**，防止路径/文本里的空格引发灾难。

### 3.3 默认值 `${VAR:-默认}` —— train.sh 的核心写法

```bash
CONFIG_YAML="${1:-${CONFIG_YAML:-examples/realRobots/zerith_pi05/train_files/run_umi_shenzhen_relative_6d.yaml}}"
COMMIT="${2:-${COMMIT:-latest}}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
export BOS_ENDPOINT="${BOS_ENDPOINT:-s3.bj.bcebos.com}"
```

语法：`${变量:-默认值}` = **变量为空或未定义时，用默认值**。

`train.sh` 的双回退（第 63-64 行）：
`${1:-${CONFIG_YAML:-默认yaml}}` 拆开看：
1. 命令行位置参数 `$1` 给了？→ 用它；
2. 没给？看环境变量 `CONFIG_YAML`；
3. 还没有？用写死的默认 yaml。

**优先级：命令行参数 > 环境变量 > 写死的默认值。**

**环境变量临时覆盖**（你用的命令）：
```bash
BATCH=2 bash scripts/train.sh ...   # 只对这一次 bash 命令临时设 BATCH=2
```

### 3.4 其它默认值写法（备查）

| 写法 | 含义 |
| --- | --- |
| `${VAR:-默认}` | 空/未定义 → 默认（**最常用**） |
| `${VAR-默认}` | 仅未定义 → 默认（空串不算） |
| `${VAR:=默认}` | 空/未定义 → 默认**并赋值给 VAR** |

---

## 4. 命令替换与管道

### 4.1 命令替换 `$(...)`

把括号里命令的**输出**替换到当前位置：

```bash
DL_TMP="$(mktemp -d "$CODE_DIR/.srczip.XXXXXX")"   # mktemp 的输出 → 变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

### 4.2 管道 `|`

左边命令的输出 → 右边命令的输入：

```bash
nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
#  ①列出所有GPU        ②数行数  ③删空格
# 结果 = GPU 数量
```

再拆一行（train.sh 第 132 行）：
```bash
REAL_SHA="$(basename "$ZIP" .zip | awk -F_ '{print $NF}' | tr 'A-Z' 'a-z')"
# basename 去掉路径和.zip后缀 → awk按_切分取最后一段(sha) → 大写转小写
```

### 4.3 常用小命令（管道里的主角）

| 命令 | 作用 | 示例 |
| --- | --- | --- |
| `basename 路径 [后缀]` | 取文件名，可去后缀 | `basename /a/b.sh .sh` → `b` |
| `dirname 路径` | 取目录部分 | `dirname /a/b.sh` → `/a` |
| `wc -l` | 数行数 | `wc -l 文件` |
| `head -N` | 取前 N 行 | `head -1` |
| `tail -N` | 取后 N 行 | `tail -5` |
| `tr A B` | 字符替换 | `tr -d ' '` 删空格；`tr 'A-Z' 'a-z'` 转小写 |
| `awk -F_ '{print $NF}'` | 按`_`切分取最后字段 | `$NF`=最后一列 |
| `grep 关键词` | 过滤行 | `grep Error` |
| `sort` / `uniq` | 排序 / 去重 | 常组合：`sort \| uniq -c` |

---

## 5. 位置参数与特殊变量

```bash
# 用法: bash train.sh [config.yaml] [commit] [--local]
```

| 特殊变量 | 含义 |
| --- | --- |
| `$0` | 脚本名 |
| `$1` `$2` ... | 第 1、2… 个位置参数 |
| `$#` | 参数个数 |
| `$@` | 所有参数（每个独立） |
| `$*` | 所有参数（拼成一个字符串） |
| `$?` | 上一条命令的退出码 |

`${1:-}` = "参数1为空就当空串"，配合 `set -u` 不报错。

```bash
# train.sh 判断帮助请求
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "用法: $0 [config.yaml] [commit] [--local]" >&2
  exit 0
fi
```

---

## 6. if 条件判断

### 6.1 语法骨架

```bash
if <条件>; then
  ...
elif <条件>; then
  ...
else
  ...
fi        # if 倒过来写，表示结束
```

### 6.2 `[ ... ]` 条件表达式（test 命令的简写）

```bash
if [ -z "${BOS_ACCESS_KEY_ID:-}" ] && [ -f "${BOS_ENV_FILE:-/root/.bosenv}" ]; then
```

常用运算符：

| 写法 | 含义 |
| --- | --- |
| `[ "$a" = "$b" ]` | 字符串相等 |
| `[ "$a" != "$b" ]` | 字符串不等 |
| `[ -z "$a" ]` | 字符串为空 |
| `[ -n "$a" ]` | 字符串非空 |
| `[ -f 路径 ]` | 是普通文件 |
| `[ -d 路径 ]` | 是目录 |
| `[ -e 路径 ]` | 路径存在 |
| `[ -L 路径 ]` | 是软链接 |
| `[ "$a" -eq "$b" ]` | 数值相等（`-ne`不等 `-gt`大于 `-lt`小于） |

### 6.3 逻辑与短路

```bash
A && B     # A 成功才执行 B
A || B     # A 失败才执行 B
```

train.sh 实例：
```bash
[ "$SWANLAB_LOCAL" = 1 ] && export SWANLAB_MODE=local
[ -n "$ZIP" ] || { echo "ERROR: 没拉到源码 zip" >&2; exit 1; }
mkdir -p "$output_dir"
cp ... && cp ...    # 前一个成功才做下一个
```

`{ ... }`：把多条命令打包成一块；`{ echo; exit 1; }` = 打印错误并退出。

---

## 7. for 循环与 case 分支

### 7.1 for 循环

```bash
for _a in "$@"; do       # 遍历每个参数
  ...处理 $_a...
done

# train.sh 里建软链的循环：
for _l in /root/data:/mnt/cfs/data /root/code:/mnt/cfs/code; do
  link="${_l%%:*}"       # 取冒号前的部分
  target="${_l#*:}"      # 取冒号后的部分
  if [ -L "$link" ]; then :
  elif [ -e "$link" ]; then echo "ERROR" >&2; exit 1
  else ln -s "$target" "$link"; fi
done
```

注意 `:` 是 bash 的**空命令**（什么都不做），用于"if 分支里占位"。

### 7.2 case 多分支

```bash
case "$_a" in
  --local) SWANLAB_LOCAL=1 ;;    # 匹配 --local
  *) _pos+=("$_a") ;;            # * 匹配其它一切
esac
```

语法要点：`case` 倒过来写 `esac` 结束；每个分支用 `;;` 结尾；`*` 是"其它"兜底分支。

### 7.3 set -- 重置位置参数

```bash
set -- "${_pos[@]}"
```
把数组内容重新设置成 `$1 $2 ...`。train.sh 用它把 `--local` 摘出去后，剩下的参数归位。

---

## 8. 环境变量、export 与 source

### 8.1 export：让子进程也能读到

```bash
export VAR=值    # 导出：当前 shell + 所有子进程都能读
VAR=值           # 不导出：只有当前 shell 能读
```

train.sh 导出训练参数给 launcher 用：
```bash
export RUN_ID MAX_STEPS BATCH SAVE_EVERY LOG_EVERY NUM_WORKERS PREFETCH PORT WANDB_PROJECT NUM_GPUS
```

### 8.2 source（. 命令）：原样执行文件内容

```bash
. 文件        # 把文件内容"复制粘贴"进当前 shell 执行
source 文件   # 同上（bash 关键字）
```

train.sh 用它加载密钥文件：
```bash
set -a; . "${BOS_ENV_FILE:-/root/.bosenv}"; set +a   # set -a=自动export，之后source进来的赋值也变环境变量
. .venv/bin/activate                                 # 激活 python 虚拟环境
```

对比：`bash 子脚本` 是**开新进程**跑（环境不共享），`source` 是**当前进程**跑（环境共享）。

---

## 9. 重定向与文件描述符

| 写法 | 含义 |
| --- | --- |
| `> 文件` | 标准输出 → 写入文件（覆盖） |
| `>> 文件` | 标准输出 → 追加 |
| `< 文件` | 从文件读输入 |
| `2> 文件` | 错误输出 → 文件 |
| `2>&1` | 错误输出并入标准输出 |
| `2>/dev/null` | 丢弃错误（`/dev/null`=黑洞） |
| `>/dev/null` | 丢弃正常输出 |
| `>&2` | 把输出发到标准错误 |

文件描述符编号：0=标准输入，1=标准输出，2=标准错误。

train.sh 实例：
```bash
nvidia-smi -L 2>/dev/null | wc -l        # 没装显卡也不报错
echo "ERROR: ..." >&2                     # 错误信息走 stderr
trap 'rm -rf "$DL_TMP" 2>/dev/null || true' EXIT INT TERM   # 删临时目录失败也无所谓
```

---

## 10. 数组

```bash
_pos=()              # 定义空数组
_pos+=("$_a")        # 追加元素
"${_pos[@]}"         # 展开：每个元素一个独立参数（最常用）
```

train.sh 用法（收集非 --local 参数）：
```bash
SWANLAB_LOCAL=0
_pos=()
for _a in "$@"; do
  case "$_a" in
    --local) SWANLAB_LOCAL=1 ;;
    *) _pos+=("$_a") ;;
  esac
done
set -- "${_pos[@]}"
```

**为什么用数组不用字符串拼接**：数组能正确处理含空格的元素；字符串拼接会错乱。

---

## 11. glob 通配符

Shell 自带的文件名展开（不用调命令）：

```bash
*.zip        # 所有 .zip 结尾的文件
starvla_*    # 以 starvla_ 开头的文件
?.txt        # 单个字符
```

train.sh 实例：
```bash
ls -t "$DL_TMP"/starvla_*.zip 2>/dev/null | head -1
```

---

## 12. trap 信号陷阱

```bash
trap '命令' EXIT INT TERM    # 脚本退出/被Ctrl+C/被kill时，先执行命令
trap - EXIT INT TERM         # 清空该 trap
```

```bash
# train.sh：保证临时目录被删，不残留垃圾
DL_TMP="$(mktemp -d "$CODE_DIR/.srczip.XXXXXX")"
trap 'rm -rf "$DL_TMP" 2>/dev/null || true' EXIT INT TERM
...
rm -rf "$DL_TMP"; trap - EXIT INT TERM   # 用完清掉，防止误删
```

`|| true`：rm 失败也当成功，避免触发 `set -e` 退出。

---

## 13. 子 shell 与 flock 并发锁

```bash
(
  flock 9
  if [ -d "$REPO_DIR/.venv" ]; then
    echo ">> [段2] 复用已有环境"
  else
    rm -rf "$REPO_DIR" && mkdir -p "$REPO_DIR"
    unzip -q "$ZIP" -d "$REPO_DIR"
    ( cd "$REPO_DIR" && uv sync --python 3.10 )
  fi
) 9>"$CODE_DIR/.repo_$REAL_SHA.lock"
```

- `( ... )`：**子 shell**，里面 cd/变量不影响外面。
- `9>`：把文件描述符 9 绑到锁文件。
- `flock 9`：**文件锁**——同时只允许一个进程持有，其它排队。多任务共享 `/mnt/cfs/code` 不打架就靠它。
- `&&` 链：前一步成功才继续。

---

## 14. 字符串操作

train.sh 里的路径切分：
```bash
for _l in /root/data:/mnt/cfs/data; do
  link="${_l%%:*}"    # → /root/data
  target="${_l#*:}"   # → /mnt/cfs/data
done
```

| 写法 | 含义 | 例 |
| --- | --- | --- |
| `${VAR%%pattern}` | 从末尾删最长匹配 | 取冒号前 |
| `${VAR%pattern}` | 从末尾删最短匹配 | |
| `${VAR#pattern}` | 从开头删最短匹配 | 取冒号后 |
| `${VAR##pattern}` | 从开头删最长匹配 | |

---

## 15. 常用命令速查表

| 命令 | 作用 |
| --- | --- |
| `mkdir -p 目录` | 递归建目录，存在不报错 |
| `ln -s 目标 链接` | 建软链接 |
| `rm -rf 路径` | 强制递归删除（危险！） |
| `unzip -q 包 -d 目录` | 静默解压 |
| `cd 目录` / `pwd` | 切目录 / 打印当前目录 |
| `echo 文本` | 打印一行 |
| `exit N` | 以退出码 N 退出 |
| `source 文件` / `. 文件` | 在当前 shell 执行文件 |
| `mktemp -d 模板` | 建唯一临时目录 |
| `flock N` | 抢文件锁 N |
| `ls -t` / `head -1` | 按时间排序列出 / 取第一行 |
| `basename` / `dirname` | 取文件名 / 取目录 |
| `which 命令` | 找命令位置 |

---

## 16. bash 文件 vs ./ 直接执行

| | `bash scripts/train.sh` | `./train.sh` |
| --- | --- | --- |
| 解释器 | bash（显式） | Shebang 指定 |
| 权限要求 | 只需读权限 | 需要执行权限（`chmod +x`） |
| Shebang 是否生效 | 被忽略 | 生效 |

`BASH_SOURCE[0]` = 脚本自身路径，配合 `dirname`/`cd`/`pwd` 自定位，让脚本**从任何目录、任何方式调用**都能找到兄弟文件。

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

---

## 17. 综合练习：翻译一段真实脚本

把 train.sh 第 125-150 行"翻译"成大白话：

```bash
mkdir -p "$CODE_DIR"
DL_TMP="$(mktemp -d "$CODE_DIR/.srczip.XXXXXX")"          # 建唯一临时目录
trap 'rm -rf "$DL_TMP" 2>/dev/null || true' EXIT INT TERM  # 退出必删
uv run --no-project --with boto3 python "$BOS_IO" src-zip "$COMMIT" --dest-dir "$DL_TMP/"   # 拉源码zip
ZIP="$(ls -t "$DL_TMP"/starvla_*.zip 2>/dev/null | head -1)"   # 找最新的zip
[ -n "$ZIP" ] || { echo "ERROR: 没拉到源码 zip" >&2; exit 1; } # 没有就报错
REAL_SHA="$(basename "$ZIP" .zip | awk -F_ '{print $NF}' | tr 'A-Z' 'a-z')"   # 从文件名解析真实sha
REPO_DIR="$CODE_DIR/repo_$REAL_SHA"
(
  flock 9                                                  # 加锁排队
  if [ -d "$REPO_DIR/.venv" ]; then                        # 环境已建？
    echo ">> [段2] 复用已有环境"
  else
    rm -rf "$REPO_DIR" && mkdir -p "$REPO_DIR"             # 清旧建新
    unzip -q "$ZIP" -d "$REPO_DIR"                         # 解压
    ( cd "$REPO_DIR" && uv sync --python 3.10 )            # 子shell里建venv
  fi
) 9>"$CODE_DIR/.repo_$REAL_SHA.lock"
rm -rf "$DL_TMP"; trap - EXIT INT TERM                     # 清理临时目录
cd "$REPO_DIR"
. .venv/bin/activate                                       # 激活venv
```

**大白话**：在 `/root/code/` 下建个随机临时目录 → 从 BOS 按 commit 拉源码 zip → 找到最新那个 → 从文件名抠出真实 sha → 去 `repo_<sha>/` 加锁解压，若 venv 已存在就跳过 → 激活 venv，之后所有 `python`/`pip`/`accelerate` 都指向这份代码和这个环境。

---

## 18. 实战：run_umi_pi05_open_eval.sh 语法讲解

> 示例：`examples/realRobots/zerith_pi05/eval_files/run_umi_pi05_open_eval.sh`（UMI·PI05 开环评测 launcher）。
> 它跟 train.sh 风格不同：**大量用 `[[ ]]` 双中括号、数组收集透传参数、命令替换动态生成 CLI 参数**。本节逐段讲它用到、而前文没展开的语法。

### 18.1 脚本自定位（BASH_SOURCE[0]）

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
```

同 §16：不管从哪个目录调用，都能定位到脚本自己、再推出仓库根并切过去。`../../../..` 是因为脚本在 `examples/realRobots/zerith_pi05/eval_files/`，向上 4 级即仓库根。

### 18.2 命令存在性检查 `command -v`

```bash
command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}
```

- `command -v uv`：查 uv 是否在 PATH，在则打印路径（返回 0）
- `>/dev/null 2>&1`：输出和错误全丢黑洞，只留退出码
- `|| { ... }`：查不到（非 0）→ 打印提示并以失败退出

### 18.3 preflight 守卫：`[[ ]] || { ...; exit 1; }`

```bash
[[ -d "${RUN_DIR}" ]] || { echo "RUN_DIR not found: ${RUN_DIR}" >&2; exit 1; }
[[ -f "${EVAL_YAML}" ]] || { echo "EVAL_YAML not found: ${EVAL_YAML}" >&2; exit 1; }
```

条件不满足 → 打印到 stderr（`>&2`）→ 退出。脚本开头的"参数守卫"标配写法。

### 18.4 `[[ ]]` vs `[ ]`（本脚本主力，前文只讲了 `[ ]`）

```bash
[[ -n "${NUM_PROCESSES:-}" && "${NUM_PROCESSES}" -gt 1 ]]  # 设了且 >1
[[ -n "${DATA_MIX_OVERRIDE}" ]]                             # 非空
[[ -z "${LAST_OUTPUT}" ]]                                   # 为空
[[ -n "${SAVED_BASE_VLM_ABS}" && ! -e "${SAVED_BASE_VLM_ABS}" && -n "${BASE_VLM:-}" ]]  # 三重条件
```

| | `[ ... ]` | `[[ ... ]]` |
| --- | --- | --- |
| 来源 | test 命令 | bash 内建关键字 |
| 逻辑与/或 | 只能 `-a`/`-o` 或拆多个 `[ ]` | 直接 `&&` / `\|\|` |
| 空值 | 不引号包住容易炸 | 裸变量也不因空值报错 |
| 通配符/正则 | 不支持 | 支持 glob / `=~` 正则 |
| 数值比较 | `-eq -ne -gt -lt -ge -le` | 相同 |

结论：新脚本优先用 `[[ ]]`。

### 18.5 if / elif / else 多分支

```bash
if [[ -f "${RUN_DIR}/config.full.yaml" ]]; then
  SAVED_CONFIG="${RUN_DIR}/config.full.yaml"
elif [[ -f "${RUN_DIR}/config.yaml" ]]; then
  SAVED_CONFIG="${RUN_DIR}/config.yaml"
else
  echo "Missing config.full.yaml / config.yaml in ${RUN_DIR}" >&2
  exit 1
fi
```

优先完整 config，退化为旧 config，都没有就报错。`elif` 可无限续。

### 18.6 case 用 glob 匹配"路径形态"

```bash
case "${SAVED_BASE_VLM}" in
  ./*) SAVED_BASE_VLM_ABS="${REPO_ROOT}/${SAVED_BASE_VLM#./}" ;;   # 相对 ./xxx → 仓库根下
  /*)  SAVED_BASE_VLM_ABS="${SAVED_BASE_VLM}" ;;                   # 绝对 /xxx → 原样
  *)   SAVED_BASE_VLM_ABS="" ;;                                    # 其它（空 / HF id）→ 置空
esac
```

- `case` 分支里的模式就是 **glob**：`./*` 匹配"以 `./` 开头"，`/*` 匹配"以 `/` 开头"，`*` 兜底一切
- `${SAVED_BASE_VLM#./}`：从开头删掉最短匹配 `./`（§14 字符串操作）
- 判断 base_vlm 是相对路径、绝对路径、还是 HF repo id（PI05 无此键 → 空 → 走 `*` 分支）

### 18.7 命令替换里跑多行 python

```bash
SAVED_BASE_VLM=$(uv run python -c 'from omegaconf import OmegaConf; import sys
cfg = OmegaConf.load(sys.argv[1])
try:
    print(cfg.framework.qwenvl.base_vlm)
except Exception:
    print("")' "${SAVED_CONFIG}")
```

- `$(...)` 捕获 `uv run python` 的 stdout（§4.1）
- 单引号包多行 python：单引号内**不做任何变量/转义展开**，代码原样传给 `-c`
- `print(...)` 的输出即命令替换结果；try/except 拿不到键时输出空行 → 变量为空
- 末尾 `${SAVED_CONFIG}` 是传给 python 的 `sys.argv[1]`

### 18.8 数组：收集透传参数（本脚本核心套路）

```bash
DO_VISUALIZE=1
STAGE2_ONLY=0
STAGE1_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --visualize)    DO_VISUALIZE=1 ;;
    --no-visualize) DO_VISUALIZE=0 ;;
    --stage2-only)  STAGE2_ONLY=1 ;;
    *) STAGE1_ARGS+=("${arg}") ;;   # 其余参数原样收集
  esac
done
```

同 §7.2/§10 的 train.sh 套路：**launcher 只认自己的 flag，剩下的"原封不动"透传给 Stage 1**。

**数组长度 + "glob 生成数组"**（Stage 2 部分）：

```bash
EPISODE_DIRS=("${LAST_OUTPUT}"/episode_*/)        # glob 展开成数组
if [[ ${#EPISODE_DIRS[@]} -eq 0 ]]; then          # 一个都没匹配到？
  echo "WARNING: no episode_*/ in ${LAST_OUTPUT}" >&2
else
  PYTHONPATH="${REPO_ROOT}" uv run python "${STAGE2_SCRIPT}" "${EPISODE_DIRS[@]}"
fi
```

- `"${#EPISODE_DIRS[@]}"` = 数组**元素个数**（新语法）
- `"${dir}"/episode_*/` 赋值时由 shell 做 glob 展开，每个匹配目录成一个元素
- ⚠️ 默认（无 `shopt -s nullglob`）glob **没匹配到时不报错，而是保留字面量 `.../episode_*`**（数组长度=1，`-eq 0` 不会触发）。本脚本这里靠 Stage 2 自己跳过"缺 predictions.npz 的目录"兜底。要"没匹配即空数组"需先 `shopt -s nullglob`。

### 18.9 条件性生成 CLI 参数（最重要的语法点）

```bash
PYTHONPATH="${REPO_ROOT}" uv run python -m starVLA.training.eval_starvla \
  --run-dir "${RUN_DIR}" \
  --eval-yaml "${EVAL_YAML}" \
  $( [[ -n "${DATA_ROOT_OVERRIDE}" ]] && echo --data-root-override "${DATA_ROOT_OVERRIDE}" ) \
  $( [[ -n "${DATA_MIX_OVERRIDE}" ]] && echo --data-mix "${DATA_MIX_OVERRIDE}" ) \
  "${STAGE1_ARGS[@]}" 2>&1 | tee "${STAGE1_LOG}"
```

拆解 `$( [[ -n "${DATA_ROOT_OVERRIDE}" ]] && echo --data-root-override "${DATA_ROOT_OVERRIDE}" )`：

1. `[[ -n ... ]]` 判断环境变量非空
2. 为真 → `&&` 执行 `echo --data-root-override /root/data/z-umi/data` → 命令替换输出这串**参数**
3. 为假 → `&&` 短路，什么都不输出 → 这组 flag 从命令行消失

结果：**设了才传，不设就不传**，让 Stage 1 内部走"从 ckpt config 读"的默认路径。这是 bash 里"可选参数"最经典的写法。

- `\` 行尾续行，多行拼成一条命令
- `${STAGE1_ARGS[@]}` 逐个展开收集的参数（含空格也不拆坏）
- `2>&1 | tee "${STAGE1_LOG}"`：stderr 并进 stdout 后 tee 一份到临时日志（§18.10 用它）

### 18.10 临时日志 + trap 清理

```bash
STAGE1_LOG="$(mktemp)"
trap 'rm -f "${STAGE1_LOG}"' EXIT
```

- `mktemp` 建唯一临时文件（§15）
- `trap ... EXIT`：脚本**任何方式退出**（正常/报错）都删日志，不留垃圾（§12）

### 18.11 grep + tail + sed 从日志精确提取 output_dir

```bash
LAST_OUTPUT="$( { grep '__EVAL_OUTPUT_DIR__' "${STAGE1_LOG}" 2>/dev/null || true; } \
  | tail -1 \
  | sed 's/.*__EVAL_OUTPUT_DIR__[[:space:]]*//')"
```

- `{ grep ... || true; }`：命令组；`|| true` 防 grep 没匹配（非 0 退出）触发 `set -e` 把脚本弄挂
- `tail -1`：取最后一行（`__EVAL_OUTPUT_DIR__` 主进程只打印一次）
- `sed 's/.*__EVAL_OUTPUT_DIR__[[:space:]]*//'`：把"匹配点前的任意内容 + 后续空白"删掉，只留路径
  - `.*` 贪婪匹配任意字符，`[[:space:]]*` 空白字符类，`//` 表示替换为空

### 18.12 ls -td 目录 glob 回退

```bash
LAST_OUTPUT="$(ls -td playground/Evals/*/ 2>/dev/null | head -1)"
LAST_OUTPUT="${LAST_OUTPUT%/}"     # 去掉末尾斜杠
```

- `playground/Evals/*/`：**末尾带 `/` 的 glob 只匹配目录**，不匹配文件
- `ls -t`：按修改时间倒序 → 第一行 = 最新；`-d`：只列目录本身，不展开内容
- `2>/dev/null`：glob 无匹配时 ls 报错，丢掉
- `head -1`：取最新那个
- `${LAST_OUTPUT%/}`：删掉末尾一个 `/`（§14）

> 这是 Stage 1 没打 `__EVAL_OUTPUT_DIR__` 标记时的**按 mtime 回退**（多 run 共存可能串目录，所以主路径用 §18.11 的标记，这里只兜底）。

### 18.13 本节涉及语法 vs 手册正文

| 本节 | 语法点 | 手册章节 |
| --- | --- | --- |
| 18.1 | `BASH_SOURCE[0]` 自定位 | §16 |
| 18.2 | `command -v` / `\|\| { }` | §4 / §6 / §15 |
| 18.3 / 18.5 | `[[ ]]` preflight / if-elif-else | §6 |
| 18.4 | `[[ ]]` vs `[ ]` 对比 | 新增 |
| 18.6 | case + glob + `${VAR#pattern}` | §7 / §14 |
| 18.7 | `$( )` 跑多行 python | §4 |
| 18.8 | 数组收集 / `${#arr[@]}` / glob 数组 | §10 |
| 18.9 | 条件性生成 CLI 参数 | 新增（重点） |
| 18.10 | `mktemp` + `trap` | §12 / §15 |
| 18.11 | grep / tail / sed 提取 | §4 / §15 |
| 18.12 | `ls -td */` + `${VAR%/}` | §11 / §14 |

---

## 附：速查卡（可打印）

```
赋值:        VAR=值（= 两边不能有空格）
取值:        "$VAR" / "${VAR}"
默认值:      "${VAR:-默认}"
环境变量:    export VAR=值
执行文件:    source 文件 | . 文件
临时覆盖:    VAR=值 命令
判断:        if [ 条件 ]; then ...; fi
循环:        for x in 列表; do ...; done
分支:        case $x in a) ;; *) ;; esac
取值替换:    $(命令)
管道:        命令1 | 命令2
错误丢弃:    命令 2>/dev/null
输出到错误:  echo xxx >&2
退出:        exit 0/1
数组:        arr=(); arr+=("x"); "${arr[@]}"
锁:          ( flock N; ... ) N>锁文件
清理钩子:    trap '命令' EXIT INT TERM
```
