# VSCode 调试配置 launch.json 详解（以 UMI PI05 开环评测为例）

> 讲解 `.vscode/launch.json` 里"Eval: UMI PI05 open-loop (stage1)"这份配置的写法，以及每个字段的含义。
> 目标读者：会在 VSCode 里 F5 调试 Python 的人。

---

## 1. 先说结论：它在做什么

这份配置 = **把 launcher（run_umi_pi05_open_eval.sh）最终拼出来的那条 Stage 1 命令，翻译成 VSCode 能直接断点调试的格式**。

bash 命令：

```bash
PYTHONPATH=/root/workspace/starvla uv run python -m starVLA.training.eval_starvla \
  --run-dir <RUN_DIR> \
  --eval-yaml <EVAL_YAML> \
  --data-root-override /root/data/z-umi/data \
  --checkpoint-step 10000 --episodes 0 1 \
  --dataset-name <...>
```

launch.json 配置（字段逐项对应，见 §5 对照表）。

---

## 2. launch.json 是什么

- 文件：项目根目录 `.vscode/launch.json`
- 作用：VSCode 的 **"调试启动配置"**——按下 F5 时，VSCode 按这里定义的方式启动程序并挂上调试器。
- 格式：**JSONC**（带注释的 JSON）。所以可以写 `//` 注释（如上文的 ⚠️ 提醒），这是合法写法，普通 JSON 不行。
- 顶层结构：

```jsonc
{
    "version": "0.2.0",              // 固定值，别动
    "configurations": [ ... ]        // 配置列表，F5 时可选其中一个
}
```

- `"version": "0.2.0"`：launch.json 格式的 schema 版本，VSCode 约定固定写法，不要改。
- `"configurations": [...]`：数组，里面每个 `{...}` 就是一个可选的调试配置。你可以在 Run 面板下拉里选哪一个。

---

## 3. 逐字段详解

### 3.1 `"name"`

```json
"name": "Eval: UMI PI05 open-loop (stage1)"
```

- 含义：**这份配置显示在 Run 面板下拉列表里的名字**，随便起，只为你自己区分。
- 惯例：动词+对象，让人一眼知道是什么（这里是"评测 PI05 开环 Stage 1"）。

### 3.2 `"type"`

```json
"type": "debugpy"
```

- 含义：**用哪个调试器**。Python 必须写 `"debugpy"`（VSCode 的 Python 扩展内置的调试器）。
- 历史：旧版本叫 `"python"`，新版统一为 `"debugpy"`。如果你的 Python 扩展版本老，可能需要写 `"python"`。

### 3.3 `"request"`

```json
"request": "launch"
```

- 含义：**调试模式**，二选一：

| 值 | 含义 | 场景 |
| --- | --- | --- |
| `"launch"` | VSCode **自己启动**程序并附上调试器 | 最常用（我们这份） |
| `"attach"` | 程序**已经在外头跑着**（如 ssh 远程起的），VSCode 去"挂"上去 | 调试已运行的服务/多卡任务 |

### 3.4 `"module"`（对应 `-m`）

```json
"module": "starVLA.training.eval_starvla"
```

- 含义：**以模块方式运行**，等价于命令行 `python -m starVLA.training.eval_starvla`。
- 注意：`-m` 后面是**点分模块路径**（包.模块），不是文件路径。
- 与之互斥的另一种写法是 `"program": "路径/文件.py"`（运行脚本文件）。本配置用 `module`，是因为原命令用的是 `-m`。

### 3.5 `"python"`（对应 `uv run` 选解释器）

```json
"python": "${workspaceFolder}/.venv/bin/python"
```

- 含义：**用哪个 python 解释器**跑这份程序。
- 为什么这么写：原命令是 `uv run python`——uv 只是"激活项目 venv 再跑"，等价于直接调 `.venv/bin/python`。这里就指向它，效果一致（不用跑 uv 也能 debug）。
- `${workspaceFolder}`：VSCode 内置**变量**，展开为"当前打开的工作区根目录"，即 `/root/workspace/starvla`。用它而不是写死绝对路径，换机器也能用。
- 常用内置变量：

| 变量 | 展开为 |
| --- | --- |
| `${workspaceFolder}` | 工作区根目录 |
| `${file}` | 当前活动文件 |
| `${env:NAME}` | 环境变量 NAME |

### 3.6 `"cwd"`（工作目录）

```json
"cwd": "${workspaceFolder}"
```

- 含义：**程序启动时的当前工作目录**（相当于在哪个目录执行）。
- 为什么必须设：`eval_starvla.py` 里相对路径（如 `--eval-yaml examples/...`）都要从仓库根解析；bash 版本里 launcher 会 `cd "${REPO_ROOT}"`，launch.json 里就用 `"cwd"` 等价实现。
- 若不设，VSCode 默认用"上次会话的工作目录"，可能不是仓库根，相对路径就会找不到。

### 3.7 `"console"`

```json
"console": "integratedTerminal"
```

- 含义：**程序的标准输入/输出（print、stdout）显示在哪**，三选一：

| 值 | 效果 |
| --- | --- |
| `"integratedTerminal"` | 在 VSCode 内置终端里跑，能看到完整输出、能交互（推荐，评测日志很长选这个） |
| `"internalConsole"` | 在"调试控制台"面板，能看但交互弱、长输出会被截断 |
| `"externalTerminal"` | 弹独立系统终端窗口 |

- 评测要打大量日志（`tee`、进度条），选 `integratedTerminal` 最合适。

### 3.8 `"env"`（环境变量，对应 `PYTHONPATH=...`）

```json
"env": {
    "PYTHONPATH": "${workspaceFolder}"
}
```

- 含义：**启动前设置的环境变量**，等价于 bash 里 `PYTHONPATH=/root/workspace/starvla python ...`。
- 为什么需要 `PYTHONPATH`：`eval_starvla.py` 用 `import starVLA...` 导入项目包；`starVLA` 包就在仓库根，只有把仓库根加进 `PYTHONPATH`，python 才能 `import starVLA` 成功。bash 版本靠 `PYTHONPATH="${REPO_ROOT}"`，这里等价。
- `"env"` 的键值：键=变量名，值=内容；`${workspaceFolder}` 变量同样可用。

### 3.9 `"args"`（命令行参数）

```json
"args": [
    "--run-dir", "/root/data/model_outputs/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0805_214632",
    "--eval-yaml", "examples/realRobots/zerith_pi05/eval_files/eval_umi_pi05_relative_6d_overlay.yaml",
    "--data-root-override", "/root/data/z-umi/data",
    "--checkpoint-step", "10000",
    "--episodes", "0", "1",
    "--dataset-name", "20260803035455_lerobot_5852395f_noep81_forceclose"
]
```

- 含义：**逐个命令行参数**，等价于 bash 里 `python -m ... 后面那串`。
- 格式要点：**每个参数单独一个字符串**，数组元素顺序就是参数顺序。

| 元素 | 对应 bash |
| --- | --- |
| `"--run-dir"` + 路径 | `--run-dir <RUN_DIR>` |
| `"--episodes"` + `"0"` + `"1"` | `--episodes 0 1`（值也拆成独立元素） |
| `"--dataset-name"` + 名字 | `--dataset-name <...>` |

- **值和参数名必须分开成两个元素**：`"--checkpoint-step", "10000"`，不能写成一个 `"--checkpoint-step 10000"`。

---

## 4. 完整配置 + 与 bash 命令的对照

```jsonc
{
    "name": "Eval: UMI PI05 open-loop (stage1)",
    "type": "debugpy",
    "request": "launch",
    "module": "starVLA.training.eval_starvla",
    "python": "${workspaceFolder}/.venv/bin/python",
    "cwd": "${workspaceFolder}",
    "console": "integratedTerminal",
    "env": { "PYTHONPATH": "${workspaceFolder}" },
    "args": [
        "--run-dir", "/root/data/model_outputs/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0805_214632",
        "--eval-yaml", "examples/realRobots/zerith_pi05/eval_files/eval_umi_pi05_relative_6d_overlay.yaml",
        "--data-root-override", "/root/data/z-umi/data",
        "--checkpoint-step", "10000",
        "--episodes", "0", "1",
        "--dataset-name", "20260803035455_lerobot_5852395f_noep81_forceclose"
    ]
}
```

| launch.json 字段 | bash 命令对应 | 作用 |
| --- | --- | --- |
| `"python"` | `uv run python` 里的解释器 | 用哪个 python |
| `"module"` | `-m starVLA.training.eval_starvla` | 运行哪个模块 |
| `"cwd"` | launcher 里的 `cd "${REPO_ROOT}"` | 在哪执行（相对路径基准） |
| `"env": {"PYTHONPATH": ...}` | `PYTHONPATH="${REPO_ROOT}"` | import 找到项目包 |
| `"args"` | `--run-dir ...` 之后整串 | 命令行参数 |
| `"console"` | `2>&1 \| tee` 的"上屏"部分 | 输出显示在哪 |
| `"request": "launch"` | 直接起进程 | 调试方式 |
| `"type": "debugpy"` | —— | 调试器选型 |

---

## 5. 调试器的工作方式（为什么能断点）

`"request": "launch"` 时 VSCode 做的事：

1. 用 `"python"` 指定的解释器，在 `"cwd"` 下，带 `"env"` 和 `"args"`，以 `-m <module>` 启动你的程序；
2. 同时启动 debugpy 调试服务器，**在进程启动那一刻就接管**；
3. 代码里打在任意 `.py` 的**断点**（红点）命中即暂停，可看变量/单步/看调用栈；
4. 跑到 `build_dataset → pick_mixture_entry` 抛 `ValueError` 时，VSCode 会**在异常处停住**（"调试程序: 异常"断点），你可以在调用栈里点开每一层看当时的变量值——这正是调试这条报错路径的最佳方式。

---

## 6. 常见变体（备查）

### 6.1 想调试别的文件：用 program 代替 module

```jsonc
{
    "name": "跑指定文件",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",            // 调试当前打开的文件
    "console": "integratedTerminal"
}
```

这就是 launch.json 里那份默认配置（"Python 调试程序: 当前文件"）——`${file}` 动态指向当前活动文件。

### 6.2 只想看项目自己的代码：justMyCode

```jsonc
{
    "justMyCode": true    // 默认 true：跳过第三方库（不进 .venv 里的代码）
}
```

调试评测时如果想知道 `pick_mixture_entry` 内部发生了什么，可以临时设 `"justMyCode": false`，这样断点能进 `checkpoint_utils.py` 等库文件。

### 6.3 attach 已运行的进程

```jsonc
{
    "name": "attach 到外部进程",
    "type": "debugpy",
    "request": "attach",
    "connect": { "host": "localhost", "port": 5678 }  // 对方进程 debugpy 监听端口
}
```

---

## 7. 使用步骤（怎么用这份配置）

1. 打开仓库根目录 `/root/workspace/starvla` 为工作区；
2. 在 `starVLA/training/eval_starvla.py`（或 `starVLA/eval/core/checkpoint_utils.py`）要看的行**打断点**（点行号左边）；
3. 左侧 Run 面板（Ctrl+Shift+D）→ 顶部下拉选 **"Eval: UMI PI05 open-loop (stage1)"**；
4. 按 **F5** 或点绿色 ▶；
5. 程序在断点处停下，就可以单步（F10）、步入（F11）、查看变量、看调用栈。

> ⚠️ 提醒：这份配置带 `--dataset-name`，直接跑会在 `build_dataset` 抛 `ValueError`。**这反而是调试的好机会**——在 `checkpoint_utils.py:171` 打断点，就能看到 `pick_mixture_entry` 为什么拒绝（可用数据集列表 vs 你要的那个）。要让它跑通，见注释里的两个解法。

---

## 8. 相关链接

- VSCode launch.json 官方文档：https://code.visualstudio.com/docs/editor/debugging
- Python 调试（debugpy）配置：https://code.visualstudio.com/docs/python/debugging
- 本配置对应的命令行来源：`docs/zerith_docs/umi/umi_pi05_open_eval.md`
- Bash 侧等价命令讲解：`docs/zerith_docs/guides/bash_basics.md` §18
