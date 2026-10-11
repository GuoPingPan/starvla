# 代理冲突问题排查与修复记录

> 环境：Ubuntu 24.04 / bash（登录 shell）
> 问题现象：`curl`、`uv`、`boto3` 连外网报 `Connection refused (os error 111)`，指向 `127.0.0.1:7897`
> 结论：本机存在两套代理变量（大小写）指向不同端口，且登录时被覆盖成死端口；最终通过 `.bashrc` 收敛到唯一活隧道解决。

---

## 1. 问题现象

### 现象 A：curl 直连失败

```bash
curl -I https://www.baidu.com
# curl: (7) Failed to connect to 127.0.0.1 port 7897 after 0 ms: Couldn't connect to server
```

### 现象 B：显式指定代理成功

```bash
curl -x http://127.0.0.1:64500 -I https://www.baidu.com
# HTTP/1.1 200 Connection established → HTTP/1.1 200 OK
```

### 现象 C：train.sh 段 2 拉代码失败（uv 拉 boto3 / boto3 连 BOS）

```
error: Request failed after 3 retries
  Caused by: Failed to fetch: `https://mirrors.aliyun.com/pypi/simple/boto3/`
  Caused by: tunnel error: failed to create underlying connection
  Caused by: tcp connect error → Connection refused (os error 111)
```

以及 boto3 的完整堆栈（关键行）：

```
botocore.exceptions.ProxyConnectionError: Failed to connect to proxy URL: "http://127.0.0.1:7897"
```

---

## 2. 排查过程

### 2.1 先确认"网络本身通不通"

用 curl 直连 aliyun/pypi 都返回 200（走代理后通），说明**外网可达，问题出在代理配置**。

### 2.2 列出所有代理环境变量（发现两套）

```bash
env | grep -i proxy
```

| 变量 | 值 | 端口状态 |
| --- | --- | --- |
| `http_proxy` / `https_proxy`（小写） | `http://127.0.0.1:64500` | ✅ 活着 |
| `HTTP_PROXY` / `HTTPS_PROXY`（大写） | `http://127.0.0.1:7897` | ❌ 死了 |
| `ALL_PROXY` / `all_proxy` | `http://127.0.0.1:7897` | ❌ 死了 |

**核心发现：同一台机器有两套代理，小写指 64500、大写指 7897。**

### 2.3 确认两个端口谁在监听

```bash
ss -tlnp | grep -E "64500|7897"
# LISTEN 127.0.0.1:64500  sshd-session   ← 有服务
# (7897 无任何输出)                        ← 没服务
```

**只有 64500 在监听，7897 无监听** → 指向 7897 的配置必然 Connection refused。

### 2.4 查明"为什么 curl 也走 7897"

curl 本该优先读小写，但现象是 curl 默认也连 7897。进一步查加载来源：

```bash
cat ~/.profile
# if [ -f ~/.codex_proxy_env.sh ]; then . ~/.codex_proxy_env.sh; fi

cat ~/.codex_proxy_env.sh
# export HTTP_PROXY=http://***      ← 全是大写端口 7897
# export HTTPS_PROXY=http://***
# export ALL_PROXY=http://***
# export http_proxy=http://***      ← 连小写也被覆盖成 7897！
# export https_proxy=http://***
# export all_proxy=http://***
```

**根因**：`~/.profile`（登录时加载）会 source Codex 的代理脚本，把**大小写所有**代理都设成 `127.0.0.1:7897`（Codex 反向 SSH 隧道，当前未启用）。

### 2.5 确认 Shell 加载顺序（决定修复放哪）

```bash
cat ~/.bash_profile
# if [ -f ~/.bashrc ]; then . ~/.bashrc; fi
```

加载顺序：

```
bash(登录shell)
 ├─ 1. /etc/profile
 ├─ 2. ~/.profile          ← Codex 代理(7897)在这里写入
 ├─ 3. ~/.bash_profile
 │     └─ . ~/.bashrc      ← 修复放在这里，后加载可覆盖
 └─ 交互提示符
```

---

## 3. 为什么不同程序表现不同（大小写代理之争）

Unix 代理变量有两套命名，**不同工具优先读的套别不一样**：

| 程序 | 优先读 | 行为 |
| --- | --- | --- |
| `curl` / `wget` | 小写 `http_proxy` | 被 .profile 覆盖后也走 7897 |
| `uv` / `pip`（reqwest/requests 系） | **大写** `HTTP_PROXY` | 走 7897 → refused |
| `boto3`（botocore/urllib3） | **大写** `HTTP_PROXY` | 走 7897 → refused（见报错堆栈） |
| 命令行 `curl -x http://127.0.0.1:64500` | 显式指定，优先级最高 | 走 64500 → 200 |

> 这正是"curl 手动带 `-x 64500` 能通，但 uv/boto3 全炸"的原因：它们读了不同套别的变量。

---

## 4. 修复方案

### 4.1 临时方案（每次跑命令前）

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# 此时 curl/uv/boto3 都会回退到小写 http_proxy
```

或一次性：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  bash scripts/train.sh <yaml> <commit>
```

### 4.2 永久方案（推荐，写入 ~/.bashrc 末尾）

```bash
cat >> ~/.bashrc <<'EOF'

# [fix] 覆盖 .profile 的 Codex 代理(7897 隧道未启用)，改用 64500 隧道
export http_proxy=http://127.0.0.1:64500
export https_proxy=http://127.0.0.1:64500
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
EOF
```

**为什么有效**：
- `.bashrc` 在 `.profile` **之后**加载 → 后赋值覆盖先赋值；
- 小写强制指向活端口 64500（覆盖掉 Codex 写入的 7897）；
- 大写全部 unset → uv/boto3 读不到坏地址，自动回退小写。

### 4.3 验证修复

```bash
# 1) 模拟新登录 shell 看变量
bash -lc 'echo "HTTP_PROXY=${HTTP_PROXY:-空} | http_proxy=${http_proxy:-空} | all_proxy=${all_proxy:-空}"'
# 期望：HTTP_PROXY=空 | http_proxy=http://127.0.0.1:64500 | all_proxy=空

# 2) 实测连通性
bash -lc 'curl -sI -m 10 https://www.baidu.com | head -1'
# 期望：HTTP/1.1 200 OK
```

修复前后对比（模拟登录 shell）：

| 变量 | 修复前 | 修复后 |
| --- | --- | --- |
| `HTTP_PROXY` | `http://127.0.0.1:7897` | 空 |
| `http_proxy` | `http://127.0.0.1:7897` | `http://127.0.0.1:64500` |
| `all_proxy` | `http://127.0.0.1:7897` | 空 |

---

## 5. 涉及的端口与隧道背景

| 端口 | 来源 | 状态 | 说明 |
| --- | --- | --- | --- |
| `127.0.0.1:64500` | `sshd-session` 反向隧道 | ✅ 监听 | 可用出口 |
| `127.0.0.1:7897` | Codex 反向 SSH 隧道（`~/.codex_proxy_env.sh`） | ❌ 无监听 | 未启用 |

**反向 SSH 隧道原理**：内网机器主动 SSH 到跳板机，把远程端口反向转发回本地，使 `127.0.0.1:<port>` 成为能出外网的出口。隧道没拉起时，端口无监听 → 任何指向它的程序都会 `Connection refused`。

### 5.1 SSH 隧道深入：LocalForward vs RemoteForward

> 背景：曾有人提出"把 `RemoteForward 64500 127.0.0.1:7897` 改成 `RemoteForward 7897 127.0.0.1:7897` 就能让本地 7897 变通"。这节解释为什么**方向错了、解决不了**。

#### 5.1.1 转发方向速记（一句话）

```
LocalForward  ：监听在「本地」→ 经 SSH 服务器转发出去
RemoteForward ：监听在「远端」→ 反向转发回本地
```

| 指令 | 监听端口落在哪一端 | 目标 `host:port` 从哪一端解析 |
| --- | --- | --- |
| `LocalForward [bind:]port host:port` | **本地**（运行 ssh 的机器） | **远端**角度解析 |
| `RemoteForward [bind:]port host:port` | **远端**（SSH 服务器） | **本地**角度解析 |

#### 5.1.2 图解

**LocalForward**（常用：把远端服务映射到本地）

```
本地(客户端)                         远端(SSH服务器)
ssh -L 本地8080:目标机:80 目标机
  ┌────────────────┐              ┌────────────────┐
  │ 127.0.0.1:8080 │←──SSH隧道──→│  连到 目标机:80 │
  │  ← 在这里监听   │              └────────────────┘
  └────────────────┘
  curl http://127.0.0.1:8080  → 实际访问 目标机:80
```

**RemoteForward**（常用：让远端能访问"本地才有的服务"）

```
本地(客户端)                         远端(SSH服务器)
ssh -R 远端8080:本地服务:80 远端机
  ┌────────────────┐              ┌────────────────┐
  │  本地服务:80    │←──SSH隧道──→│ 127.0.0.1:8080  │
  │  ← 目标在这里   │              │ ← 在这里监听    │
  └────────────────┘              └────────────────┘
  远端机访问 http://127.0.0.1:8080  → 实际连到本地服务:80
```

#### 5.1.3 本机例子逐句拆解

```bash
Host 120.48.62.164
  RemoteForward 64500 127.0.0.1:7897
```

翻译成人话：**在 120.48.62.164 上监听 64500，任何连过来的请求反向转发回"本地"的 127.0.0.1:7897。**

- 监听的是**远端** 120.48.62.164:64500；
- 转发目标是**本地** 127.0.0.1:7897。

#### 5.1.4 为什么改成 `RemoteForward 7897 127.0.0.1:7897` 不行

```
120.48.62.164:7897 开始监听  ← 端口开在远端！
        ↑
（不是本地！）
        ↓
连 120.48.62.164:7897 的请求
        ↓
反向转发回本地 127.0.0.1:7897  ← 本地 7897 依然没人监听
```

两个致命问题：

1. **本地 `127.0.0.1:7897` 依旧无监听**。你的 curl/uv/boto3 连的是**本地** 7897，而这条指令开的是**远端** 7897 → 本地照旧 `Connection refused`。
2. **转发目标还是本地 7897**。就算远端 7897 有人连，请求被反向转发回本地 `127.0.0.1:7897`——本地没服务 → 依旧失败。等于"把死端口绕一圈又指回死端口"。

#### 5.1.5 如果真的想让"本地 7897"变通

正确方向是 `LocalForward`，且目标要指向一个**真的能出网**的地方（比如已验证的 64500）：

```bash
LocalForward 7897 127.0.0.1:64500
# 本地 7897 监听 → 转发到本地 64500 → 出网
```

但这是**绕路**：64500 已经能用，直接让程序用 64500 即可（即本节的 bashrc 修复），无需多一层转发。

#### 5.1.6 排查"端口到底监听在哪端"

| 想确认 | 命令 |
| --- | --- |
| 本地有哪些端口监听 | `ss -tlnp \| grep <端口>`（本机） |
| 远端有哪些端口监听 | `ssh 120.48.62.164 'ss -tlnp \| grep <端口>'`（登录到远端看） |

> 经验：程序连 `127.0.0.1:<port>` 报 refused，第一件事就是在本机 `ss -tlnp` 看这个端口有没有监听——**监听不存在时，任何代理配置都是白搭**。这正是本案例的根因：本地 7897 无监听，而 64500 有监听。

---

## 6. 排查工具速查

| 目的 | 命令 |
| --- | --- |
| 看当前所有代理变量 | `env \| grep -i proxy` |
| 看端口监听 | `ss -tlnp \| grep <端口>` |
| 测某个代理通不通 | `curl -x http://127.0.0.1:<port> -I https://www.baidu.com` |
| 看 Shell 加载了什么 | `bash -x -lc 'echo ok'`（跟踪每行） |
| 模拟新登录 shell 看变量 | `bash -lc 'env \| grep -i proxy'` |
| 临时清掉坏代理 | `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy` |

---

## 7. 相关文档

- 冒烟测试与训练命令：`docs/zerith_docs/umi/umi_pi05_smoke_test.md`
- 训练流水线详解：`docs/zerith_docs/infra/train_sh_run_flow.md`
- Bash 语法速学：`docs/zerith_docs/guides/bash_basics.md`
