# Git 分支合并操作指南（rebase + 强推 + MR）

本文档记录把个人分支 `dev/Alex` 与主线 `dev/main` 合并的操作流程，选用 **rebase** 方案以保持线性历史，最后强推并提交 Merge Request。

## 1. 背景概念

### 1.1 查看当前分支及其跟踪的远程分支

```bash
git branch --show-current          # 当前分支名，如 dev/Alex
git rev-parse --abbrev-ref --symbolic-full-name @{upstream}   # 跟踪的远程分支，如 origin/dev/Alex
```

### 1.2 merge vs rebase

| | Merge | Rebase |
|---|---|---|
| 历史 | 保留分叉，有合并节点 | 线性，无分叉 |
| 提交 hash | 不变 | 改写 |
| 已推送分支 | 安全 | 需强推，有风险 |
| 适用 | 团队协作、长期分支 | 个人分支、提交未推送时 |

### 1.3 rebase 方向（容易搞混）

规则：**当前分支的提交会被重放到目标分支之上。**

```bash
git checkout dev/Alex
git rebase dev/main   # dev/Alex 的提交 → 重放到 dev/main 之上（推荐做法）
```

```bash
git checkout dev/main
git rebase dev/Alex   # 反方向：dev/main 的提交 → 重放到 dev/Alex 之上（一般不这么做）
```


### 1.4 git fetch vs git pull

- `git fetch origin`：只把远端最新提交下载到本地，**不改变工作区**、不合并
- `git pull`：相当于 `fetch` + `merge`，会真正更新本地分支

## 2. 完整操作流程

### 步骤 1：同步本地

```bash
git fetch origin                # 下载远端最新状态

git checkout dev/main
git pull                        # 更新本地 dev/main 到远端最新

git checkout dev/Alex
git pull                        # 更新本地 dev/Alex
```

### 步骤 2：Rebase 到 dev/main 上

```bash
git rebase dev/main
```

- 成功则直接进入步骤 3
- 失败（有冲突）时看第 3 节

### 步骤 3：强推远端 dev/Alex

```bash
git push --force-with-lease origin dev/Alex
```

> 用 `--force-with-lease` 而非 `--force`：推送前会检查远端是否有他人新推送的提交，有则拒绝，避免覆盖他人工作。

### 步骤 4：在 GitLab 发起 Merge Request

```bash
# 若安装了 glab CLI
glab mr create --source-branch dev/Alex --target-branch dev/main
```

无 CLI 时浏览器访问远端仓库 → New Merge Request → 选 `dev/Alex` → `dev/main`。

## 3. 冲突处理

Rebase 逐提交重放时若某提交改动的文件与目标基线有冲突，会暂停并提示：

```
CONFLICT (content): Merge conflict in <file>
error: could not apply <hash>... <提交信息>
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add/rm <conflicted_files>", then run "git rebase --continue".
```

### 3.1 分析冲突

```bash
git status            # 查看冲突文件
git show <hash> -- <file>   # 查看出问题的提交对这个文件具体改了哪些行
```

冲突标记含义（`<<<<<<< HEAD` 是目标基线（dev/main）的内容，`>>>>>>> <hash>` 是被重放提交的内容）：

```
<<<<<<< HEAD
... dev/main 的内容 ...
=======
... 被重放提交的内容 ...
>>>>>>> 670f2b1 (...)
```

### 3.2 解决冲突（实例）

本次实际遇到的冲突：提交 `670f2b1` 只改动了一行（取消某 entry 的注释），但 dev/main 在该区域新增了 `_trim`/`_trim0` 两段，Git 无法自动定位。

解决原则：**两边内容都要**——dev/main 新增的段保留，被重放提交对该行的修改保留。即删除三行冲突标记 `<<<<<<< HEAD` / `=======` / `>>>>>>> <hash>`，其余内容原样保留。

### 3.3 校验并继续

```bash
# 确认无残留冲突标记（rg 不可用时用 grep）
grep -n "<<<<<<<\|=======\|>>>>>>>" <file> || echo "无冲突标记"

python -c "import ast; ast.parse(open('<file>').read()); print('Python 语法 OK')"   # py 文件建议做语法检查

git add <file>
git rebase --continue
```

### 3.4 编辑器报错

若终端没有配置编辑器，rebase --continue 时可能报 `cannot run editor`，可用环境变量指定 no-op 编辑器（保留默认提交信息）：

```bash
GIT_EDITOR=true git rebase --continue
```

### 3.5 中途放弃

```bash
git rebase --abort     # 放弃本次 rebase，回到开始前状态
git rebase --skip      # 跳过当前出问题的提交
```

## 4. 操作完成后确认

```bash
git status          # 工作区干净、无 rebase 进行中
git log --oneline -5
```

Rebase 后本地与远端会显示"分叉"（如 `12 and 7 different commits`），这是重写历史后的正常现象，属预期行为。

## 5. 注意事项

- 强推前务必确认 dev/Alex 没有其他协作者基于它开发
- `git pull` 用普通方式即可；rebase 完成后不要直接 `git pull`（会引入 merge commit 抹掉线性历史），直接强推
- 若 dev/Alex 已在远端且多人使用，优先考虑 merge 而非 rebase
