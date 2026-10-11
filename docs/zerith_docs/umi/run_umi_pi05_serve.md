# UMI · PI05 部署启动命令说明

## 1. 命令本体

```bash
CHECKPOINT=/home/zerith/.models/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0806_042239/final_model/pytorch_model.pt \
bash examples/realRobots/zerith_pi05/deploy_files/run_umi_pi05_serve.sh
```

这是一个**单次生效的环境变量 + 脚本调用**：`CHECKPOINT=...` 仅对后面这次 `bash ...` 有效，不会污染当前 shell。

它做的事情一句话总结：**把下载好的 PI05 模型权重以 Zenoh 服务形式常驻加载，供 z-rio 机器人客户端调用。**

---

## 2. 启动脚本解析（run_umi_pi05_serve.sh）

### 2.1 环境变量赋值

```bash
CHECKPOINT=${CHECKPOINT:-/home/zerith/.models/run_umi_shenzhen_relative_6d_20000step_0727_100151}
```

`${VAR:-默认值}` 语义：**变量已设置则用它，未设置才用默认值**。所以我们命令行传的
`CHECKPOINT=/home/zerith/.models/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0806_042239/final_model/pytorch_model.pt`
**覆盖了脚本里的默认 checkpoint**（默认是另一个模型 `run_umi_shenzhen_relative_6d_20000step_0727_100151`）。

即：本次要部署的是 `forceclose` 版本、时间戳 `0806_042239` 的 final 权重。

### 2.2 默认值与变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHECKPOINT` | `/home/zerith/.models/run_umi_shenzhen_relative_6d_20000step_0727_100151` | ckpt run 目录或 `.pt` 文件（本次被覆盖） |
| `DEPLOY_YAML` | 同目录 `deploy_umi_pi05_relative_6d.yaml` | 部署 overlay 配置 |
| `SERVE_PYTHON` | `/home/zerith/miniforge3/envs/starVLA/bin/python` | serve 用的 python（需 zenoh + sentencepiece） |

### 2.3 preflight 检查

1. `CHECKPOINT` 文件必须存在，否则报 `CHECKPOINT not found` 退出。
2. `DEPLOY_YAML` 文件必须存在。
3. 校验 python 环境已安装 `zenoh`（serve 必需），没有则报错退出。

### 2.4 最终执行的命令

```bash
PYTHONPATH=/home/zerith/ws/orion/starvla \
  /home/zerith/miniforge3/envs/starVLA/bin/python \
  -m deployment.model_server.server_umi_pi05_rio \
  serve \
  --checkpoint /home/zerith/.models/run_umi_shenzhen_relative_6d_v2_forceclose_20000step/0806_042239/final_model/pytorch_model.pt \
  --deploy-yaml examples/realRobots/zerith_pi05/deploy_files/deploy_umi_pi05_relative_6d.yaml
```

> 设计原则：launcher（shell）只管路径 / 环境 / preflight；部署参数（device / dtype / zenoh 地址等）全在 deploy-yaml 的 `deploy:` 段，不走 CLI。

---

## 3. 服务端做了什么（server_umi_pi05_rio.py）

### 3.1 配置加载（Deployment.load）

1. `resolve_checkpoint`：解析 ckpt 路径（支持 `.pt` 文件 / run 目录 / 扁平或标准布局）。
2. `read_mode_config`：读 ckpt 同目录的 `config.yaml` + `dataset_statistics.json`（反归一化统计量）。
3. `merge_deploy_overlay`：把 deploy-yaml 与 ckpt config **OmegaConf.merge**（overlay 覆盖 ckpt 同名字段）。
   - 优先级：**CLI > deploy-yaml > ckpt config > 代码默认**。
4. `build_tokenizer_override`：若 ckpt config 里的 tokenizer 路径（训练机 `/nix/...`）本机缺失，则用 deploy-yaml 里的本地副本覆盖。
5. `DeployConfig.from_merged`：解析 `deploy:` 段参数。

### 3.2 deploy overlay（deploy_umi_pi05_relative_6d.yaml）

```yaml
framework:
  tokenizer:
    model_path: /home/zerith/.cache/openpi/big_vision/paligemma_tokenizer.model   # 覆盖训练机缺失路径

deploy:
  device: cuda:0               # torch device
  dtype: bf16                  # 精度
  listen_addr: 127.0.0.1:5570  # Zenoh 监听地址（z-rio 客户端连这里）
  key_prefix: z-rio/starvla    # Zenoh key 前缀
  unnorm_key: null             # dataset_statistics.json 顶层键；null=单键自动选
  shutdown_timeout: 5.0        # 清理超过该秒数强制退出（释放显存）
```

### 3.3 模型加载（UMIPI05Policy）

- 反归一化：`PolicyNormProcessor` 重建训练 transform 链，用 `dataset_statistics.json` 的 `_per_key`（形状 `(H, D)`）做逐时间步归一化，`unapply_actions` 走完整 `unapply`（denorm + **相对→绝对 SE(3) 合成** + topic 拆分）。
- `baseframework.from_pretrained` 加载 PI05 模型 → `cuda:0` + `bf16` + `eval()`。
- 关键坑（已处理）：
  - PI05 扁平 config：`action_dim`/`action_horizon` 在顶层，没有 `framework.action_model`。
  - **图像必须按训练方式拉伸到 224×224**（训练用 `resize`，PI0 推理默认 `resize_with_pad` 会产生黑边 → OOD），见 `_to_train_pil`。
  - `state` 是 post 输入（相对→绝对 SE(3) 的基准），不是 forward 输入（forward 时按 0 填充）。

### 3.4 Zenoh 服务（StarVlaZenohServer）

- 以 `peer` 模式监听 `tcp/127.0.0.1:5570`。
- 暴露 3 个 queryable（供 z-rio `rio/policies/starvla.py` 客户端调用）：
  - `<key_prefix>/health` → 返回 `metadata`（协议版本、action/state 表示、图像尺寸等契约字段）
  - `<key_prefix>/infer` → 收 `{video, language, state}`，返回**绝对** action dict + `org_action`
  - `<key_prefix>/reset` → episode 切换
- 线协议 = **msgpack-numpy**（与 z-rio `_pack_array` 一致）。
- forward 用 `threading.Lock` 串行化（CUDA + 多线程 forward 不安全）。

### 3.5 数据流

```
z-rio 客户端
   │  {video: {cam_wrist_0}, language, state(quat xyzw)}
   ▼
<z-rio/starvla/infer> ── msgpack-numpy ──► UMIPI05Policy.predict
   │   1. 图像拉伸到 224×224 (_to_train_pil)
   │   2. model.predict_action (forward; state 填 0)
   │   3. unapply_actions: denorm + 相对→绝对 SE(3) 合成 + topic 拆分
   ▼
   {action: {<topic>: 绝对动作(旋转 quat xyzw, 夹爪米)}, org_action: 模型原始输出}
```

---

## 4. 执行前提

1. 已通过 `bos_ckpt.sh download` 把 ckpt 下载到本地（本命令指向的路径必须存在）。
2. 用装有 `zenoh` + `sentencepiece` 的 python（默认 starVLA conda env）。
3. tokenizer 本地副本存在（`/home/zerith/.cache/openpi/big_vision/paligemma_tokenizer.model`），或设置 `STARVLA_PALIGEMMA_TOKENIZER` 环境变量兜底。
4. 有可用 GPU（`cuda:0`）。

---

## 5. 相关文件

| 文件 | 作用 |
|------|------|
| `examples/realRobots/zerith_pi05/deploy_files/run_umi_pi05_serve.sh` | 启动 launcher（环境 / preflight / 最终调用） |
| `examples/realRobots/zerith_pi05/deploy_files/deploy_umi_pi05_relative_6d.yaml` | 部署 overlay（tokenizer 路径 + deploy: 段） |
| `deployment/model_server/server_umi_pi05_rio.py` | Zenoh 服务端实现 |
| `deployment/model_server/policy_norm_processor.py` | 反归一化（完整 transform unapply） |

---

## 6. 常见操作

```bash
# 默认模型（不覆盖 CHECKPOINT）
bash examples/realRobots/zerith_pi05/deploy_files/run_umi_pi05_serve.sh

# 覆盖 checkpoint + 指定不同 python
CHECKPOINT=/path/to/run_dir \
SERVE_PYTHON=/path/to/python \
  bash examples/realRobots/zerith_pi05/deploy_files/run_umi_pi05_serve.sh

# 改部署参数（device/dtype/监听地址）：编辑 deploy_umi_pi05_relative_6d.yaml 的 deploy: 段
```

> 注意：`download` 默认拿到的是 `final` 权重；`--out` 可指定其它下载目录，启动时相应调整 `CHECKPOINT`。
