# Tasks · 更名 z_umi → zerith_pi05

## 1. 目录与代码

- [x] 1.1 `git mv examples/realRobots/z_umi examples/realRobots/zerith_pi05`
- [x] 1.2 全仓路径替换：`examples/realRobots/z_umi` → `examples/realRobots/zerith_pi05`（scripts/ examples/ docs/ openspec/ starVLA/ deployment/）
- [x] 1.3 Python 点号 import 替换：`examples.realRobots.z_umi.*` → `examples.realRobots.zerith_pi05.*`（data-analyze、RoboDojo data_registry、z_umi 自身 data_config）
- [x] 1.4 `scripts/ddl_to_data_config.py` 路径拼装 `/ "z_umi"` → `/ "zerith_pi05"` + 注释同步
- [x] 1.5 ~~兼容 shim：`examples/realRobots/z_umi/train_files/run_umi_pi05_relative_6d.sh` exec 转发新 launcher~~
      （2026-09-21 已删除：AIHC 队列启动改为提交时同步 `scripts/train.sh + cloud_io.py` 到 CFS
      `/mnt/cfs/code/` 执行，与镜像解耦，见 `launch_train.sh` 与 `docs/zerith_docs/infra/aihc_queue_training.md` §5 坑 2）

## 2. spec 与文档

- [x] 2.1 `agent.md`：目录树（§2）、子项目表（§3.1/§5）、附录 A 标题与新增「命名变更记录」blockquote
- [x] 2.2 `openspec/specs/umi/pi05-open-eval/spec.md`：Purpose + References 路径更新
- [x] 2.3 `openspec/specs/umi/umi-code-changes/spec.md`：标题 + Purpose 历史路径表述更新
- [x] 2.4 `docs/zerith_docs/umi/umi_training_plan.md`（单一真实来源）顶部更名说明
- [x] 2.5 `docs/zerith_docs/` 其余文档路径/散文引用同步（启动命令、train_sh_run_flow、CI 文档、项目总结等）
- [x] 2.6 本 change（proposal + tasks）登记

## 3. 刻意保留（不改）

- [x] 3.1 物理数据根 `/root/data/z-umi`（本地/共享盘/机器变体）与云端 `datasets/z-umi/` URI 前缀
- [x] 3.2 wandb 默认项目名 `pi05_z_umi_relative_6d`（面板历史连续；如需更名另立变更）

## 4. 验证

- [x] 4.1 `git grep -n "z_umi"` 残留核对：仅 shim 头注释、agent.md/openspec 更名记录、wandb 项目名字符串、`umi_non_zumi_code_changes.md` 文件名（历史文档名，暂留）
- [x] 4.2 `python -m py_compile`：zerith_pi05 / data-analyze / RoboDojo 三个 `data_config.py` + `ddl_to_data_config.py` 通过
- [x] 4.3 venv 导入验证：三个 data_registry 的 data_config 模块导入成功，`umi_2gripper_relative_6d` 等 robot_type 与 mixtures 键完整
- [x] 4.4 registry 自动发现验证：`discover_and_merge()` 后 `DATASET_NAMED_MIXTURES` / `ROBOT_TYPE_CONFIG_MAP` 含 UMI 条目（扫描规则 `examples/*/train_files/data_registry/` 与目录名无关）
