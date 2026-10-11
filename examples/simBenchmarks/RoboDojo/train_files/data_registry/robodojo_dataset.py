"""RoboDojo (bimanual unified_robot) — LeRobot Dataset 子类。

继承 ``UMILeRobotDataset`` 复用其「在线 norm + transform 驱动统计」机制（相对位姿 /
rotation_6d 变换会改变值分布，必须在线算 norm），做三处适配：

1. ``_ensure_modality_json``：RoboDojo 原始数据集不带 ``meta/modality.json``，缺失时
   从本目录 ``robodojo_modality_template.json``（双臂 16D 拆分）拷贝。
2. ``_pack_sample``：UMI 版丢弃语言，RoboDojo 每条 episode 带 task 描述，这里恢复语言。
3. ``_get_trajectories`` + ``_resolve_norm_statistics``：支持 ``data_cfg.task_index``
   单任务过滤（在 _all_steps / norm 计算前过滤），并按任务和动作表示隔离 norm 统计。

与 UMI 的区别：
* RoboDojo parquet **自带 action 列**（16D 绝对末端位姿），无需从 state 合成 ——
  ``UMILeRobotDataset.get_state_or_action`` 在 action 列存在时自动走标准读取路径。
* 四元数约定 wxyz（l_w,l_wx,l_wy,l_wz），与所有 JIT 旋转函数一致，几何正确。
"""
import os
from pathlib import Path

import numpy as np

from examples.realRobots.zerith_pi05.train_files.data_registry.umi_dataset import UMILeRobotDataset


class RoboDojoLeRobotDataset(UMILeRobotDataset):
    """RoboDojo 双臂数据集：bimanual modality + 语言 + 单任务过滤 + 复用 UMI 在线 norm。"""

    # ------------------------------------------------------------------ #
    # modality.json 生成
    # ------------------------------------------------------------------ #
    def _ensure_modality_json(self, modality_meta_path: Path) -> None:
        if modality_meta_path.exists():
            return
        template_path = Path(__file__).parent / "robodojo_modality_template.json"
        if not template_path.exists():
            raise FileNotFoundError(
                f"找不到 modality 元数据 {modality_meta_path}，且模板缺失: {template_path}。"
            )
        modality_meta_path.parent.mkdir(parents=True, exist_ok=True)
        modality_meta_path.write_text(template_path.read_text())
        print(f"[modality] {modality_meta_path.name} 缺失，已从 robodojo 模板生成: {modality_meta_path}")

    # ------------------------------------------------------------------ #
    # 单任务过滤：data_cfg.task_index / task_indices
    # ------------------------------------------------------------------ #
    def _build_episode_task_map(self) -> dict:
        """episode_index -> task_index，从主 data parquet 读一次（每条轨迹 task 恒定）。"""
        if getattr(self, "_ep_task_map", None) is not None:
            return self._ep_task_map
        import glob
        import pandas as pd

        files = sorted(glob.glob(str(self.dataset_path / "data" / "chunk-*" / "*.parquet")))
        frames = [pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in files]
        df = pd.concat(frames, ignore_index=True)
        self._ep_task_map = df.groupby("episode_index")["task_index"].first().to_dict()
        return self._ep_task_map

    def _allowed_tasks(self) -> set | None:
        cfg = self.data_cfg or {}
        ti = cfg.get("task_index", None)
        tis = cfg.get("task_indices", None)
        if ti is None and tis is None:
            return None
        allowed = set()
        if ti is not None:
            allowed.add(int(ti))
        if tis is not None:
            allowed.update(int(x) for x in tis)
        return allowed

    def _get_trajectories(self):
        """基类返回全部轨迹后，按 task 过滤（在 _all_steps / norm 之前生效）。"""
        ids, lengths = super()._get_trajectories()
        allowed = self._allowed_tasks()
        if allowed is None:
            return ids, lengths
        mp = self._build_episode_task_map()
        mask = np.array([mp.get(int(tid)) in allowed for tid in ids])
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[robodojo] task 过滤 task_index={sorted(allowed)}: "
                  f"保留 {int(mask.sum())}/{len(ids)} episodes")
        return ids[mask], lengths[mask]

    # ------------------------------------------------------------------ #
    # per-task norm 缓存隔离
    # ------------------------------------------------------------------ #
    def _resolve_norm_statistics(self):
        """按任务和动作表示隔离统计，关节动作使用全局逐维统计。"""
        from starVLA.dataloader.gr00t_lerobot.datasets import (
            _load_stats_cache,
            _save_stats_cache,
        )
        from starVLA.dataloader.gr00t_lerobot.transform.relative_pose import RelativePoseTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import PerStepNormalizationTransform

        allowed = self._allowed_tasks()
        suffix = "" if allowed is None else "_task" + "-".join(str(t) for t in sorted(allowed))
        relative_pose = any(isinstance(t, RelativePoseTransform) for t in self.transforms.transforms)
        stride = int((self.data_cfg or {}).get("norm_stride", 10))
        action_keys = self.modality_keys.get("action", [])
        action_horizon = int(len(self.delta_indices[action_keys[0]])) if action_keys else 0
        joint_action = "action.left_arm_joint" in action_keys
        per_step = any(isinstance(t, PerStepNormalizationTransform) for t in self.transforms.transforms)
        if joint_action:
            stats_name = "stats_gr00t_abs_joint_14d_v1"
            stats_cache_config = {
                "norm_stride": stride,
                "action_horizon": action_horizon,
                "per_step_action_norm": per_step,
                "action_representation": "absolute_joint_14d_v1",
            }
        elif relative_pose:
            stats_name = "stats_gr00t_wxyz_v1"
            stats_cache_config = {
                "norm_stride": stride,
                "action_horizon": action_horizon,
                "per_step_action_norm": True,
                "relative_pose_quaternion_convention": "wxyz_v1",
            }
        else:
            stats_name = "stats_gr00t_abs_ee_6d_v1"
            stats_cache_config = {
                "norm_stride": stride,
                "action_horizon": action_horizon,
                "per_step_action_norm": True,
                "action_representation": "absolute_ee_6d_v1",
            }
        stats_path = self.dataset_path / f"meta/{stats_name}{suffix}.json"
        import torch.distributed as dist

        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_rank0:
            le_statistics = _load_stats_cache(stats_path, stats_cache_config, invalidate_legacy=True)
            if le_statistics is None:
                le_statistics = self._calculate_norm_statistics(stride, per_step=per_step)
                _save_stats_cache(stats_path, stats_cache_config, le_statistics)
        # Every rank must reach this barrier, including a rank that observes a
        # newly written cache. Otherwise its barrier can pair with an earlier
        # _get_all_steps barrier while rank 0 is still computing statistics.
        if dist.is_initialized():
            dist.barrier()
        if not is_rank0:
            le_statistics = _load_stats_cache(stats_path, stats_cache_config, invalidate_legacy=False)
            if le_statistics is None:
                raise RuntimeError(
                    f"rank0 未成功写出统计缓存, 非 rank0 读不到: {stats_path}。"
                )
        return le_statistics

    def _get_all_steps(self):
        """task 感知的 steps 缓存。

        基类 ``_get_all_steps`` 读缓存时**不校验 config_key**，单任务过滤后会把别的
        任务的 steps 缓存串用（task24 误用 task0 的 steps）。这里用带 task 后缀的
        文件名隔离：task0 → steps_data_index_task0.pkl，task24 → ..._task24.pkl。
        """
        import pickle
        import torch.distributed as dist

        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        allowed = self._allowed_tasks()
        suffix = "" if allowed is None else "_task" + "-".join(str(t) for t in sorted(allowed))
        steps_path = self.dataset_path / "meta" / f"steps_data_index{suffix}.pkl"

        if is_main():
            rebuild = True
            if steps_path.exists():
                try:
                    with open(steps_path, "rb") as f:
                        pickle.load(f)["steps"]
                    rebuild = False
                except Exception as e:
                    print(f"[robodojo] Failed to load cached steps ({e}), will rebuild.")
            if rebuild:
                all_steps = self._get_all_steps_single_process()
                cache_data = {"steps": all_steps, "num_trajectories": len(self.trajectory_ids)}
                steps_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = steps_path.with_suffix(".tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, steps_path)
                print(f"[robodojo] Cached steps saved to {steps_path}")

        # Match the statistics barrier on every rank, even when this cache was
        # already present at entry. First-time creation may race with readers.
        if dist.is_initialized():
            dist.barrier()

        with open(steps_path, "rb") as f:
            return pickle.load(f)["steps"]

    def _calculate_norm_statistics(self, stride: int = 10, per_step: bool = True):
        """与 UMILeRobotDataset._calculate_norm_statistics 一致，**唯一改动**：
        遍历 ``self.trajectory_ids`` 而非 ``range(num_traj)``。

        原版用 ``range(num_traj)`` 当 traj_id，只在「traj_id 恰为 0..N-1」时正确
        （UMI 单数据集）。RoboDojo 单任务过滤后 traj_id 是非连续的（如 task24 的
        2400-2499），原版会 get_trajectory_index 失败。改用真实 trajectory_ids。

        另与 UMI 同款修复：只跑「几何」transform 链（剔除 NormalizationTransform /
        PerStepNormalizationTransform）算 pre-norm 统计，避免同 embodiment mixture 多数据集
        下共享 normalizer 被先加载数据集填满后，把本数据集原始值用别人的 min_max 先归一化
        再统计 → stats 污染。详见 UMILeRobotDataset._calculate_norm_statistics。
        """
        import time
        from tqdm import tqdm

        state_keys = list(self.modality_keys.get("state", []))
        action_keys = list(self.modality_keys["action"])
        collect_keys = state_keys + action_keys
        accum = {k: [] for k in collect_keys}
        t0 = time.time()
        n_used = 0

        # 只跑「几何」transform 链（剔除归一化）——归一化 transform 的 normalizer 是可变共享
        # 状态，mixture 多数据集下会被先加载数据集填满，若统计时仍跑归一化会把本数据集原始值
        # 用别人的 min_max 先归一化 → stats 污染。详见 UMILeRobotDataset 同名函数。
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
            NormalizationTransform,
            PerStepNormalizationTransform,
        )
        geom_transforms = ComposedModalityTransform(transforms=[
            t for t in self.transforms.transforms
            if not isinstance(t, (NormalizationTransform, PerStepNormalizationTransform))
        ])

        pbar = tqdm(list(self.trajectory_ids), desc="[norm-stats]", dynamic_ncols=True)
        for traj_id in pbar:
            traj_idx = self.get_trajectory_index(traj_id)
            traj_len = int(self.trajectory_lengths[traj_idx])
            if traj_len <= 1:
                continue
            pbar.set_postfix(traj_len=traj_len, samples=n_used)
            self.curr_traj_data = self.get_trajectory_data(traj_id)

            for base_index in range(0, traj_len, stride):
                raw = {}
                for modality, keys in (("state", state_keys), ("action", action_keys)):
                    for key in keys:
                        raw[key] = self.get_state_or_action(traj_id, modality, key, base_index)
                out = geom_transforms.apply({k: np.asarray(v) for k, v in raw.items()})
                for key in collect_keys:
                    if key in out:
                        accum[key].append(np.asarray(out[key], dtype=np.float32))
                n_used += 1

        statistics = {}
        for key, arrs in accum.items():
            if not arrs:
                continue
            horizon = len(self.delta_indices[key])
            if per_step and horizon > 1:
                all_vals = np.stack(arrs, axis=0).astype(np.float64)
            else:
                all_vals = np.concatenate(arrs, axis=0).astype(np.float64)
            statistics[key] = {
                "min": np.min(all_vals, axis=0).tolist(),
                "max": np.max(all_vals, axis=0).tolist(),
                "mean": np.mean(all_vals, axis=0).tolist(),
                "std": np.std(all_vals, axis=0).tolist(),
                "q01": np.quantile(all_vals, 0.01, axis=0).tolist(),
                "q99": np.quantile(all_vals, 0.99, axis=0).tolist(),
            }
        print(f"[norm-stats] 在线统计完成：采样 {n_used} samples (stride={stride}, "
              f"per_step={per_step}), 覆盖 {len(statistics)} subkeys, 耗时 {time.time() - t0:.1f}s")
        return statistics

    # ------------------------------------------------------------------ #
    # 完整性检查：跳过 language（靠 task 文本，无数组 modality metadata）
    # ------------------------------------------------------------------ #
    def _check_integrity(self):
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"
        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                # language 无数组 modality metadata（task 文本走 self.tasks），跳过
                if key.split(".")[0] == "language":
                    continue
                if key in ("lapa_action", "dream_actions"):
                    continue
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    # ------------------------------------------------------------------ #
    # 样本打包：3 相机 + 语言 + action(+state)
    # ------------------------------------------------------------------ #
    def _pack_sample(self, data: dict) -> dict:
        from PIL import Image

        step_images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]
            image = Image.fromarray(image).resize((224, 224))
            step_images.append(image)

        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(np.float16)

        default_lang = (self.data_cfg or {}).get("default_lang", "")
        lang_keys = self.modality_keys.get("language", [])
        if lang_keys and lang_keys[0] in data:
            lang_val = data[lang_keys[0]]
            language = lang_val[0] if isinstance(lang_val, (list, tuple)) else lang_val
            language = language if language else default_lang
        else:
            language = default_lang

        sample = {
            "action": action,
            "image": step_images,
            "lang": language,
            "robot_tag": self.tag,
        }

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                state.append(data[state_key])
            if state:
                state = np.concatenate(state, axis=1).astype(np.float16)
                sample["state"] = state

        return sample
