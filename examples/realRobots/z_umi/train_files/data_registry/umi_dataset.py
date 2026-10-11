"""UMI 专用的 LeRobot Dataset 子类。

继承 LeRobotSingleDataset 并重写方法来处理 UMI 数据集的特殊情况：
1. 数据集没有 action 列，需要从状态数据生成 delta action
2. 某些状态列（如 effector_0_joint_state）是一维数组
3. 没有语言注释
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class UMILeRobotDataset(LeRobotSingleDataset):
    """UMI 数据集专用的 LeRobot Dataset 子类。

    处理 UMI 数据集的特殊情况：
    - 没有独立的 action 列，从状态数据计算 delta action
    - 某些状态列是一维数组（如夹爪状态）
    - 没有语言注释
    """

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """获取状态或action数据，重写以处理UMI数据集的特殊情况。

        对于UMI数据集：
        1. 确保一维数组转换为二维（如 effector_0_joint_state）
        2. 如果是action模态且数据集中没有action列，则从状态数据生成
        """
        # 获取步骤索引
        step_indices = self.delta_indices[key] + base_index
        # trajectory_id 是 episode 的逻辑 id，而 trajectory_lengths / 采样权重等数组都是
        # 按位置索引存储的，所以先用 get_trajectory_index 把 id 映射到位置索引，再取出该
        # 轨迹的帧数 max_length。max_length 用于在下面的 retrieve_data_and_pad /
        # _get_action_from_state 里对越界的 step_indices 做边界填充/裁剪。
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        key = key.replace(modality + ".", "")

        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key

        # 确保有当前轨迹数据
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"

        # 特殊处理：UMI 数据集没有独立的 action 列（modality.json 里 action.original_key 为 null），
        # 此时 le_key（退化为 subkey 名）不在数据列中，需要从 state 合成绝对下一帧 action。
        if modality == "action" and le_key not in self.curr_traj_data.columns:
            return self._get_action_from_state(key, step_indices, max_length)

        # 标准处理：从数据列读取并切片（与 _get_action_from_state 共用 _load_modality_array）
        data_array = self._load_modality_array(modality, key)

        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # 填充数据
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def _load_modality_array(self, modality: str, subkey: str) -> np.ndarray:
        """从当前轨迹数据中读取某个 state/action 列，返回 (T, D) 二维数组。

        统一了 ``get_state_or_action`` 的标准读取路径与 ``_get_action_from_state``
        的 state 读取路径（两者原本各自重复 stack/reshape/slice 逻辑）：
        1. 解析 lerobot modality meta 的 original_key（为空则退化为 subkey 名）
        2. stack 该列、一维数组升维到二维（UMI 的夹爪列是一维）
        3. 按 meta 的 start/end 切出所需维度
        """
        le_cfg = getattr(self.lerobot_modality_meta, modality)[subkey]
        original_key = le_cfg.original_key or subkey

        assert self.curr_traj_data is not None, f"No data found when loading {modality}.{subkey}"
        assert original_key in self.curr_traj_data.columns, f"No {original_key} found in trajectory"

        # [perf] 同一条轨迹内同一列只 np.stack 一次：逐 sample 调 get_state_or_action 时
        # 避免反复 stack 整列（re-stack 是在线 norm 的主要开销）。轨迹切换 → curr_traj_data
        # 换对象 → id 变 → 缓存自动失效。缓存的是「整列 stack 结果」，多个 subkey 共用同一
        # original_key（如 position/rotation 都来自 end_0_pose）时也只 stack 一次，按 start/end 切片。
        traj_obj_id = id(self.curr_traj_data)
        if getattr(self, "_stacked_traj_id", None) != traj_obj_id:
            self._stacked_traj_id = traj_obj_id
            self._stacked_cols = {}
        stacked = self._stacked_cols.get(original_key)
        if stacked is None:
            data_array = np.stack(self.curr_traj_data[original_key])
            # UMI 特殊处理：一维数组（如 effector_0_joint_state）转二维
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            assert data_array.ndim == 2, f"Expected 2D array, got key {original_key} is {data_array.shape} array"
            self._stacked_cols[original_key] = data_array
            stacked = data_array

        le_indices = np.arange(le_cfg.start, le_cfg.end)
        return stacked[:, le_indices]

    def _get_action_from_state(
        self,
        subkey: str,
        step_indices: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """从状态数据合成 action（绝对下一帧）。

        对于 UMI 相对位姿 pipeline，action 必须是绝对下一帧的 state[t+1]，
        这样 ``RelativePoseTransform`` 才能做几何正确的 SE(3) 相对化：
            p_rel = R(state[t])^T · (state[t+1] - state[t])

        以前这里返回 element-wise delta (state[t+1] - state[t])，会被 transform
        当成绝对位姿去减当前 state，结果 ≈ -R^T·state[t] —— 完全是 garbage。

        边界处理：最后一帧用 state[-1] 复制（hold）而不是 0，避免在 chunk 末端
        引入突变。

        Args:
            subkey: 已剥掉 ``action.`` 前缀的 subkey（如 ``end_0_position``）。
                UMI 的 ``umi_modality_template.json`` 里 action 与 state 的 subkey
                同名，所以直接用同一个 subkey 去 ``_load_modality_array("state", ...)``
                读取对应的 state 列即可（旧代码里的 ``key.replace("action.", "state.")``
                其实是空操作——key 进来时已经没有 ``action.`` 前缀了）。
            step_indices: 要取的时间步索引（可能越界）。
            max_length: 当前轨迹帧数，用于裁剪越界索引。

        Note:
            这里用 clamp（hold 首尾帧）取数，而不是走 ``retrieve_data_and_pad``：
            UMI 的 action 模态 ``absolute=false``，``retrieve_data_and_pad`` 会用
            ``"zero"`` 策略把末端帧填 0，而 clamp 保留 hold 住的末帧值。这些越界帧
            随后都会被 ``action_pad`` 在 loss 里 mask 掉，但为保持与改动前完全一致的
            数值行为，这里仍沿用 clamp。
        """
        # 读取对应的 state 列（action/state 同名 subkey，见 docstring）
        state_data = self._load_modality_array("state", subkey)

        # 绝对下一帧 action: action_data[t] = state_data[t+1]
        # 最后一帧 hold（复制 state[-1]），避免 chunk 末端引入 0 突变。
        action_data = np.zeros_like(state_data)
        if len(state_data) > 1:
            action_data[:-1] = state_data[1:]
            action_data[-1] = state_data[-1]
        else:
            action_data = state_data.copy()

        # action_data 维度已与 state 的 slice 一致，无需再用 action 的 start/end 裁剪。
        # 越界 step_indices clamp 到 [0, max_length-1]（hold 首尾帧）后取数。
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        return action_data[step_indices]

    def _finalize_sample(self, sample: dict, trajectory_id, base_index) -> dict:
        """Attach UMI-specific per-sample fields that need trajectory context.

        Overrides the base no-op hook so these fields are produced on BOTH the
        single-dataset path (``LeRobotSingleDataset.__getitem__``) and the
        mixture path (``LeRobotMixtureDataset.__getitem__`` — the real training
        path, which previously bypassed ``__getitem__`` and dropped these).

        - ``action_pad``: (action_horizon,) bool mask, True = valid timestep,
          False = temporal pad. 轨迹末尾凑不齐 action_horizon 的 timestep 标记为
          False，loss 计算时忽略。
        - ``action_dim_actual``: 实际动作特征维度，来自 yaml 配置（不指定则从
          数据 shape 自动检测）。

        为什么在这里算 pad mask，而不是让 ``get_state_or_action`` 直接返回
        ``action_is_pad``：
            基类 ``get_state_or_action`` 的返回契约是 bare ``np.ndarray``，被
            ``get_data_by_modality`` / ``get_step_data`` / transforms 链路广泛依赖。
            把 is_pad 穿透进去要改基类签名，影响所有数据集（不止 UMI），代价过大。
            而这里根据 ``base_index`` 和轨迹长度重算 mask 与
            ``retrieve_data_and_pad`` 的填充是等价的：action 的 delta_indices 从 0
            开始、base_index >= 0，所以只可能出现末端填充，valid_steps =
            min(action_horizon, traj_length - base_index) 恰好对应未被填充的帧。
            并且这个 hook 同时被 single / mixture 两条路径调用，是最稳妥的注入点。
        """
        action = sample["action"]
        traj_index = self.get_trajectory_index(trajectory_id)
        traj_length = int(self.trajectory_lengths[traj_index])
        valid_steps = min(action.shape[0], max(0, traj_length - base_index))
        action_pad = np.zeros(action.shape[0], dtype=bool)
        action_pad[:valid_steps] = True
        sample["action_pad"] = action_pad

        sample["action_dim_actual"] = int(self.data_cfg.get("action_dim_actual", action.shape[-1]))
        return sample

    def _pack_sample(self, data: dict) -> dict:
        """重写 _pack_sample，UMI 数据集没有语言注释。"""
        from PIL import Image
        import numpy as np

        step_images = []
        for video_key in self.modality_keys["video"]:
            image = data[video_key][0]
            # 注 resize 放到这里了
            image = Image.fromarray(image).resize((224, 224))
            step_images.append(image)

        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(np.float16)

        # action 的 temporal pad mask 不在这里算——_pack_sample 拿不到 trajectory
        # 上下文（base_index / trajectory_id）。它在 _finalize_sample 里计算并注入
        # （见该方法的 action_pad 字段），那里两条 __getitem__ 路径都会经过。
        # UMI 无语言注释：用默认文本占位（可通过 data_cfg.default_lang 覆盖）。
        # 不能用 None——PI0/PI05 collate 会把 str(None) 当成字面量 "None" 作为语言
        # 条件喂给 tokenizer；空串则被 tokenize 成 BOS + "\n"（≥2 token），中性且安全。
        default_lang = (self.data_cfg or {}).get("default_lang", "")
        sample = {
            "action": action,
            "image": step_images,
            "lang": default_lang,
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

    def _ensure_modality_json(self, modality_meta_path: Path) -> None:
        """UMI 原始数据集不带 ``meta/modality.json``，缺它就从模板自动拷贝一份。

        所有 UMI 数据集的列结构一致（state 绝对 quaternion、action 从 state 合成），
        而 rotation_6d / SE(3) 相对位姿都在 transform 层做、与 modality.json 无关，
        故直接复用同目录下的 ``umi_modality_template.json`` 即可，无需逐数据集手写。
        已存在则不覆盖；模板也缺失时抛清晰错误（而不是静默 assert）。
        """
        if modality_meta_path.exists():
            return
        template_path = Path(__file__).parent / "umi_modality_template.json"
        if not template_path.exists():
            raise FileNotFoundError(
                f"找不到 modality 元数据 {modality_meta_path}，且模板缺失: {template_path}。"
                f"请手动放置一份 {modality_meta_path.name} 到 {modality_meta_path.parent}/。"
            )
        modality_meta_path.parent.mkdir(parents=True, exist_ok=True)
        # 原子写：先写同目录下的唯一临时文件，再 os.replace 原子替换目标文件。
        # write_text 的 open('w') 会先把文件截断为 0 字节再写内容，非原子；多进程/
        # 多节点在共享盘(NFS)上并发生成同一个 modality.json 时，一个进程截断的瞬间
        # 另一个进程 json.load 会读到 0 字节空文件而崩 (char 0)。os.replace 在同分区下
        # 是原子 rename，读端要么看到“不存在”要么看到完整文件，永不读到中间空态。
        # mkstemp 生成唯一随机文件名，避免并发进程互相覆盖对方的临时文件。
        content = template_path.read_text()
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{modality_meta_path.name}.",
            suffix=".tmp",
            dir=modality_meta_path.parent,
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, modality_meta_path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        print(f"[modality] {modality_meta_path.name} 缺失，已从模板生成: {modality_meta_path}")

    def _get_metadata(self, embodiment_tag):
        """构建数据集元数据（modalities + 空 statistics）。

        norm 统计延迟到 ``set_transforms_metadata`` 在线计算：本函数在 ``__init__``
        早期（datasets.py:622）执行，那时 get_state_or_action / get_trajectory_data
        等还没初始化（:635-638 才就绪），无法算 norm。而 ``set_transforms_metadata``
        在 :639 被调用、依赖已就绪，故把 norm 放那里算并 patch 进 metadata。
        """
        from starVLA.dataloader.gr00t_lerobot.schema import (
            LeRobotModalityMetadata,
            DatasetMetadata,
            DatasetStatistics,
            DatasetModalities,
        )
        from starVLA.dataloader.gr00t_lerobot.datasets import (
            LE_ROBOT_MODALITY_FILENAME,
            LE_ROBOT_INFO_FILENAME,
        )
        import json

        # 1. 加载 modality 元数据（缺失则从 umi_modality_template.json 自动生成）
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        self._ensure_modality_json(modality_meta_path)
        with open(modality_meta_path, "r") as f:
            lerobot_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        self._lerobot_modality_meta = lerobot_modality_meta

        # 2. 加载 info 元数据
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)

        # 3. 返回元数据（statistics 留空，由 set_transforms_metadata 在线 patch）
        simplified_modality_meta = self._build_simplified_modality_meta(lerobot_modality_meta)
        metadata = DatasetMetadata(
            statistics=DatasetStatistics(state={}, action={}),
            modalities=DatasetModalities(**simplified_modality_meta),
            embodiment_tag=embodiment_tag,
        )
        return metadata

    def set_transforms_metadata(self, metadata):
        """首次调用时在线算 norm 并 patch 进 metadata，再喂给 transforms。

        时机：``__init__`` 在 datasets.py:639 调用本方法，此刻
        modality_keys / delta_indices / trajectory_lengths / get_trajectory_data 均已就绪，
        故可在此复用 get_state_or_action + self.transforms 在线算 norm。

        评测模式 (env STARVLA_EVAL_PER_KEY_JSON): 直接用模型固化的 per-subkey norm
        (_per_key), 跳过数据集 stats 计算 —— 彻底脱离 stats_gr00t.json (后者可能被多卡/
        错配重算污染, 实测会把 action min 撑到 -30)。训练时 save_dataset_statistics 已固化。
        """
        if not getattr(self, "_norm_stats_resolved", False):
            import os as _os
            _override = _os.environ.get("STARVLA_EVAL_PER_KEY_JSON")
            if _override:
                try:
                    import json as _json
                    from starVLA.dataloader.gr00t_lerobot.schema import DatasetStatistics

                    _ns = _json.load(open(_override))
                    _pk = _ns.get(self.tag, {}).get("_per_key") if isinstance(_ns, dict) else None
                    if _pk:
                        self._metadata = self._metadata.model_copy(
                            update={"statistics": DatasetStatistics.model_validate(_pk)}
                        )
                        self._norm_stats_resolved = True
                        if int(_os.environ.get("RANK", "0")) == 0:
                            print(
                                f"[umi] 🔒 用模型固化 norm (_per_key, tag={self.tag}), 跳过数据集 stats",
                                flush=True,
                            )
                        super().set_transforms_metadata(self._metadata)
                        return
                except Exception as _e:
                    print(f"[umi] ⚠️ 加载 _per_key 失败, 用数据集 stats: {_e}")
            le_statistics = self._resolve_norm_statistics()
            self._metadata = self._metadata.model_copy(
                update={"statistics": self._build_statistics_meta(le_statistics)}
            )
            self._norm_stats_resolved = True
        super().set_transforms_metadata(self._metadata)

    def _resolve_norm_statistics(self):
        """加载缓存或在线计算 norm 统计。

        cache key 纳入 ``norm_stride`` / ``action_horizon`` / ``per_step_action_norm``：
        任一变化都会自动失效重算。
        """
        from starVLA.dataloader.gr00t_lerobot.datasets import (
            _load_stats_cache,
            _save_stats_cache,
        )
        stats_path = self.dataset_path / "meta/stats_gr00t.json"
        stride = int((self.data_cfg or {}).get("norm_stride", 10))
        # action chunk 长度 H = action 任意子键的 delta_indices 长度（UMI = 50）。
        action_keys = self.modality_keys.get("action", [])
        action_horizon = int(len(self.delta_indices[action_keys[0]])) if action_keys else 0
        # UMI action 固定 per-step 归一化（每个时间步独立 min/max）。改此行为需删 meta/stats_gr00t.json。
        stats_cache_config = {
            "norm_stride": stride,
            "action_horizon": action_horizon,
            "per_step_action_norm": True,
        }
        cached = _load_stats_cache(stats_path, stats_cache_config, invalidate_legacy=True)
        if cached is not None:
            print(f"[norm-stats] 命中缓存 {stats_path} (stride={stride}, action_horizon={action_horizon})")
            return cached

        # 多 rank 并发时只让 rank0 计算 + 写缓存, 其余 rank barrier 等它写完再读
        # (与 datasets._load_or_compute_statistics 的协调方式一致)。省掉 N 倍重复统计,
        # 也避免并发写。单进程 (dist 未初始化) 时 rank0=True, 走正常计算+写路径。
        import torch.distributed as dist

        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_rank0:
            le_statistics = self._calculate_norm_statistics(stride)   # per_step 默认 True
            _save_stats_cache(stats_path, stats_cache_config, le_statistics)
        if dist.is_initialized():
            dist.barrier()  # 等 rank0 把缓存写盘
        if not is_rank0:
            le_statistics = _load_stats_cache(stats_path, stats_cache_config, invalidate_legacy=False)
            if le_statistics is None:
                raise RuntimeError(
                    f"rank0 未成功写出统计缓存, 非 rank0 读不到: {stats_path}。"
                    f"删掉该文件后重试, 或检查 rank0 日志。"
                )
        return le_statistics

    def _calculate_norm_statistics(self, stride: int = 10, per_step: bool = True):
        """在线、transform 驱动的 norm 统计（与训练 transform 完全一致，消灭离线双写）。

        对采样的 base_index：raw(state[t0] + action chunk) → ``self.transforms``。
        此时归一化 transform（state 的 ``NormalizationTransform``、action 的
        ``PerStepNormalizationTransform``）的 normalizer 字典仍为空（本函数在
        set_transforms_metadata 把 stats 喂进 transforms 之前运行），``apply`` 会
        passthrough，故输出即 relative + rotation_6d 的 pre-normalization 值——
        正是要算 min_max 的对象，无需分割链。

        Args:
            stride: base_index 抽样步长。
            per_step: True 时 action 子键按 ``(H, D')`` 统计（每步独立 min/max），
                state 子键始终按 ``(D',)`` 统计。False 时全部按 ``(D',)``（传统行为）。

        复用：``get_trajectory_data``（parquet 加载/缓存）+ ``get_state_or_action``
        （UMI action 合成/hold/切片）+ ``self.transforms``。零定制 parquet 代码、不读 video。

        Returns:
            ``{"state.<subkey>": {...}, "action.<subkey>": {...}}``，供 _build_statistics_meta。
        """
        import time
        from tqdm import tqdm

        # ---- 1. 准备：要算哪些 key、采样步长 ----
        # modality_keys 形如 {"state": ["state.end_0_position", ...], "action": [...]}。
        # 对每个 state/action subkey 各算一份统计量（min_max 归一化时按 subkey 取 min/max）。
        state_keys = list(self.modality_keys.get("state", []))
        action_keys = list(self.modality_keys["action"])
        collect_keys = state_keys + action_keys
        # stride：base_index 抽样步长，由 _resolve_norm_statistics 传入（默认 10）。
        # re-stack 已在 _load_modality_array 用「轨迹级列缓存」消除；但逐 sample 调
        # self.transforms.apply（deepcopy + torch JIT + numpy↔tensor）才是当前主开销
        #（~4.5ms/sample）。故 stride=1（全量 ~31万 sample）≈ 24min 不实用；默认 10
        #（~3万 sample ≈ 140s，一次性、之后缓存命中）。要「全量 + 快」需改成向量化（方案 B）。

        # accum[key] = list of (T, D) 数组，每个 sample 贡献一份，最后 concat 算统计量。
        accum = {k: [] for k in collect_keys}
        t0 = time.time()
        n_used = 0

        # ---- 2. 遍历每条轨迹 ----
        num_traj = len(self.trajectory_lengths)
        pbar = tqdm(range(num_traj), desc="[norm-stats]", dynamic_ncols=True)
        for traj_id in pbar:
            traj_idx = self.get_trajectory_index(traj_id)       # 逻辑 id → 位置索引
            traj_len = int(self.trajectory_lengths[traj_idx])   # 该轨迹帧数
            if traj_len <= 1:
                continue
            pbar.set_postfix(traj_len=traj_len, samples=n_used)
            # 把整条轨迹的 parquet 读进 self.curr_traj_data。下面的 get_state_or_action
            # 内部会反复读它（_load_modality_array 每次 stack 整列），故每条轨迹只读一次。
            self.curr_traj_data = self.get_trajectory_data(traj_id)

            # ---- 3. 轨迹内按 stride 抽样 anchor t0 ----
            for base_index in range(0, traj_len, stride):
                # 一个 sample 的 raw 输入 = 当前观测 state[t0] + 未来 action chunk。
                # 维度：state 用 observation_indices=[0]  → 形状 (1, D)（仅当前帧）；
                #       action 用 action_indices=[0..49] → 形状 (50, D)（未来 50 帧），
                #       且 action 是「绝对下一帧 state[t+1]」（_get_action_from_state 合成）。
                raw = {}
                for modality, keys in (("state", state_keys), ("action", action_keys)):
                    for key in keys:
                        raw[key] = self.get_state_or_action(traj_id, modality, key, base_index)

                # ---- 4. 跑整条 transform 链（= _UMI2GripperRelative6DConfig.transforms）----
                # 链路：StateActionToTensor → RelativePoseTransform(SE(3) 把 action 相对到 state[t0])
                #       → QuaternionTo6DRotation(state+action 的 quat→rotation_6d)
                #       → 归一化：state 走 NormalizationTransform，action 走 PerStepNormalizationTransform
                #         （此时两者的 normalizer 字典均为空 → passthrough！）
                # 为什么归一化会 passthrough：本函数在 set_transforms_metadata
                # 「把统计量喂进 transforms」之前运行，normalizer 还没建，apply 直接跳过。
                # 因此 out 里是 **pre-normalization** 的值：action = SE(3) 相对 + rotation_6d，
                # state = 绝对 rotation_6d —— 正是 min_max 要算 min/max 的对象，无需分割链。
                out = self.transforms.apply({k: np.asarray(v) for k, v in raw.items()})

                # out 每个 key 的形状：state.* → (1, D')，action.* → (50, D')
                # （rotation 从 4D quat 变成 6D rotation_6d，故 D' 与 raw 不同）。
                for key in collect_keys:
                    if key in out:
                        accum[key].append(np.asarray(out[key], dtype=np.float32))
                n_used += 1

        # ---- 5. 每个 subkey 跨所有 sample 合并，算逐维统计量 ----
        # per_step=True 时，action 子键（horizon>1）按 (H, D') 统计：每个时间步独立 min/max；
        # state 子键（horizon=1）以及 per_step=False 时一律按 (D',) 统计（全 chunk 共用一组）。
        statistics = {}
        for key, arrs in accum.items():
            if not arrs:
                continue
            horizon = len(self.delta_indices[key])  # action=50, state=1
            if per_step and horizon > 1:
                # 每个 sample 贡献 (H, D') → stack 成 (N, H, D')，沿 N 归约 → (H, D')
                all_vals = np.stack(arrs, axis=0).astype(np.float64)
            else:
                # 摊平所有步：state → (N, D')；per_step=False 的 action → (N*H, D')
                all_vals = np.concatenate(arrs, axis=0).astype(np.float64)
            statistics[key] = {
                "min": np.min(all_vals, axis=0).tolist(),   # ← min_max 实际只消费 min/max
                "max": np.max(all_vals, axis=0).tolist(),
                "mean": np.mean(all_vals, axis=0).tolist(),
                "std": np.std(all_vals, axis=0).tolist(),
                "q01": np.quantile(all_vals, 0.01, axis=0).tolist(),  # 其它模式(q99/mean_std)备用
                "q99": np.quantile(all_vals, 0.99, axis=0).tolist(),
            }
        print(f"[norm-stats] 在线统计完成：采样 {n_used} samples (stride={stride}, "
              f"per_step={per_step}), 覆盖 {len(statistics)} subkeys, 耗时 {time.time() - t0:.1f}s")
        return statistics

    def _build_simplified_modality_meta(self, lerobot_modality_meta):
        """构建简化的模态元数据。"""
        simplified = {}

        for modality in ["state", "action"]:
            simplified[modality] = {}
            le_modality_dict = getattr(lerobot_modality_meta, modality)

            for subkey, le_cfg in le_modality_dict.items():
                state_action_dtype = np.dtype(le_cfg.dtype)
                is_continuous = np.issubdtype(state_action_dtype, np.floating)

                simplified[modality][subkey] = {
                    "absolute": le_cfg.absolute,
                    "rotation_type": le_cfg.rotation_type,
                    "shape": [le_cfg.end - le_cfg.start],
                    "continuous": is_continuous,
                }

        # 处理视频模态
        simplified["video"] = {}
        for video_key in lerobot_modality_meta.video.keys():
            simplified["video"][video_key] = {
                "resolution": [224, 224],  # 固定分辨率
                "channels": 3,
                "fps": 60,
            }

        return simplified

    def _build_statistics_meta(self, le_statistics):
        """构建统计数据元数据。"""
        from starVLA.dataloader.gr00t_lerobot.schema import DatasetStatistics, DatasetStatisticalValues

        # 分离 state 和 action 统计量
        state_stats = {}
        action_stats = {}

        for key, modality_stats in le_statistics.items():
            # 跳过非 state/action 的统计量（支持多种格式）
            # 格式1: observation.state.xxx / observation.action.xxx
            # 格式2: state.xxx / action.xxx
            if key.startswith("observation.state."):
                clean_key = key.replace("observation.", "")
                modality = "state"
            elif key.startswith("observation.action."):
                clean_key = key.replace("observation.", "")
                modality = "action"
            elif key.startswith("state."):
                clean_key = key
                modality = "state"
            elif key.startswith("action."):
                clean_key = key
                modality = "action"
            else:
                continue

            stats_values = DatasetStatisticalValues(
                mean=np.array(modality_stats["mean"], dtype=np.float32),
                std=np.array(modality_stats["std"], dtype=np.float32),
                min=np.array(modality_stats["min"], dtype=np.float32),
                max=np.array(modality_stats["max"], dtype=np.float32),
                q01=np.array(modality_stats.get("q01", modality_stats["min"]), dtype=np.float32),
                q99=np.array(modality_stats.get("q99", modality_stats["max"]), dtype=np.float32),
            )

            if modality == "state":
                state_key = clean_key.replace("state.", "")
                state_stats[state_key] = stats_values
            else:
                action_key = clean_key.replace("action.", "")
                action_stats[action_key] = stats_values

        return DatasetStatistics(state=state_stats, action=action_stats)


# ============================================================
# 单样本 norm 分析入口（重构调试用）
# ============================================================

def _main():
    """加载指定路径的数据 → 调 ``_get_metadata`` 看 norm 材料 → 调 ``__getitem__`` 看归一化后样本。

    用于逐步分析 norm 流程，不依赖训练入口。CPU 即可跑。

    用法（从 repo root 或任意目录）::
        uv run python examples/realRobots/z_umi/train_files/data_registry/umi_dataset.py \\
            --dataset_path /nix/data/z-umi/20260722114220_lerobot_b80dcf10

        # 覆盖 robot_type / 采样步 / 是否带 state
        ... --robot_type umi_2gripper_relative_6d --index 5 --include_state
    """
    import argparse
    import sys
    from pathlib import Path

    # 允许从仓库任意位置直接运行本文件
    _repo_root = Path(__file__).resolve().parents[5]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    parser = argparse.ArgumentParser(description="UMILeRobotDataset 单样本 norm 分析")
    parser.add_argument(
        "--dataset_path",
        default="/nix/data/z-umi/20260722114220_lerobot_b80dcf10",
        help="lerobot 数据集目录（其下应含 meta/ 和 data/）",
    )
    parser.add_argument("--robot_type", default="umi_2gripper_relative_6d")
    # [REFACTOR] norm 不再依赖 action_mode（见 NORM_FLOW.md §12）
    # parser.add_argument("--action_mode", default="abs", choices=["abs", "delta", "rel"])
    parser.add_argument("--video_backend", default="torchvision_av")
    parser.add_argument("--include_state", action="store_true", help="sample 里是否带上 state")
    parser.add_argument("--index", type=int, default=5, help="__getitem__ 取的扁平 step 索引")
    parser.add_argument("--anchor", type=int, default=5, help="bonus 演示用的轨迹内 anchor")
    parser.add_argument("--norm_stride", type=int, default=10,
                        help="在线 norm 的 base_index 抽样步长（1=全量，越大越快越粗）")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    data_root_dir = dataset_path.parent
    data_name = dataset_path.name

    print("=" * 72)
    print(f"dataset_path = {dataset_path}")
    print(f"data_root_dir = {data_root_dir} | data_name = {data_name}")
    print(f"robot_type = {args.robot_type} | "
          f"include_state = {args.include_state}")
    print("=" * 72)

    data_cfg = {
        # [REFACTOR] norm 不再依赖 action_mode
        # "action_mode": args.action_mode,
        "include_state": args.include_state,
        "lerobot_version": "v3.0",
        "video_backend": args.video_backend,
        "norm_stride": args.norm_stride,
    }
    ds = make_LeRobotSingleDataset(data_root_dir, data_name, args.robot_type, data_cfg=data_cfg)

    # ---------- [1] 数据集概览 ----------
    print("\n[1] 数据集概览")
    print(f"  modality_keys.action = {ds.modality_keys['action']}")
    print(f"  modality_keys.state  = {ds.modality_keys.get('state', [])}")
    print(f"  num trajectories     = {len(ds.trajectory_lengths)}")
    print(f"  trajectory_lengths[:5] = {list(ds.trajectory_lengths[:5])}")
    print(f"  len(ds) (采样总数)    = {len(ds)}")

    # ---------- [2] metadata.statistics：在线算出的 norm 材料 ----------
    # 注意：_get_metadata 现在只返回空 statistics（norm 延迟到 set_transforms_metadata 在线算），
    # 真正的统计量在 __init__ 里已 patch 进 ds.metadata，故这里直接读 ds.metadata.statistics。
    print("\n[2] ds.metadata.statistics —— 在线算出的 norm 材料")
    metadata = ds.metadata
    for modality in ("state", "action"):
        stats = getattr(metadata.statistics, modality)
        print(f"  [{modality}]")
        for subkey, vals in stats.items():
            mn = np.asarray(vals.min)
            print(
                f"    {subkey:<28} shape={mn.shape} "
                f"min={np.round(mn, 3).tolist()} max={np.round(np.asarray(vals.max), 3).tolist()} "
                f"mean={np.round(np.asarray(vals.mean), 3).tolist()}"
            )
    print("  （归一化模式 = min_max：normalized = 2*(x-min)/(max-min)-1）")

    # ---------- [3] __getitem__：离开 dataloader 的归一化样本 ----------
    idx = min(args.index, max(0, len(ds) - 1))
    print(f"\n[3] __getitem__({idx}) —— 离开 dataloader 的样本")
    sample = ds[idx]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:<18} ndarray shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, (list, tuple)) and v and hasattr(v[0], "size"):
            print(f"  {k:<18} {type(v).__name__}[{len(v)}] e.g. {type(v[0]).__name__}")
        else:
            print(f"  {k:<18} {type(v).__name__}: {v!r}")

    action = np.asarray(sample["action"], dtype=np.float32)  # [H, D]
    print(f"\n  action.shape = {action.shape}  (期望 [action_horizon, action_dim_actual])")
    print(f"  action 全局 min={action.min():.4f} max={action.max():.4f}  (min_max 后应在 ~[-1,1])")
    print(f"  action 每维 min = {np.round(action.min(axis=0), 3).tolist()}")
    print(f"  action 每维 max = {np.round(action.max(axis=0), 3).tolist()}")
    if "action_pad" in sample:
        pad = np.asarray(sample["action_pad"])
        print(f"  action_pad: 有效步 {int(pad.sum())}/{len(pad)}  (尾部 padding 会被 loss mask)")
    if "action_dim_actual" in sample:
        print(f"  action_dim_actual = {sample['action_dim_actual']}  (loss 里 [..., dim:32] 维被 mask)")

    # ---------- [4] bonus: 单 key 的 raw → normalized 对比 ----------
    print(f"\n[4] bonus: raw → transforms.apply 后的逐 key 归一化（traj 0, anchor={args.anchor}）")
    state_keys = ds.modality_keys.get("state", [])
    action_keys = ds.modality_keys["action"]
    traj_len = int(ds.trajectory_lengths[ds.get_trajectory_index(0)])
    anchor = min(args.anchor, max(0, traj_len - 1))
    raw = ds.get_step_data(0, anchor)
    # 必须同时带 state key：RelativePoseTransform 要读 state 作为 SE(3) base。
    raw_numpy = {
        **{k: np.asarray(raw[k], dtype=np.float32) for k in state_keys},
        **{k: np.asarray(raw[k], dtype=np.float32) for k in action_keys},
    }
    norm = ds.transforms.apply(dict(raw_numpy))
    for k in action_keys:
        r = np.asarray(raw_numpy[k], dtype=np.float64)
        n = np.asarray(norm[k], dtype=np.float64)
        print(
            f"  {k:<28} raw[{r.shape[-1]}] [{r.min():.3f},{r.max():.3f}] -> "
            f"norm[{n.shape[-1]}] [{n.min():.3f},{n.max():.3f}]"
        )

    print("\n完成。")


if __name__ == "__main__":
    _main()
