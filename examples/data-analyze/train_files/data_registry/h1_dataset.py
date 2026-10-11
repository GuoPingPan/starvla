"""H1 DDL job d18480a6 数据集类与 robot_type 注册 (自 examples/realRobots/H1_ddl_d18480a6 迁入,
统一由 data-analyze/train_files/data_config.py 引用; 一个模型一个路径)。

The initial PI05 action subset is 30 dimensions:
two end-effector poses (14), two arm joint states (14), and two grippers (2).
"""

import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    NormalizationTransform,
    StateActionToTensor,
)


_VIDEO_KEYS = ["video.cam_high_0", "video.cam_wrist_0", "video.cam_wrist_1"]
_STATE_KEYS = [
    "state.end_0_pose",
    "state.end_1_pose",
    "state.arm_0_joint_state",
    "state.arm_1_joint_state",
    "state.effector_0_joint_state",
    "state.effector_1_joint_state",
]
_ACTION_KEYS = [
    "action.end_0_pose",
    "action.end_1_pose",
    "action.arm_0_joint_state",
    "action.arm_1_joint_state",
    "action.effector_0_joint_state",
    "action.effector_1_joint_state",
]
_LANGUAGE_KEYS = ["annotation.human.task"]


def _ensure_modality_metadata(dataset_path):
    path = dataset_path / "meta" / "modality.json"
    if path.exists():
        return
    state_fields = [
        ("end_0_pose", 7, "observation.state.end_0_pose"),
        ("end_1_pose", 7, "observation.state.end_1_pose"),
        ("arm_0_joint_state", 7, "observation.state.arm_0_joint_state"),
        ("arm_1_joint_state", 7, "observation.state.arm_1_joint_state"),
        ("effector_0_joint_state", 1, "observation.state.effector_0_joint_state"),
        ("effector_1_joint_state", 1, "observation.state.effector_1_joint_state"),
    ]
    action_fields = [
        ("end_0_pose", 7, "action.end_0_pose"),
        ("end_1_pose", 7, "action.end_1_pose"),
        ("arm_0_joint_state", 7, "action.arm_0_joint_state"),
        ("arm_1_joint_state", 7, "action.arm_1_joint_state"),
        ("effector_0_joint_state", 1, "action.effector_0_joint_state"),
        ("effector_1_joint_state", 1, "action.effector_1_joint_state"),
    ]

    def fields(spec):
        return {
            name: {
                "start": 0,
                "end": width,
                "absolute": True,
                "dtype": "float32",
                "original_key": original_key,
            }
            for name, width, original_key in spec
        }

    payload = {
        "state": fields(state_fields),
        "action": fields(action_fields),
        "video": {
            "cam_high_0": {"original_key": "observation.images.cam_high_0"},
            "cam_wrist_0": {"original_key": "observation.images.cam_wrist_0"},
            "cam_wrist_1": {"original_key": "observation.images.cam_wrist_1"},
        },
        "annotation": {"human.task": {"original_key": "task_index"}},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _CompactColumn:
    """pandas Series 的最小替身, 只实现 BaseSingleDataset 用到的接口。

    支持: np.stack(col) (逐行迭代 -> (N,D)), col.to_numpy(), col.iloc[i],
    len(col), col.tolist()。底层数据是紧凑的 float32/int64 ndarray。
    """

    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    def to_numpy(self):
        return self._arr

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)

    def __iter__(self):
        return iter(self._arr)

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, i):
        return self._arr[i]

    def tolist(self):
        return self._arr.tolist()

    @property
    def iloc(self):
        return self  # .iloc[i] -> self._arr[i]


class _CompactEpisodeData:
    """单 episode 数据的 DataFrame 最小替身 (列 -> _CompactColumn)。"""

    __slots__ = ("_cols", "columns")

    def __init__(self, cols: dict, columns: list):
        self._cols = cols
        self.columns = columns

    def __getitem__(self, key):
        return _CompactColumn(self._cols[key])


class H1LeRobotDataset(LeRobotSingleDataset):
    """H1 adapter for scalar gripper columns in LeRobot v3 parquet files.

    额外做了紧凑化 file 级缓存: pandas 把 list<float> 列存为逐元素 numpy 对象的
    object 列 (单文件 ~0.85GB vs 紧凑 float32 ~25MB), 22 个数据集的不设限缓存会
    直接把 memcg (~119GB) 撑爆。这里把整文件转成 {列: ndarray} 紧凑存档 + 按
    episode_index 掩码切片, LRU 上限 compact_file_cache_size (默认 4)。
    BaseSingleDataset 对 curr_traj_data 只用 4 个接口 (columns / ["timestamp"]
    .to_numpy() / np.stack(col) / .iloc[i]), _CompactEpisodeData 全部满足。
    """

    compact_file_cache_size = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compact_file_cache = OrderedDict()

    def _load_compact_file(self, parquet_path):
        cache = self._compact_file_cache
        if parquet_path in cache:
            cache.move_to_end(parquet_path)
            return cache[parquet_path]

        df = pd.read_parquet(parquet_path)
        file_cols = {}
        for col in df.columns:
            values = df[col]
            first = values.iloc[0] if len(values) else None
            if isinstance(first, (list, np.ndarray)) or (
                hasattr(first, "__len__") and not isinstance(first, str)
            ):
                try:
                    file_cols[col] = np.asarray(
                        [np.asarray(x, dtype=np.float32) for x in values],
                        dtype=np.float32,
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            file_cols[col] = values.to_numpy()
        del df

        cache[parquet_path] = file_cols
        while self.compact_file_cache_size and len(cache) > self.compact_file_cache_size:
            cache.popitem(last=False)
        return file_cols

    def get_trajectory_data(self, trajectory_id):
        return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def get_trajectory_data_lerobot_v3(self, trajectory_id):
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
        chunk_index = episode_meta["data/chunk_index"]
        file_index = self.get_episode_file_index(trajectory_id)
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            chunk_index=chunk_index, file_index=file_index
        )
        assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
        file_cols = self._load_compact_file(parquet_path)
        mask = file_cols["episode_index"] == trajectory_id
        cols = {name: arr[mask] for name, arr in file_cols.items()}
        return _CompactEpisodeData(cols, list(cols.keys()))

    def get_state_or_action(self, trajectory_id, modality, key, base_index):
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}"
        subkey = key.replace(modality + ".", "")
        modality_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = modality_cfg[subkey].original_key or subkey
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array = np.stack(self.curr_traj_data[le_key])
        if data_array.ndim == 1:
            data_array = data_array[:, None]
        assert data_array.ndim == 2, f"Expected 2D array, got {le_key}: {data_array.shape}"
        cfg = modality_cfg[subkey]
        data_array = data_array[:, np.arange(cfg.start, cfg.end)]
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )


class H1ManipulationConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = _VIDEO_KEYS
    state_keys = _STATE_KEYS
    action_keys = _ACTION_KEYS
    language_keys = _LANGUAGE_KEYS
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                NormalizationTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                NormalizationTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )

    def make_dataset(self, **kwargs):
        _ensure_modality_metadata(kwargs["dataset_path"])
        return H1LeRobotDataset(**kwargs)


ROBOT_TYPE_CONFIG_MAP = {"h1_manipulation": H1ManipulationConfig()}

DATASET_NAMED_MIXTURES = {
    "h1_ddl_d18480a6": [
        (
            "20260909155441_lerobot_f61b924b",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_214/PR-0011_BIZ-STORE-001_214_20260909155441_lerobot_f61b924b_20260909155441_lerobot_f61b924b.zip",
        ),
        (
            "20260909155646_lerobot_165e92fd",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_218/PR-0011_BIZ-STORE-001_218_20260909155646_lerobot_165e92fd_20260909155646_lerobot_165e92fd.zip",
        ),
        (
            "20260909155823_lerobot_fc8b7a1f",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_217/PR-0011_BIZ-STORE-001_217_20260909155823_lerobot_fc8b7a1f_20260909155823_lerobot_fc8b7a1f.zip",
        ),
        (
            "20260909155821_lerobot_2a9ebb61",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_212/PR-0011_BIZ-STORE-001_212_20260909155821_lerobot_2a9ebb61_20260909155821_lerobot_2a9ebb61.zip",
        ),
        (
            "20260909160736_lerobot_74634506",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_213/PR-0011_BIZ-STORE-001_213_20260909160736_lerobot_74634506_20260909160736_lerobot_74634506.zip",
        ),
        (
            "20260909161139_lerobot_d6c6ab11",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_216/PR-0011_BIZ-STORE-001_216_20260909161139_lerobot_d6c6ab11_20260909161139_lerobot_d6c6ab11.zip",
        ),
        (
            "20260909161129_lerobot_1669a254",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_215/PR-0011_BIZ-STORE-001_215_20260909161129_lerobot_1669a254_20260909161129_lerobot_1669a254.zip",
        ),
        (
            "20260909162216_lerobot_e76f38a1",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_211/PR-0011_BIZ-STORE-001_211_20260909162216_lerobot_e76f38a1_20260909162216_lerobot_e76f38a1.zip",
        ),
        (
            "20260909175028_lerobot_208b0c3d",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_218/PR-0011_BIZ-STORE-001_218_20260909175028_lerobot_208b0c3d_20260909175028_lerobot_208b0c3d.zip",
        ),
        (
            "20260909175125_lerobot_a7993ccb",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_213/PR-0011_BIZ-STORE-001_213_20260909175125_lerobot_a7993ccb_20260909175125_lerobot_a7993ccb.zip",
        ),
        (
            "20260909175323_lerobot_813ddbae",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_217/PR-0011_BIZ-STORE-001_217_20260909175323_lerobot_813ddbae_20260909175323_lerobot_813ddbae.zip",
        ),
        (
            "20260909175456_lerobot_74867465",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_216/PR-0011_BIZ-STORE-001_216_20260909175456_lerobot_74867465_20260909175456_lerobot_74867465.zip",
        ),
        (
            "20260909175643_lerobot_0fd80cdc",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_212/PR-0011_BIZ-STORE-001_212_20260909175643_lerobot_0fd80cdc_20260909175643_lerobot_0fd80cdc.zip",
        ),
        (
            "20260909180326_lerobot_7af49f52",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_215/PR-0011_BIZ-STORE-001_215_20260909180326_lerobot_7af49f52_20260909180326_lerobot_7af49f52.zip",
        ),
        (
            "20260909181641_lerobot_47cd03c3",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_211/PR-0011_BIZ-STORE-001_211_20260909181641_lerobot_47cd03c3_20260909181641_lerobot_47cd03c3.zip",
        ),
        (
            "20260909191822_lerobot_044a14a3",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_217/PR-0011_BIZ-STORE-001_217_20260909191822_lerobot_044a14a3_20260909191822_lerobot_044a14a3.zip",
        ),
        (
            "20260909191824_lerobot_ff43c3e3",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_211/PR-0011_BIZ-STORE-001_211_20260909191824_lerobot_ff43c3e3_20260909191824_lerobot_ff43c3e3.zip",
        ),
        (
            "20260909191853_lerobot_5f599eee",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_219/PR-0011_BIZ-STORE-001_219_20260909191853_lerobot_5f599eee_20260909191853_lerobot_5f599eee.zip",
        ),
        (
            "20260909192101_lerobot_d5ea6c17",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_213/PR-0011_BIZ-STORE-001_213_20260909192101_lerobot_d5ea6c17_20260909192101_lerobot_d5ea6c17.zip",
        ),
        (
            "20260909192142_lerobot_5e1d0916",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_216/PR-0011_BIZ-STORE-001_216_20260909192142_lerobot_5e1d0916_20260909192142_lerobot_5e1d0916.zip",
        ),
        (
            "20260909192220_lerobot_142fd6bb",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_215/PR-0011_BIZ-STORE-001_215_20260909192220_lerobot_142fd6bb_20260909192220_lerobot_142fd6bb.zip",
        ),
        (
            "20260909192307_lerobot_97018056",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_212/PR-0011_BIZ-STORE-001_212_20260909192307_lerobot_97018056_20260909192307_lerobot_97018056.zip",
        ),
    ],
}
