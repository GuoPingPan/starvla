"""UMI — 数据配置/embodiment/数据集混合。

自动被 ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge`` 发现。

提供两种机器人配置：
- umi_2gripper: 原版配置（quaternion 旋转）
- umi_2gripper_6d: 6D 旋转表示配置（更平滑，适合神经网络学习）

数据格式 (UMI 2-gripper):
* observation.state.end_0_pose (7,) = 位置(3) + 旋转四元数(4)
* observation.state.effector_0_joint_state (1,) = 夹爪
* action.end_0_pose (7,) = 位置(3) + 旋转四元数(4)
* action.effector_0_joint_state (1,) = 夹爪
* 视频: cam_wrist_0
"""
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    NormalizationTransform,
    PerStepNormalizationTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.relative_pose import (
    RelativePoseTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.quaternion_to_6d import QuaternionTo6DRotation
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from examples.realRobots.z_umi.train_files.data_registry.umi_dataset import UMILeRobotDataset


# ============================================================
# 公共配置（两个 robot_type 共享）
# ============================================================

_STATE_KEYS = ["state.end_0_position", "state.end_0_rotation", "state.effector_0_joint_state"]
_ACTION_KEYS = ["action.end_0_position", "action.end_0_rotation", "action.effector_0_joint_state"]
_VIDEO_KEYS = ["video.cam_wrist_0"]
_OBSERVATION_INDICES = [0]
_ACTION_INDICES = list(range(50))


def _build_modality_config():
    return {
        "video":  ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_VIDEO_KEYS),
        "state":  ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_STATE_KEYS),
        "action": ModalityConfig(delta_indices=_ACTION_INDICES,      modality_keys=_ACTION_KEYS),
    }


# ============================================================
# Relative-6D 配置（SE(3) 相对位姿 + rotation_6d）
# ============================================================

class _UMI2GripperRelative6DConfig:
    """UMI 双夹爪单臂配置（SE(3) 相对位姿 + rotation_6d 表示）。

    与 ``umi_2gripper_6d`` 的区别：
    - ``umi_2gripper_6d``：直接对绝对四元数做 4D→6D 转换（无相对化）。
    - ``umi_2gripper_relative_6d``：使用 ``RelativePoseTransform`` 把 action
      位姿相对到当前 state（SE(3) 群运算，几何正确），再做 4D→6D 转换。

    数据维度变化：
    - 输入 state：position (3,) + quaternion (4,) + gripper (1,) = 8D
    - 输出 state：position (3,) + rotation_6d (6,) + gripper (1,) = 10D
    - 输入 action：position (3,) + quaternion (4,) + gripper (1,) = 8D
    - 输出 action：relative_position (3,) + relative_rotation_6d (6,) + gripper (1,) = 10D
    """

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = _VIDEO_KEYS
    state_keys = _STATE_KEYS
    action_keys = _ACTION_KEYS
    language_keys = []

    observation_indices = _OBSERVATION_INDICES
    action_indices = _ACTION_INDICES

    def __init__(self):
        _VIRTUAL_POSE = "action.end_0_pose"
        split_pose_keys = {
            _VIRTUAL_POSE: {
                "position": "action.end_0_position",
                "rotation": "action.end_0_rotation",
                "state_position": "state.end_0_position",
                "state_rotation": "state.end_0_rotation",
            }
        }
        self.transforms=[
            # ---------------- state + action: 转 tensor ----------------
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),

            # ---------------- action: SE(3) 相对位姿（此时 rotation 仍是 4D 四元数） ----------------
            RelativePoseTransform(
                apply_to=[_VIRTUAL_POSE],
                split_pose_keys=split_pose_keys,
                input_rotation_type="quaternion",
                fast_forward=True,
            ),

            # ---------------- state + action: 四元数 → rotation_6d ----------------
            QuaternionTo6DRotation(apply_to=["state.end_0_rotation"], input_convention="xyzw"),
            QuaternionTo6DRotation(apply_to=["action.end_0_rotation"], input_convention="xyzw"),
            # ---------------- state + action: min_max 归一化 ----------------
            # 6D 旋转 / SE(3) 相对位姿已由上面的 transform 处理，这里只做纯归一化。
            # state 无 horizon（observation_indices=[0]）→ per-dim (D,) min_max；
            # action 是相对 t0 的累积位姿、位移随步数增长 → 逐时间步 (H, D') 归一化
            #（每步独立统计，避免远端大位移主导、压扁近期步分辨率）。
            # mode='q99'：用 q01/q99 边界而非 min/max。UMI action 离群严重（少数轨迹大幅
            # 位移 / step0 录制伪影把 min/max 撑大 2-20x，主体被压进 [-1,1] 的 ~30%）。
            # q99 把 ~98% 主体铺满 [-1,1]，分辨率提升 ~2x；离群点落到 [-1,1] 外（不裁剪，可逆）。
            NormalizationTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "min_max" for k in self.state_keys},
            ),
            PerStepNormalizationTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "q99" for k in self.action_keys},
            ),
        ]

    def modality_config(self):
        return _build_modality_config()

    def transform(self):
        return ComposedModalityTransform(transforms=self.transforms)

    def make_dataset(self, **kwargs):
        return UMILeRobotDataset(**kwargs)


# ============================================================
# 注册表
# ============================================================

ROBOT_TYPE_CONFIG_MAP = {
    "umi_2gripper_relative_6d": _UMI2GripperRelative6DConfig(),
}


DATASET_NAMED_MIXTURES = {
    "shenzhen_relative_6d": [
        # (dataset_id, weight, robot_type, bos_uri)  —— 第4元素 = BOS 上该数据集的 zip URI;
        # train.sh 的 download_data.py 读它来拉数据; 训练 (lerobot_datasets.py) 会忽略第4元素。
        ("20260720061720_lerobot_df899c7d", 1.0, "umi_2gripper_relative_6d",
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_126/PR-0008_BIZ-STORE-001_126_20260720061720_lerobot_df899c7d_20260720061720_lerobot_df899c7d.zip"),
    ],
    "shenzhen_relative_6d_2": [
        ("20260726133821_lerobot_7d6be9ce", 1.0, "umi_2gripper_relative_6d",
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_191/PR-0008_BIZ-STORE-001_191_20260726133821_lerobot_7d6be9ce_20260726133821_lerobot_7d6be9ce.zip"),
        # ("20260803035455_lerobot_5852395f", 1.0, "umi_2gripper_relative_6d",
        #  "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_03/PR-0008_BIZ-STORE-001_03_20260803035455_lerobot_5852395f_20260803035455_lerobot_5852395f.zip"),
    ],
    "shenzhen_relative_6d_2_nooutlier": [
        # 本地派生数据集：从 shenzhen_relative_6d_2 的两个数据集各剔 1 条异常长轨迹
        # (远超 ~320帧/5.4s 正常簇的离群)：
        #   7d6be9ce 剔 ep58 (794帧/13.2s, 中位数283的2.8x) → ..._noep58 (374→373 ep)
        #   5852395f 剔 ep81 (998帧/16.6s, 中位数284的3.5x) → ..._noep81 (154→153 ep)
        # 视频文件原样保留 (被剔 ep 时段成孤儿段, 无害)，仅重排 episode_index/from/to_index。
        # 第4元素留空：本地派生数据无 BOS uri; lerobot_datasets.py 用 *_ 忽略第4元素,
        # 仅 train.sh 的 download_data.py 读它 (如需经 train.sh 拉取需先上传 BOS)。
        ("20260726133821_lerobot_7d6be9ce_noep58", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260803035455_lerobot_5852395f_noep81", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    "shenzhen_relative_6d_2_nooutlier_forceclose": [
        # 本地派生数据集：在 _nooutlier 基础上做 "force-close" hack（不改结构，只改夹爪值）。
        # 原 UMI 数据每条轨迹夹爪: 闭合(0) → 张开(peak~0.027) → 再次闭合(回落到 ~0.008 平台, 没到0)。
        #   注: 0=完全闭合, ~0.027=张开。
        # hack 目标形状: 0 → peak → 纯0（去掉峰后的平台）。变体 B (s=0.0015/帧):
        #   从峰值按固定斜率 s=0.0015 线性降到 0，之后纯 0 到结尾；峰前(闭合→张开)完全不动。
        # 仅改 data parquet 的 observation.state.effector_0_joint_state 列；action 加载时由
        #   state[t+1] 派生，自动跟着变。stats/steps 缓存已删，归一化统计训练在线重算。
        # 构建脚本: examples/realRobots/z_umi/train_files/bar/build_gripper_forceclose.py
        ("20260726133821_lerobot_7d6be9ce_noep58_forceclose", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260803035455_lerobot_5852395f_noep81_forceclose", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    "hefei_relative_6d": [
        ("20260722114220_lerobot_b80dcf10", 1.0, "umi_2gripper_relative_6d",
         "bos://robot-data-from-disk/disk2lakehouse/umi-20260610180125/grasp_glue-20260713110000/umi-20260610180125_grasp_glue-20260713110000_20260722114220_lerobot_b80dcf10_20260722114220_lerobot_b80dcf10.zip"),

    ],

    "hefei_relative_6d_no280": [
        # 本地派生数据集：从 hefei (b80dcf10) 剔除异常长轨迹 ep280 (4334帧/72.2s, 其余轨迹 ~410帧) 后生成。
        # 772→771 episodes, 317254→312920 frames。视频原样保留 (ep280 时段成孤儿段, 无害)。
        # 第4元素留空：本地派生数据无 BOS uri; lerobot_datasets.py 会忽略第4元素,
        # 仅 train.sh 的 download_data.py 会读它 (如需经 train.sh 拉取需先上传 BOS)。
        ("20260722114220_lerobot_b80dcf10_no280", 1.0, "umi_2gripper_relative_6d", ""),
    ],

}
