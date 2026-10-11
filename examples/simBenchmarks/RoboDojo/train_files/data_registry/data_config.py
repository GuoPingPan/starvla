"""RoboDojo (bimanual unified_robot) — 数据配置 / embodiment / 数据集混合。

自动被 ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge`` 发现
（扫描 ``examples/**/train_files/data_registry``）。

适配自 z_umi 的 relative_6d 配置，扩展为**双臂**：
- SE(3) 相对位姿 + rotation_6d 表示，对左右两条臂分别施加。
- 数据格式 (RoboDojo ee_lerobot_v30_video, 16D state/action)：
    observation.state / action = [left_pos(3) + left_quat_wxyz(4) + left_gripper(1)
                                  + right_pos(3) + right_quat_wxyz(4) + right_gripper(1)]
- 变换后维度：
    state  = left_pos(3)+left_rot6d(6)+left_grip(1)+right_pos(3)+right_rot6d(6)+right_grip(1) = 20D
    action = 相对位姿版同结构 = 20D  (action_dim_actual=20, pad 到 32 匹配 PI05 预训练 ckpt)
- 视频：cam_high + cam_left_wrist + cam_right_wrist (3 相机)
- 语言：每条 episode 的任务描述 (task_index → task 文本)

任务过滤：data_cfg.task_index (int) / task_indices (list) 只训练指定任务的 episodes。
"""
from typing import List

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    NormalizationTransform,
    PerStepNormalizationTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.relative_pose import RelativePoseTransform
from examples.realRobots.z_umi.train_files.data_registry.data_config import QuaternionTo6DRotation
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from examples.simBenchmarks.RoboDojo.train_files.data_registry.robodojo_dataset import RoboDojoLeRobotDataset


# ============================================================
# 双臂 split-key 定义
# ============================================================
# state 子键顺序（决定最终 state 拼接顺序）
_STATE_KEYS = [
    "state.end_left_position", "state.end_left_rotation", "state.left_gripper",
    "state.end_right_position", "state.end_right_rotation", "state.right_gripper",
]
_ACTION_KEYS = [
    "action.end_left_position", "action.end_left_rotation", "action.left_gripper",
    "action.end_right_position", "action.end_right_rotation", "action.right_gripper",
]
_VIDEO_KEYS = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
_LANGUAGE_KEYS = ["annotation.human.task"]

_OBSERVATION_INDICES = [0]
_ACTION_INDICES = list(range(50))  # action_horizon = 50


def _build_modality_config():
    return {
        "video":    ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_VIDEO_KEYS),
        "state":    ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_STATE_KEYS),
        "action":   ModalityConfig(delta_indices=_ACTION_INDICES, modality_keys=_ACTION_KEYS),
        "language": ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_LANGUAGE_KEYS),
    }


# ============================================================
# 双臂 SE(3) 相对位姿 + rotation_6d 配置
# ============================================================
class _RoboDojoBimanualRelative6DConfig:
    """RoboDojo 双臂配置（SE(3) 相对位姿 + rotation_6d），适配自 z_umi 单臂版本。

    数据维度变化（每条臂）：
    - 输入 state：position (3,) + quaternion_wxyz (4,) + gripper (1,) = 8D
    - 输出 state：position (3,) + rotation_6d (6,) + gripper (1,) = 10D
    - 输入 action：绝对下一帧位姿同 8D
    - 输出 action：相对 state[t0] 的 SE(3) 位姿 + rotation_6d + gripper = 10D
    双臂拼接 → state/action 各 20D。
    """

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = _VIDEO_KEYS
    state_keys = _STATE_KEYS
    action_keys = _ACTION_KEYS
    language_keys = _LANGUAGE_KEYS

    observation_indices = _OBSERVATION_INDICES
    action_indices = _ACTION_INDICES
    state_indices = _OBSERVATION_INDICES

    def __init__(self):
        # 两条臂各自一个虚拟 pose，供 RelativePoseTransform 做 SE(3) 相对化
        split_pose_keys = {
            "action.end_left_pose": {
                "position": "action.end_left_position",
                "rotation": "action.end_left_rotation",
                "state_position": "state.end_left_position",
                "state_rotation": "state.end_left_rotation",
            },
            "action.end_right_pose": {
                "position": "action.end_right_position",
                "rotation": "action.end_right_rotation",
                "state_position": "state.end_right_position",
                "state_rotation": "state.end_right_rotation",
            },
        }
        self.transforms = [
            # state + action → tensor
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),

            # action: SE(3) 相对位姿（左右臂分别相对到各自 state[t0]，此时仍是 4D 四元数 wxyz）
            RelativePoseTransform(
                apply_to=["action.end_left_pose", "action.end_right_pose"],
                split_pose_keys=split_pose_keys,
                input_rotation_type="quaternion",
                fast_forward=True,
            ),

            # state + action: 四元数 wxyz → rotation_6d（4 条 rotation key）
            QuaternionTo6DRotation(apply_to=["state.end_left_rotation"], input_convention="wxyz"),
            QuaternionTo6DRotation(apply_to=["state.end_right_rotation"], input_convention="wxyz"),
            QuaternionTo6DRotation(apply_to=["action.end_left_rotation"], input_convention="wxyz"),
            QuaternionTo6DRotation(apply_to=["action.end_right_rotation"], input_convention="wxyz"),

            # state: min_max 归一化（每维一组统计）
            NormalizationTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "min_max" for k in self.state_keys},
            ),
            # action: 逐时间步 q99 归一化（相对位姿位移随步数增长，每步独立统计）
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
        return RoboDojoLeRobotDataset(**kwargs)


# ============================================================
# 注册表
# ============================================================
ROBOT_TYPE_CONFIG_MAP = {
    "robodojo_bimanual_relative_6d": _RoboDojoBimanualRelative6DConfig(),
}

DATASET_NAMED_MIXTURES = {
    # 全量多任务（35 任务）。单任务训练用同一 mixture + data_cfg.task_index=N。
    # 第4元素 = BOS 上该数据集的 URI（download_data.py 读它拉数据；训练忽略第4元素）。
    # 尾 "/" = 目录（bos_io.download_dir 递归拉，幂等，适合 ~100GB 大数据集）；否则按 zip。
    "robodojo_ee_all": [
        ("RoboDojo_ee_lerobot_v30_video", 1.0, "robodojo_bimanual_relative_6d",
         "bos://umi-gripper/Orion/robodojo/RoboDojo_ee_lerobot_v30_video/"),
    ],
}
