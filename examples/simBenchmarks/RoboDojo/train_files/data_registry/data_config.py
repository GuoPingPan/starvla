"""RoboDojo (bimanual unified_robot) — 数据配置 / embodiment / 数据集混合。

自动被 ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge`` 发现
（扫描 ``examples/**/train_files/data_registry``）。

适配自 zerith_pi05 的 relative_6d 配置，扩展为**双臂**：
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
from examples.realRobots.zerith_pi05.train_files.data_registry.data_config import QuaternionTo6DRotation
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

# Replay joint observations and commanded IK targets share the same 14D order.
_JOINT_STATE_KEYS = [
    "state.left_arm_joint", "state.left_gripper",
    "state.right_arm_joint", "state.right_gripper",
]
_JOINT_ACTION_KEYS = [
    "action.left_arm_joint", "action.left_gripper",
    "action.right_arm_joint", "action.right_gripper",
]

_OBSERVATION_INDICES = [0]
_ACTION_INDICES = list(range(50))  # action_horizon = 50


def _build_modality_config():
    return {
        "video":    ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_VIDEO_KEYS),
        "state":    ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_STATE_KEYS),
        "action":   ModalityConfig(delta_indices=_ACTION_INDICES, modality_keys=_ACTION_KEYS),
        "language": ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_LANGUAGE_KEYS),
    }


def _build_joint_modality_config():
    return {
        "video": ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_JOINT_STATE_KEYS),
        "action": ModalityConfig(delta_indices=_ACTION_INDICES, modality_keys=_JOINT_ACTION_KEYS),
        "language": ModalityConfig(delta_indices=_OBSERVATION_INDICES, modality_keys=_LANGUAGE_KEYS),
    }


# ============================================================
# 双臂 SE(3) 相对位姿 + rotation_6d 配置
# ============================================================
class _RoboDojoBimanualRelative6DConfig:
    """RoboDojo 双臂配置（SE(3) 相对位姿 + rotation_6d），适配自 zerith_pi05 单臂版本。

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
                input_quaternion_convention="wxyz",
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


class _RoboDojoBimanualAbsolute6DConfig(_RoboDojoBimanualRelative6DConfig):
    """Keep the dataset's absolute EE targets; only convert quaternion to 6D."""

    def __init__(self):
        super().__init__()
        self.transforms = [
            transform for transform in self.transforms
            if not isinstance(transform, RelativePoseTransform)
        ]


class _RoboDojoBimanualAbsoluteJointConfig(_RoboDojoBimanualRelative6DConfig):
    """Absolute six joint targets and continuous gripper command per arm."""

    state_keys = _JOINT_STATE_KEYS
    action_keys = _JOINT_ACTION_KEYS
    state_key_dims = dict(zip(_JOINT_STATE_KEYS, [6, 1, 6, 1]))
    action_key_dims = dict(zip(_JOINT_ACTION_KEYS, [6, 1, 6, 1]))

    def __init__(self):
        # No EE pose, quaternion, or delta transforms are applicable to joints.
        self.transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            NormalizationTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            NormalizationTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ]

    def modality_config(self):
        return _build_joint_modality_config()


# ============================================================
# 注册表
# ============================================================
ROBOT_TYPE_CONFIG_MAP = {
    "robodojo_bimanual_relative_6d": _RoboDojoBimanualRelative6DConfig(),
    "robodojo_bimanual_absolute_6d": _RoboDojoBimanualAbsolute6DConfig(),
    "robodojo_bimanual_absolute_joint": _RoboDojoBimanualAbsoluteJointConfig(),
}

DATASET_NAMED_MIXTURES = {
    # make_kong official_push_postures_v2: 195 successful seed/layout rollouts,
    # converted from NPZ + three MP4s to LeRobot v3.0. The dataset is staged on
    # the training CFS at /mnt/cfs/data/robodojo/RoboDojo_make_kong_v30.
    # No BOS URI: stage the converted dataset on CFS before launching training.
    "robodojo_make_kong_v2": [
        ("RoboDojo_make_kong_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    "robodojo_make_kong_v2_abs": [
        ("RoboDojo_make_kong_v30", 1.0, "robodojo_bimanual_absolute_6d", ""),
    ],
    "robodojo_make_kong_joint_replay": [
        ("RoboDojo_make_kong_joint_replay_v30", 1.0, "robodojo_bimanual_absolute_joint", ""),
    ],
    "robodojo_play_xylophone_joint": [
        ("RoboDojo_play_Xylophone_joint_v30", 1.0, "robodojo_bimanual_absolute_joint", ""),
    ],
    # 全量多任务（35 任务）。单任务训练用同一 mixture + data_cfg.task_index=N。
    # 第4元素 = BOS 上该数据集的 URI（download_data.py 读它拉数据；训练忽略第4元素）。
    # 尾 "/" = 目录（bos_io.download_dir 递归拉，幂等，适合 ~100GB 大数据集）；否则按 zip。
    "robodojo_ee_all": [
        ("RoboDojo_ee_lerobot_v30_video", 1.0, "robodojo_bimanual_relative_6d",
         "bos://umi-gripper/Orion/robodojo/RoboDojo_ee_lerobot_v30_video/"),
    ],
    # ── 4 任务消融（官方 task 3=cover_blocks / 4=deposit_coin / 9=fold_clothes /
    #    29=stack_blocks）────────────────────────────────────────────────────
    # 官方 4 任务物理子集（scripts/dev/subset_robodojo_tasks.py 生成；task_index 重编为
    # 0..3，视频 symlink 官方全量数据集，故第4元素留空 —— 数据已在 CFS 本地）。
    "robodojo_ee_4task": [
        ("RoboDojo_ee_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    # 官方 4 任务子集 + mcap 闭环采集生成数据（scripts/dev/mcap_to_lerobot_v30.py 从
    # /mnt/cfs/robodojo_expert_mcap_100 的 foxglove MCAP 转出，404 集 / ~129k 帧）。
    "robodojo_ee_4task_plus_gen": [
        ("RoboDojo_ee_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_mcap100_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    # ── 6 任务扩展（+ play_tic_tac_toe=task21 / press_by_number=task25）────
    "robodojo_ee_6task": [
        ("RoboDojo_ee_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_ee_2task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    "robodojo_ee_6task_plus_gen": [
        ("RoboDojo_ee_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_ee_2task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_mcap100_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_mcap100_2task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    # ── 对齐版生成数据 (action[t]:=state[t+1], 修正 mcap action 滞后 ~7°/1mm;
    #    异常 episode 已剔除: 4task 2 个跳变 / 2task 26 个跳变; 四元数重归一) ──
    "robodojo_ee_4task_plus_gen_align": [
        ("RoboDojo_ee_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_mcap100_4task_align_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    "robodojo_ee_6task_plus_gen_align": [
        ("RoboDojo_ee_4task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_ee_2task_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_mcap100_4task_align_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_mcap100_2task_align_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    # ── cover_blocks 单任务三分支消融 (官方 task3 / 生成对齐 102 集) ────────
    "robodojo_cover_base": [
        ("RoboDojo_ee_cover_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    "robodojo_cover_gen": [
        ("RoboDojo_gen_cover_align_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    "robodojo_cover_base_gen": [
        ("RoboDojo_ee_cover_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_cover_align_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    # ── fold_clothes 单任务三分支消融 (官方 task9 / 生成对齐 98 集, 剔2异常) ──
    "robodojo_fold_base": [
        ("RoboDojo_ee_fold_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    "robodojo_fold_gen": [
        ("RoboDojo_gen_fold_align_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
    "robodojo_fold_base_gen": [
        ("RoboDojo_ee_fold_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
        ("RoboDojo_gen_fold_align_v30", 1.0, "robodojo_bimanual_relative_6d", ""),
    ],
}
