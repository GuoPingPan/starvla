"""zerith 示例：6D rotation + relative pose 的 DataConfig（从上游 ``data_config.py`` 外移，避免改上游文件）。

历史上这段示例被追加在上游 ``gr00t_lerobot/data_config.py`` 末尾（patch P002），但全仓库无任何
robot_type 真正使用 ``relative_6d_example``，属于"参考用、未接线"的示例。为减小上游补丁面，移到
本独立文件（纯增量）。需要时 ``import`` 本模块即把示例注册进 ``ROBOT_TYPE_CONFIG_MAP``。

新 transform 一律走直接子模块导入（``transform.pose_rotation`` / ``transform.relative_pose`` 等），
不经 ``transform/__init__.py`` 重导出，因此不依赖对上游 ``__init__.py`` 的修改。
"""

# 注册到上游 ROBOT_TYPE_CONFIG_MAP（导入本模块即生效；本模块默认不被任何训练流程导入，故为 opt-in 示例）
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class Relative6DPoseDataConfig:
    """Example DataConfig demonstrating 6D rotation and relative pose control.

    Pipeline:
        raw (euler) → quaternion → relative pose → 6D rotation → normalize → concat
    """

    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_pose",  # [x, y, z, euler_x, euler_y, euler_z] - 6D
        "state.gripper",
    ]
    action_keys = [
        "action.eef_pose",  # [x, y, z, euler_x, euler_y, euler_z] - 6D
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig

        video_modality = ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys)
        state_modality = ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys)
        action_modality = ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys)
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices, modality_keys=self.language_keys
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
            StateActionToTensor,
            StateActionTransform,
        )

        # Import the new transforms (直接子模块导入，不依赖 transform/__init__.py 重导出)
        from starVLA.dataloader.gr00t_lerobot.transform.pose_rotation import (
            EulerPoseToQuaternionTransform,
            PoseRotationTransform,
        )
        from starVLA.dataloader.gr00t_lerobot.transform.relative_pose import RelativePoseTransform

        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            EulerPoseToQuaternionTransform(apply_to=["state.eef_pose"], euler_convention="XYZ"),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.eef_pose": "min_max", "state.gripper": "binary"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            EulerPoseToQuaternionTransform(apply_to=["action.eef_pose"], euler_convention="XYZ"),
            RelativePoseTransform(
                apply_to=["action.eef_pose"],
                state_keys_map={"action.eef_pose": "state.eef_pose"},
                input_rotation_type="quaternion",
                fast_forward=True,
            ),
            PoseRotationTransform(
                apply_to=["action.eef_pose"],
                rotation_type="rotation_6d",
                input_rotation_type="quaternion",
                fast_forward=True,
            ),
            StateActionTransform(
                apply_to=["action.eef_pose", "action.gripper"],
                normalization_modes={"action.eef_pose": "min_max", "action.gripper": "binary"},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


# Register the example config (opt-in：仅当本模块被 import 时)
ROBOT_TYPE_CONFIG_MAP["relative_6d_example"] = Relative6DPoseDataConfig()
