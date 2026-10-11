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
from starVLA.dataloader.gr00t_lerobot.transform import (
    VideoCenterCrop,
    VideoColorJitter,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from examples.realRobots.zerith_pi05.train_files.data_registry.umi_dataset import UMILeRobotDataset


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
            # ---------------- video: 图像前处理（cover 全部图像变换，最终 resize 到 224） ----------------
            # 顺序: ToTensor -> CenterCrop(正方形) -> Resize(224) -> ColorJitter -> ToNumpy
            # - ColorJitter 放在 Resize 之后 (224×224 而非 2704×2028, ~110x 像素量减少):
            #   brightness/contrast/saturation 是逐像素仿射, 与双线性 resize 数学可交换;
            #   hue (RGB↔HSV 非线性) 顺序差异可忽略。eval no-op 不受影响。
            # - CenterCrop 按 crop_ratio=1.0 裁正方形（运行时读真实 H/W，自适应不同分辨率数据集），
            #   消除长宽比、避免 resize 拉伸
            # - Resize 到 224：_pack_sample 不再 resize，最终尺寸由此决定
            # 所有参数写死默认值；要调实验配置改这里。
            VideoToTensor(apply_to=self.video_keys),
            VideoCenterCrop(apply_to=self.video_keys, crop_ratio=1.0),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05,
            ),
            VideoToNumpy(apply_to=self.video_keys),

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
            # ---------------- state + action: q99 归一化 ----------------
            # 6D 旋转 / SE(3) 相对位姿已由上面的 transform 处理，这里只做纯归一化。
            # state 无 horizon（observation_indices=[0]）→ per-dim (D,) q99；
            # action 是相对 t0 的累积位姿、位移随步数增长 → 逐时间步 (H, D') 归一化
            #（每步独立统计，避免远端大位移主导、压扁近期步分辨率）。
            # mode='q99'：用 q01/q99 边界而非 min/max。state 绝对位姿与 action 都有整段
            # 漂移坏 ep（位姿突跳几米~20m）把 min/max 撑大 5-25x，主体被压进 [-1,1] 的
            # ~5-30%；q99 把 ~98% 主体铺满 [-1,1]，离群点落到 [-1,1] 外（不裁剪，可逆）。
            # 注：stats 文件已含 q01/q99，改 mode 无需重算 stats（缓存命中即可用）；
            #     但已训 ckpt 用的是 min_max，q99 需重训生效。
            NormalizationTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "q99" for k in self.state_keys},
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
    "bench_demo":[
        ("20260806145048_lerobot_3f973840", 1.0, "umi_2gripper_relative_6d",
        "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_275/PR-0008_BIZ-STORE-001_2_275_20260806145048_lerobot_3f973840_20260806145048_lerobot_3f973840.zip"),
    ],
    "shenzhen_relative_6d": [
        # (dataset_id, weight, robot_type, bos_uri)  —— 第4元素 = BOS 上该数据集的 zip URI;
        # train.sh 的 download_data.py 读它来拉数据; 训练 (lerobot_datasets.py) 会忽略第4元素。
        ("20260720061720_lerobot_df899c7d", 1.0, "umi_2gripper_relative_6d",
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_126/PR-0008_BIZ-STORE-001_126_20260720061720_lerobot_df899c7d_20260720061720_lerobot_df899c7d.zip"),
    ],
    "shenzhen_relative_6d_3": [
        ("20260804152533_lerobot_d6949716", 1.0, "umi_2gripper_relative_6d",
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_271/PR-0008_BIZ-STORE-001_2_271_20260804152533_lerobot_d6949716_20260804152533_lerobot_d6949716.zip"),
        ("20260804131149_lerobot_8ee73e4e", 1.0, "umi_2gripper_relative_6d",
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_270/PR-0008_BIZ-STORE-001_2_270_20260804131149_lerobot_8ee73e4e_20260804131149_lerobot_8ee73e4e.zip")
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
        # 构建脚本: examples/realRobots/zerith_pi05/train_files/bar/build_gripper_forceclose.py
        ("20260726133821_lerobot_7d6be9ce_noep58_forceclose", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803035455_lerobot_5852395f_noep81_forceclose", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    "shenzhen_relative_6d_2_nooutlier_forceclose_trim": [
        # 本地派生：在 _forceclose 基础上, 把每条轨迹末尾"纯0平台"截到只留 0.5s (30帧@60fps)。
        # forceclose 把峰后整段拉成0, 末尾0平台很长 (median ~149帧/2.5s, 占整条轨迹近半,
        # 多为"夹爪已闭合静止"冗余帧)。截到30帧后帧数 ~ -42%/-41%:
        #   7d6be9ce: 105294→60554 (-42.5%)   5852395f: 42573→25271 (-40.6%)
        # 结构性改动(仿 _nooutlier): 按 episode 删尾行 + 重排全局 index + 重算
        #   episodes(length/dataset_from_to_index/to_timestamp) + info.json(total_frames);
        #   to_timestamp 收到 from_ts+length/fps 维持 (to_ts-from_ts==length/fps)。
        # 视频不动(尾帧成孤儿段, 无害): loader 解码只用 from_ts+每集timestamp列(datasets.py:1832-1842)。
        # 峰前(含峰)+下降段完全不变, 仅删纯0平台的多余帧; episodes 数不变。stats/steps 缓存已删。
        # 构建脚本: examples/realRobots/zerith_pi05/train_files/bar/build_gripper_forceclose_trim.py
        ("20260726133821_lerobot_7d6be9ce_noep58_forceclose_trim", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803035455_lerobot_5852395f_noep81_forceclose_trim", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    "shenzhen_relative_6d_2_nooutlier_forceclose_trim0": [
        # 本地派生：同 _trim 但 KEEP=0 —— 末尾"纯0平台"全删, 轨迹止于下降段最后一个非0帧
        # (夹爪几乎闭合~0.0016 但未精确到0)。比 _trim 多删那 0.5s(30帧)保持段:
        #   7d6be9ce: 105294→49364 (-53.1%)   5852395f: 42573→20681 (-51.4%)
        # 其余与 _trim 完全相同(结构性改动/视频不动/峰前+下降段不变/episodes数不变/stats在线重算)。
        # 构建脚本: TRIM_KEEP=0 TRIM_SUFFIX=_trim0 python3 bar/build_gripper_forceclose_trim.py
        ("20260726133821_lerobot_7d6be9ce_noep58_forceclose_trim0", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803035455_lerobot_5852395f_noep81_forceclose_trim0", 1.0, "umi_2gripper_relative_6d", ""),
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


    # <<<< AUTO-GENERATED from ddl-job-4715dbbe >>>>
    # ddl_job_4715dbbe: 35 个数据集
    "ddl_job_4715dbbe": [
        ("20260719185726_lerobot_0f437b52", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取红狗_猫用宠物营养补充剂_10g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/umi-20260610180125/grasp_ointment-20260713091601/umi-20260610180125_grasp_ointment-20260713091601_20260719185726_lerobot_0f437b52_20260719185726_lerobot_0f437b52.zip"),
        ("20260719194252_lerobot_6b874f60", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取奥信达_强力补鞋胶_散装-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/umi-20260610180125/grasp_glue-20260713110000/umi-20260610180125_grasp_glue-20260713110000_20260719194252_lerobot_6b874f60_20260719194252_lerobot_6b874f60.zip"),
        ("20260721040252_lerobot_f99915db", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取哇哈哈_AD钙_220g-单sku8个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_37/PR-0008_BIZ-STORE-001_37_20260721040252_lerobot_f99915db_20260721040252_lerobot_f99915db.zip"),
        ("20260721055517_lerobot_f26f9106", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取高亮远射_手电筒_散装-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_34/PR-0008_BIZ-STORE-001_2_34_20260721055517_lerobot_f26f9106_20260721055517_lerobot_f26f9106.zip"),
        ("20260721061238_lerobot_90e6a697", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取惠宜选_苹果手表磁吸充电器_17g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_117/PR-0008_BIZ-STORE-001_117_20260721061238_lerobot_90e6a697_20260721061238_lerobot_90e6a697.zip"),
        ("20260721062108_lerobot_2c7955ce", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取德优米_X52A手机风冷降温器_50g-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_106/PR-0008_BIZ-STORE-001_106_20260721062108_lerobot_2c7955ce_20260721062108_lerobot_2c7955ce.zip"),
        ("20260721092621_lerobot_9a54a112", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取宠佼佼_鱼油化毛膏_120g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_27/PR-0008_BIZ-STORE-001_2_27_20260721092621_lerobot_9a54a112_20260721092621_lerobot_9a54a112.zip"),
        ("20260721100220_lerobot_d37ab51a", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取鑫朗日_清洁软胶_160g±10g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_24/PR-0008_BIZ-STORE-001_2_24_20260721100220_lerobot_d37ab51a_20260721100220_lerobot_d37ab51a.zip"),
        ("20260721102016_lerobot_f9c8ce40", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取双汇_润口香甜王玉米风味香肠_75g-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_61/PR-0008_BIZ-STORE-001_61_20260721102016_lerobot_f9c8ce40_20260721102016_lerobot_f9c8ce40.zip"),
        ("20260721102040_lerobot_a939fffc", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取旺旺_碎冰冰果味饮料葡萄味_78ml-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_60/PR-0008_BIZ-STORE-001_60_20260721102040_lerobot_a939fffc_20260721102040_lerobot_a939fffc.zip"),
        ("20260722114220_lerobot_b80dcf10", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取奥信达_强力补鞋胶_散装-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/umi-20260610180125/grasp_glue-20260713110000/umi-20260610180125_grasp_glue-20260713110000_20260722114220_lerobot_b80dcf10_20260722114220_lerobot_b80dcf10.zip"),
        ("20260723062337_lerobot_3666d289", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取好丽友_蛋黄派_46g-单sku6个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_94/PR-0008_BIZ-STORE-001_94_20260723062337_lerobot_3666d289_20260723062337_lerobot_3666d289.zip"),
        ("20260724095346_lerobot_b416dbe0", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取梁丰_麦丽素_25g-单sku7个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_46/PR-0008_BIZ-STORE-001_2_46_20260724095346_lerobot_b416dbe0_20260724095346_lerobot_b416dbe0.zip"),
        ("20260724095454_lerobot_23523b4b", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取高亮远射_手电筒_散装-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_34/PR-0008_BIZ-STORE-001_2_34_20260724095454_lerobot_23523b4b_20260724095454_lerobot_23523b4b.zip"),
        ("20260724095457_lerobot_ee041cdf", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取宠佼佼_鱼油化毛膏_120g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_27/PR-0008_BIZ-STORE-001_2_27_20260724095457_lerobot_ee041cdf_20260724095457_lerobot_ee041cdf.zip"),
        ("20260724095529_lerobot_a3e2fb6b", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取暴肌独角兽_全麦麸五黑紫米欧包奶酪味_50g-单sku6个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_156/PR-0008_BIZ-STORE-001_156_20260724095529_lerobot_a3e2fb6b_20260724095529_lerobot_a3e2fb6b.zip"),
        ("20260724095618_lerobot_dfd0a83d", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取狗牙儿_比萨卷烧烤味_20g-单sku5个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_49/PR-0008_BIZ-STORE-001_2_49_20260724095618_lerobot_dfd0a83d_20260724095618_lerobot_dfd0a83d.zip"),
        ("20260724095632_lerobot_e7e4ae84", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取红狗_猫用宠物营养补充剂_10g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/umi-20260610180125/grasp_ointment-20260713091601/umi-20260610180125_grasp_ointment-20260713091601_20260724095632_lerobot_e7e4ae84_20260724095632_lerobot_e7e4ae84.zip"),
        ("20260724095815_lerobot_1196f165", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取阿尔卑斯_草莓牛奶味_10g-单sku5个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_157/PR-0008_BIZ-STORE-001_157_20260724095815_lerobot_1196f165_20260724095815_lerobot_1196f165.zip"),
        ("20260724100205_lerobot_7272f39d", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取鑫朗日_清洁软胶_160g±10g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_24/PR-0008_BIZ-STORE-001_2_24_20260724100205_lerobot_7272f39d_20260724100205_lerobot_7272f39d.zip"),
        ("20260724100217_lerobot_fac8e100", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取达利园_蛋黄派_散装-单sku7个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_153/PR-0008_BIZ-STORE-001_153_20260724100217_lerobot_fac8e100_20260724100217_lerobot_fac8e100.zip"),
        ("20260724100416_lerobot_b5e9b282", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取庞胖胖_椰椰爆打奶油老式面包_散装-单sku5个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_160/PR-0008_BIZ-STORE-001_160_20260724100416_lerobot_b5e9b282_20260724100416_lerobot_b5e9b282.zip"),
        ("20260724100509_lerobot_197d0c5f", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取旺旺_碎冰冰果味饮料葡萄味_78ml-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_60/PR-0008_BIZ-STORE-001_60_20260724100509_lerobot_197d0c5f_20260724100509_lerobot_197d0c5f.zip"),
        ("20260724100525_lerobot_b09e78b5", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取惠宜选_苹果手表磁吸充电器_17g-单sku9个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_117/PR-0008_BIZ-STORE-001_117_20260724100525_lerobot_b09e78b5_20260724100525_lerobot_b09e78b5.zip"),
        ("20260724100527_lerobot_d86c4ca6", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取双汇_润口香甜王玉米风味香肠_75g-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_61/PR-0008_BIZ-STORE-001_61_20260724100527_lerobot_d86c4ca6_20260724100527_lerobot_d86c4ca6.zip"),
        ("20260724100644_lerobot_7dac99a3", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取德优米_X52A手机风冷降温器_50g-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_106/PR-0008_BIZ-STORE-001_106_20260724100644_lerobot_7dac99a3_20260724100644_lerobot_7dac99a3.zip"),
        ("20260724100644_lerobot_fd9fa6bd", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取奥信达_强力补鞋胶_散装-单sku10个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/umi-20260610180125/grasp_glue-20260713110000/umi-20260610180125_grasp_glue-20260713110000_20260724100644_lerobot_fd9fa6bd_20260724100644_lerobot_fd9fa6bd.zip"),
        ("20260724101506_lerobot_b4d97096", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取好丽友_蛋黄派_46g-单sku6个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_94/PR-0008_BIZ-STORE-001_94_20260724101506_lerobot_b4d97096_20260724101506_lerobot_b4d97096.zip"),
        ("20260724101545_lerobot_0bda20df", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取哇哈哈_AD钙_220g-单sku8个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_37/PR-0008_BIZ-STORE-001_37_20260724101545_lerobot_0bda20df_20260724101545_lerobot_0bda20df.zip"),
        ("20260727060951_lerobot_22517704", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取狗牙儿_比萨卷烧烤味_20g-单sku5个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_49/PR-0008_BIZ-STORE-001_2_49_20260727060951_lerobot_22517704_20260727060951_lerobot_22517704.zip"),
        ("20260727061626_lerobot_6d1aaaf9", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取梁丰_麦丽素_25g-单sku7个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_2_46/PR-0008_BIZ-STORE-001_2_46_20260727061626_lerobot_6d1aaaf9_20260727061626_lerobot_6d1aaaf9.zip"),
        ("20260727061729_lerobot_ae6995d5", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取暴肌独角兽_全麦麸五黑紫米欧包奶酪味_50g-单sku6个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_156/PR-0008_BIZ-STORE-001_156_20260727061729_lerobot_ae6995d5_20260727061729_lerobot_ae6995d5.zip"),
        ("20260727062644_lerobot_58699c00", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取庞胖胖_椰椰爆打奶油老式面包_散装-单sku5个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_160/PR-0008_BIZ-STORE-001_160_20260727062644_lerobot_58699c00_20260727062644_lerobot_58699c00.zip"),
        ("20260727062830_lerobot_e14d35d7", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取阿尔卑斯_草莓牛奶味_10g-单sku5个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_157/PR-0008_BIZ-STORE-001_157_20260727062830_lerobot_e14d35d7_20260727062830_lerobot_e14d35d7.zip"),
        ("20260727063238_lerobot_ff9da23a", 1.0, "umi_2gripper_relative_6d",  # lerobot_mcap_抓取达利园_蛋黄派_散装-单sku7个物体-大框
         "bos://robot-data-from-disk/disk2lakehouse/PR-0008/BIZ-STORE-001_153/PR-0008_BIZ-STORE-001_153_20260727063238_lerobot_ff9da23a_20260727063238_lerobot_ff9da23a.zip"),
    ],
    # <<<< END AUTO-GENERATED from ddl-job-4715dbbe >>>>

    # <<<< AUTO-GENERATED from ddl_job_4715dbbe_nooutlier >>>>
    # ddl_job_4715dbbe_nooutlier: 35 个数据集, 每个剔 top-3 最长 ep (含 3 条 4334帧/72s 极异常)。
    # 派生自 ddl_job_4715dbbe; 视频硬链接不动; 第4元素 BOS uri 留空 (本地派生)。
    "ddl_job_4715dbbe_nooutlier": [
        ("20260719185726_lerobot_0f437b52_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260719194252_lerobot_6b874f60_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721040252_lerobot_f99915db_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721055517_lerobot_f26f9106_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061238_lerobot_90e6a697_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721062108_lerobot_2c7955ce_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260721092621_lerobot_9a54a112_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260721100220_lerobot_d37ab51a_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260721102016_lerobot_f9c8ce40_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260721102040_lerobot_a939fffc_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260722114220_lerobot_b80dcf10_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260723062337_lerobot_3666d289_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724095346_lerobot_b416dbe0_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724095454_lerobot_23523b4b_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724095457_lerobot_ee041cdf_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724095529_lerobot_a3e2fb6b_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724095618_lerobot_dfd0a83d_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724095632_lerobot_e7e4ae84_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724095815_lerobot_1196f165_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100205_lerobot_7272f39d_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100217_lerobot_fac8e100_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100416_lerobot_b5e9b282_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100509_lerobot_197d0c5f_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100525_lerobot_b09e78b5_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100527_lerobot_d86c4ca6_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100644_lerobot_7dac99a3_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724100644_lerobot_fd9fa6bd_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724101506_lerobot_b4d97096_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260724101545_lerobot_0bda20df_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260727060951_lerobot_22517704_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260727061626_lerobot_6d1aaaf9_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260727061729_lerobot_ae6995d5_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260727062644_lerobot_58699c00_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260727062830_lerobot_e14d35d7_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        # ("20260727063238_lerobot_ff9da23a_nooutlier", 1.0, "umi_2gripper_relative_6d", "")
    ],
    # <<<< END AUTO-GENERATED from ddl-job-4715dbbe-nooutlier >>>>

    # <<<< AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup >>>>
    # 去重 mixture: 从 _nooutlier 删 18 个重复数据集 (同 source_mcap_uri 重复入库, 占 51.6%),
    # 保留 17 个独立数据集 (每重复组取 ORDER 最靠前; 补鞋胶3连删2). 引用现有 _nooutlier 目录, 不复制.
    # 重复分析见 bar/find_duplicates.py; 去重后坏 ep 40→20 (7 数据集, 待 _clean 剔除).
    "ddl_job_4715dbbe_nooutlier_dedup": [
        ("20260719185726_lerobot_0f437b52_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260719194252_lerobot_6b874f60_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721040252_lerobot_f99915db_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721055517_lerobot_f26f9106_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061238_lerobot_90e6a697_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721062108_lerobot_2c7955ce_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721092621_lerobot_9a54a112_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100220_lerobot_d37ab51a_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102016_lerobot_f9c8ce40_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102040_lerobot_a939fffc_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723062337_lerobot_3666d289_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095346_lerobot_b416dbe0_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095529_lerobot_a3e2fb6b_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095618_lerobot_dfd0a83d_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095815_lerobot_1196f165_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100217_lerobot_fac8e100_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100416_lerobot_b5e9b282_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    # <<<< END AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup >>>>

    # <<<< AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean >>>>
    # 最终训练 mixture: 去重(17 数据集) + 剔位姿漂移坏 ep. 7 个有坏 ep 的用 _clean 版,
    # 10 个无坏 ep 的用原 _nooutlier. state 用 q99 归一化 (见 _UMI2GripperRelative6DConfig).
    # 去重见 bar/find_duplicates.py + register_dedup_mixture.py;
    # 坏 ep 见 bar/scan_all_pose_jumps.py(>1m); 剔除见 bar/build_nooutlier_clean.py.
    "ddl_job_4715dbbe_nooutlier_dedup_clean": [
        ("20260719185726_lerobot_0f437b52_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260719194252_lerobot_6b874f60_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721040252_lerobot_f99915db_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721055517_lerobot_f26f9106_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061238_lerobot_90e6a697_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721062108_lerobot_2c7955ce_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721092621_lerobot_9a54a112_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100220_lerobot_d37ab51a_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102016_lerobot_f9c8ce40_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102040_lerobot_a939fffc_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723062337_lerobot_3666d289_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095346_lerobot_b416dbe0_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095529_lerobot_a3e2fb6b_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095618_lerobot_dfd0a83d_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095815_lerobot_1196f165_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100217_lerobot_fac8e100_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100416_lerobot_b5e9b282_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    # <<<< END AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean >>>>

    # <<<< AUTO-GENERATED: umi_6sessions_clean >>>>
    # 6 个会话的去重干净子集 (用户挑的会话, 每会话取 dedup 保留的 1 个 hash):
    #   grasp_glue / BIZ-STORE-001_37 / _2_34 / _106 / _2_49 / _160
    # 版本: 有 _clean(剔位姿漂移坏ep) 用 _clean, 否则 _nooutlier(剔异常长ep). state q99 归一化.
    # 会话→hash 映射见 bar/scan_sessions.py; 范围/版本见对话(每会话1个, 最干净).
    "umi_6sessions_clean": [
        ("20260719194252_lerobot_6b874f60_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),  # 补鞋胶(grasp_glue)
        ("20260721040252_lerobot_f99915db_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),  # AD钙(BIZ-37)
        ("20260721055517_lerobot_f26f9106_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),  # 手电筒(BIZ-2_34)
        ("20260721062108_lerobot_2c7955ce_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),  # 风冷(BIZ-106)
        ("20260724095618_lerobot_dfd0a83d_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),  # 比萨卷(BIZ-2_49)
        ("20260724100416_lerobot_b5e9b282_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),  # 老面包(BIZ-160)
    ],
    # <<<< END AUTO-GENERATED: umi_6sessions_clean >>>>

    # <<<< AUTO-GENERATED: umi_6sessions_clean_nohjerk >>>>
    # umi_6sessions_clean 的剔高jerk离群ep版: 3 个独有数据集(2c7955ce/dfd0a83d/b5e9b282)的 _clean
    # 再剔轨迹聚类 cluster1 的高jerk抖动ep(共70条, 3-4%/集); 3 个共享集(6b874f60/f99915db/f26f9106)
    # 离群极少, 沿用原 _nooutlier. 离群来自 trajectory_cluster.py (非漂移坏ep, 是手持UMI高频抖动);
    # 派生脚本 bar/build_umi_nohjerk.py (复用 build_nooutlier_clean 剔ep逻辑, 视频硬链接).
    "umi_6sessions_clean_nohjerk": [
        ("20260719194252_lerobot_6b874f60_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),  # 补鞋胶(共享)
        ("20260721040252_lerobot_f99915db_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),  # AD钙(共享)
        ("20260721055517_lerobot_f26f9106_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),  # 手电筒(共享)
        ("20260721062108_lerobot_2c7955ce_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),  # 风冷
        ("20260724095618_lerobot_dfd0a83d_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),  # 比萨卷
        ("20260724100416_lerobot_b5e9b282_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),  # 老面包
    ],
    # <<<< END AUTO-GENERATED: umi_6sessions_clean_nohjerk >>>>

    # <<<< AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean_nohjerk >>>>
    # ddl_job_4715dbbe_nooutlier_dedup_clean (17数据集) 的剔高jerk离群ep版:
    # 13 个有离群的目录生成 _nohjerk (剔轨迹聚类 cluster>0 的188条, 1.67%; 非漂移坏ep, 手持UMI高频抖动),
    # 4 个零离群(0f437b52/f99915db/f26f9106/f9c8ce40)沿用原目录.
    # 离群来自 analyze_dedup_clean.py (trajcluster_dedup_clean_out/DROP_EPS.json);
    # 派生脚本 bar/build_dedup_clean_nohjerk.py (复用 build_nooutlier_clean 剔ep逻辑, 视频硬链接).
    # 注: 2c7955ce/b5e9b282/dfd0a83d 的 _clean_nohjerk 用 dedup 17集聚类的离群(23/42/34)覆盖了
    # 之前 umi_6sessions_clean 聚类的版本(18/27/25), umi_6sessions_clean_nohjerk 也随之更彻底.
    "ddl_job_4715dbbe_nooutlier_dedup_clean_nohjerk": [
        ("20260719185726_lerobot_0f437b52_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260719194252_lerobot_6b874f60_nooutlier_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721040252_lerobot_f99915db_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721055517_lerobot_f26f9106_nooutlier", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061238_lerobot_90e6a697_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721062108_lerobot_2c7955ce_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721092621_lerobot_9a54a112_nooutlier_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100220_lerobot_d37ab51a_nooutlier_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102016_lerobot_f9c8ce40_nooutlier_clean", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102040_lerobot_a939fffc_nooutlier_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723062337_lerobot_3666d289_nooutlier_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095346_lerobot_b416dbe0_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095529_lerobot_a3e2fb6b_nooutlier_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095618_lerobot_dfd0a83d_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095815_lerobot_1196f165_nooutlier_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100217_lerobot_fac8e100_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100416_lerobot_b5e9b282_nooutlier_clean_nohjerk", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    # <<<< END AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean_nohjerk >>>>

    # <<<< AUTO-GENERATED: f99915db 问题帧率三档 (训练对比 loss, 同集控制变量) >>>>
    # 同集 f99915db 按帧级异常分析 (bar/frame_level_anomaly.py) 的"问题帧率(pfr)"分三档,
    # 每档派生子集 (keep 该档 ep, drop 其余). 唯一差异是问题帧率, 看其对训练 loss 影响.
    # pfrEq0=完全无问题帧(145ep/54576帧) / pfr0to1=0-1%(301ep/113258帧) / pfr1to5=1-5%(283ep/106970帧).
    # 派生脚本 bar/build_band_subsets.py (复用 build_ddl 剔ep逻辑). 直观看: bar/band_rerun/ 每档2条rerun.
    "band_pfrEq0_f99915db": [
        ("20260721040252_lerobot_f99915db_nooutlier_pfrEq0", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    "band_pfr0to1_f99915db": [
        ("20260721040252_lerobot_f99915db_nooutlier_pfr0to1", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    "band_pfr1to5_f99915db": [
        ("20260721040252_lerobot_f99915db_nooutlier_pfr1to5", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    # <<<< END AUTO-GENERATED: f99915db 问题帧率三档 >>>>

    # <<<< AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean_noproblem >>>>
    # ddl_job_4715dbbe_nooutlier_dedup_clean (17集) 的剔 problem ep 版:
    # problem = pfr>15% (问题帧率/持续抖动) OR 任一 max>p95 (单点突变) — 两维并集, 共 3357 ep (29.8%).
    # 基于帧级 vel/acc/jerk 分析 (bar/frame_level_anomaly.py + max_pfr_stats.py + apply_thresholds.py).
    # 派生后 11265→7908 ep (保留 70.2%), 4.56M→3.17M 帧 (保留 69.6%). 高噪集保留少(90e6a697 30%/fac8e100 33%),
    # 低噪集保留多(f26f9106 97%/6b874f60 95%). 派生 bar/build_noproblem.py. problem_eps: threshold_apply_out/problem_eps.json.
    "ddl_job_4715dbbe_nooutlier_dedup_clean_noproblem": [
        ("20260719185726_lerobot_0f437b52_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260719194252_lerobot_6b874f60_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721040252_lerobot_f99915db_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721055517_lerobot_f26f9106_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061238_lerobot_90e6a697_nooutlier_clean_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721062108_lerobot_2c7955ce_nooutlier_clean_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721092621_lerobot_9a54a112_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100220_lerobot_d37ab51a_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102016_lerobot_f9c8ce40_nooutlier_clean_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102040_lerobot_a939fffc_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723062337_lerobot_3666d289_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095346_lerobot_b416dbe0_nooutlier_clean_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095529_lerobot_a3e2fb6b_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095618_lerobot_dfd0a83d_nooutlier_clean_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095815_lerobot_1196f165_nooutlier_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100217_lerobot_fac8e100_nooutlier_clean_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100416_lerobot_b5e9b282_nooutlier_clean_noproblem", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    # <<<< END AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean_noproblem >>>>

    # <<<< AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean_noproblem_smooth >>>>
    # _noproblem 的轨迹平滑版: 中值滤波 state — pos(3)/quat(4) 窗5(quat归一化) + gripper 窗7(拉大)。
    # 只改 state 数值列(不剔ep/不改结构), 视频硬链, 按 ep 分组平滑避免跨ep。删 stats 重算。
    # 用户: 位移/旋转窗5够(问题不大), 夹爪拉大到窗7。平滑对象=合格数据(_noproblem, 剔problem后)。
    # 派生 bar/build_smooth.py。规模同 _noproblem (7908 ep / 3.17M 帧, 结构不变)。
    "ddl_job_4715dbbe_nooutlier_dedup_clean_noproblem_smooth": [
        ("20260719185726_lerobot_0f437b52_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260719194252_lerobot_6b874f60_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721040252_lerobot_f99915db_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721055517_lerobot_f26f9106_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061238_lerobot_90e6a697_nooutlier_clean_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721062108_lerobot_2c7955ce_nooutlier_clean_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721092621_lerobot_9a54a112_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100220_lerobot_d37ab51a_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102016_lerobot_f9c8ce40_nooutlier_clean_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721102040_lerobot_a939fffc_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723062337_lerobot_3666d289_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095346_lerobot_b416dbe0_nooutlier_clean_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095529_lerobot_a3e2fb6b_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095618_lerobot_dfd0a83d_nooutlier_clean_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095815_lerobot_1196f165_nooutlier_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100217_lerobot_fac8e100_nooutlier_clean_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100416_lerobot_b5e9b282_nooutlier_clean_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    # <<<< END AUTO-GENERATED: ddl_job_4715dbbe_nooutlier_dedup_clean_noproblem_smooth >>>>

    # <<<< AUTO-GENERATED: umi_all_nohjerk_noproblem_smooth >>>>
    # 全量汇总: data_root_dir 下所有 *nohjerk_noproblem_smooth 目录 (85集, 26340 ep / 10.78M 帧)。
    # = nooutlier(+clean) + 剔高jerk离群ep + 剔problem ep(pfr>15% OR 单点突变) + 轨迹平滑(中值滤波)。
    # 与 ddl_job_4715dbbe_nooutlier_dedup_clean_noproblem_smooth (17集, dedup 后) 的区别:
    # 覆盖全部 session (未做 dedup 跨集去重), 无 episode_speed.json → speed token 自动为空。
    "umi_all_nohjerk_noproblem_smooth": [
        ("20260721034429_lerobot_072f59cd_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721034609_lerobot_2722737f_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721035353_lerobot_0d7c2558_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721040111_lerobot_80c27843_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721055231_lerobot_27cc19ce_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721060740_lerobot_6a8534a7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061250_lerobot_c035f83f_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721061853_lerobot_b3e78a8d_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721093341_lerobot_2950fd08_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100144_lerobot_04ba8652_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100717_lerobot_85a8014d_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721100845_lerobot_7d0b4c83_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260721101929_lerobot_07a359d7_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260722091104_lerobot_6491df37_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723060139_lerobot_0194e4fa_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723060826_lerobot_a1b443be_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723061008_lerobot_e7d97679_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723061525_lerobot_bd0cf0ac_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723062302_lerobot_0f2432e0_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723074814_lerobot_79c234ef_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723075517_lerobot_048a0e72_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723083613_lerobot_9fed3b59_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723084145_lerobot_f7248328_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723085927_lerobot_d10f16b8_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723090122_lerobot_791893e7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723090253_lerobot_c93ab935_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723092457_lerobot_5cc321d3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260723093040_lerobot_968799a2_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095457_lerobot_ee041cdf_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724095552_lerobot_bc37a961_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100509_lerobot_197d0c5f_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100525_lerobot_b09e78b5_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100644_lerobot_7dac99a3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724100644_lerobot_fd9fa6bd_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260724101506_lerobot_b4d97096_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260727060905_lerobot_d5825884_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260727060951_lerobot_22517704_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260727061729_lerobot_ae6995d5_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260727062644_lerobot_58699c00_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260727062830_lerobot_e14d35d7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260727062926_lerobot_56d6af55_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260727063238_lerobot_ff9da23a_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728055114_lerobot_1ea4c859_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728055712_lerobot_a8d68f9d_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728060525_lerobot_991e0f1b_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728060602_lerobot_6e3ed8a3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728060818_lerobot_c3e4cb07_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728061013_lerobot_4fb34a59_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728061812_lerobot_d6e57e7e_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260728093737_lerobot_4392dd8a_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731025107_lerobot_6db76510_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731025823_lerobot_b11fd3a9_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731030536_lerobot_41fb2a92_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731031248_lerobot_8360c7e1_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731031942_lerobot_1207e6b9_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731032208_lerobot_8baae60d_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731032215_lerobot_6f146ede_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731032708_lerobot_edb79113_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731032829_lerobot_8be7ff1e_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731033225_lerobot_871e3702_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731033332_lerobot_dffb12f7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731033752_lerobot_7a0bb0e7_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731093809_lerobot_42d30c38_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731094710_lerobot_e08fc34a_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260731100518_lerobot_149caf82_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803021314_lerobot_8b2043e1_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803021925_lerobot_84e13a29_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803022615_lerobot_25e5299a_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803023514_lerobot_b3fc239f_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803032618_lerobot_b1993aa3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803033352_lerobot_b1e93a34_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803034251_lerobot_b621497c_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803034634_lerobot_2c5ac9d4_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803091308_lerobot_aa5d309c_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803091803_lerobot_6ee5e835_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803092111_lerobot_d30f19d6_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803092621_lerobot_53c862e1_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803092837_lerobot_5ff764fc_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803093156_lerobot_5bd31acc_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803093901_lerobot_154e1ec8_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260803094221_lerobot_bddd2861_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260810092238_lerobot_40b818bf_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260810094053_lerobot_465bef4c_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260810094158_lerobot_35c2d9e0_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
        ("20260810094623_lerobot_55acd0c6_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", ""),
    ],
    # <<<< END AUTO-GENERATED: umi_all_nohjerk_noproblem_smooth >>>>

    # <<<< AUTO-GENERATED: umi_all_nohjerk_noproblem_smooth_oss >>>>
    # 同 umi_all_nohjerk_noproblem_smooth (85集, 26340 ep / 10.78M 帧), 但带阿里云 OSS 目录 URI
    # (oss://algo-public0/orion/datasets/z-umi/<id>/, 由本机 z-umi 数据镜像上传),
    # 供 OSS 环境直接拉取; 目录 URI (尾 /) 递归同步, 幂等 (同大小跳过/补缺)。
    "umi_all_nohjerk_noproblem_smooth_oss": [
        ("20260721034429_lerobot_072f59cd_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721034429_lerobot_072f59cd_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721034609_lerobot_2722737f_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721034609_lerobot_2722737f_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721035353_lerobot_0d7c2558_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721035353_lerobot_0d7c2558_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721040111_lerobot_80c27843_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721040111_lerobot_80c27843_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260721055231_lerobot_27cc19ce_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721055231_lerobot_27cc19ce_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721060740_lerobot_6a8534a7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721060740_lerobot_6a8534a7_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260721061250_lerobot_c035f83f_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721061250_lerobot_c035f83f_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721061853_lerobot_b3e78a8d_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721061853_lerobot_b3e78a8d_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721093341_lerobot_2950fd08_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721093341_lerobot_2950fd08_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721100144_lerobot_04ba8652_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721100144_lerobot_04ba8652_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721100717_lerobot_85a8014d_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721100717_lerobot_85a8014d_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721100845_lerobot_7d0b4c83_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721100845_lerobot_7d0b4c83_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260721101929_lerobot_07a359d7_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260721101929_lerobot_07a359d7_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260722091104_lerobot_6491df37_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260722091104_lerobot_6491df37_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723060139_lerobot_0194e4fa_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723060139_lerobot_0194e4fa_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260723060826_lerobot_a1b443be_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723060826_lerobot_a1b443be_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723061008_lerobot_e7d97679_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723061008_lerobot_e7d97679_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723061525_lerobot_bd0cf0ac_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723061525_lerobot_bd0cf0ac_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723062302_lerobot_0f2432e0_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723062302_lerobot_0f2432e0_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723074814_lerobot_79c234ef_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723074814_lerobot_79c234ef_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260723075517_lerobot_048a0e72_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723075517_lerobot_048a0e72_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260723083613_lerobot_9fed3b59_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723083613_lerobot_9fed3b59_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260723084145_lerobot_f7248328_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723084145_lerobot_f7248328_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723085927_lerobot_d10f16b8_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723085927_lerobot_d10f16b8_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260723090122_lerobot_791893e7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723090122_lerobot_791893e7_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260723090253_lerobot_c93ab935_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723090253_lerobot_c93ab935_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723092457_lerobot_5cc321d3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723092457_lerobot_5cc321d3_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260723093040_lerobot_968799a2_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260723093040_lerobot_968799a2_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260724095457_lerobot_ee041cdf_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260724095457_lerobot_ee041cdf_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260724095552_lerobot_bc37a961_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260724095552_lerobot_bc37a961_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260724100509_lerobot_197d0c5f_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260724100509_lerobot_197d0c5f_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260724100525_lerobot_b09e78b5_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260724100525_lerobot_b09e78b5_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260724100644_lerobot_7dac99a3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260724100644_lerobot_7dac99a3_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260724100644_lerobot_fd9fa6bd_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260724100644_lerobot_fd9fa6bd_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260724101506_lerobot_b4d97096_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260724101506_lerobot_b4d97096_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260727060905_lerobot_d5825884_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260727060905_lerobot_d5825884_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260727060951_lerobot_22517704_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260727060951_lerobot_22517704_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260727061729_lerobot_ae6995d5_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260727061729_lerobot_ae6995d5_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260727062644_lerobot_58699c00_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260727062644_lerobot_58699c00_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260727062830_lerobot_e14d35d7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260727062830_lerobot_e14d35d7_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260727062926_lerobot_56d6af55_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260727062926_lerobot_56d6af55_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260727063238_lerobot_ff9da23a_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260727063238_lerobot_ff9da23a_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728055114_lerobot_1ea4c859_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728055114_lerobot_1ea4c859_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728055712_lerobot_a8d68f9d_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728055712_lerobot_a8d68f9d_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728060525_lerobot_991e0f1b_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728060525_lerobot_991e0f1b_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728060602_lerobot_6e3ed8a3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728060602_lerobot_6e3ed8a3_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728060818_lerobot_c3e4cb07_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728060818_lerobot_c3e4cb07_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728061013_lerobot_4fb34a59_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728061013_lerobot_4fb34a59_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728061812_lerobot_d6e57e7e_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728061812_lerobot_d6e57e7e_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260728093737_lerobot_4392dd8a_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260728093737_lerobot_4392dd8a_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731025107_lerobot_6db76510_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731025107_lerobot_6db76510_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731025823_lerobot_b11fd3a9_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731025823_lerobot_b11fd3a9_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260731030536_lerobot_41fb2a92_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731030536_lerobot_41fb2a92_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260731031248_lerobot_8360c7e1_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731031248_lerobot_8360c7e1_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731031942_lerobot_1207e6b9_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731031942_lerobot_1207e6b9_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731032208_lerobot_8baae60d_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731032208_lerobot_8baae60d_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260731032215_lerobot_6f146ede_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731032215_lerobot_6f146ede_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731032708_lerobot_edb79113_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731032708_lerobot_edb79113_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731032829_lerobot_8be7ff1e_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731032829_lerobot_8be7ff1e_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260731033225_lerobot_871e3702_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731033225_lerobot_871e3702_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731033332_lerobot_dffb12f7_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731033332_lerobot_dffb12f7_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260731033752_lerobot_7a0bb0e7_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731033752_lerobot_7a0bb0e7_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731093809_lerobot_42d30c38_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731093809_lerobot_42d30c38_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260731094710_lerobot_e08fc34a_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731094710_lerobot_e08fc34a_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260731100518_lerobot_149caf82_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260731100518_lerobot_149caf82_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803021314_lerobot_8b2043e1_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803021314_lerobot_8b2043e1_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260803021925_lerobot_84e13a29_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803021925_lerobot_84e13a29_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260803022615_lerobot_25e5299a_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803022615_lerobot_25e5299a_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260803023514_lerobot_b3fc239f_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803023514_lerobot_b3fc239f_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803032618_lerobot_b1993aa3_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803032618_lerobot_b1993aa3_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803033352_lerobot_b1e93a34_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803033352_lerobot_b1e93a34_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260803034251_lerobot_b621497c_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803034251_lerobot_b621497c_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803034634_lerobot_2c5ac9d4_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803034634_lerobot_2c5ac9d4_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803091308_lerobot_aa5d309c_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803091308_lerobot_aa5d309c_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803091803_lerobot_6ee5e835_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803091803_lerobot_6ee5e835_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803092111_lerobot_d30f19d6_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803092111_lerobot_d30f19d6_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260803092621_lerobot_53c862e1_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803092621_lerobot_53c862e1_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803092837_lerobot_5ff764fc_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803092837_lerobot_5ff764fc_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803093156_lerobot_5bd31acc_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803093156_lerobot_5bd31acc_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260803093901_lerobot_154e1ec8_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803093901_lerobot_154e1ec8_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260803094221_lerobot_bddd2861_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260803094221_lerobot_bddd2861_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260810092238_lerobot_40b818bf_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260810092238_lerobot_40b818bf_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260810094053_lerobot_465bef4c_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260810094053_lerobot_465bef4c_nooutlier_clean_nohjerk_noproblem_smooth/"),
        ("20260810094158_lerobot_35c2d9e0_nooutlier_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260810094158_lerobot_35c2d9e0_nooutlier_nohjerk_noproblem_smooth/"),
        ("20260810094623_lerobot_55acd0c6_nooutlier_clean_nohjerk_noproblem_smooth", 1.0, "umi_2gripper_relative_6d", "oss://algo-public0/orion/datasets/z-umi/20260810094623_lerobot_55acd0c6_nooutlier_clean_nohjerk_noproblem_smooth/"),
    ],
    # <<<< END AUTO-GENERATED: umi_all_nohjerk_noproblem_smooth_oss >>>>
}
