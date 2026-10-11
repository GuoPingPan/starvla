# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    DATASET_NAMED_MIXTURES,
    EmbodimentTag,
)

logger = logging.getLogger(__name__)


def _collate_tensors_enabled() -> bool:
    """图像张量化开关: env STARVLA_COLLATE_TENSORS 优先, 未设时读 yaml 注入的
    全局标记 (build_dataloader 从 datasets.vla_data.collate_image_tensors 读出后
    写入本模块级变量 —— collate_fn 在 worker 进程执行, 拿不到 cfg 对象)。"""
    env = os.environ.get("STARVLA_COLLATE_TENSORS")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(_YAML_COLLATE_FLAG[0])


_YAML_COLLATE_FLAG = [False]


def set_collate_flag_from_cfg(vla_dataset_cfg) -> None:
    """把 yaml 的 collate_image_tensors 写进 env (worker 进程继承) + 模块标记。

    env STARVLA_COLLATE_TENSORS 已显式设置时以 env 为准 (不覆盖)。
    """
    yaml_on = bool(vla_dataset_cfg.get("collate_image_tensors", False)) if hasattr(vla_dataset_cfg, "get") else False
    if "STARVLA_COLLATE_TENSORS" not in os.environ and yaml_on:
        os.environ["STARVLA_COLLATE_TENSORS"] = "1"
    _YAML_COLLATE_FLAG[0] = yaml_on


def _pil_to_chw_float(image) -> torch.Tensor:
    """PIL → (3,H,W) float32 [-1,1] (与 PI0._pil_to_tensor 相同的归一化)。

    注意: 这里假定图像已经是模型目标分辨率 (transform 链末端 resize 过);
    与主进程 _pil_to_tensor 的 resize_with_pad 不完全等价 (后者还会保比例 pad),
    因此本函数只在 collate_image_tensors 开启且 transform 已保证 224×224 时使用。
    """
    arr = np.asarray(image.convert("RGB")).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
    return t / 255.0 * 2.0 - 1.0


def collate_fn(batch):
    """恒等 collate (样本 list 直通主进程)。

    可选加速 (STARVLA_COLLATE_TENSORS=1): 在 **worker 进程内** 把 PIL 图像转成
    CHW float [-1,1] 张量 —— 等价于把 PI0._prepare_examples 的逐样本 CPU 预处理
    下沉到 dataloader worker 并行执行。主进程 _prepare_examples 检测到张量输入后
    跳过 PIL 路径 (见 PI0._prepare_examples)。

    实测动机: 单线程主进程的 PIL→tensor 在 8 个解码 worker 的 CPU 抢占下会被
    拉慢 10-100× (aten::stack 5ms→59ms, 整段 _prepare_examples 0.3→4.4s),
    GPU util 掉到 13%。前提: transform 链末端已 resize 到模型目标分辨率
    (UMI/LIBERO 的 DataConfig 均满足), 否则保持 PIL 让主进程 resize。
    """
    if not _collate_tensors_enabled():
        return batch
    from PIL import Image

    for sample in batch:
        img = sample.get("image")
        if img is None:
            continue
        imgs = list(img) if isinstance(img, (list, tuple)) else [img]
        if all(isinstance(im, Image.Image) for im in imgs):
            sample["image"] = [_pil_to_chw_float(im) for im in imgs]
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    split_mode: str | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        print(f"Warning: DataConfig for robot_type={robot_type!r} has no embodiment_tag, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    # 解码器参数透传 (如 decord 多线程爆炸: A800 实测 64 worker × 默认 auto 线程 →
    # ~1000 线程, step 7.19s; num_threads=1 后 3.35s, GPU util 恢复 ~70%)。
    # yaml 用法: datasets.vla_data.video_backend_kwargs: {num_threads: 1}
    video_backend_kwargs = None
    if data_cfg is not None:
        raw_kwargs = data_cfg.get("video_backend_kwargs", None)
        if raw_kwargs:
            video_backend_kwargs = dict(raw_kwargs) if not isinstance(raw_kwargs, dict) else dict(raw_kwargs)

    # Opt-in factory hook: a DataConfig may define ``make_dataset(dataset_name=..., **ds_kwargs)``
    # to swap in a custom dataset class (e.g. with per-task filtering / chunk stride).
    # When absent, fall through to the default LeRobotSingleDataset construction below.
    if hasattr(data_config, "make_dataset"):
        return data_config.make_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            dataset_name=data_name,
            split_mode=split_mode,
        )

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        video_backend_kwargs=video_backend_kwargs,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        split_mode=split_mode,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"[dataloader] Using mixture '{data_mix}': {[(d, w, r) for d, w, r, *_ in mixture_spec]}")
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type, *_ in mixture_spec:  # *_ : 容忍可选的第4元素(如 BOS 路径), 训练忽略
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg, split_mode=mode), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/simBenchmarks/LIBERO/train_files/bar/starvla_cotrain_libero.yaml", help="Path to YAML config")
    parser.add_argument("--data_mix", type=str, default=None, help="Override data_mix from config")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override data_root_dir from config")
    args = parser.parse_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_root_dir = Path(vla_dataset_cfg.data_root_dir)
    if args.data_mix is not None:
        vla_dataset_cfg.data_mix = args.data_mix
    if args.data_root_dir is not None:
        vla_dataset_cfg.data_root_dir = Path(args.data_root_dir)

    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 3:
            break
        count += 1
        pass