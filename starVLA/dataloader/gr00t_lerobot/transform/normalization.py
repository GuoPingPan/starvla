"""
统计量计算函数 - 通过 dataloader 采样计算。

核心思想：
- 复用现有的 transform pipeline（不含 NormalizationTransform）
- 对 transform 后的数据计算统计量
- 不关心数据是如何转换的（action_mode、windowing 等）

使用方式：
    from starVLA.dataloader.gr00t_lerobot.transform.stats_computer import compute_stats_from_dataset

    # 创建 dataset（使用 stats_transforms，不含 NormalizationTransform）
    dataset = MyDataset(transforms=stats_transforms)

    # 计算统计量
    statistics = compute_stats_from_dataset(
        dataset=dataset,
        keys=["state.eef_position", "action.eef_position"],
    )

    # 保存统计量
    save_statistics(statistics, "meta/stats_gr00t.json")
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


def compute_stats_from_dataset(
    dataset,
    keys: List[str],
    show_progress: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: int = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    从经过 transform 的数据计算统计量（使用 DataLoader 多进程加载）。

    核心思想：
    - dataset 应该使用 stats_transforms（不含 NormalizationTransform）
    - 使用 DataLoader 多进程批量加载，提高效率
    - 数据已经经过 action_mode、windowing、chunking 等处理

    Args:
        dataset: 已经包含 transform pipeline 的 dataset
                 注意：应该使用 stats_transforms（不含 NormalizationTransform）
        keys: 需要计算统计量的 key 列表，如 ["state.eef_position", "action.eef_position"]
        show_progress: 是否显示进度条
        batch_size: DataLoader batch size
        num_workers: DataLoader worker 数量
        max_samples: 最大样本数（用于快速测试，None 表示全部）

    Returns:
        统计量字典，格式为:
        {
            "state.eef_position": {"min": [...], "max": [...], "mean": [...], "std": [...], "q01": [...], "q99": [...]},
            "action.eef_position": {"min": [...], "max": [...], "mean": [...], "std": [...], "q01": [...], "q99": [...]},
        }
    """
    from torch.utils.data import DataLoader

    statistics = {}
    # 创建数据累加器
    accumulators = {key: [] for key in keys}

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=lambda x: x,  # 不 collate，保持原始格式
    )

    # 遍历 dataloader，收集数据
    sample_count = 0
    for batch in tqdm(dataloader, desc="Computing statistics") if show_progress else dataloader:
        if max_samples and sample_count >= max_samples:
            break

        for data in batch:
            sample_count += 1
            if max_samples and sample_count > max_samples:
                break

            # 收集每个 key 的数据
            for key in keys:
                if key not in data:
                    continue

                value = data[key]
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                elif isinstance(value, list):
                    value = np.array(value)

                # 确保是 numpy 数组
                if isinstance(value, np.ndarray):
                    # 处理不同维度的数据
                    if value.ndim == 2:
                        accumulators[key].append(value.reshape(-1))
                    elif value.ndim == 1:
                        accumulators[key].append(value)
                    else:
                        accumulators[key].append(value.reshape(-1))

    # 计算最终统计量
    for key in keys:
        if not accumulators[key]:
            print(f"Warning: No data collected for key {key}")
            continue

        # 合并所有数据
        all_data = np.concatenate(accumulators[key], axis=0)

        # 计算统计量
        statistics[key] = {
            "min": np.min(all_data, axis=0).tolist(),
            "max": np.max(all_data, axis=0).tolist(),
            "mean": np.mean(all_data, axis=0).tolist(),
            "std": np.std(all_data, axis=0).tolist(),
        }

        # 计算分位数
        if len(all_data) > 100:
            statistics[key]["q01"] = np.quantile(all_data, 0.01, axis=0).tolist()
            statistics[key]["q99"] = np.quantile(all_data, 0.99, axis=0).tolist()
        else:
            statistics[key]["q01"] = statistics[key]["min"]
            statistics[key]["q99"] = statistics[key]["max"]

    return statistics


def save_statistics(
    statistics: Dict[str, Dict[str, List[float]]],
    path: Union[str, Path],
    cache_config: Optional[Dict] = None,
) -> None:
    """
    保存统计量到 JSON 文件。

    Args:
        statistics: 统计量字典
        path: 保存路径
        cache_config: 可选的缓存配置（用于验证缓存是否有效）
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 如果提供了 cache_config，使用与现有格式兼容的格式
    if cache_config is not None:
        payload = {
            "__format_version": 2,
            "__cache_config": cache_config,
            "statistics": statistics,
        }
    else:
        payload = statistics

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_statistics(path: Union[str, Path]) -> Dict[str, Dict[str, List[float]]]:
    """
    从 JSON 文件加载统计量。

    Args:
        path: 文件路径

    Returns:
        统计量字典
    """
    import json

    with open(path, "r") as f:
        data = json.load(f)

    # 处理两种格式
    if "__format_version" in data:
        # 新格式（包含 cache_config）
        return data["statistics"]
    else:
        # 旧格式（直接是 statistics）
        return data
