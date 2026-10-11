import json
import os
import re
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader, Sampler
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


class SkipFirstEpochSampler(Sampler):
    """resume 数据流对齐: 首个 epoch 从 skip_items 起跳, 之后回全长。

    背景: mixture 采样是 (epoch, index, seed) 的确定函数, 但流位置 (index) 由
    sampler 决定。is_resume 时按 checkpoint 步数折算出已消费的全局 item 数,
    第一个 epoch 直接从该偏移开始 —— 与连续训练看到完全相同的样本序列。
    skip 只在第一次 __iter__ 生效 (一次性消费), epoch 翻滚后回到全长,
    与连续训练的下一 epoch 行为一致。
    """

    def __init__(self, dataset, skip_items: int):
        self.dataset = dataset
        self.skip_items = max(0, int(skip_items))

    def __iter__(self):
        start = self.skip_items
        self.skip_items = 0
        return iter(range(start, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


def _scan_latest_train_step(checkpoint_dir) -> int:
    """扫描 steps_N 权重文件 + steps_N_ds/ 全量状态目录, 返回最大 N。

    与 train_starvla 的 resume 发现逻辑同命名约定 (权重兜底 + 全量状态取较新),
    轻微重复是刻意的 —— dataloader 构造先于 trainer 存在, 不能复用其实例方法。
    """
    if not os.path.isdir(checkpoint_dir):
        return 0
    best = 0
    for name in os.listdir(checkpoint_dir):
        m = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", name)
        d = re.match(r"steps_(\d+)_ds$", name)
        if m and os.path.isfile(os.path.join(checkpoint_dir, name)):
            best = max(best, int(m.group(1)))
        elif d and os.path.isdir(os.path.join(checkpoint_dir, name)):
            best = max(best, int(d.group(1)))
    return best

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe", mode="train"): # TODO now here only is get dataset, we need mv dataloader to here
    if os.environ.get("STARVLA_SYNTHETIC", "0") in ("1", "true", "yes") or dataset_py == "synthetic":
        from starVLA.dataloader.synthetic import make_synthetic_dataloader

        return make_synthetic_dataloader(cfg, mode=mode)

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn, set_collate_flag_from_cfg
        vla_dataset_cfg = cfg.datasets.vla_data

        # 图像张量化下沉 worker 的开关 (yaml datasets.vla_data.collate_image_tensors)。
        # 写入环境变量: DataLoader worker (fork/spawn) 都能继承, collate_fn 内读取。
        set_collate_flag_from_cfg(vla_dataset_cfg)

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            mode=mode,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )
        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        # 固定 DataLoader generator: worker base_seed 变成 (seed) 的确定函数而非
        # 全局 CPU RNG 派生 —— 流内容与 RNG 历史解耦, resume 后完全可复现。
        _seed = 42
        try:
            _seed = int(cfg.seed)
        except Exception:
            pass
        _gen = torch.Generator()
        _gen.manual_seed(_seed + (0 if mode == "train" else 10_000))
        dataloader_kwargs["generator"] = _gen

        if mode == "train":
            _is_resume = False
            try:
                _is_resume = bool(cfg.trainer.is_resume)
            except Exception:
                pass
            if _is_resume and os.environ.get("STARVLA_RESUME_SKIP", "1").lower() not in ("0", "false", "no", "off"):
                _B = int(vla_dataset_cfg.get("per_device_batch_size", 1))
                _W = dist.get_world_size() if dist.is_initialized() else 1
                _N = _scan_latest_train_step(Path(cfg.output_dir) / "checkpoints")
                if _N > 0:
                    _L = len(vla_dataset)
                    _consumed = _N * _W * _B
                    _epochs_done, _skip = divmod(_consumed, _L)
                    if hasattr(vla_dataset, "set_epoch"):
                        vla_dataset.set_epoch(_epochs_done)
                    dataloader_kwargs["sampler"] = SkipFirstEpochSampler(vla_dataset, _skip)
                    logger.info(
                        f"resume 数据流对齐: ckpt_steps={_N} world={_W} batch={_B} → "
                        f"epoch={_epochs_done}, 首轮跳过 {_skip}/{_L} items"
                    )

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        # 只在训练侧固化 dataset_statistics.json: 评测侧 mixture 走同一份缓存统计,
        # 重算/覆盖无意义且可能与训练侧竞争。split 划分在 stats 计算之后, 两边数值一致。
        if mode == "train" and (not dist.is_initialized() or dist.get_rank() == 0):
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
