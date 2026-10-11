"""StarVLA 多卡开环评测入口 — 复刻 ``train_starvla.py`` 的结构。

════════════════════════════════════════════════════════════════════════
为什么这样写
════════════════════════════════════════════════════════════════════════

跟 ``train_starvla.py`` 1:1 对称: 复用 ``TrainerUtils`` / ``baseframework`` /
``make_LeRobotSingleDataset`` / ``collate_fn``, 把训练循环改成 inference-only,
多卡能力作为副产品自动获得 (Accelerate + DistributedSampler + gather_object)。

注: 没直接 import ``train_starvla.setup_accelerator`` — 它带 DeepSpeed plugin,
eval 无 backward 时会触发 "train_micro_batch_size_per_gpu" 缺失错误。Eval 走本地
``setup_eval_accelerator`` 用裸 Accelerator, 同样支持多卡 (DDP)。

跟旧单卡入口 ``examples/Z-umi/eval_files/eval_open_loop.py`` 行为对齐:
  - 数据通路完全一致 (同 dataset 类, 同 collate_fn, 同 transforms)
  - per-anchor 同时存 normalized model 输出 + raw 物理量
  - schema v4 不变, Stage 2 (stage2_visualize.py) 无需修改

新增能力:
  - 多卡: ``accelerate launch --multi-gpu --num_processes N -m starVLA.training.eval_starvla ...``
  - 多 episode: ``--episodes 0 1 2`` (值是 **eval 列表的 0 基位置索引**, 非原始 episode_id;
    内部自动映射到真实 episode_id) 或 YAML ``eval.episodes: [0, 1, 2]``

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════

  # 单卡
  python -m starVLA.training.eval_starvla \\
    --run-dir /nix/data/umi_qwenpi_v3_A800_overfit_ep0 \\
    --eval-yaml examples/Z-umi/eval_files/eval_umi_overfit_overlay.yaml

  # 多卡 (每个 rank 处理 episode 内不同 anchor, gather_object 回 rank 0)
  accelerate launch --multi-gpu --num_processes 4 -m starVLA.training.eval_starvla \\
    --run-dir /nix/data/... \\
    --eval-yaml examples/Z-umi/eval_files/eval_umi_overfit_overlay.yaml

  # 多 episode
  python -m starVLA.training.eval_starvla ... --episodes 0 1 2

产物 (每个 episode 一个子目录, schema v4 跟 io.py:21 一致):
  <output_root>/<run_name>_step<N>_seed<S>/
    episode_<id>/
      predictions.npz
      metrics.json
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ── 复用 train_starvla 的 setup_accelerator (DeepSpeed plugin 在纯推理下会强
#    要 batch_size 属性, 见 prepare_eval 注释) — eval 走本地 setup_eval_accelerator。
from starVLA.model.framework.base_framework import baseframework
from starVLA.dataloader.lerobot_datasets import collate_fn, make_LeRobotSingleDataset
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.eval.core.io import OBS_IMAGES_FILENAME, write_metrics, write_obs_images, write_predictions
from starVLA.eval.core.transforms import align_quaternion_sign_slice
from starVLA.eval.core.checkpoint_utils import (
    checkpoint_path,
    discover_checkpoint_steps,
    pick_mixture_entry,
    to_numpy,
    torch_dtype,
)


# ─────────────────────────── 分布式 gather ─────────────────────────────


def _gather_object_list(local_list: list, accelerator: Accelerator) -> list:
    """跨 rank all-gather Python object list, 拼成一份给所有 rank。

    Accelerate 1.5 没有 ``gather_object`` API (新版才有)。这里直接走
    ``torch.distributed.all_gather_object`` — 各 rank 把自己的 list 上交,
    返回时所有 rank 都拿到 ``[rank0_list, rank1_list, ...]``, flatten 即可。
    单进程模式直接返回 local_list。
    """
    if not accelerator.use_distributed:
        return list(local_list)
    gathered: list[list] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, list(local_list))
    flat: list = []
    for sublist in gathered:
        if sublist:
            flat.extend(sublist)
    return flat


def _resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    """PIL 按长边缩到 ``long_side`` (保比例), 返回 HWC uint8。已更小则原样返回 (不放大)。"""
    h, w = int(frame.shape[0]), int(frame.shape[1])
    scale = long_side / float(max(h, w))
    if scale < 1.0:
        from PIL import Image

        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        frame = np.asarray(Image.fromarray(frame).resize((nw, nh)))
    return frame


# ─────────────────────────── accelerator ──────────────────────────────


def setup_eval_accelerator() -> Accelerator:
    """Eval-only Accelerator — 不带 DeepSpeed plugin。

    跟 ``train_starvla.setup_accelerator`` 的区别: 训练用 DeepSpeed 来 ZeRO 分片
    optimizer state + 大模型, 但 eval 没 backward / 没优化器, DeepSpeed 反而会
    强制要求 dataloader 有 ``batch_size`` 属性 (accelerate 1.5 在 prepare 单 model
    时报 ``train_micro_batch_size_per_gpu`` 缺失)。Eval 用裸 Accelerator 走 DDP
    即可满足多卡需求。
    """
    accelerator = Accelerator()
    accelerator.print(accelerator.state)
    return accelerator


# ─────────────────────────── config 加载 ──────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """优先级: CLI > eval_yaml > 代码默认。所有参数都可选。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="训练 run 目录 (含 config.yaml + checkpoints/)。")
    parser.add_argument("--eval-yaml", type=Path, required=True,
                        help="Eval overlay YAML, 跟 ckpt config 合并。")
    parser.add_argument("--data-root-override", type=Path, default=None,
                        help="覆盖 config 里的 data_root_dir (训练机和评测机路径通常不同)。")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="从 mixture 里挑哪个 dataset。mixture >1 时必须指定。")
    parser.add_argument("--data-mix", type=str, default=None,
                        help="覆盖 ckpt config 里的 datasets.vla_data.data_mix (换数据集评测)。"
                             "可用 mixture 见 data_registry/data_config.py 的 DATASET_NAMED_MIXTURES。"
                             "注意: model.norm_stats 仍来自 ckpt 固化的 dataset_statistics.json, "
                             "换不同分布数据集后开环数字不可定量比较, 仅用于 pipeline 验证/可视化。")
    parser.add_argument("--checkpoint-step", type=int, default=None,
                        help="单 ckpt step 号。null = 用 eval.yaml 或 auto-discover 最新。")
    parser.add_argument("--episode-id", type=int, default=None,
                        help="单 episode, 按 eval 列表的 0 基位置索引 (旧字段, 等价 --episodes N)。"
                             "例 --episode-id 3 = eval 列表第 4 个; 不指定 = 全量。")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="多 episode, 按 eval 列表的 0 基位置索引 (非原始 episode_id)。"
                             "例 --episodes 0 1 2 3 = eval 列表前 4 个; 不指定 = 全量 eval。"
                             "startup 会打印「位置→真实 episode_id」映射。")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="DataLoader batch_size。null = eval.yaml 默认 (4)。")
    parser.add_argument("--max-anchors", type=int, default=None,
                        help="smoke test 用, 限制每个 episode 的 anchor 数。")
    parser.add_argument("--seed", type=int, default=None,
                        help="单 seed。null = eval.yaml (42)。")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="覆盖自动派生的 output_dir。")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def load_merged_config(
    run_dir: Path,
    eval_yaml: Path,
    data_root_override: Path | None,
    dataset_name_override: str | None,
    data_mix_override: str | None = None,
) -> tuple[Any, Path]:
    """读 ckpt config.yaml + eval overlay YAML, OmegaConf.merge 合并。

    OmegaConf.merge 语义: 后者覆盖前者。eval overlay 里写的字段会覆盖 ckpt config
    的同名字段。eval 不重写训练参数, 只覆盖必要项。

    Returns:
        (merged_cfg, ckpt_config_path)
    """
    ckpt_config_path = run_dir / "config.full.yaml"
    if not ckpt_config_path.exists():
        ckpt_config_path = run_dir / "config.yaml"
    if not ckpt_config_path.exists():
        raise FileNotFoundError(f"No saved config found in {run_dir}")

    ckpt_cfg = OmegaConf.load(ckpt_config_path)
    eval_cfg = OmegaConf.load(eval_yaml) if eval_yaml.exists() else OmegaConf.create({"eval": {}})
    merged = OmegaConf.merge(ckpt_cfg, eval_cfg)

    if data_root_override is not None:
        merged.datasets.vla_data.data_root_dir = str(data_root_override)
    if data_mix_override is not None:
        merged.datasets.vla_data.data_mix = data_mix_override
    data_root = Path(merged.datasets.vla_data.data_root_dir)
    if not data_root.exists():
        raise FileNotFoundError(
            f"data_root_dir does not exist: {data_root}. "
            f"Pass --data-root-override to point at the local data mirror."
        )

    if dataset_name_override is not None:
        merged.eval.dataset_name = dataset_name_override

    return merged, ckpt_config_path


def build_dataset(cfg) -> Any:
    """跟训练走同一条 dataset 构造路径: ``make_LeRobotSingleDataset``。

    返回的 dataset 类型由 robot_type 决定 (umi_2gripper / umi_2gripper_test 等),
    本函数不关心差异, 走 polymorphism。
    """
    data_cfg = cfg.datasets.vla_data
    dataset_name, robot_type = pick_mixture_entry(
        str(data_cfg.data_mix),
        cfg.eval.get("dataset_name"),
    )
    return make_LeRobotSingleDataset(
        Path(data_cfg.data_root_dir),
        dataset_name,
        robot_type,
        data_cfg=data_cfg,
        split_mode="eval",
    )


# ───────────────────────── VLAEvaluator ──────────────────────────────


class _IndexedSubset(torch.utils.data.Dataset):
    """Wrap a base dataset + a list of global indices, 注入 ``traj_id`` / ``anchor`` 到 sample。

    必要性: ``LeRobotSingleDataset._pack_sample`` (datasets.py:1925) 只返回
    ``{action, image, lang, robot_tag, state?}``, 不带索引字段。单卡下可以靠
    ``samples_done`` 计数反推, 但 DistributedSampler 下顺序不可靠, 必须显式带索引。

    collate_fn 是 identity (返回 List[dict]), 所以注入的字段会原样进 batch。
    """

    def __init__(self, base_dataset: Any, global_indices: range):
        self.base = base_dataset
        self.indices = list(global_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        gidx = self.indices[i]
        traj_id, anchor = self.base.all_steps[gidx]
        sample = self.base[gidx]
        sample["traj_id"] = int(traj_id)
        sample["anchor"] = int(anchor)
        return sample


class VLAEvaluator(TrainerUtils):
    """镜像 ``VLATrainer(TrainerUtils)``, 但去掉 optimizer/scheduler/wandb/resume。

    跟 VLATrainer 的对应关系:
      - ``__init__``        ↔ VLATrainer.__init__ (train_starvla.py:126)
      - ``prepare_eval``    ↔ VLATrainer.prepare_training (train_starvla.py:137), 跳过 wandb/resume/freeze
      - ``eval_all``        ↔ VLATrainer.train (train_starvla.py:356), 多 episode 外层 loop
      - ``_infer_episode``  ↔ VLATrainer.eval_action_model (train_starvla.py:408), 用 predict_action
    """

    def __init__(
        self,
        cfg: Any,
        model: Any,
        accelerator: Accelerator,
        eval_cfg: Any,
        run_dir: Path,
        ckpt_step: int,
        dataset: Any,
        output_dir: Path,
    ):
        self.config = cfg
        self.model = model
        self.accelerator = accelerator
        self.eval_cfg = eval_cfg
        self.run_dir = run_dir
        self.ckpt_step = ckpt_step
        self.dataset = dataset
        self.output_dir = output_dir

        # Eval-time hyperparams (已 resolve 过 CLI > yaml > default)
        self.batch_size = int(eval_cfg.get("batch_size", 4))
        self.max_anchors = eval_cfg.get("max_anchors")  # None or int
        self.seed = int(eval_cfg.get("seed", 42))
        self.device = eval_cfg.get("device", "cuda")
        self.dtype_name = eval_cfg.get("dtype", "bf16")
        self.progress_every = int(eval_cfg.get("progress_every", 10))

        # action_horizon: 取 chunk 长度, 用来过滤尾部 padding 样本
        self.action_horizon = len(self.dataset.modality_configs["action"].delta_indices)

        # 表示规约: 从 dataset metadata 一次性推出每个 action key 的 (normalized 维度,
        # raw 维度, rotation_type) + 累计偏移。所有"表示相关"的判断 (切分预测、四元数
        # 符号对齐、写元数据) 都查它 —— 不为每种机器人写代码。见 _build_action_spec。
        self.action_spec = self._build_action_spec()

        # horizon_step: 从 action chunk 里挑哪一步放进 schema-v4 的标量 action_gt 数组。
        # 纯粹是 chunk 下标, 跟"下一帧 state"无关 (那是 UMI raw action 的具体语义)。
        # 默认 1 (UMI 上对应 anchor+1 帧, viz 里跟 obs 视觉重合); 可由 eval.horizon_step 覆盖。
        self.horizon_step = int(self.eval_cfg.get("horizon_step", 1))

        # 观测图像 sidecar (obs_images.npz): 给 Stage 2 的 rrd 带图像, 跟 anchor_frame
        # timeline 对齐。rank 0 本地取图 + 降采样后写出; dump 失败只警告不中断 eval。
        # dump_obs_images 默认 True; obs_image_long_side = 降采样长边像素 (保比例)。
        self.dump_obs_images = bool(self.eval_cfg.get("dump_obs_images", True))
        self.obs_image_long_side = int(self.eval_cfg.get("obs_image_long_side", 320))

    # ── 对应 VLATrainer.prepare_training ──
    def prepare_eval(self) -> None:
        """Eval-side 准备。跳过 optimizer/scheduler/wandb/checkpoint-resume/freeze。

        关键: 用全局 seed, **不加 rank 偏移**。否则不同 world_size 下同一 anchor
        会拿到不同 flow-matching 噪声, 多卡评测不可复现。
        """
        set_seed(self.seed)

        # model 已经在外层 from_pretrained + .to(device) 完成, 这里只走 accelerator.prepare
        # 注: accelerator.prepare(model) 会按需 wrap DeepSpeed; 推理路径无 backward,
        # 但 unwrap_model().predict_action 是必须的 (DeepSpeed-wrapped 模块不暴露 predict_action)。
        self.model = self.accelerator.prepare(self.model)

    # ── 对应 VLATrainer.train ──
    def eval_all(self) -> dict[int, dict[str, np.ndarray]]:
        """多 episode 外层 loop。每 episode 独立 _IndexedSubset + prepare + gather + 写盘。"""
        episodes = self._resolve_episode_list()
        all_results: dict[int, dict[str, np.ndarray]] = {}

        for ep in episodes:
            self.accelerator.print(f"\n=== Episode {ep} ===")

            # 每 episode 重建 dataloader (不同 episode 区间不同), 走 accelerator.prepare
            # 注入 DistributedSampler, 把 anchor 切分到各 rank
            eval_dataloader = self._build_episode_dataloader(ep)
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

            local_records = self._infer_episode(ep, eval_dataloader)

            # 各 rank 的 list[dict] 拼接到所有 rank。变长数据用 all_gather_object,
            # 不用 gather(tensor) 因为后者要求等长 padding。
            gathered = _gather_object_list(local_records, self.accelerator)

            if self.accelerator.is_main_process:
                results = self._collate_records(gathered)
                ep_dir = self.output_dir / f"episode_{ep}"
                self._dump_obs_images(ep_dir, gathered)  # 失败只警告, 不阻断 _write_artifacts
                self._write_artifacts(ep_dir, ep, results)
                all_results[ep] = results
                self._print_summary(ep, results)

            self.accelerator.wait_for_everyone()

        return all_results

    # ── 对应 VLATrainer.eval_action_model, 推理循环 ──
    def _infer_episode(
        self, episode_id: int, eval_dataloader: DataLoader
    ) -> list[dict[str, Any]]:
        """单 episode 内 anchor 跨 rank 自动切分, predict_action + per-sample decode。

        表示无关: decode/GT 都在数据集原生 raw 空间 (经训练 transform 链的 unapply 还原),
        不假设 action 是位姿 / 四元数 / 关节角。每条 record 是一个 dict, 含:
          - anchor (int)
          - state_at_anchor (D_state,) raw anchor 状态 (按 state key 拼接)
          - action_gt (D_raw,) raw GT, 取自 action chunk 的第 horizon_step 步
          - action_pred_chunks (H, D_raw) raw 预测, 完整 chunk
          - latency_seconds (float)
        D_raw 由表示决定 (见 metrics.json 的 representation 块)。
        """
        local_records: list[dict[str, Any]] = []

        # 每 episode 1 次 set_seed: 复现 flow-matching 噪声。
        # 所有 rank 用同一 seed → 多卡下同一 anchor 拿到同一初始噪声 (可复现性关键)。
        set_seed(self.seed)

        action_keys = self.dataset.modality_keys["action"]
        state_keys = self.dataset.modality_keys["state"]

        samples_done = 0
        n_samples = len(eval_dataloader.dataset)

        for batch in eval_dataloader:
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            # ⭐ 跟 train_starvla.py:412 完全一致: unwrap + predict_action
            output = self.accelerator.unwrap_model(self.model).predict_action(examples=batch)
            if self.device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            normalized_preds = np.asarray(output["normalized_actions"], dtype=np.float64)
            # predict_action 输出 (B, H, framework.action_dim), action_dim 是 padding 后的
            # 宽度 (e.g. 32, 匹配预训练 ckpt)。只有前 action_dim_actual 维有意义。
            framework_action_dim = normalized_preds.shape[-1]

            for i, sample in enumerate(batch):
                # _IndexedSubset 已经把 traj_id / anchor 注入 sample dict。
                # 多卡下 DistributedSampler 不保证顺序, 不能靠 samples_done 计数反推。
                traj_id = int(sample["traj_id"])
                anchor = int(sample["anchor"])

                action_dim_actual = int(sample.get("action_dim_actual", framework_action_dim))
                pred_norm = normalized_preds[i][..., :action_dim_actual]  # (H, action_dim_actual)

                raw_step = self.dataset.get_step_data(traj_id, anchor)
                raw_state = {k: np.asarray(raw_step[k], dtype=np.float64) for k in state_keys}

                # GT: 数据集原生的 raw action chunk (get_step_data 返回 transform 之前的量,
                # 跟 _decode_to_raw 输出在同一空间, 直接可比)。UMI 上 raw action 恰好是
                # 绝对下一帧 pose; 关节空间数据集上是关节角; delta 数据集上是 delta ——
                # 这里不关心具体语义, 统一按 action key 拼接。
                gt_raw_abs = np.concatenate(
                    [np.asarray(raw_step[k], dtype=np.float64) for k in action_keys], axis=-1
                )

                # Prediction: normalized → raw, 走训练 transform 链的精确逆 (unapply)。
                pred_raw_abs = self._decode_to_raw(pred_norm, raw_state)

                # 四元数符号对齐 (双覆盖, 避免 viz/比较的假阳性分量误差): 只对
                # rotation_type==QUATERNION 的 key 做, 按 spec 定位它在 raw 拼接向量里的切片。
                # 非四元数表示 (关节 / euler / raw 6d) 不动 —— 不再假设 [3:7] 是四元数。
                for entry in self.action_spec:
                    if entry["rotation_type"] != "QUATERNION":
                        continue
                    sl = slice(entry["raw_offset"], entry["raw_offset"] + entry["raw_dim"])
                    pred_raw_abs[:, sl] = align_quaternion_sign_slice(
                        pred_raw_abs[:, sl], gt_raw_abs[:, sl]
                    )

                anchor_state = np.concatenate(
                    [raw_state[k][0] for k in state_keys], axis=-1
                ).astype(np.float64)

                local_records.append({
                    "anchor": anchor,
                    "traj_id": traj_id,
                    "state_at_anchor": anchor_state,
                    "action_gt": gt_raw_abs[self.horizon_step],
                    "action_pred_chunks": pred_raw_abs,
                    "latency_seconds": latency / len(batch),
                })

            samples_done += len(batch)
            if self.progress_every > 0 and self.accelerator.is_main_process and (
                samples_done <= self.progress_every
                or samples_done % self.progress_every == 0
                or samples_done >= n_samples
            ):
                print(f"  Replayed {samples_done}/{n_samples} (last anchor={anchor})", flush=True)

        return local_records

    # ─────────────────── 迁过来的 stateless 工具 ───────────────────

    def _find_episode_range(self, episode_id: int) -> tuple[int, int]:
        """找 episode_id 在 dataset.all_steps 里的连续区间 [start, end)。

        区间内只保留 ``frame + action_horizon <= traj_length`` 的样本 — 这些是
        能取到完整 16-step chunk 且不需要尾部 padding 的 "干净" 样本。
        (来自旧 eval_open_loop.py:184)

        Returns:
            (start, end) — all_steps 里的索引区间, 给 Subset(range(start, end)) 用。
        """
        # 若启用了 train/eval split, 被划到训练集的 episode 不在本 (评测) 数据集里 →
        # 提前给出清晰错误, 而不是让 get_trajectory_index 报含糊的 "found array([])"。
        if int(episode_id) not in {int(t) for t in np.asarray(self.dataset.trajectory_ids).ravel()}:
            raise ValueError(
                f"Episode {episode_id} 不在当前数据集的 trajectory_ids 里 "
                f"(仅 {len(self.dataset.trajectory_ids)} 条轨迹)。若启用了 train/eval "
                f"split (datasets.vla_data.split), 该 episode 可能属于另一侧划分。"
            )
        traj_index = self.dataset.get_trajectory_index(episode_id)
        traj_length = int(self.dataset.trajectory_lengths[traj_index])

        start, end = None, 0
        for i, (traj_id, frame) in enumerate(self.dataset.all_steps):
            if traj_id != episode_id:
                if start is not None:
                    break
                continue
            if frame + self.action_horizon > traj_length:
                if start is not None:
                    break
                continue
            if start is None:
                start = i
            end = i + 1
        if start is None:
            raise ValueError(f"No valid samples for episode {episode_id}")
        return start, end

    def _build_action_spec(self) -> list[dict]:
        """从 dataset.metadata 构建每个 action key 的表示规约 (表示无关评测的核心)。

        每个 entry (按 ``modality_keys['action']`` 顺序):
          - name:        modality key, e.g. "action.end_0_rotation"
          - norm_dim:    normalized/模型输出维度 (取自 statistics, 转换+归一化之后; e.g.
                         relative_6d 的 rotation 在归一化前 quat(4)→6d(6), 故 norm_dim=6)
          - raw_dim:     raw 物理量维度 (取自 metadata.modalities, 转换之前; e.g. 4)
          - rotation_type: raw 旋转类型名 ("QUATERNION" / "EULER" / "ROTATION_6D" / None)
          - norm_offset / raw_offset: 在对应拼接向量里的累计偏移

        norm_dim 用来切分模型输出; raw_dim + rotation_type 用来解读 decode 后的输出
        (e.g. 决定哪个切片是 quaternion、要不要做符号对齐)。两者都来自数据集元数据,
        所以同一套代码对 quaternion / 6d / euler / 关节角 都成立。
        """
        action_keys = self.dataset.modality_keys["action"]
        spec: list[dict] = []
        norm_off = raw_off = 0
        for key in action_keys:
            subkey = key.replace("action.", "")
            # 统计量可能是 1D (D,)（per-dim）或 2D (H, D)（per-step，如 UMI action），
            # 取最后一维即特征维 d'。对 1D 而言 shape[-1] == shape[0]，无副作用。
            norm_dim = int(
                np.asarray(self.dataset.metadata.statistics.action[subkey].mean).shape[-1]
            )
            mmeta = self.dataset.metadata.modalities.action[subkey]
            raw_dim = int(np.asarray(mmeta.shape).tolist()[0])
            rot = getattr(mmeta, "rotation_type", None)
            rot_name = getattr(rot, "name", str(rot)) if rot is not None else None
            spec.append(
                {
                    "name": key,
                    "norm_dim": norm_dim,
                    "raw_dim": raw_dim,
                    "rotation_type": rot_name,
                    "norm_offset": norm_off,
                    "raw_offset": raw_off,
                }
            )
            norm_off += norm_dim
            raw_off += raw_dim
        return spec

    def _decode_to_raw(
        self,
        pred_norm: np.ndarray,
        raw_state: dict[str, np.ndarray],
    ) -> np.ndarray:
        """把 normalized 模型输出 (H, action_dim_actual) 还原成数据集原生 raw 物理量。

        表示无关: 直接复用 dataset 自带的训练 transform 链做精确逆 (apply/unapply),
        不重新实现归一化 / 旋转表示 / SE(3) 相对 的逆运算 —— 这样跟训练 100% 一致,
        对 quaternion / rotation_6d / euler / 关节角 等任意表示都成立 (roundtrip 已验证)。

        步骤:
          1. anchor state (raw) 经 transforms.apply → 跟训练一致的 normalized state。
             (含 RelativePoseTransform 的 pipeline 里, apply 在缺 action key 时对 state
             仍做 quat→6d + 归一化; 不含的 pipeline 里只是归一化 state。)
          2. normalized 预测按 spec.norm_dim 切成 per-key split-key dict。
          3. transforms.unapply({state_norm, action_norm}) 按 reverse 顺序还原:
             归一化.unapply → (若有) rotation_6d→quaternion → (若有) RelativePose.unapply
             (SE(3) 相对→绝对, 用 state 当 base)。

        apply 期望 numpy (StateActionToTensor 会转 tensor), unapply 期望 tensor。

        Returns:
            (H, D_raw) —— 按 ``modality_keys['action']`` 顺序拼接的 raw 物理量。
            D_raw 由表示决定 (UMI 位姿=8, 关节臂=7, ...), 不写死。
        """
        transforms = self.dataset.transforms
        action_keys = self.dataset.modality_keys["action"]
        state_keys = self.dataset.modality_keys["state"]

        # 1. anchor state → normalized (跟训练同构)
        state_raw = {k: np.asarray(raw_state[k], dtype=np.float32) for k in state_keys}
        state_norm = transforms.apply(dict(state_raw))

        # 2. 按 spec 切分 normalized 预测
        dims = [e["norm_dim"] for e in self.action_spec]
        pred_t = torch.as_tensor(pred_norm, dtype=torch.float32)
        if sum(dims) != pred_t.shape[-1]:
            raise ValueError(
                f"per-key normalized dims {dims} (sum={sum(dims)}) != prediction dim "
                f"{pred_t.shape[-1]}; action_dim_actual 跟 dataset statistics 不一致"
            )
        action_norm, cursor = {}, 0
        for key, d in zip(action_keys, dims):
            action_norm[key] = pred_t[..., cursor : cursor + d]
            cursor += d

        # 3. unapply 全量 dict → raw 物理量
        restored = transforms.unapply({**state_norm, **action_norm})
        raw = np.concatenate(
            [to_numpy(restored[k]) for k in action_keys], axis=-1
        ).astype(np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("Decoded raw action contains NaN or Inf")
        return raw

    # ─────────────────── 新增辅助 ───────────────────

    def _resolve_episode_list(self) -> list[int]:
        """解析要评测的 episode 列表, 返回真实 episode_id。

        优先级: --episodes > --episode-id > eval.episodes (YAML) > eval.episode_id (YAML) > 全量。

        ⚠️ 位置索引语义: --episodes / --episode-id / eval.episodes / eval.episode_id 的值都是
        **eval 列表里的 0 基位置索引**, 不是原始 episode_id。人按 0,1,2,3,... 顺序指定,
        这里映射到真实 eval episode_id (升序)。例: hefei eval=[0,10,20,...] 时
        --episodes 0 1 2 3 → {0,10,20,30}。全量 (不指定) 返回全部 eval episode_id。
        split 方案 (交错式) 不变, 与训练侧互斥 → eval 的都是训练未见过的轨迹。
        """
        # eval 列表: split_mode=eval 过滤后的全部 trajectory_id, 升序 (位置索引的基底)
        eval_traj_ids = sorted(int(t) for t in np.asarray(self.dataset.trajectory_ids).ravel())
        n_eval = len(eval_traj_ids)

        # CLI 已在 main() 的 _resolve_eval_params 里 resolve 进 eval_cfg, 这里只读
        requested, src = None, None
        episodes = self.eval_cfg.get("episodes")
        if episodes is not None:
            requested, src = [int(ep) for ep in episodes], "episodes"
        else:
            episode_id = self.eval_cfg.get("episode_id")
            if episode_id is not None:
                requested, src = [int(episode_id)], "episode-id"

        if requested is None:
            # 全量
            if n_eval:
                self.accelerator.print(
                    f"[eval] 全量评测: {n_eval} 条 eval episode "
                    f"(id {eval_traj_ids[0]}..{eval_traj_ids[-1]})"
                )
            return eval_traj_ids

        # 位置索引 → 真实 episode_id (越界给清晰报错, 不含糊)
        mapped, bad = [], []
        for idx in requested:
            if 0 <= idx < n_eval:
                mapped.append(eval_traj_ids[idx])
            else:
                bad.append(idx)
        if bad:
            raise ValueError(
                f"--{src} 位置索引 {bad} 越界: eval 列表共 {n_eval} 条 "
                f"(有效位置 0..{n_eval - 1})。注意这些值是 eval 列表里的位置, 不是原始 episode_id; "
                f"全量评测请省略 --episodes/--episode-id。"
            )
        pairs = ", ".join(f"{i}→ep{t}" for i, t in zip(requested, mapped))
        self.accelerator.print(
            f"[eval] {src} 位置索引→真实 episode_id: [{pairs}]  (eval 共 {n_eval} 条)"
        )
        return mapped

    def _build_episode_dataloader(self, episode_id: int) -> DataLoader:
        """单 episode 的索引化 Subset + DataLoader。

        跟旧 eval_open_loop.py:287 结构一致, 但用 ``_IndexedSubset`` 替代原版 ``Subset``,
        显式把 ``traj_id`` / ``anchor`` 注入 sample dict (多卡下 DistributedSampler
        不保证顺序, 不能靠计数反推)。让 ``accelerator.prepare()`` 注入 DistributedSampler,
        把 anchor 切到各 rank。
        """
        start, end = self._find_episode_range(episode_id)
        n_total = end - start
        n_samples = (
            min(n_total, self.max_anchors) if self.max_anchors is not None else n_total
        )
        indexed = _IndexedSubset(self.dataset, range(start, start + n_samples))
        return DataLoader(
            indexed,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0,
        )

    def _collate_records(
        self, gathered: list[dict[str, Any]]
    ) -> dict[str, np.ndarray]:
        """gather_object 拿到的 list[dict] 排序 + stack 成 schema v4 数组。

        DistributedSampler 不保证跨 rank 交错升序, 必须按 anchor 重排。
        """
        gathered_sorted = sorted(gathered, key=lambda r: r["anchor"])
        return {
            "anchors": np.asarray([r["anchor"] for r in gathered_sorted], dtype=np.int64),
            "state_at_anchor": np.stack([r["state_at_anchor"] for r in gathered_sorted]),
            "action_gt": np.stack([r["action_gt"] for r in gathered_sorted]),
            "action_pred_chunks": np.stack([r["action_pred_chunks"] for r in gathered_sorted]),
            "latency_seconds": np.asarray(
                [r["latency_seconds"] for r in gathered_sorted], dtype=np.float64
            ),
        }

    def _dump_obs_images(self, ep_dir: Path, gathered: list[dict[str, Any]]) -> None:
        """rank 0 本地把每个 anchor 的 obs 视频帧降采样后写 ``obs_images.npz`` sidecar。

        给 Stage 2 的 rrd 带图像 (跟 anchor_frame timeline 对齐)。无 video key 的数据集
        直接跳过; 取图/降采样/写盘任一步失败只警告, 不影响 predictions/metrics。

        - 单 episode 全部 anchor 共享 traj_id, 取图廉价; 只在 rank 0 跑, 不跨 rank 传图。
        - 降采样: PIL 按长边缩到 ``self.obs_image_long_side``, 保比例, uint8。
        """
        if not self.dump_obs_images or not self.accelerator.is_main_process:
            return
        video_keys = self.dataset.modality_keys.get("video", [])
        if not video_keys:
            return

        ep_dir.mkdir(parents=True, exist_ok=True)
        gathered_sorted = sorted(gathered, key=lambda r: r["anchor"])
        anchors = np.asarray([r["anchor"] for r in gathered_sorted], dtype=np.int64)

        try:
            images_per_cam: dict[str, np.ndarray] = {}
            for cam in video_keys:
                frames: list[np.ndarray] = []
                for r in gathered_sorted:
                    # get_video 返回 (T,H,W,C); obs delta_indices=[0] → 单帧, 取 [0]。
                    frame = np.asarray(self.dataset.get_video(r["traj_id"], cam, r["anchor"]))[0]
                    frames.append(_resize_long_side(frame, self.obs_image_long_side))
                images_per_cam[cam] = np.stack(frames, axis=0).astype(np.uint8)
            path = write_obs_images(ep_dir, video_keys, images_per_cam, anchors)
            self.accelerator.print(
                f"  wrote obs images → {path.name} (cams={video_keys}, n={len(anchors)})"
            )
        except Exception as exc:  # noqa: BLE001 — 图像 dump 是 best-effort, 不阻断 eval
            self.accelerator.print(
                f"  WARNING: obs image dump failed ({exc!r}); rrd 将不带图像 (eval 继续)"
            )

    def _write_artifacts(
        self, ep_dir: Path, episode_id: int, results: dict[str, np.ndarray]
    ) -> None:
        """写 predictions.npz (schema v4) + metrics.json。"""
        ep_dir.mkdir(parents=True, exist_ok=True)
        preds_path = write_predictions(ep_dir, results)
        artifacts: dict[str, str] = {"predictions": preds_path.name}
        # obs_images.npz 由 _dump_obs_images 先写; 在场才登记, 让 Stage 2 / 下游知道。
        if (ep_dir / OBS_IMAGES_FILENAME).exists():
            artifacts["obs_images"] = OBS_IMAGES_FILENAME
        report = {
            "eval_config": OmegaConf.to_container(self.eval_cfg, resolve=True),
            "episode_id": int(episode_id),
            "checkpoint_step": int(self.ckpt_step),
            "run_dir": str(self.run_dir),
            "artifacts": artifacts,
            # 表示元数据 (schema v4 只增不改): 告诉下游 action_pred_chunks / action_gt
            # 每个 dim 是什么 (key 名 / 维度 / 旋转类型), 让 metrics/viz 能表示无关地解读。
            "representation": {
                "action_keys": [
                    {
                        "name": e["name"],
                        "norm_dim": e["norm_dim"],
                        "raw_dim": e["raw_dim"],
                        "rotation_type": e["rotation_type"],
                    }
                    for e in self.action_spec
                ],
                "norm_dim_total": sum(e["norm_dim"] for e in self.action_spec),
                "raw_dim_total": sum(e["raw_dim"] for e in self.action_spec),
                "action_horizon": self.action_horizon,
                "horizon_step": self.horizon_step,
            },
        }
        write_metrics(ep_dir, report)

    def _print_summary(self, episode_id: int, results: dict[str, np.ndarray]) -> None:
        n = len(results["anchors"])
        mean_latency = float(np.mean(results["latency_seconds"]))
        print(
            f"\n[ep={episode_id} step={self.ckpt_step} seed={self.seed}] "
            f"samples: {n}  mean latency: {mean_latency:.3f}s\n"
            f"  artifacts: {self.output_dir / f'episode_{episode_id}'}\n"
            f"  Stage 2: PYTHONPATH=. python examples/realRobots/z_umi/eval_files/stage2_visualize.py "
            f"{self.output_dir / f'episode_{episode_id}'}"
        )


# ─────────────────────────── main ─────────────────────────────────────


def _resolve_eval_params(args: argparse.Namespace, eval_cfg: Any, run_dir: Path) -> tuple[int, Any]:
    """把 CLI 参数合并进 eval_cfg (CLI > yaml > default), 返回 (ckpt_step, merged eval_cfg)。"""
    # 用一个可写 dict 包装, 后续 _resolve_episode_list / __init__ 都从这读
    if not OmegaConf.is_config(eval_cfg):
        eval_cfg = OmegaConf.create(eval_cfg)

    # 把 CLI 覆盖写进 eval_cfg
    if args.episodes is not None:
        eval_cfg.episodes = list(args.episodes)
    elif args.episode_id is not None:
        eval_cfg.episodes = [int(args.episode_id)]

    if args.batch_size is not None:
        eval_cfg.batch_size = int(args.batch_size)
    if args.max_anchors is not None:
        eval_cfg.max_anchors = int(args.max_anchors)
    if args.seed is not None:
        eval_cfg.seed = int(args.seed)
    if args.device is not None:
        eval_cfg.device = args.device
    if args.dtype is not None:
        eval_cfg.dtype = args.dtype
    if args.progress_every is not None:
        eval_cfg.progress_every = int(args.progress_every)

    # 默认值兜底
    eval_cfg.batch_size = int(eval_cfg.get("batch_size", 4))
    eval_cfg.seed = int(eval_cfg.get("seed", 42))
    eval_cfg.device = eval_cfg.get("device", "cuda")
    eval_cfg.dtype = eval_cfg.get("dtype", "bf16")

    # ckpt step: CLI > eval.yaml > auto-discover 最新
    if args.checkpoint_step is not None:
        step = int(args.checkpoint_step)
    elif eval_cfg.get("checkpoint_step") is not None:
        step = int(eval_cfg.checkpoint_step)
    else:
        step = int(discover_checkpoint_steps(run_dir)[-1])

    return step, eval_cfg


def _resolve_output_dir(
    args: argparse.Namespace, eval_cfg: Any, run_dir: Path, ckpt_step: int, seed: int
) -> Path:
    """自动派生: <output_root>/<run_basename>_step<N>_seed<S>/"""
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_root = Path(eval_cfg.get("output_root", "playground/Evals"))
        # 换数据集 (--data-mix) 时追加 _mix-{name}, 避免覆盖原数据集产物。
        mix_suffix = f"_mix-{args.data_mix}" if getattr(args, "data_mix", None) else ""
        output_dir = output_root / f"{run_dir.name}_step{ckpt_step}_seed{seed}{mix_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    # Eval 是纯推理 (产物全落本地 npz/json/rrd), 设计上不接任何 tracker。
    # 防御: 环境里若残留 SWANLAB_MODE=local (例如此前跑过 train.sh --local 或手动 export),
    # 某些库会据此自动 swanlab.init 并尝试启本地看板, 缺 swanboard (swanlab[dashboard]) 时
    # 抛 ImportError 污染/中断 eval。强制 disabled, 让任何 swanlab.init 退化成 no-op。
    os.environ["SWANLAB_MODE"] = "disabled"

    args = parse_args()
    run_dir = args.run_dir.resolve()

    # ── 1. config + dataset ─────────────────────────────────────────
    cfg, _ = load_merged_config(
        run_dir, args.eval_yaml, args.data_root_override, args.dataset_name, args.data_mix
    )
    # 强制评测用模型固化的 per-subkey norm (dataset_statistics.json 的 _per_key), 脱离
    # 数据集 stats_gr00t.json (后者可能被多卡/错配重算污染)。数据集构造时 (_get_metadata)
    # 读此 env, 用 _per_key 替换 metadata.statistics → transform init 时就按模型 norm 建。
    os.environ["STARVLA_EVAL_PER_KEY_JSON"] = str(run_dir / "dataset_statistics.json")
    dataset = build_dataset(cfg)

    # ── 2. resolve eval params (CLI > yaml > default) ───────────────
    ckpt_step, eval_cfg = _resolve_eval_params(args, cfg.eval, run_dir)

    device = eval_cfg.device
    dtype = torch_dtype(eval_cfg.dtype)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    output_dir = _resolve_output_dir(args, eval_cfg, run_dir, ckpt_step, int(eval_cfg.seed))

    # ── 3. accelerator + model ──────────────────────────────────────
    # Eval 走 setup_eval_accelerator (不带 DeepSpeed, 见函数注释)。
    accelerator = setup_eval_accelerator()
    accelerator.print(f"\nEval run_dir: {run_dir}, ckpt_step: {ckpt_step}, output_dir: {output_dir}")

    ckpt_path = checkpoint_path(run_dir, ckpt_step)
    accelerator.print(f"Loading checkpoint: {ckpt_path}")
    # baseframework.from_pretrained 内部 read_mode_config → build_framework → load_state_dict
    # 它读 ckpt 自带的 config.yaml, 跟我们 merged cfg 解耦, 避免 schema mismatch。
    model = baseframework.from_pretrained(str(ckpt_path))
    model = model.to(device=device, dtype=dtype).eval()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── 4. evaluator ────────────────────────────────────────────────
    evaluator = VLAEvaluator(
        cfg=cfg,
        model=model,
        accelerator=accelerator,
        eval_cfg=eval_cfg,
        run_dir=run_dir,
        ckpt_step=ckpt_step,
        dataset=dataset,
        output_dir=output_dir,
    )
    evaluator.prepare_eval()
    evaluator.eval_all()

    # 给 launcher (run_umi_pi05_open_eval.sh) 一个确定性的 output_dir 标记, 替代它旧的
    # `ls -td | head -1` 猜测 —— 多个 eval 产物共存时后者会按 mtime 误抓别的 run 的目录。
    # 主进程单行打印; launcher grep 这一行定位 Stage 2 要可视化的目录。
    accelerator.print(f"__EVAL_OUTPUT_DIR__ {output_dir.resolve()}")

    # ── 5. 清理 ─────────────────────────────────────────────────────
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
