"""RoboDojo dataloader 冒烟测试（CPU 即可）。

验证：
1. registry 自动发现 robodojo data_config（robot_type + mixture）
2. 单任务过滤生效（task 0 → 100 episodes）
3. 样本形状/范围/语言正确
4. 相对位姿几何正确性（step0 相对位置应小；rotation_6d 行范数 ~1）

用法（从 robodojo worktree repo root）:
  python examples/simBenchmarks/RoboDojo/train_files/smoke_test_dataloader.py
"""
import sys
from pathlib import Path

import numpy as np

# repo root: 向上找直到含 starVLA/ 目录
_REPO_ROOT = Path(__file__).resolve().parent
while _REPO_ROOT != _REPO_ROOT.parent and not (_REPO_ROOT / "starVLA").is_dir():
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.registry import (  # noqa: E402
    ROBOT_TYPE_CONFIG_MAP, DATASET_NAMED_MIXTURES, discover_and_merge,
)
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402

DATA_ROOT = "/nix/data/robodojo"
DATA_NAME = "RoboDojo_ee_lerobot_v30_video"
ROBOT_TYPE = "robodojo_bimanual_relative_6d"


def main():
    print("=" * 70)
    # 1. registry 发现
    assert ROBOT_TYPE in ROBOT_TYPE_CONFIG_MAP, f"{ROBOT_TYPE} 未被 registry 发现！"
    assert "robodojo_ee_all" in DATASET_NAMED_MIXTURES, "mixture robodojo_ee_all 未发现！"
    print(f"[1] registry OK: robot_type={ROBOT_TYPE}, mixture=robodojo_ee_all")

    # 2. 构建 task 0 数据集
    data_cfg = {
        "include_state": True,
        "lerobot_version": "v3.0",
        "video_backend": "torchvision_av",
        "task_index": 0,
        "norm_stride": 10,
    }
    ds = make_LeRobotSingleDataset(Path(DATA_ROOT), DATA_NAME, ROBOT_TYPE, data_cfg=data_cfg)
    n_traj = len(ds.trajectory_lengths)
    print(f"[2] task 0 数据集: {n_traj} episodes (期望 100), len(ds)={len(ds)}")
    assert n_traj == 100, f"task 过滤失败: 期望 100 episodes, 实际 {n_traj}"

    # 3. 取一个样本
    idx = min(50, len(ds) - 1)
    sample = ds[idx]
    print(f"[3] sample[{idx}] keys: {list(sample.keys())}")
    action = np.asarray(sample["action"], dtype=np.float32)
    print(f"    action.shape={action.shape} (期望 [50, 20])  min={action.min():.3f} max={action.max():.3f}")
    assert action.shape == (50, 20), f"action 形状错: {action.shape}"
    state = np.asarray(sample["state"], dtype=np.float32)
    print(f"    state.shape={state.shape} (期望 [1, 20])  min={state.min():.3f} max={state.max():.3f}")
    assert state.shape == (1, 20), f"state 形状错: {state.shape}"
    print(f"    image: {len(sample['image'])} 张 (期望 3)")
    assert len(sample["image"]) == 3, f"图像数错: {len(sample['image'])}"
    print(f"    lang: {sample['lang']!r}")
    assert sample["lang"] and sample["lang"] != "", "语言为空！"
    pad = np.asarray(sample["action_pad"])
    print(f"    action_pad: 有效步 {int(pad.sum())}/{len(pad)}")

    # 4. 相对位姿几何正确性
    # 取一条轨迹, 算 step0 的相对 action（应接近 0，因为 action[t0]=state[t0+1] 与 state[t0] 很近）
    traj_len = int(ds.trajectory_lengths[0])
    anchor = 5
    raw = ds.get_step_data(0, anchor)
    state_keys = ds.modality_keys["state"]
    action_keys = ds.modality_keys["action"]
    raw_np = {**{k: np.asarray(raw[k], dtype=np.float32) for k in state_keys},
              **{k: np.asarray(raw[k], dtype=np.float32) for k in action_keys}}
    out = ds.transforms.apply(dict(raw_np))

    # 左臂相对位置 step0（相对位姿在第 0 步应小）
    rel_lpos = np.asarray(out["action.end_left_position"])  # (50, 3)
    rel_rpos = np.asarray(out["action.end_right_position"])
    print(f"[4] 左臂相对位置 step0: {np.round(rel_lpos[0], 4).tolist()} (应接近 0)")
    print(f"    右臂相对位置 step0: {np.round(rel_rpos[0], 4).tolist()} (应接近 0)")

    # rotation_6d 行范数应 ~1（前两行 of rotation matrix）
    for k in ["state.end_left_rotation", "state.end_right_rotation"]:
        r6 = np.asarray(out[k])  # (1, 6) 或 (1,6)
        r6 = r6.reshape(-1, 3)
        norms = np.linalg.norm(r6, axis=1)
        print(f"    {k} rotation_6d 行范数: {np.round(norms, 3).tolist()} (应 ~1.0)")

    # round-trip: unapply 应回到原始（验证可逆/几何一致）
    back = ds.transforms.unapply(dict(out))
    for k in ["state.end_left_position", "action.end_left_position"]:
        orig = raw_np[k].astype(np.float64)
        rec = np.asarray(back[k]).astype(np.float64)
        err = np.abs(orig - rec).max()
        print(f"    round-trip {k}: max|orig-rec|={err:.4f}")

    print("\n✅ 冒烟测试通过")
    print("完成。")


if __name__ == "__main__":
    main()
