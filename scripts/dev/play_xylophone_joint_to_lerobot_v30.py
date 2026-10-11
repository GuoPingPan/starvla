#!/usr/bin/env python3
"""Convert successful play_Xylophone rollouts to a 14D absolute-joint LeRobot v3 dataset.

The source is the completed ``official_retarget_v2_tail_grasp`` production
tree (or its ``success`` directory). At time t, the recorded arm joints and
normalized EE gripper state describe the observation before action t. The
action is the *applied* absolute IK arm target and normalized gripper command.
The source also records physical gripper joints in metres; those are not the
normalized gripper values used by the policy. The source videos contain T+1
frames; the terminal frame is retained for H264 decoding but is not a row.

All source trajectories and videos are read-only. An episode with any applied
joint target more than pi radians from the corresponding observed joint is
excluded as an IK branch jump; all other targets are preserved exactly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from make_kong_npz_to_lerobot_v30 import (
    CAMERAS, FPS, _check_video, _check_video_pts, _convert_video, _probe_video,
)
from make_kong_joint_replay_to_lerobot_v30 import JOINT_NAMES, _modality


TASK_TEXT = "Pick up the mallet and strike all xylophone keys from left to right."
SEED_PATTERN = re.compile(r"seed_(\d+)$")
LAYOUT_PATTERN = re.compile(r"layout_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="completed production root or its success directory")
    parser.add_argument("--dst", type=Path, help="new LeRobot v3 dataset root")
    parser.add_argument("--max-episodes", type=int, help="first N episodes for a smoke check")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel video validation/copy workers")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate all selected trajectories and source video timing; write nothing")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.validate_only and args.dst is None:
        parser.error("--dst is required unless --validate-only is set")
    return args


def _scalar(npz: np.lib.npyio.NpzFile, key: str, dtype: type) -> object:
    value = np.asarray(npz[key])
    if value.shape != ():
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    return dtype(value.item())


def _source_roots(src: Path) -> tuple[Path, Path]:
    src = src.resolve()
    root = src.parent if src.name == "success" else src
    success = root / "success"
    if not success.is_dir():
        raise FileNotFoundError(success)
    return root, success


def _episodes(root: Path, success: Path, max_episodes: int | None) -> list[dict]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    if (summary.get("status") != "complete" or summary.get("task") != "play_Xylophone"
            or summary.get("robot") != "arx_x5"
            or summary.get("control_mode") != "ee_to_ik_joint_position"
            or summary.get("trajectory_schema_version") != 2
            or summary.get("augmentation") != "none"):
        raise ValueError(f"Unexpected or incomplete production summary: {summary_path}")

    paths = sorted(success.glob("seed_*/layout_*/trajectory.npz"))
    if (len(paths) != summary.get("success_samples")
            or len(paths) != summary.get("expected")
            or summary.get("video_count") != len(CAMERAS) * len(paths)):
        raise ValueError(f"Production summary disagrees with success tree: {summary_path}")
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise ValueError(f"No successful trajectories under {success}")

    episodes = []
    seen = set()
    for index, path in enumerate(paths):
        folder = path.parent
        seed_match = SEED_PATTERN.fullmatch(folder.parent.name)
        layout_match = LAYOUT_PATTERN.fullmatch(folder.name)
        if not seed_match or not layout_match:
            raise ValueError(f"Unexpected trajectory path: {path}")
        seed, layout = int(seed_match[1]), int(layout_match[1])
        if (seed, layout) in seen:
            raise ValueError(f"Duplicate seed/layout {(seed, layout)}")
        seen.add((seed, layout))
        sample = json.loads((folder / "sample.json").read_text())
        if (sample.get("seed") != seed or sample.get("layout_id") != layout
                or sample.get("status") != "success"
                or sample.get("control_mode") != "ee_to_ik_joint_position"):
            raise ValueError(f"Sample metadata mismatch: {folder}")

        with np.load(path, allow_pickle=False) as npz:
            length = len(npz["actions"])
            if (length < 1 or _scalar(npz, "schema_version", int) != 2
                    or _scalar(npz, "eval_seed", int) != seed
                    or _scalar(npz, "layout_id", int) != layout
                    or _scalar(npz, "run_id", str) != sample.get("last_run_id")
                    or _scalar(npz, "policy_version", str) != sample.get("policy_version")
                    or _scalar(npz, "control_mode", str) != "ee_to_ik_joint_position"
                    or _scalar(npz, "arm_joint_position_unit", str) != "rad"
                    or _scalar(npz, "gripper_joint_position_unit", str) != "m"
                    or _scalar(npz, "action_type", str) != "ee"):
                raise ValueError(f"Trajectory identity/control mismatch: {path}")
            ee_actions = np.asarray(npz["actions"], dtype=np.float32)
            robot_states = np.asarray(npz["robot_states"], dtype=np.float32)
            arm_joints = np.asarray(npz["arm_joints"], dtype=np.float32)
            arm_targets = np.asarray(npz["arm_joint_targets"], dtype=np.float32)
            gripper_joints = np.asarray(npz["gripper_joints"], dtype=np.float32)
            if (ee_actions.shape != (length, 16)
                    or robot_states.shape != (length, 16)
                    or arm_joints.shape != (length, 2, 6)
                    or arm_targets.shape != (length, 2, 6)
                    or gripper_joints.shape != (length, 2, 2)
                    or npz["terminal_arm_joints"].shape != (2, 6)
                    or npz["terminal_gripper_joints"].shape != (2, 2)
                    or sample.get("steps") != length
                    or not np.array_equal(npz["step_counts"], np.arange(length))):
                raise ValueError(f"Trajectory arrays/step count mismatch: {path}")
            if (not _scalar(npz, "success", bool)
                    or not _scalar(npz, "environment_success", bool)
                    or _scalar(npz, "unstable", bool)
                    or _scalar(npz, "terminal_remaining_checks", int) != 0
                    or _scalar(npz, "terminal_reward", float) != 1.0):
                raise ValueError(f"Trajectory did not finish with native success: {path}")
            if not all(np.isfinite(array).all() for array in
                       (ee_actions, robot_states, arm_joints, arm_targets, gripper_joints)):
                raise ValueError(f"Nonfinite joint state/action: {path}")
            states = np.concatenate((arm_joints[:, 0], robot_states[:, 7:8],
                                     arm_joints[:, 1], robot_states[:, 15:16]), axis=1)
            actions = np.concatenate((arm_targets[:, 0], ee_actions[:, 7:8],
                                      arm_targets[:, 1], ee_actions[:, 15:16]), axis=1)
            if (states.shape != (length, 14) or actions.shape != (length, 14)
                    or np.any(states[:, [6, 13]] < 0)
                    or np.any(states[:, [6, 13]] > 1)
                    or np.any(actions[:, [6, 13]] < 0)
                    or np.any(actions[:, [6, 13]] > 1)):
                raise ValueError(f"Invalid normalized gripper or joint vector: {path}")
            outlier_count = int(np.count_nonzero(np.abs(arm_targets) > 2 * math.pi))
            max_abs_target = float(np.max(np.abs(arm_targets)))
            target_state_delta = np.abs(arm_targets - arm_joints)
            branch_jump_count = int(np.count_nonzero(target_state_delta > math.pi))
            max_abs_delta = float(np.max(target_state_delta))

        videos = {}
        for camera in CAMERAS:
            video = folder / f"{camera}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise FileNotFoundError(video)
            videos[camera] = video
        episodes.append({
            "episode_index": index, "seed": seed, "layout_id": layout,
            "folder": folder, "run_id": sample["last_run_id"], "length": length,
            "states": states, "actions": actions, "videos": videos,
            "policy_version": sample["policy_version"],
            "joint_targets_over_2pi": outlier_count,
            "max_abs_joint_target_rad": max_abs_target,
            "joint_target_state_delta_over_pi": branch_jump_count,
            "max_abs_joint_target_state_delta_rad": max_abs_delta,
        })
    return episodes


def _write_tasks(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pandas_meta = {
        "index_columns": ["__index_level_0__"],
        "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode",
                            "numpy_type": "str", "metadata": {"encoding": "UTF-8"}}],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
             "numpy_type": "int64", "metadata": None},
            {"name": None, "field_name": "__index_level_0__", "pandas_type": "object",
             "numpy_type": "str", "metadata": None},
        ],
        "attributes": {}, "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    table = pa.Table.from_arrays(
        [pa.array([0], type=pa.int64()), pa.array([TASK_TEXT], type=pa.large_string())],
        names=["task_index", "__index_level_0__"],
    ).replace_schema_metadata({b"pandas": json.dumps(pandas_meta).encode()})
    pq.write_table(table, path)


def _write_dataset(dst: Path, src: Path, episodes: list[dict], excluded: list[dict],
                   source_count: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_frames = sum(ep["length"] for ep in episodes)
    states = np.concatenate([ep["states"] for ep in episodes])
    actions = np.concatenate([ep["actions"] for ep in episodes])
    fixed14 = pa.list_(pa.field("element", pa.float32(), nullable=False), 14)
    arrays = [
        pa.FixedSizeListArray.from_arrays(pa.array(states.ravel(), type=pa.float32()), 14),
        pa.FixedSizeListArray.from_arrays(pa.array(actions.ravel(), type=pa.float32()), 14),
    ]
    fields = [pa.field("observation.state", fixed14), pa.field("action", fixed14)]
    for name, values, dtype in [
        ("timestamp", np.concatenate([np.arange(ep["length"], dtype=np.float32) / FPS
                                      for ep in episodes]), pa.float32()),
        ("frame_index", np.concatenate([np.arange(ep["length"], dtype=np.int64)
                                        for ep in episodes]), pa.int64()),
        ("episode_index", np.concatenate([np.full(ep["length"], ep["episode_index"], dtype=np.int64)
                                          for ep in episodes]), pa.int64()),
        ("index", np.arange(total_frames, dtype=np.int64), pa.int64()),
        ("task_index", np.zeros(total_frames, dtype=np.int64), pa.int64()),
    ]:
        arrays.append(pa.array(values, type=dtype))
        fields.append(pa.field(name, dtype))
    data_dir = dst / "data/chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields)),
                   data_dir / "file-000.parquet")

    rows = []
    provenance = []
    start = 0
    for ep in episodes:
        length, index = ep["length"], ep["episode_index"]
        row = {
            "episode_index": index, "tasks": [TASK_TEXT], "length": length,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": start, "dataset_to_index": start + length,
        }
        for video_key in CAMERAS.values():
            prefix = f"videos/{video_key}"
            row[f"{prefix}/chunk_index"] = 0
            row[f"{prefix}/file_index"] = index
            row[f"{prefix}/from_timestamp"] = 0.0
            row[f"{prefix}/to_timestamp"] = length / FPS
        rows.append(row)
        provenance.append({
            "episode_index": index, "seed": ep["seed"], "layout_id": ep["layout_id"],
            "run_id": ep["run_id"], "policy_version": ep["policy_version"],
            "source": str(ep["folder"].relative_to(src)), "length": length,
            "joint_targets_over_2pi": ep["joint_targets_over_2pi"],
            "max_abs_joint_target_rad": ep["max_abs_joint_target_rad"],
            "joint_target_state_delta_over_pi": ep["joint_target_state_delta_over_pi"],
            "max_abs_joint_target_state_delta_rad": ep["max_abs_joint_target_state_delta_rad"],
        })
        start += length
    episodes_dir = dst / "meta/episodes/chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), episodes_dir / "file-000.parquet")

    meta_dir = dst / "meta"
    _write_tasks(meta_dir / "tasks.parquet")
    (meta_dir / "modality.json").write_text(json.dumps(_modality(), indent=2) + "\n")
    video_info = {
        "dtype": "video", "shape": [3, 480, 640], "names": ["channels", "height", "width"],
        "info": {"video.height": 480, "video.width": 640, "video.codec": "h264",
                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                 "video.fps": FPS, "video.channels": 3, "has_audio": False},
    }
    info = {
        "codebase_version": "v3.0", "robot_type": "unified_robot",
        "total_episodes": len(episodes), "total_frames": total_frames, "total_tasks": 1,
        "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
        "fps": FPS, "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14], "names": [JOINT_NAMES]},
            "action": {"dtype": "float32", "shape": [14], "names": [JOINT_NAMES]},
            **{key: video_info for key in CAMERAS.values()},
            **{key: {"dtype": dtype, "shape": [1], "names": None} for key, dtype in (
                ("timestamp", "float32"), ("frame_index", "int64"),
                ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64"),
            )},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    with (meta_dir / "episode_sources.jsonl").open("w") as stream:
        for record in provenance:
            stream.write(json.dumps(record) + "\n")
    with (meta_dir / "excluded_sources.jsonl").open("w") as stream:
        for ep in excluded:
            stream.write(json.dumps({
                "seed": ep["seed"], "layout_id": ep["layout_id"],
                "source": str(ep["folder"].relative_to(src)), "length": ep["length"],
                "reason": "applied_joint_target_minus_observed_joint_exceeds_pi",
                "joint_target_state_delta_over_pi": ep["joint_target_state_delta_over_pi"],
                "max_abs_joint_target_state_delta_rad": ep["max_abs_joint_target_state_delta_rad"],
                "joint_targets_over_2pi": ep["joint_targets_over_2pi"],
                "max_abs_joint_target_rad": ep["max_abs_joint_target_rad"],
            }) + "\n")
    (meta_dir / "source_validation.json").write_text(json.dumps({
        "source_episodes_checked": source_count,
        "accepted_episodes": len(episodes),
        "excluded_episodes": len(excluded),
        "accepted_frames": total_frames,
        "exclusion_threshold_rad": math.pi,
        "exclusion_rule": "any(abs(arm_joint_targets-arm_joints)>pi)",
        "joint_targets_preserved_without_wrap_or_clamp": True,
    }, indent=2) + "\n")
    return total_frames


def _validate_video(job: tuple[Path, int]) -> None:
    path, length = job
    _check_video(_probe_video(path), length + 1, path)
    _check_video_pts(path, length + 1)


def main() -> int:
    args = parse_args()
    root, success = _source_roots(args.src)
    source_episodes = _episodes(root, success, args.max_episodes)
    excluded = [ep for ep in source_episodes if ep["joint_target_state_delta_over_pi"]]
    episodes = [ep for ep in source_episodes if not ep["joint_target_state_delta_over_pi"]]
    for index, ep in enumerate(episodes):
        ep["episode_index"] = index
    if not episodes:
        raise ValueError("No trajectories remain after IK branch-jump exclusion")
    for ep in excluded:
        print(f"Excluded seed={ep['seed']} layout={ep['layout_id']}: "
              f"{ep['joint_target_state_delta_over_pi']} target-state deltas >pi, "
              f"max={ep['max_abs_joint_target_state_delta_rad']:.3f} rad", flush=True)
    outliers = sum(ep["joint_targets_over_2pi"] for ep in source_episodes)
    total_frames = sum(ep["length"] for ep in episodes)
    print(f"Checked {len(source_episodes)} successful joint trajectories; "
          f"accepted {len(episodes)} episodes/{total_frames} frames, "
          f"excluded {len(excluded)} IK branch-jump episodes; "
          f"source had {outliers} applied joint targets outside [-2pi,2pi]", flush=True)
    if args.max_episodes is None and (len(source_episodes) != 165 or len(episodes) != 159
                                      or total_frames != 46441):
        raise ValueError("Full production acceptance count changed from 165/159/46441")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("Missing required executable: ffprobe")
    if args.validate_only:
        jobs = [(video, ep["length"]) for ep in source_episodes
                for video in ep["videos"].values()]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_validate_video, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 60 == 0 or completed == len(futures):
                    print(f"Videos validated {completed}/{len(futures)}", flush=True)
        print("Validation complete; no files written", flush=True)
        return 0

    dst = args.dst.resolve()
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.parent / f".{dst.name}.partial-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    try:
        # Excluded trajectories must still have valid source videos. Accepted
        # videos are checked by _convert_video while they are copied.
        excluded_jobs = [(video, ep["length"]) for ep in excluded
                         for video in ep["videos"].values()]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for future in as_completed([executor.submit(_validate_video, job)
                                        for job in excluded_jobs]):
                future.result()
        jobs = []
        for ep in episodes:
            for camera, video_key in CAMERAS.items():
                output = (staging / "videos" / video_key / "chunk-000"
                          / f"file-{ep['episode_index']:03d}.mp4")
                jobs.append((ep["videos"][camera], output, ep["length"]))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_convert_video, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 60 == 0 or completed == len(futures):
                    print(f"Videos copied {completed}/{len(futures)}", flush=True)
        total_frames = _write_dataset(staging, root, episodes, excluded,
                                      len(source_episodes))
        staging.rename(dst)
    except Exception:
        shutil.rmtree(staging)
        raise
    print(f"Done: {len(episodes)} episodes, {total_frames} frames -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError,
            subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
