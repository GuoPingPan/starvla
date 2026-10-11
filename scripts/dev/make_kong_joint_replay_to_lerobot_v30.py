#!/usr/bin/env python3
"""Convert validated make_kong joint replays into an independent LeRobot v3 dataset.

Supported production layouts::

    schema v3: <src>/success/seed_0/layout_000/trajectory.npz
               <src>/success/seed_0/layout_000/verification.json
               <src>/success/seed_0/layout_000/cam_{head,left_wrist,right_wrist}.mp4
    schema v1: <src>/0_layout_000/<run_id>/trajectory_joint.npz
               <src>/0_layout_000/<run_id>/episode_*_cam_*_success.mp4

For v3, each 14D observation.state is made from the pre-action measured
arm_joint_states (left six, right six) and robot_states[:, [7, 15]] for the
normalized left and right gripper state seen by the policy. Those gripper
values reflect the last applied motor target and can differ from the previous
gripper command while the controller is moving. The 14D action is the
recorded absolute joint_actions: arm position targets in radians and
normalized gripper opening commands in [0, 1].

The replay must have T pre-action joint observations, T actual joint
targets, and T+1 newly recorded video frames. At row t, the video and state
describe the observation immediately before joint_actions[t] was commanded.
The terminal video frame is retained as an H264 reference but is not a
training row. Original EE videos are deliberately never used: even a small
replay drift could make their images disagree with the captured joint state.

Both formats use the official RoboDojo 14D order: left arm's six joints,
left gripper, right arm's six joints, right gripper.

Example::

    python scripts/dev/make_kong_joint_replay_to_lerobot_v30.py \\
      --src /cfs/users/kindel/robodojo_data/make_kong/joint_replay_all_2gpu_20260924_075401 \\
      --dst /path/to/RoboDojo_make_kong_joint_replay_v30

Use --max-episodes 1 with a temporary destination for an isolated smoke test.
No source files are modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from make_kong_npz_to_lerobot_v30 import (
    CAMERAS,
    FPS,
    TASK_TEXT,
    _convert_video,
    _write_tasks,
)


JOINT_NAMES = [
    *(f"left_joint_{i}" for i in range(7)),
    *(f"right_joint_{i}" for i in range(7)),
]
EPISODE_NAME = re.compile(r"(?:seed_)?(?P<seed>\d+)_layout_(?P<layout>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="joint replay capture root")
    parser.add_argument("--dst", type=Path, required=True, help="new LeRobot v3 dataset root")
    parser.add_argument("--max-episodes", type=int, default=None, help="first N episodes for a smoke check")
    parser.add_argument("--workers", type=int, default=4, help="parallel video validation and copy jobs")
    return parser.parse_args()


def _scalar(npz: np.lib.npyio.NpzFile, name: str, dtype: type) -> object:
    value = np.asarray(npz[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar, got {value.shape}")
    return dtype(value.item())


def _video_paths(folder: Path) -> dict[str, Path]:
    videos = {}
    for camera in CAMERAS:
        matches = list(folder.glob(f"episode_*_{camera}_success.mp4"))
        if len(matches) != 1 or matches[0].stat().st_size == 0:
            raise ValueError(f"Expected one nonempty {camera} success video in {folder}; got {matches}")
        videos[camera] = matches[0]
    return videos


def _episodes_v1(src: Path, max_episodes: int | None) -> list[dict]:
    paths = sorted(src.rglob("trajectory_joint.npz"))
    candidates = []
    seen = set()
    for path in paths:
        match = EPISODE_NAME.fullmatch(path.parent.parent.name)
        if match is None:
            raise ValueError(f"Expected <seed>_layout_<layout>/<run_id>/trajectory_joint.npz: {path}")
        seed, layout_id = int(match["seed"]), int(match["layout"])
        key = seed, layout_id
        if key in seen:
            raise ValueError(f"Multiple replay runs for seed/layout {key}; select a unique source root")
        seen.add(key)
        candidates.append((seed, layout_id, path))
    candidates.sort(key=lambda item: item[:2])
    if max_episodes is not None:
        candidates = candidates[:max_episodes]

    episodes = []
    for index, (seed, layout_id, path) in enumerate(candidates):
        with np.load(path, allow_pickle=False) as npz:
            if _scalar(npz, "seed", int) != seed or _scalar(npz, "layout_id", int) != layout_id:
                raise ValueError(f"Replay identity disagrees with path: {path}")
            if not _scalar(npz, "replay_valid", bool) or not _scalar(npz, "replay_success", bool):
                raise ValueError(f"Replay did not pass validity and success checks: {path}")
            if _scalar(npz, "schema_version", int) != 1:
                raise ValueError(f"Unsupported joint replay schema: {path}")
            states = np.asarray(npz["joint_states"], dtype=np.float32)
            actions = np.asarray(npz["joint_actions"], dtype=np.float32)
            length = len(actions)
            if length < 1 or states.shape != (length, 14) or actions.shape != (length, 14):
                raise ValueError(f"Expected nonempty joint_states/joint_actions (T,14): {path}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"Nonfinite joint state/action: {path}")
            ik_status = np.asarray(npz["ik_status"])
            if ik_status.dtype != np.dtype("bool") or ik_status.shape != (length, 2) or not ik_status.all():
                raise ValueError(f"All T x 2 IK solves must succeed: {path}")
            if (_scalar(npz, "source_length", int) != length
                    or not np.array_equal(npz["step_counts"], np.arange(length))
                    or npz["source_actions"].shape != (length, 16)
                    or npz["source_robot_states"].shape != (length, 16)):
                raise ValueError(f"Source EE arrays do not align with replay: {path}")
            source_actions = np.asarray(npz["source_actions"], dtype=np.float32)
            replay_states = np.asarray(npz["replay_robot_states"], dtype=np.float32)
            if (replay_states.shape != (length, 16)
                    or not np.array_equal(actions[:, [6, 13]], source_actions[:, [7, 15]])
                    or not np.array_equal(states[:, [6, 13]], replay_states[:, [7, 15]])):
                raise ValueError(f"Joint gripper slots do not match EE replay records: {path}")
            errors = {}
            for name in ("ee_pos_error_m", "ee_rot_error_deg", "object_pos_error_m", "object_rot_error_deg"):
                values = np.asarray(npz[name])
                if (values.ndim < 1 or values.shape[0] != length or values.size < 1
                        or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()):
                    raise ValueError(f"Missing or nonfinite replay error {name}: {path}")
                errors[f"max_{name}"] = float(np.max(values))
        episodes.append({
            "episode_index": index, "seed": seed, "layout_id": layout_id,
            "folder": path.parent, "run_id": path.parent.name, "length": length,
            "states": states, "actions": actions, "videos": _video_paths(path.parent),
            "errors": errors,
        })
    return episodes


def _production_video_paths(folder: Path) -> dict[str, Path]:
    videos = {}
    for camera in CAMERAS:
        path = folder / f"{camera}.mp4"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        videos[camera] = path
    return videos


def _episodes_v3(src: Path, paths: list[Path], max_episodes: int | None) -> list[dict]:
    if max_episodes is None:
        summary_path = src / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        if (summary.get("production_status") != "complete"
                or summary.get("sample_count") != len(paths)
                or summary.get("video_count") != 3 * len(paths)
                or not summary.get("all_verified_rollouts_accepted")):
            raise ValueError(f"Incomplete or inconsistent production summary: {summary_path}")

    candidates = []
    seen = set()
    for path in paths:
        folder = path.parent
        seed_match = re.fullmatch(r"seed_(\d+)", folder.parent.name)
        layout_match = re.fullmatch(r"layout_(\d+)", folder.name)
        if seed_match is None or layout_match is None:
            raise ValueError(f"Unexpected production path: {path}")
        seed, layout_id = int(seed_match[1]), int(layout_match[1])
        if (seed, layout_id) in seen:
            raise ValueError(f"Duplicate seed/layout {(seed, layout_id)} under {src}")
        seen.add((seed, layout_id))
        candidates.append((seed, layout_id, path))
    candidates.sort(key=lambda item: item[:2])
    if max_episodes is not None:
        candidates = candidates[:max_episodes]

    episodes = []
    for index, (seed, layout_id, path) in enumerate(candidates):
        folder = path.parent
        sample = json.loads((folder / "sample.json").read_text())
        verification = json.loads((folder / "verification.json").read_text())
        if (sample.get("seed") != seed or sample.get("layout_id") != layout_id
                or sample.get("status") != "success" or not sample.get("native_success")
                or not verification.get("accepted_joint_rollout")
                or not verification.get("joint_fields_valid")
                or not verification.get("joint_actions_match_applied_targets")
                or not verification.get("replay_step_counts_contiguous")
                or not verification.get("source_actions_complete")
                or not verification.get("actions_exactly_match_source")
                or not verification.get("native_success")
                or verification.get("terminal_reward") != 1.0):
            raise ValueError(f"Unaccepted or inconsistent sample: {folder}")

        with np.load(path, allow_pickle=False) as npz:
            if (_scalar(npz, "schema_version", int) != 3
                    or _scalar(npz, "eval_seed", int) != seed
                    or _scalar(npz, "layout_id", int) != layout_id):
                raise ValueError(f"Replay schema or identity mismatch: {path}")
            actions = np.asarray(npz["joint_actions"], dtype=np.float32)
            length = len(actions)
            arm_states = np.asarray(npz["arm_joint_states"], dtype=np.float32)
            arm_targets = np.asarray(npz["arm_joint_targets"], dtype=np.float32)
            robot_states = np.asarray(npz["robot_states"], dtype=np.float32)
            ee_actions = np.asarray(npz["actions"], dtype=np.float32)
            if (length < 1 or actions.shape != (length, 14)
                    or arm_states.shape != (length, 12)
                    or arm_targets.shape != (length, 12)
                    or robot_states.shape != (length, 16)
                    or ee_actions.shape != (length, 16)):
                raise ValueError(f"Joint replay shapes do not align: {path}")
            states = np.column_stack((arm_states[:, :6], robot_states[:, 7],
                                      arm_states[:, 6:], robot_states[:, 15])).astype(np.float32)
            expected_actions = np.column_stack((arm_targets[:, :6], ee_actions[:, 7],
                                                arm_targets[:, 6:], ee_actions[:, 15]))
            if (not np.isfinite(states).all() or not np.isfinite(actions).all()
                    or not np.isfinite(expected_actions).all()
                    or not np.array_equal(actions, expected_actions)
                    or np.any(states[:, [6, 13]] < 0) or np.any(states[:, [6, 13]] > 1)
                    or np.any(actions[:, [6, 13]] < 0) or np.any(actions[:, [6, 13]] > 1)):
                raise ValueError(f"Invalid joint state or applied target: {path}")
            if (sample.get("steps") != length
                    or verification.get("replay_steps") != length
                    or _scalar(npz, "source_action_count", int) != length
                    or not _scalar(npz, "source_actions_complete", bool)
                    or not np.array_equal(npz["step_counts"], np.arange(length))
                    or not _scalar(npz, "success", bool)
                    or not _scalar(npz, "environment_success", bool)
                    or _scalar(npz, "unstable", bool)
                    or _scalar(npz, "terminal_remaining_checks", int) != 0
                    or _scalar(npz, "terminal_reward", float) != 1.0
                    or abs(_scalar(npz, "control_dt_seconds", float) - 1 / FPS) > 1e-6):
                raise ValueError(f"Replay timing or success mismatch: {path}")
            if _scalar(npz, "run_id", str) != sample.get("replay_run_id"):
                raise ValueError(f"Replay run ID mismatch: {path}")
        episodes.append({
            "episode_index": index, "seed": seed, "layout_id": layout_id,
            "folder": folder, "run_id": sample["replay_run_id"], "length": length,
            "states": states, "actions": actions,
            "videos": _production_video_paths(folder), "errors": {},
        })
    return episodes


def _episodes(src: Path, max_episodes: int | None) -> list[dict]:
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    production_paths = sorted((src / "success").glob("seed_*/layout_*/trajectory.npz"))
    if production_paths:
        return _episodes_v3(src, production_paths, max_episodes)
    if not next(src.rglob("trajectory_joint.npz"), None):
        raise ValueError(f"No supported joint replay trajectories under {src}")
    return _episodes_v1(src, max_episodes)


def _modality() -> dict:
    def arm_fields(original_key: str) -> dict:
        fields = {}
        for name, start, end in (
            ("left_arm_joint", 0, 6), ("left_gripper", 6, 7),
            ("right_arm_joint", 7, 13), ("right_gripper", 13, 14),
        ):
            fields[name] = {
                "original_key": original_key, "start": start, "end": end,
                "dtype": "float32", "absolute": True, "rotation_type": None,
            }
        return fields

    return {
        "state": arm_fields("observation.state"),
        "action": arm_fields("action"),
        "video": {key.rsplit(".", 1)[-1]: {"original_key": key}
                  for key in CAMERAS.values()},
        "annotation": {"human.task": {"original_key": "task_index"}},
    }


def _write_dataset(dst: Path, src: Path, episodes: list[dict]) -> int:
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
        length = ep["length"]
        index = ep["episode_index"]
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
            "run_id": ep["run_id"], "source": str(ep["folder"].relative_to(src)),
            "length": length, **ep["errors"],
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
    with (meta_dir / "episode_sources.jsonl").open("w") as file:
        for record in provenance:
            file.write(json.dumps(record) + "\n")
    return total_frames


def main() -> int:
    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("Missing required executable: ffprobe")
    episodes = _episodes(src, args.max_episodes)
    print(f"Checked {len(episodes)} validated joint replays; preparing {dst}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.parent / f".{dst.name}.partial-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    try:
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
                    print(f"Videos {completed}/{len(futures)}", flush=True)
        total_frames = _write_dataset(staging, src, episodes)
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
