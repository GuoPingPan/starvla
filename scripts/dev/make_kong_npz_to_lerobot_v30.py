#!/usr/bin/env python3
"""Convert successful make_kong replay rollouts into a LeRobot v3 dataset.

Only ``<src>/success/seed_*/layout_*/`` is read. The production tree also
contains ``raw_eval_result`` hard links to the same files; reading both would
duplicate every episode. A rollout has T action-before-state rows and T+1
video frames (the last frame is the terminal observation). The output keeps
the source videos byte-for-byte, while episode metadata exposes only T
training rows and timestamps 0 through (T-1)/25. Frame t is aligned to the
state observed immediately before action t; the terminal frame is never
requested by the training loader. Keeping the complete H264 stream avoids
cutting a required B-frame reference. Actions are the actual 16D EE commands
sent to the simulator; they are not replaced with the next observed state.

Example:
    python scripts/dev/make_kong_npz_to_lerobot_v30.py \
        --src /cfs/users/kindel/robodojo_data/make_kong/official_push_postures_v2_all_4gpu_20260923_101009 \
        --dst /mnt/cfs/34vghc/data/robodojo/RoboDojo_make_kong_v30

For a quick isolated check, add ``--max-episodes 1`` and choose an empty
temporary destination. The source tree is never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


FPS = 25
TASK_TEXT = "Wait for the opponent to discard a tile, then declare a kong with the matching tiles."
CAMERAS = {
    "cam_head": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
STATE_NAMES = [
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="completed production root")
    parser.add_argument("--dst", type=Path, required=True, help="new LeRobot v3 dataset root")
    parser.add_argument("--max-episodes", type=int, default=None, help="convert the first N episodes for a smoke check")
    parser.add_argument("--workers", type=int, default=4, help="parallel video copy and validation jobs")
    return parser.parse_args()


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def _probe_video(path: Path) -> dict:
    data = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]))
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    return streams[0]


def _check_video(stream: dict, frame_count: int, path: Path) -> None:
    expected = {
        "codec_name": "h264", "pix_fmt": "yuv420p", "width": 640, "height": 480,
        "r_frame_rate": "25/1", "avg_frame_rate": "25/1", "nb_frames": str(frame_count),
    }
    mismatches = {key: (stream.get(key), value) for key, value in expected.items()
                  if stream.get(key) != value}
    if mismatches:
        raise ValueError(f"Video metadata mismatch for {path}: {mismatches}")


def _check_video_pts(path: Path, frame_count: int) -> None:
    """Verify that loader requests 0..T-1 map to contiguous source frames."""
    packet_pts = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path),
    ])
    pts = sorted(float(line) for line in packet_pts.splitlines() if line.strip())
    if len(pts) != frame_count or any(abs(value - index / FPS) > 1e-4
                                      for index, value in enumerate(pts)):
        raise ValueError(f"Video PTS do not cover contiguous 25 fps frames: {path}")


def _episodes(src: Path, max_episodes: int | None) -> list[dict]:
    success_dir = src / "success"
    if not success_dir.is_dir():
        raise FileNotFoundError(f"Missing success directory: {success_dir}")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    folders = sorted(success_dir.glob("seed_*/layout_*"),
                     key=lambda p: (int(p.parent.name.removeprefix("seed_")),
                                    int(p.name.removeprefix("layout_"))))
    if not folders:
        raise ValueError(f"No successful episodes under {success_dir}")
    if max_episodes is None:
        summary_path = src / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete" or summary.get("success_samples") != len(folders):
                raise ValueError("Production summary is incomplete or disagrees with success directories")
    else:
        folders = folders[:max_episodes]

    episodes = []
    for episode_index, folder in enumerate(folders):
        if not folder.is_dir():
            raise ValueError(f"Not an episode directory: {folder}")
        seed = int(folder.parent.name.removeprefix("seed_"))
        layout_id = int(folder.name.removeprefix("layout_"))
        sample = json.loads((folder / "sample.json").read_text())
        if (sample.get("status") != "success" or sample.get("seed") != seed
                or sample.get("layout_id") != layout_id):
            raise ValueError(f"Sample identity mismatch: {folder}")
        with np.load(folder / "trajectory.npz", allow_pickle=False) as npz:
            states = np.asarray(npz["robot_states"], dtype=np.float32)
            actions = np.asarray(npz["actions"], dtype=np.float32)
            length = len(actions)
            if (int(npz["schema_version"]) != 2 or states.shape != (length, 16)
                    or actions.shape != (length, 16) or length < 1):
                raise ValueError(f"Invalid state/action schema: {folder}")
            if (int(npz["eval_seed"]) != seed or int(npz["layout_id"]) != layout_id
                    or not bool(npz["success"]) or not bool(npz["environment_success"])
                    or bool(npz["unstable"]) or int(npz["terminal_remaining_checks"]) != 0):
                raise ValueError(f"NPZ identity or success mismatch: {folder}")
            if sample.get("steps") != length or not np.array_equal(npz["step_counts"], np.arange(length)):
                raise ValueError(f"Step count mismatch: {folder}")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"Nonfinite state/action: {folder}")
        for camera in CAMERAS:
            path = folder / f"{camera}.mp4"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        episodes.append({
            "episode_index": episode_index, "seed": seed, "layout_id": layout_id,
            "folder": folder, "length": length, "states": states, "actions": actions,
        })
    return episodes


def _convert_video(job: tuple[Path, Path, int]) -> None:
    source, output, length = job
    _check_video(_probe_video(source), length + 1, source)
    _check_video_pts(source, length + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    with source.open("rb") as source_file, output.open("wb") as output_file:
        while chunk := source_file.read(1024 * 1024):
            source_hash.update(chunk)
            output_file.write(chunk)
    output_hash = hashlib.sha256()
    with output.open("rb") as output_file:
        while chunk := output_file.read(1024 * 1024):
            output_hash.update(chunk)
    if output_hash.digest() != source_hash.digest():
        raise ValueError(f"Copied video differs from source: {source}")
    _check_video(_probe_video(output), length + 1, output)


def _write_tasks(path: Path) -> None:
    """Write the same pandas-index shape used by official RoboDojo tasks.parquet."""
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


def _write_dataset(dst: Path, episodes: list[dict], src: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_frames = sum(episode["length"] for episode in episodes)
    all_states = np.concatenate([episode["states"] for episode in episodes])
    all_actions = np.concatenate([episode["actions"] for episode in episodes])
    fixed16 = pa.list_(pa.field("element", pa.float32(), nullable=False), 16)
    row_arrays = [
        pa.FixedSizeListArray.from_arrays(pa.array(all_states.ravel(), type=pa.float32()), 16),
        pa.FixedSizeListArray.from_arrays(pa.array(all_actions.ravel(), type=pa.float32()), 16),
    ]
    row_fields = [pa.field("observation.state", fixed16), pa.field("action", fixed16)]
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
        row_arrays.append(pa.array(values, type=dtype))
        row_fields.append(pa.field(name, dtype))
    data_dir = dst / "data/chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_arrays(row_arrays, schema=pa.schema(row_fields)),
                   data_dir / "file-000.parquet")

    ep_rows = []
    provenance = []
    start = 0
    for episode in episodes:
        length = episode["length"]
        index = episode["episode_index"]
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
        ep_rows.append(row)
        provenance.append({
            "episode_index": index, "seed": episode["seed"],
            "layout_id": episode["layout_id"],
            "source": str(episode["folder"].relative_to(src)), "length": length,
        })
        start += length
    episodes_dir = dst / "meta/episodes/chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), episodes_dir / "file-000.parquet")

    meta_dir = dst / "meta"
    _write_tasks(meta_dir / "tasks.parquet")
    template = (Path(__file__).resolve().parents[2] / "examples/simBenchmarks/RoboDojo"
                / "train_files/data_registry/robodojo_modality_template.json")
    shutil.copyfile(template, meta_dir / "modality.json")
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
            "observation.state": {"dtype": "float32", "shape": [16], "names": [STATE_NAMES]},
            "action": {"dtype": "float32", "shape": [16], "names": [STATE_NAMES]},
            **{key: video_info for key in CAMERAS.values()},
            **{key: {"dtype": dtype, "shape": [1], "names": None} for key, dtype in [
                ("timestamp", "float32"), ("frame_index", "int64"),
                ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64"),
            ]},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    with (meta_dir / "episode_sources.jsonl").open("w") as f:
        for record in provenance:
            f.write(json.dumps(record) + "\n")
    return total_frames


def main() -> int:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for program in ("ffprobe",):
        if shutil.which(program) is None:
            raise RuntimeError(f"Missing required executable: {program}")
    episodes = _episodes(src, args.max_episodes)
    print(f"Checked {len(episodes)} successful episodes; preparing {dst}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = dst.parent / f".{dst.name}.partial-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    try:
        jobs = []
        for episode in episodes:
            index = episode["episode_index"]
            for camera, video_key in CAMERAS.items():
                source = episode["folder"] / f"{camera}.mp4"
                output = staging / "videos" / video_key / "chunk-000" / f"file-{index:03d}.mp4"
                jobs.append((source, output, episode["length"]))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_convert_video, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 60 == 0 or completed == len(futures):
                    print(f"Videos {completed}/{len(futures)}", flush=True)
        total_frames = _write_dataset(staging, episodes, src)
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
