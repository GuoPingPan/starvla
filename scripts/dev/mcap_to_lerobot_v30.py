#!/usr/bin/env python3
"""Convert RoboDojo foxglove-MCAP collected episodes to a LeRobot v3.0 dataset.

Input layout (per task):
    <src>/<task>/<episode_dir>/<episode_dir>.mcap + metadata.json

Output: LeRobot v3.0 dataset identical in layout/convention to the official
``RoboDojo_ee_lerobot_v30_video`` release:

- 16D state/action = [L: pos3 + quat_wxyz4 + grip1] + [R: same]
  (mcap poses are foxglove x,y,z,qx,qy,qz,qw -> reorder to wxyz;
   gripper = effector joint_state "gripper" position, [0,1], 1=open)
- 3 videos: cam_high_0 -> cam_high, cam_wrist_0 -> cam_left_wrist,
  cam_wrist_1 -> cam_right_wrist (H264 remuxed to mp4, no re-encode)
- videos/<key>/chunk-XXX/file-YYY.mp4, one file per episode (file_index = episode_index)
- data/chunk-000/file-000.parquet (all episodes), meta/{info.json, tasks.parquet,
  episodes/chunk-000/file-000.parquet}

Task texts are canonicalized to the official RoboDojo task strings so that
language prompts match across datasets.

Usage:
    python mcap_to_lerobot_v30.py --src /mnt/cfs/robodojo_expert_mcap_100 \
        --dst /root/data/robodojo/RoboDojo_gen_mcap100_4task_v30 \
        --task stack_blocks:29 --task cover_blocks:3 \
        --task deposit_coin:4 --task fold_clothes:9 \
        --official-root /root/data/robodojo/RoboDojo_ee_lerobot_v30_video \
        --workers 32
The per-task ":N" suffix is the official task_index used only to look up the
canonical task text from the official dataset metadata; output task_index is
re-numbered 0..N-1 in the order given.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CAM_MAP = {
    "observation/images/cam_high_0/color_h264": "observation.images.cam_high",
    "observation/images/cam_wrist_0/color_h264": "observation.images.cam_left_wrist",
    "observation/images/cam_wrist_1/color_h264": "observation.images.cam_right_wrist",
}
STATE_TOPICS = ("observation/state/end_0/pose", "observation/state/end_1/pose")
ACTION_TOPICS = ("action/end_0/pose", "action/end_1/pose")
GRIP_S_TOPICS = ("observation/state/effector_0/joint_state", "observation/state/effector_1/joint_state")
GRIP_A_TOPICS = ("action/effector_0/joint_state", "action/effector_1/joint_state")

EPS_PER_DATA_FILE = 1000  # rows chunking for data parquet (single file up to this)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="root containing <task>/<episode_dir>/")
    p.add_argument("--dst", required=True, help="output dataset root")
    p.add_argument("--task", action="append", required=True,
                   help="task_name[:official_task_index], repeatable")
    p.add_argument("--official-root", default="/root/data/robodojo/RoboDojo_ee_lerobot_v30_video",
                   help="official dataset used to canonicalize task texts")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--align-action", action="store_true", default=False,
                   help="action[t] := state[t+1] (官方语义; 修正 mcap action 滞后)")
    p.add_argument("--exclude", action="append", default=[],
                   help="跳过的 episode 目录名 (可多次, 异常检测剔除用)")
    return p.parse_args()


def official_task_texts(official_root: str) -> dict[int, str]:
    import pyarrow.parquet as pq

    df = pq.read_table(Path(official_root) / "meta/episodes/chunk-000/file-000.parquet",
                       columns=["episode_index", "tasks"]).to_pandas()
    texts = {}
    for _, r in df.iterrows():
        t = r["tasks"][0] if isinstance(r["tasks"], (list, tuple, np.ndarray)) else str(r["tasks"])
        texts.setdefault(int(r["episode_index"]) // 100, t)
    return texts


def read_episode(mcap_path: str) -> dict:
    """Read one mcap; return dict with state(16), action(16), video chunk lists, ts0_ns."""
    from mcap.reader import make_reader

    poses = {t: [] for t in STATE_TOPICS + ACTION_TOPICS}
    grips = {t: [] for t in GRIP_S_TOPICS + GRIP_A_TOPICS}
    videos = {t: [] for t in CAM_MAP}
    times = []
    with open(mcap_path, "rb") as f:
        for _schema, ch, msg in make_reader(f).iter_messages():
            t = ch.topic
            if t in poses:
                d = json.loads(msg.data)["pose"]
                q = d["orientation"]
                poses[t].append([d["position"]["x"], d["position"]["y"], d["position"]["z"],
                                 q["w"], q["x"], q["y"], q["z"]])
                times.append(msg.log_time)
            elif t in grips:
                grips[t].append(json.loads(msg.data)["joints"][0]["position"])
            elif t in videos:
                videos[t].append(base64.b64decode(json.loads(msg.data)["data"]))

    def arm(side):  # side 0=left,1=right
        st = np.asarray(poses[STATE_TOPICS[side]], dtype=np.float64)
        ac = np.asarray(poses[ACTION_TOPICS[side]], dtype=np.float64)
        gs = np.asarray(grips[GRIP_S_TOPICS[side]], dtype=np.float64)
        ga = np.asarray(grips[GRIP_A_TOPICS[side]], dtype=np.float64)
        state = np.concatenate([st, gs[:, None]], axis=1)
        action = np.concatenate([ac, ga[:, None]], axis=1)
        return state, action

    s0, a0 = arm(0)
    s1, a1 = arm(1)
    state = np.concatenate([s0, s1], axis=1)
    action = np.concatenate([a0, a1], axis=1)
    return {"state": state, "action": action, "videos": videos,
            "ts0_ns": times[0] if times else 0}


def remux_h264(chunks: list[bytes], dst_mp4: Path, fps: int = 25) -> int:
    """Concatenate annex-b H264 NAL chunks, remux to mp4 (no re-encode). Return n frames."""
    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as tf:
        raw_path = tf.name
    try:
        with open(raw_path, "wb") as f:
            for c in chunks:
                f.write(c)
        dst_mp4.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts",
               "-r", str(fps), "-i", raw_path, "-c", "copy", str(dst_mp4)]
        subprocess.run(cmd, check=True)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(dst_mp4)],
            capture_output=True, text=True, check=True)
        return int(probe.stdout.strip())
    finally:
        os.unlink(raw_path)


def convert_one(job: dict) -> dict:
    """Worker: convert a single episode. Returns row-level records + wrote video files."""
    try:
        ep = read_episode(job["mcap"])
        n = len(ep["state"])
        ok = (n == len(ep["action"])
              and all(len(ep["videos"][t]) == n for t in CAM_MAP)
              and n == job["frame_num"])
        if not ok:
            return {"ok": False, "ep": job["ep_dir"], "reason": f"frame mismatch rows={n} meta={job['frame_num']}"}

        # action 对齐: mcap 的 action topic 是滞后的控制指令 (旋转平均滞后 ~7°);
        # 官方 RoboDojo 语义是 action[t] == state[t+1] (下一帧目标, 误差 0.03mm)。
        # 用 state[t+1] 整列 (16D: 位置+四元数+夹爪, 双臂) 重写 action, 末帧沿用原值。
        if job.get("align_action", False):
            action = ep["action"].copy()
            action[:-1] = ep["state"][1:]
            ep["action"] = action

        # 四元数重归一 (mcap 源个别帧 |q| 低至 0.9974; 几何等价, 保证单位范数)
        for arr in (ep["state"], ep["action"]):
            for sl in (slice(3, 7), slice(11, 15)):
                q = arr[:, sl]
                arr[:, sl] = q / np.linalg.norm(q, axis=1, keepdims=True)

        for topic, vkey in CAM_MAP.items():
            dst = Path(job["dst"]) / "videos" / vkey / f"chunk-{job['video_chunk']:03d}" / f"file-{job['episode_index']:03d}.mp4"
            if dst.exists() and not job["overwrite"]:
                continue
            nf = remux_h264(ep["videos"][topic], dst)
            if nf != n:
                return {"ok": False, "ep": job["ep_dir"], "reason": f"{vkey} decoded {nf} != {n}"}

        return {
            "ok": True, "ep": job["ep_dir"], "episode_index": job["episode_index"],
            "task_index": job["task_index"], "state": ep["state"].astype(np.float32),
            "action": ep["action"].astype(np.float32), "n": n,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "ep": job["ep_dir"], "reason": repr(e)}


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists() and not args.overwrite:
        print(f"ERROR: {dst} already exists (use --overwrite)"); sys.exit(1)

    off_texts = official_task_texts(args.official_root)
    tasks = []  # (task_name, out_task_index, canonical_text)
    for i, spec in enumerate(args.task):
        name, _, off_idx = spec.partition(":")
        text = off_texts[int(off_idx)] if off_idx else name
        tasks.append((name, i, text))
    for name, i, text in tasks:
        print(f"task[{i}] {name}: {text[:70]}")

    jobs, skipped = [], []
    for name, tidx, _ in tasks:
        tdir = src / name
        if not tdir.is_dir():
            print(f"ERROR: missing task dir {tdir}"); sys.exit(1)
        for ep_dir in sorted(tdir.iterdir()):
            if not ep_dir.is_dir():
                continue
            meta_p = ep_dir / "metadata.json"
            mcaps = glob.glob(str(ep_dir / "*.mcap"))
            if not meta_p.exists() or not mcaps:
                skipped.append((str(ep_dir), "missing metadata.json or .mcap")); continue
            meta = json.loads(meta_p.read_text())
            if ep_dir.name in set(args.exclude):
                skipped.append((str(ep_dir), "excluded (anomaly)")); continue
            if meta.get("instruction") not in [t[2] for t in tasks]:
                skipped.append((str(ep_dir), f"instruction mismatch: {meta.get('instruction')!r}")); continue
            jobs.append({"ep_dir": ep_dir.name, "mcap": mcaps[0], "frame_num": int(meta["frame_num"]),
                         "task_index": tidx, "dst": str(dst),
                         "align_action": bool(getattr(args, "align_action", False))})
    print(f"episodes to convert: {len(jobs)}, pre-skipped: {len(skipped)}")

    # assign episode indices grouped by task order for contiguous task blocks
    jobs.sort(key=lambda j: (j["task_index"], j["ep_dir"]))
    for i, j in enumerate(jobs):
        j["episode_index"] = i
        j["video_chunk"] = i // 1000

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 50 == 0 or not r["ok"]:
                print(f"[{done}/{len(jobs)}] ok" if r["ok"] else f"[{done}/{len(jobs)}] FAIL {r['ep']}: {r.get('reason')}"[:200])

    good = sorted([r for r in results if r["ok"]], key=lambda r: r["episode_index"])
    bad = [r for r in results if not r["ok"]]
    for r in bad:
        print(f"SKIP {r['ep']}: {r['reason']}")
    for s in skipped:
        print(f"SKIP {s[0]}: {s[1]}")
    if not good:
        print("ERROR: no episodes converted"); sys.exit(1)

    # ---------------- write dataset ----------------
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_frames = sum(r["n"] for r in good)
    n_eps = len(good)
    print(f"writing dataset: {n_eps} episodes, {total_frames} frames -> {dst}")

    data_dir = dst / "data/chunk-000"; data_dir.mkdir(parents=True, exist_ok=True)
    ep_rows, rows = [], []
    gidx = 0
    for r in good:
        n, ei, ti = r["n"], r["episode_index"], r["task_index"]
        ts = (np.arange(n, dtype=np.float32) / 25.0).astype(np.float32)
        rows.append(pd.DataFrame({
            "observation.state": [np.asarray(v, dtype=np.float32) for v in r["state"]],
            "action": [np.asarray(v, dtype=np.float32) for v in r["action"]],
            "timestamp": ts,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, dtype=np.int64),
            "index": np.arange(gidx, gidx + n, dtype=np.int64),
            "task_index": np.full(n, ti, dtype=np.int64),
        }))
        task_text = [t[2] for t in tasks if t[1] == ti][0]
        ep_rows.append({
            "episode_index": ei, "tasks": [task_text], "length": n,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": gidx, "dataset_to_index": gidx + n,
            **{f"videos/{v}/chunk_index": 0 for v in CAM_MAP.values()},
            **{f"videos/{v}/file_index": ei for v in CAM_MAP.values()},
            **{f"videos/{v}/from_timestamp": 0.0 for v in CAM_MAP.values()},
            **{f"videos/{v}/to_timestamp": n / 25.0 for v in CAM_MAP.values()},
        })
        gidx += n
    df = pd.concat(rows, ignore_index=True)

    def to_arrow(d: pd.DataFrame) -> pa.Table:
        tbl = pa.Table.from_pandas(d, preserve_index=False)
        # force fixed_size_list<float>[16] for state/action like official release
        fields = []
        arrs = []
        for f, col in zip(tbl.schema, tbl.columns):
            if f.name in ("observation.state", "action"):
                lst = col.combine_chunks()
                flat = lst.flatten().cast(pa.float32())
                arrs.append(pa.FixedSizeListArray.from_arrays(flat, 16))
                fields.append(pa.field(f.name, pa.list_(pa.field("element", pa.float32(), nullable=False), 16)))
            else:
                arrs.append(col.combine_chunks())
                fields.append(f)
        return pa.Table.from_arrays(arrs, schema=pa.schema(fields))

    pq.write_table(to_arrow(df), data_dir / "file-000.parquet")

    eps_df = pd.DataFrame(ep_rows)
    eps_dir = dst / "meta/episodes/chunk-000"; eps_dir.mkdir(parents=True, exist_ok=True)
    eps_df.to_parquet(eps_dir / "file-000.parquet", index=False)

    # tasks.parquet: index = task text, column task_index (official style)
    tasks_df = pd.DataFrame({"task_index": [t[1] for t in tasks]},
                            index=pd.Index([t[2] for t in tasks], name=None))
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    tasks_df.to_parquet(dst / "meta/tasks.parquet")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unified_robot",
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 25,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16], "names": None},
            "action": {"dtype": "float32", "shape": [16], "names": None},
            **{v: {"dtype": "video", "shape": [3, 480, 640], "names": None} for v in CAM_MAP.values()},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    print(f"DONE: {n_eps} episodes ({len(bad)} failed, {len(skipped)} pre-skipped), "
          f"{total_frames} frames -> {dst}")


if __name__ == "__main__":
    main()
