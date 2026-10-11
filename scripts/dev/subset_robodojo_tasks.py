#!/usr/bin/env python3
"""Physically subset the official RoboDojo EE dataset to selected tasks.

Creates a new LeRobot v3.0 dataset (parquet rows copied, task_index renumbered
to 0..N-1, videos symlinked to the official release) so it can be mixed with
generated datasets without yaml-level task filtering (data_cfg is global across
a mixture, per-dataset filtering is not supported).

Usage:
    python subset_robodojo_tasks.py --official /root/data/robodojo/RoboDojo_ee_lerobot_v30_video \
        --dst /root/data/robodojo/RoboDojo_ee_4task_v30 --task-index 3 --task-index 4 \
        --task-index 9 --task-index 29 [--link-prefix /mnt/cfs/data/robodojo]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--task-index", type=int, action="append", required=True)
    ap.add_argument("--link-prefix", default="",
                    help="rewrite symlink targets to start with this prefix "
                         "(e.g. /mnt/cfs/data/robodojo) so links work on cluster pods")
    args = ap.parse_args()

    off, dst = Path(args.official), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"ERROR: {dst} already exists")
    want = sorted(args.task_index)

    eps = pd.read_parquet(off / "meta/episodes/chunk-000/file-000.parquet")
    # select by task block: episode_index // 100 == official task_index
    eps = eps[eps["episode_index"].map(lambda e: e // 100).isin(want)].reset_index(drop=True)
    assert len(eps) == 100 * len(want), f"expected {100*len(want)} episodes, got {len(eps)}"

    remap = {t: i for i, t in enumerate(want)}
    texts = {r["episode_index"] // 100: (r["tasks"][0] if isinstance(r["tasks"], (list, tuple, np.ndarray)) else str(r["tasks"]))
             for _, r in eps.iterrows()}

    # ---- data parquet ----
    import glob
    dfs = []
    for f in sorted(glob.glob(str(off / "data/chunk-*/*.parquet"))):
        d = pd.read_parquet(f)
        dfs.append(d[d["episode_index"].map(lambda e: e // 100).isin(want)])
    df = pd.concat(dfs, ignore_index=True).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    df["task_index"] = df["episode_index"].map(lambda e: remap[e // 100]).astype("int64")
    df["index"] = np.arange(len(df), dtype=np.int64)
    print(f"data rows: {len(df)} over {df['episode_index'].nunique()} episodes")

    (dst / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    # keep fixed_size_list<float>[16] typing identical to official
    arrs, fields = [], []
    for f, col in zip(tbl.schema, tbl.columns):
        if f.name in ("observation.state", "action"):
            lst = col.combine_chunks()
            arrs.append(pa.FixedSizeListArray.from_arrays(lst.flatten().cast(pa.float32()), 16))
            fields.append(pa.field(f.name, pa.list_(pa.field("element", pa.float32(), nullable=False), 16)))
        else:
            arrs.append(col.combine_chunks()); fields.append(f)
    pq.write_table(pa.Table.from_arrays(arrs, schema=pa.schema(fields)), dst / "data/chunk-000/file-000.parquet")

    # ---- episodes meta ----
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    keep_cols = (["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
                  "dataset_from_index", "dataset_to_index"]
                 + [c for c in eps.columns if c.startswith("videos/") and
                    (c.endswith("/chunk_index") or c.endswith("/file_index") or
                     c.endswith("/from_timestamp") or c.endswith("/to_timestamp"))])
    eps_out = eps[keep_cols].copy()
    eps_out["data/chunk_index"] = 0
    eps_out["data/file_index"] = 0
    (dst / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    eps_out.to_parquet(dst / "meta/episodes/chunk-000/file-000.parquet", index=False)

    # ---- tasks.parquet (official pandas style: index=task text, col task_index) ----
    tasks_df = pd.DataFrame({"task_index": list(range(len(want)))},
                            index=pd.Index([texts[t] for t in want]))
    tasks_df.to_parquet(dst / "meta/tasks.parquet")

    # ---- info.json ----
    info = json.loads((off / "meta/info.json").read_text())
    info.update({"total_episodes": int(len(eps_out)), "total_frames": int(len(df)),
                 "total_tasks": len(want), "splits": {"train": f"0:{len(eps_out)}"}})
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))

    # ---- video symlinks ----
    off_prefix = str(off)
    target_prefix = args.link_prefix.rstrip("/") + "/" + off.name if args.link_prefix else off_prefix
    n_links = 0
    for _, r in eps_out.iterrows():
        for col_prefix in sorted({c[:-len("/file_index")] for c in eps_out.columns
                                  if c.startswith("videos/") and c.endswith("/file_index")}):
            key = col_prefix[len("videos/"):]
            src = Path(target_prefix) / "videos" / key / f"chunk-{int(r[f'{col_prefix}/chunk_index']):03d}" / f"file-{int(r[f'{col_prefix}/file_index']):03d}.mp4"
            # destination keeps the SAME chunk/file index -> reader finds it via episodes meta
            ldir = dst / "videos" / key / f"chunk-{int(r[f'{col_prefix}/chunk_index']):03d}"
            ldir.mkdir(parents=True, exist_ok=True)
            link = ldir / f"file-{int(r[f'{col_prefix}/file_index']):03d}.mp4"
            if not link.exists():
                link.symlink_to(src)
                n_links += 1
    print(f"video symlinks: {n_links}")
    print(f"DONE -> {dst}")


if __name__ == "__main__":
    main()
