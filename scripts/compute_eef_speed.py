"""Offline computation of per-episode average end-effector (EEF) translational speed.

Produces a speed-conditioning signal for training (mirrors the spirit of pi0.7's
episode-metadata conditioning, but uses the *physical* average EEF speed instead
of episode length). Training only *loads* the result; no computation happens on
the training path.

Definition
----------
For each episode, take the EEF xyz trajectory ``p[0..N-1]`` read from the RAW
parquet (before any normalization / delta / relative-pose transform) at the
dataset control rate ``fps``:

    avg_eef_speed = mean_t( ||p[t+1] - p[t]||_2 ) * fps      [m/s for UMI]

i.e. average instantaneous translational speed. Rotation and gripper are
intentionally excluded (different physical units).

Binning
-------
Supports two modes (auto-chosen by number of --dataset-path):
- single dataset        -> per-dataset quantile bins
- multiple datasets     -> GLOBAL quantile bins pooled across all datasets, so
  ``<speed:N>`` means the same physical speed for every dataset of the same
  embodiment (required for a mixture of same-robot datasets).

Outputs (written into each dataset's ``meta/`` dir)
---------------------------------------------------
- ``meta/episode_speed.json``  (source of truth the loader reads):
    {
      "fps": 60.0, "eef_col": "observation.state.end_0_pose", "xyz_indices":[0,1,2],
      "num_bins": 10, "bin_method":"quantile_global", "bin_edges":[...],
      "episodes": {"0": {"raw":0.031, "bin":3}, ...}
    }
- ``meta/episodes.jsonl`` (v2.0 only): each line additionally gets
  ``avg_eef_speed`` and ``speed_bin`` for human inspection.

Usage
-----
  # single dataset
  python scripts/compute_eef_speed.py --dataset-path <ds>
  # whole mixture (global bins across all listed datasets)
  python scripts/compute_eef_speed.py \
      --dataset-path <ds1> --dataset-path <ds2> ... [--num-bins 10]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

EPISODE_SPEED_FILENAME = "episode_speed.json"
EPISODES_FILENAME = "episodes.jsonl"
INFO_FILENAME = "info.json"
MODALITY_FILENAME = "modality.json"
DATA_GLOB = "data/*/*.parquet"

DEFAULT_EEF_COL = "observation.state"
DEFAULT_FALLBACK_COL = "action"
DEFAULT_XYZ = [0, 1, 2]


def _resolve_fps(meta_dir: Path) -> float:
    info = json.loads((meta_dir / INFO_FILENAME).read_text())
    for _, feat in info.get("features", {}).items():
        if not isinstance(feat, dict):
            continue
        vi = feat.get("video_info", {})
        if "video.fps" in vi:
            return float(vi["video.fps"])
    return float(info.get("fps", 20.0))


def _autodetect_eef_keys(meta_dir: Path):
    """Resolve (column, [i,j,k]) from the dataset's own meta/modality.json.

    Strategy (first hit wins):
    1. state sub-key whose name contains "position" and spans 3 dims
       (e.g. UMI ``end_0_position`` -> original_key ``observation.state.end_0_pose``).
    2. explicit ``x``/``y``/``z`` sub-keys (e.g. LIBERO).
    Returns None if undeterminable.
    """
    path = meta_dir / MODALITY_FILENAME
    if not path.exists():
        return None
    try:
        mod = json.loads(path.read_text())
    except Exception:
        return None
    state = mod.get("state", {})
    if not isinstance(state, dict):
        return None
    # (1) a "*position*" 3-dim sub-key
    for name, spec in state.items():
        if (
            isinstance(spec, dict)
            and "position" in name.lower()
            and (spec.get("end", 0) - spec.get("start", 0)) == 3
        ):
            col = spec.get("original_key") or DEFAULT_EEF_COL
            s = int(spec["start"])
            return col, [s, s + 1, s + 2]
    # (2) x / y / z
    xyz = {n: state[n] for n in ("x", "y", "z") if isinstance(state.get(n), dict)}
    if len(xyz) == 3:
        try:
            idx = [int(xyz[k]["start"]) for k in ("x", "y", "z")]
        except KeyError:
            return None
        col = None
        for spec in xyz.values():
            if spec.get("original_key"):
                col = spec["original_key"]
                break
        return (col or DEFAULT_EEF_COL, idx)
    return None


def _load_episodes_meta(meta_dir: Path):
    path = meta_dir / EPISODES_FILENAME
    if not path.exists():
        return None
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_eef_table(dataset_path: Path, eef_col: str, fallback_col: str, xyz_idx):
    """Pool all parquet rows of a dataset, return (episode_ids, xyz (N,3)) sorted
    by (episode_index, frame_index)."""
    files = sorted(glob.glob(str(dataset_path / DATA_GLOB)))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_path / DATA_GLOB}")
    eps, xyzs = [], []
    for pf in files:
        cols = set(pq.ParquetFile(pf).schema_arrow.names)
        col = eef_col if eef_col in cols else (fallback_col if fallback_col in cols else None)
        if col is None:
            continue
        need = [c for c in (col, "episode_index", "frame_index") if c in cols]
        df = pd.read_parquet(pf, columns=need)
        arr = np.stack(df[col].to_list()).astype(np.float64)  # (N, D)
        if arr.ndim != 2 or arr.shape[1] <= max(xyz_idx):
            continue
        ep = df["episode_index"].to_numpy() if "episode_index" in df.columns else np.zeros(len(df))
        fr = df["frame_index"].to_numpy() if "frame_index" in df.columns else np.arange(len(df))
        order = np.lexsort((fr, ep))
        eps.append(ep[order])
        xyzs.append(arr[order][:, xyz_idx])
    if not eps:
        return None, None
    return np.concatenate(eps), np.concatenate(xyzs, axis=0)


def _per_episode_speed(ep, xyz, fps):
    """avg_eef_speed per episode from sorted (ep, xyz) arrays."""
    speeds = {}
    if ep is None:
        return speeds
    for eid in np.unique(ep):
        pts = xyz[ep == eid]
        if len(pts) < 2:
            speeds[int(eid)] = 0.0
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        speeds[int(eid)] = float(seg.mean()) * fps
    return speeds


def _quantile_edges(values: np.ndarray, num_bins: int):
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(values, qs))
    if edges.size < 2:
        edges = np.array([float(values.min()), float(values.max()) + 1e-9])
    return edges


def _assign_bins(values: np.ndarray, edges: np.ndarray):
    # higher value -> higher bin, range 0..len(edges)-2
    return np.digitize(values, edges[1:-1]).astype(int)


def _write_speed_json(
    dataset_path: Path, eef_col, xyz_idx, fps, num_bins, method, edges,
    ep_speeds: dict,
):
    ep_ids = sorted(ep_speeds)
    values = np.array([ep_speeds[e] for e in ep_ids], dtype=np.float64)
    bins = _assign_bins(values, edges)
    episodes_map = {
        str(e): {"raw": round(float(ep_speeds[e]), 6), "bin": int(bins[i])}
        for i, e in enumerate(ep_ids)
    }
    payload = {
        "fps": fps,
        "eef_col": eef_col,
        "xyz_indices": list(xyz_idx),
        "num_bins": int(num_bins),
        "bin_method": method,
        "bin_edges": [round(float(x), 6) for x in edges.tolist()],
        "speed_min": round(float(values.min()), 6),
        "speed_max": round(float(values.max()), 6),
        "speed_mean": round(float(values.mean()), 6),
        "speed_median": round(float(np.median(values)), 6),
        "n_episodes": len(ep_ids),
        "episodes": episodes_map,
    }
    out = (dataset_path / "meta" / EPISODE_SPEED_FILENAME)
    out.write_text(json.dumps(payload, indent=2))
    # annotate episodes.jsonl (v2.0) for inspection
    eps_meta = _load_episodes_meta(dataset_path / "meta")
    if eps_meta is not None:
        for ep in eps_meta:
            entry = episodes_map.get(str(ep.get("episode_index")))
            if entry is not None:
                ep["avg_eef_speed"] = entry["raw"]
                ep["speed_bin"] = entry["bin"]
        with open(dataset_path / "meta" / EPISODES_FILENAME, "w") as f:
            for ep in eps_meta:
                f.write(json.dumps(ep) + "\n")
    return payload


def _resolve_dataset_cfg(dataset_path: Path, eef_col, xyz_idx, fps_override):
    meta_dir = dataset_path / "meta"
    fps = float(fps_override) if fps_override is not None else _resolve_fps(meta_dir)
    if eef_col is None or xyz_idx is None:
        auto = _autodetect_eef_keys(meta_dir)
        if auto is not None:
            eef_col = eef_col or auto[0]
            xyz_idx = xyz_idx or auto[1]
    eef_col = eef_col or DEFAULT_EEF_COL
    xyz_idx = list(xyz_idx) if xyz_idx else list(DEFAULT_XYZ)
    return fps, eef_col, xyz_idx


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset-path", action="append", default=[], type=Path,
                    help="LeRobot dataset root; repeat for a global-bin mixture run")
    ap.add_argument("--dataset-list", default=None,
                    help="file with one dataset root path per line (blank lines/# ignored)")
    ap.add_argument("--eef-col", default=None)
    ap.add_argument("--fallback-eef-col", default=DEFAULT_FALLBACK_COL)
    ap.add_argument("--xyz-indices", default=None, help="comma-separated, default 0,1,2")
    ap.add_argument("--num-bins", type=int, default=10)
    ap.add_argument("--fps", type=float, default=None, help="override fps (else read per-dataset info.json)")
    ap.add_argument("--per-dataset-bins", action="store_true",
                    help="force per-dataset bins even with multiple datasets")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    xyz_cli = [int(i) for i in args.xyz_indices.split(",")] if args.xyz_indices else None
    dataset_paths = list(args.dataset_path)
    if args.dataset_list:
        for line in Path(args.dataset_list).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                dataset_paths.append(Path(line))
    if not dataset_paths:
        ap.error("at least one --dataset-path (or --dataset-list) is required")
    global_bins = (len(dataset_paths) > 1) and not args.per_dataset_bins

    print(f"[speed] {len(dataset_paths)} dataset(s); "
          f"binning={'GLOBAL' if global_bins else 'per-dataset'}; num_bins={args.num_bins}")

    # Pass 1: per-dataset raw speeds (+ resolved cfg)
    per_ds = []  # list of dicts
    for dp in dataset_paths:
        fps, eef_col, xyz_idx = _resolve_dataset_cfg(dp, args.eef_col, xyz_cli, args.fps)
        ep, xyz = _load_eef_table(dp, eef_col, args.fallback_eef_col, xyz_idx)
        if ep is None:
            print(f"[speed] WARN: no EEF data found in {dp} (col={eef_col}); skipping")
            continue
        speeds = _per_episode_speed(ep, xyz, fps)
        per_ds.append({"path": dp, "fps": fps, "eef_col": eef_col, "xyz_idx": xyz_idx, "speeds": speeds})
        print(f"[speed] {dp.name}: fps={fps} col={eef_col} eps={len(speeds)} "
              f"speed[min/med/max]={np.min(list(speeds.values())):.4f}/"
              f"{np.median(list(speeds.values())):.4f}/{np.max(list(speeds.values())):.4f} m/s")

    if not per_ds:
        raise RuntimeError("No datasets could be processed; check --eef-col / parquet layout.")

    # Edges
    pooled = np.concatenate([np.fromiter(d["speeds"].values(), dtype=np.float64,
                                         count=len(d["speeds"])) for d in per_ds])
    if global_bins:
        edges = _quantile_edges(pooled, args.num_bins)
        method = "quantile_global"
        print(f"[speed] GLOBAL edges: {[round(float(e),4) for e in edges.tolist()]}")
    # per-dataset edges computed below

    # Pass 2: assign bins + write
    any_written = False
    for d in per_ds:
        if not global_bins:
            vals = np.fromiter(d["speeds"].values(), dtype=np.float64, count=len(d["speeds"]))
            edges = _quantile_edges(vals, args.num_bins)
            method = "quantile"
        out_path = d["path"] / "meta" / EPISODE_SPEED_FILENAME
        if out_path.exists() and not args.overwrite:
            print(f"[speed] SKIP {d['path'].name} (exists; --overwrite to redo)")
            continue
        _write_speed_json(d["path"], d["eef_col"], d["xyz_idx"], d["fps"],
                          args.num_bins, method, edges, d["speeds"])
        any_written = True
    print(f"[speed] pooled episodes={len(pooled)} "
          f"speed[min/med/max]={pooled.min():.4f}/{np.median(pooled):.4f}/{pooled.max():.4f} m/s; "
          f"done (overwrite={args.overwrite}).")


if __name__ == "__main__":
    main()
