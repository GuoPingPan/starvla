#!/usr/bin/env python3
"""Add the common LeRobot v3 modality mapping to H1 DDL exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATE_FIELDS = [
    ("end_0_pose", 7, "observation.state.end_0_pose"),
    ("end_1_pose", 7, "observation.state.end_1_pose"),
    ("arm_0_joint_state", 7, "observation.state.arm_0_joint_state"),
    ("arm_1_joint_state", 7, "observation.state.arm_1_joint_state"),
    ("effector_0_joint_state", 1, "observation.state.effector_0_joint_state"),
    ("effector_1_joint_state", 1, "observation.state.effector_1_joint_state"),
]
ACTION_FIELDS = [
    ("end_0_pose", 7, "action.end_0_pose"),
    ("end_1_pose", 7, "action.end_1_pose"),
    ("arm_0_joint_state", 7, "action.arm_0_joint_state"),
    ("arm_1_joint_state", 7, "action.arm_1_joint_state"),
    ("effector_0_joint_state", 1, "action.effector_0_joint_state"),
    ("effector_1_joint_state", 1, "action.effector_1_joint_state"),
]


def _fields(fields):
    out = {}
    for name, width, original_key in fields:
        out[name] = {
            "start": 0,
            "end": width,
            "absolute": True,
            "dtype": "float32",
            "original_key": original_key,
        }
    return out


def build_modality():
    return {
        "state": _fields(STATE_FIELDS),
        "action": _fields(ACTION_FIELDS),
        "video": {
            "cam_high_0": {"original_key": "observation.images.cam_high_0"},
            "cam_wrist_0": {"original_key": "observation.images.cam_wrist_0"},
            "cam_wrist_1": {"original_key": "observation.images.cam_wrist_1"},
        },
        "annotation": {
            "human.task": {"original_key": "task_index"},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_modality(), ensure_ascii=False, indent=2) + "\n"
    datasets = sorted(
        p
        for p in args.data_root.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").exists()
    )
    if not datasets:
        raise SystemExit(f"no LeRobot datasets found under {args.data_root}")
    for dataset in datasets:
        path = dataset / "meta" / "modality.json"
        path.write_text(payload, encoding="utf-8")
        print(f"wrote: {path}")
    print(f"prepared {len(datasets)} dataset(s)")


if __name__ == "__main__":
    main()
