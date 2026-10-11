#!/usr/bin/env python3
"""Replay one saved make_kong EE rollout and record its actual IK joint targets.

Run with the RoboDojo simulator Python, from any working directory::

    /root/workspace/micromamba/envs/RoboDojo/bin/python \
      scripts/dev/replay_make_kong_joint.py \
      --sim-root /root/workspace/sim/RoboDojo \
      --source-npz /cfs/users/kindel/robodojo_data/make_kong/official_push_postures_v2_all_4gpu_20260923_101009/success/seed_0/layout_004/trajectory.npz \
      --output-root /root/workspace/starvla/.runtime/make_kong_joint_replay \
      --device-id 0 --headless --enable_cameras \
      --kit_args ' --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera'

The output is under RoboDojo's normal per-run eval directory and contains
``trajectory_joint.npz``, three newly rendered videos, and ``_result.json``.
The original data tree and simulator source are read-only. ``--dry-run`` checks
the source trajectory and original-source snapshot without launching Isaac Sim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ARM_NAMES = ("left", "right")
POSE_KEYS = ("left_ee_pose", "right_ee_pose")
GRIPPER_KEYS = ("left_ee_joint_state", "right_ee_joint_state")
JOINT_KEYS = ("left_arm_joint_state", "right_arm_joint_state")
VIDEO_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SOURCE_FILES = (
    "src/eval_client/eval_env.py",
    "env/seed_manager/seed_manager.py",
    "env_cfg/arx_x5.yml",
    "env_cfg/robot/dual_x5.yml",
    "env_cfg/sim/sim_config.yml",
    "env_cfg/camera/camera_config.yml",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sim-root", type=Path, required=True, help="full RoboDojo checkout")
    result.add_argument("--source-npz", type=Path, required=True, help="one successful source trajectory.npz")
    result.add_argument("--output-root", type=Path, required=True, help="new eval result root; never the source tree")
    result.add_argument("--device-id", type=int, default=0, help="physical GPU index before CUDA_VISIBLE_DEVICES")
    result.add_argument("--dry-run", action="store_true", help="validate inputs without launching Isaac Sim")
    result.add_argument("--allow-source-mismatch", action="store_true", help="permit source files different from the production snapshot")
    result.add_argument("--max-ee-pos-error-m", type=float, default=0.02)
    result.add_argument("--max-ee-rot-error-deg", type=float, default=10.0)
    result.add_argument("--max-object-pos-error-m", type=float, default=0.03)
    result.add_argument("--max-object-rot-error-deg", type=float, default=20.0)
    result.add_argument("--max-gripper-error", type=float, default=0.05)
    return result


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_check(sim_root: Path, snapshot_root: Path) -> dict[str, str]:
    mismatches = {}
    for name in SOURCE_FILES:
        original = snapshot_root / name
        current = sim_root / name
        if not original.is_file() or not current.is_file():
            mismatches[name] = "missing original snapshot or simulator file"
        elif _hash(original) != _hash(current):
            mismatches[name] = "content differs from production snapshot"
    return mismatches


def _load_source(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        required = ("actions", "robot_states", "object_poses", "terminal_robot_state",
                    "terminal_object_poses", "labels", "step_counts", "eval_seed",
                    "layout_id", "target_group")
        absent = [key for key in required if key not in source]
        if absent:
            raise ValueError(f"Source trajectory is missing {absent}")
        actions = np.asarray(source["actions"], dtype=np.float32)
        states = np.asarray(source["robot_states"], dtype=np.float32)
        objects = np.asarray(source["object_poses"], dtype=np.float32)
        terminal_state = np.asarray(source["terminal_robot_state"], dtype=np.float32)
        terminal_objects = np.asarray(source["terminal_object_poses"], dtype=np.float32)
        labels = np.asarray(source["labels"], dtype=str)
        steps = np.asarray(source["step_counts"], dtype=np.int32)
        seed = int(source["eval_seed"])
        layout_id = int(source["layout_id"])
        success = bool(source["success"]) if "success" in source else False
        target_group = int(source["target_group"])
    count = len(actions)
    if not success or count < 1 or actions.shape != (count, 16) or states.shape != (count, 16):
        raise ValueError("Source must be a successful (T,16) EE rollout")
    if objects.shape != (count, len(labels), 7) or steps.shape != (count,) or not np.array_equal(steps, np.arange(count)):
        raise ValueError("Source object poses or step indices do not align with actions")
    if terminal_state.shape != (16,) or terminal_objects.shape != (len(labels), 7):
        raise ValueError("Source terminal states do not have the expected shape")
    if target_group not in range(4):
        raise ValueError(f"Unexpected source target group {target_group}")
    if not all(np.isfinite(arr).all() for arr in (actions, states, objects, terminal_state, terminal_objects)):
        raise ValueError("Source has nonfinite state, action, or object pose")
    if len(labels) != len(set(labels)):
        raise ValueError("Source object labels are not unique")
    return dict(actions=actions, states=states, objects=objects,
                terminal_state=terminal_state, terminal_objects=terminal_objects,
                labels=labels, steps=steps, seed=seed, layout_id=layout_id,
                target_group=target_group)


def _as_vector(value: Any, count: int, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be {count} finite values; got {array.shape}")
    return array


def _pose_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    position = np.linalg.norm(actual[..., :3] - expected[..., :3], axis=-1)
    qa = actual[..., 3:7] / np.linalg.norm(actual[..., 3:7], axis=-1, keepdims=True)
    qb = expected[..., 3:7] / np.linalg.norm(expected[..., 3:7], axis=-1, keepdims=True)
    cosine = np.abs(np.sum(qa * qb, axis=-1))
    angle = np.rad2deg(2 * np.arccos(np.clip(cosine, 0.0, 1.0)))
    return position.astype(np.float32), angle.astype(np.float32)


def _observation_vectors(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    state = observation["state"]
    ee = np.stack([_as_vector(state[key], 7, key) for key in POSE_KEYS])
    gripper = np.array([
        _as_vector(state.get(key, observation.get("action", {}).get(key)), 1, key)[0]
        for key in GRIPPER_KEYS
    ], dtype=np.float32)
    joints = np.stack([_as_vector(state[key], 6, key) for key in JOINT_KEYS])
    state14 = np.concatenate((joints[0], gripper[:1], joints[1], gripper[1:]))
    return np.concatenate((ee[0], gripper[:1], ee[1], gripper[1:])), state14


def _live_objects(task_env: Any, labels: np.ndarray) -> np.ndarray:
    manager = task_env.scene_manager.layout_manager
    result = []
    for label in labels:
        position, quaternion = manager.get_instance_pose(env_idx=0, label=str(label))
        result.append(np.concatenate((_as_vector(position, 3, f"{label}.position"),
                                      _as_vector(quaternion, 4, f"{label}.quaternion"))))
    return np.stack(result)


def _action_dict(row: np.ndarray) -> dict[str, list[float]]:
    return {
        "left_ee_pose": row[:7].tolist(),
        "left_ee_joint_state": row[7:8].tolist(),
        "right_ee_pose": row[8:15].tolist(),
        "right_ee_joint_state": row[15:16].tolist(),
    }


def _capture_episode(task_env: Any, source: dict[str, Any]) -> dict[str, Any]:
    """Called by EvalEnv.run_eval in place of the policy deploy callback."""
    replay_target_group = int(task_env.push[0].split("_")[0].removeprefix("mahjong")) - 5
    state14, action14, ik_ok, ik_text = [], [], [], []
    replay_ee, replay_objects = [], []
    captured_targets: dict[str, tuple[str, np.ndarray | None]] = {}
    original_solve_ik = task_env.robot_manager.solve_ik

    def recording_solve_ik(target_pose, env_idx, robot, trans="world"):
        result = original_solve_ik(target_pose=target_pose, env_idx=env_idx, robot=robot, trans=trans)
        if env_idx == 0 and robot.type == "target":
            arm = robot.arm_name.split("_")[0]
            status = str(result.get("status", "missing_status"))
            target = None
            if status == "Success":
                target = _as_vector(result["joint_value"], 6, f"{arm}.ik_joint_value")
            captured_targets[arm] = (status, target)
        return result

    task_env.robot_manager.solve_ik = recording_solve_ik
    try:
        for step, original_action in enumerate(source["actions"]):
            if task_env.is_episode_end():
                break
            observation = task_env.get_obs()  # One newly rendered video frame per action.
            if int(task_env.take_action_cnt[0]) != step:
                raise RuntimeError(f"Replay step counter differs at {step}: {task_env.take_action_cnt[0]}")
            ee16, joint_state = _observation_vectors(observation)
            objects = _live_objects(task_env, source["labels"])
            captured_targets.clear()
            task_env.take_action(_action_dict(original_action))
            if int(task_env.take_action_cnt[0]) != step + 1:
                raise RuntimeError(f"EE action was not executed at source step {step}")
            target_rows, success_rows, status_rows = [], [], []
            for arm in ARM_NAMES:
                status, target = captured_targets.get(arm, ("not_called", None))
                success_rows.append(target is not None)
                status_rows.append(status)
                target_rows.append(target if target is not None else np.full(6, np.nan, dtype=np.float32))
            joint_action = np.concatenate((target_rows[0], original_action[7:8],
                                           target_rows[1], original_action[15:16]))
            state14.append(joint_state)
            action14.append(joint_action)
            ik_ok.append(success_rows)
            ik_text.append(status_rows)
            replay_ee.append(ee16)
            replay_objects.append(objects)
        if not task_env.is_episode_end():
            # Force official finalization for a replay that did not end by T.
            # is_episode_end streams the one terminal frame; get_obs below
            # reads that state but does not stream while end_flag is true.
            task_env.success[0] = False
            task_env.is_episode_end()
        # Original collector reads one terminal observation after termination.
        terminal_observation = task_env.get_obs()
        terminal_ee, terminal_joints = _observation_vectors(terminal_observation)
        terminal_objects = _live_objects(task_env, source["labels"])
    finally:
        task_env.robot_manager.solve_ik = original_solve_ik

    count = len(state14)
    ee = np.asarray(replay_ee, dtype=np.float32).reshape(count, 16)
    objects = np.asarray(replay_objects, dtype=np.float32).reshape(count, len(source["labels"]), 7)
    ee_position, ee_rotation = _pose_errors(ee[:, [*range(7), *range(8, 15)]].reshape(count, 2, 7),
                                            source["states"][:, [*range(7), *range(8, 15)]].reshape(len(source["actions"]), 2, 7)[:count])
    object_position, object_rotation = _pose_errors(objects, source["objects"][:count])
    gripper_error = np.abs(ee[:, [7, 15]] - source["states"][:count, [7, 15]])
    terminal_ee_position, terminal_ee_rotation = _pose_errors(
        terminal_ee[[*range(7), *range(8, 15)]].reshape(2, 7),
        source["terminal_state"][[*range(7), *range(8, 15)]].reshape(2, 7),
    )
    terminal_object_position, terminal_object_rotation = _pose_errors(
        terminal_objects, source["terminal_objects"]
    )
    terminal_gripper_error = np.abs(terminal_ee[[7, 15]] - source["terminal_state"][[7, 15]])
    return dict(
        joint_states=np.asarray(state14, dtype=np.float32).reshape(count, 14),
        joint_actions=np.asarray(action14, dtype=np.float32).reshape(count, 14),
        ik_status=np.asarray(ik_ok, dtype=np.bool_).reshape(count, 2),
        ik_status_text=np.asarray(ik_text, dtype=str).reshape(count, 2),
        ee_pos_error_m=ee_position,
        ee_rot_error_deg=ee_rotation,
        object_pos_error_m=object_position,
        object_rot_error_deg=object_rotation,
        gripper_error=gripper_error,
        terminal_ee_pos_error_m=terminal_ee_position,
        terminal_ee_rot_error_deg=terminal_ee_rotation,
        terminal_object_pos_error_m=terminal_object_position,
        terminal_object_rot_error_deg=terminal_object_rotation,
        terminal_gripper_error=terminal_gripper_error,
        replay_robot_states=ee,
        replay_object_poses=objects,
        terminal_robot_state=terminal_ee,
        terminal_joint_state=terminal_joints,
        terminal_object_poses=terminal_objects,
        replay_target_group=replay_target_group,
        replay_success=bool(task_env.success[0] and task_env.end_flag[0] and 0 not in task_env.unstable_envs),
        step_count=count,
    )


def _video_frame_count(path: Path) -> int | None:
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                                check=True, capture_output=True, text=True)
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _finish_output(task_env: Any, source: dict[str, Any], capture: dict[str, Any], args: argparse.Namespace,
                   source_path: Path, snapshot_mismatches: dict[str, str]) -> Path:
    run_dir = Path(task_env.save_dir).resolve()
    count = capture["step_count"]
    videos = [run_dir / f"episode_0000000_{key}_{'success' if capture['replay_success'] else 'fail'}.mp4"
              for key in VIDEO_KEYS]
    video_counts = [_video_frame_count(path) for path in videos]
    aligned_video = all(value == count + 1 for value in video_counts)
    checks = {
        "all_source_steps_executed": count == len(source["actions"]),
        "environment_success": capture["replay_success"],
        "all_ik_succeeded": bool(capture["ik_status"].all()),
        "ee_position_aligned": bool(np.max(capture["ee_pos_error_m"], initial=0) <= args.max_ee_pos_error_m),
        "ee_rotation_aligned": bool(np.max(capture["ee_rot_error_deg"], initial=0) <= args.max_ee_rot_error_deg),
        "object_position_aligned": bool(np.max(capture["object_pos_error_m"], initial=0) <= args.max_object_pos_error_m),
        "object_rotation_aligned": bool(np.max(capture["object_rot_error_deg"], initial=0) <= args.max_object_rot_error_deg),
        "gripper_aligned": bool(np.max(capture["gripper_error"], initial=0) <= args.max_gripper_error),
        "terminal_ee_position_aligned": bool(np.max(capture["terminal_ee_pos_error_m"]) <= args.max_ee_pos_error_m),
        "terminal_ee_rotation_aligned": bool(np.max(capture["terminal_ee_rot_error_deg"]) <= args.max_ee_rot_error_deg),
        "terminal_object_position_aligned": bool(np.max(capture["terminal_object_pos_error_m"]) <= args.max_object_pos_error_m),
        "terminal_object_rotation_aligned": bool(np.max(capture["terminal_object_rot_error_deg"]) <= args.max_object_rot_error_deg),
        "terminal_gripper_aligned": bool(np.max(capture["terminal_gripper_error"]) <= args.max_gripper_error),
        "target_group_aligned": capture["replay_target_group"] == source["target_group"],
        "video_frames_aligned": aligned_video,
        "source_snapshot_matched": not snapshot_mismatches,
    }
    valid = all(checks.values())
    result = run_dir / "trajectory_joint.npz"
    temporary = run_dir / ".trajectory_joint.tmp.npz"
    np.savez_compressed(
        temporary,
        schema_version=np.int32(1), seed=np.int32(source["seed"]), layout_id=np.int32(source["layout_id"]),
        source_npz=np.asarray(str(source_path)), source_length=np.int32(len(source["actions"])),
        step_counts=np.arange(count, dtype=np.int32),
        source_actions=source["actions"][:count], source_robot_states=source["states"][:count],
        source_object_poses=source["objects"][:count],
        source_terminal_robot_state=source["terminal_state"],
        source_terminal_object_poses=source["terminal_objects"], labels=source["labels"],
        replay_valid=np.bool_(valid), replay_success=np.bool_(capture["replay_success"]),
        **{key: value for key, value in capture.items() if key not in ("replay_success", "step_count")},
    )
    os.replace(temporary, result)
    summary = dict(source_npz=str(source_path), joint_npz=str(result), checks=checks,
                   source_snapshot_mismatches=snapshot_mismatches,
                   video_paths=[str(path) for path in videos], video_frames=video_counts,
                   source_steps=len(source["actions"]), replay_steps=count,
                   max_ee_pos_error_m=float(np.max(capture["ee_pos_error_m"], initial=0)),
                   max_ee_rot_error_deg=float(np.max(capture["ee_rot_error_deg"], initial=0)),
                   max_object_pos_error_m=float(np.max(capture["object_pos_error_m"], initial=0)),
                   max_object_rot_error_deg=float(np.max(capture["object_rot_error_deg"], initial=0)),
                   max_gripper_error=float(np.max(capture["gripper_error"], initial=0)),
                   max_terminal_ee_pos_error_m=float(np.max(capture["terminal_ee_pos_error_m"])),
                   max_terminal_ee_rot_error_deg=float(np.max(capture["terminal_ee_rot_error_deg"])),
                   max_terminal_object_pos_error_m=float(np.max(capture["terminal_object_pos_error_m"])),
                   max_terminal_object_rot_error_deg=float(np.max(capture["terminal_object_rot_error_deg"])),
                   max_terminal_gripper_error=float(np.max(capture["terminal_gripper_error"])),
                   source_target_group=source["target_group"],
                   replay_target_group=capture["replay_target_group"], replay_valid=valid)
    summary_path = run_dir / "replay_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    args_parser = parser()
    if "--dry-run" in sys.argv:
        args = args_parser.parse_args()
    else:
        # Mirror the official eval entry: launch Isaac Sim before importing its modules.
        early_args, _ = args_parser.parse_known_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(early_args.device_id)
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(args_parser)
        args = args_parser.parse_args()
    source_path = args.source_npz.resolve(strict=True)
    sim_root = args.sim_root.resolve()
    output_root = args.output_root.resolve()
    source = _load_source(source_path)
    source_root = source_path.parents[3]
    snapshot_root = source_root / "source_snapshot"
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output-root must be outside the original production data tree")
    mismatches = _snapshot_check(sim_root, snapshot_root)
    preflight = dict(seed=source["seed"], layout_id=source["layout_id"], steps=len(source["actions"]),
                     source_npz=str(source_path), simulator_root=str(sim_root), output_root=str(output_root),
                     snapshot_mismatches=mismatches)
    if args.dry_run:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0
    if mismatches and not args.allow_source_mismatch:
        raise RuntimeError(f"Simulator source differs from production snapshot: {mismatches}")
    if not sim_root.is_dir():
        raise FileNotFoundError(sim_root)
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["ROBODOJO_EVAL_RESULT_ROOT"] = str(output_root)
    os.environ.pop("ROBODOJO_DISABLE_VIDEO", None)
    os.environ["ROBODOJO_RUN_ID"] = f"joint_replay_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}"
    sys.path.insert(0, str(sim_root / "XPolicyLab"))
    sys.path.insert(0, str(sim_root))
    os.chdir(sim_root)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    task_env = None
    try:
        from omegaconf import OmegaConf
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        from src.eval_client import eval_env as eval_env_module
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        import importlib

        if BENCHMARK != "RoboDojo":
            raise RuntimeError(f"Unexpected official benchmark: {BENCHMARK}")
        task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "arx_x5.yml"))
        eval_cfg.update(task_name="make_kong", num_envs=1, device_id=args.device_id, eval_batch=False,
                        policy_name="joint_replay_capture", additional_info=f"layout_{source['layout_id']:03d}",
                        seed=source["seed"], eval_num=1, physx_monitor_enabled=False)
        deploy_cfg = dict(policy_name="joint_replay_capture", port=1, host="localhost", protocol="ws")
        config_names = eval_cfg["config"]
        environment_cfg = OmegaConf.create({
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", config_names["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", config_names["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", config_names["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", config_names["robot"] + ".yml")),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), "make_kong")),
            "eval_cfg": eval_cfg, "deploy_cfg": deploy_cfg,
        })
        OmegaConf.update(environment_cfg, "sim.scene.num_envs", 1, force_add=True)
        environment_cfg = process_randomization(environment_cfg)
        environment_cfg, _ = process_config(environment_cfg, task_name="make_kong")
        OmegaConf.update(environment_cfg, "camera.default_frequency",
                         eval_cfg["observation"].get("collect_freq", 0), force_add=True)
        environment_cfg.sim.seed = [0]

        class _NoPolicyServer:
            def __init__(self, **_kwargs):
                pass

            def call(self, **_kwargs):
                return None

            def close(self):
                pass

        # EvalEnv.__init__ otherwise opens a WebSocket. All actions come from source NPZ.
        eval_env_module.WsModelClient = _NoPolicyServer
        task_env = eval_env_module.create_eval_env(environment_cfg, simulation_app)
        capture: dict[str, Any] = {}

        def record_one_episode():
            capture.update(_capture_episode(task_env, source))

        task_env.eval_one_episode = record_one_episode
        task_env.reset(seed=[source["layout_id"]])
        task_env.run_eval()  # Official reward, support arm, video, and result flow.
        if not capture:
            raise RuntimeError("Official run_eval returned without a recorded episode")
        result = _finish_output(task_env, source, capture, args, source_path, mismatches)
        with np.load(result, allow_pickle=False) as data:
            return 0 if bool(data["replay_valid"]) else 2
    finally:
        if task_env is not None:
            task_env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
