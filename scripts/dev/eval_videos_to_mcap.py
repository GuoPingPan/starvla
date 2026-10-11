#!/usr/bin/env python3
"""eval 产物 → foxglove mcap（与训练数据同 topic/格式）。

v2 模式（--scan）: episode 目录里有 `episode_XXX_data.jsonl`（eval_env.py
的逐步记录: state/action/视频帧标记）时，产出训练数据全 schema mcap:
  - observation/images/{cam_high_0,cam_wrist_0,cam_wrist_1}/color_h264
  - observation/state/effector_{0,1}/joint_state   (gripper, 归一化)
  - observation/state/end_{0,1}/pose               (pos+quat, base_link0 系)
  - action/effector_{0,1}/joint_state, action/end_{0,1}/pose
  - /tf (end_0/end_1 相对 base_link0)
  - observation/instruction
时间戳全部取自记录的真实 wall-clock → Foxglove 回放即真实速度。
旧目录（无 jsonl）自动退化为 v1 行为: 仅视频 + instruction，等间隔时间戳。

单集用法:
  eval_videos_to_mcap.py <episode_dir> <out.mcap> [--meta '{}']
批量用法:
  eval_videos_to_mcap.py --scan <run_dir> --out <out_dir> [--meta '{}']
"""
import argparse
import base64
import io
import json
import struct
import sys
import time
from pathlib import Path

import av
from mcap.writer import Writer

CAM_MAP = {
    "cam_head": "observation/images/cam_high_0/color_h264",
    "cam_left_wrist": "observation/images/cam_wrist_0/color_h264",
    "cam_right_wrist": "observation/images/cam_wrist_1/color_h264",
}
ARM_IDX = {"left": 0, "right": 1, "": 0}
BASE_FRAME = "base_link0"
VIDEO_FPS_FALLBACK = 25

VIDEO_SCHEMA = json.dumps({
    "title": "foxglove.CompressedVideo",
    "description": "h264 frame",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
        "format": {"type": "string"},
    },
}, ensure_ascii=False)
TEXT_SCHEMA = json.dumps({
    "title": "custom.TextMap",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "fields": {"type": "array", "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}}},
    },
}, ensure_ascii=False)
JOINT_SCHEMA = json.dumps({
    "title": "foxglove.JointStates",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "joints": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "position": {"type": "number"}}}},
    },
}, ensure_ascii=False)
POSE_SCHEMA = json.dumps({
    "title": "foxglove.PoseInFrame",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "frame_id": {"type": "string"},
        "pose": {"type": "object", "properties": {
            "position": {"type": "object", "properties": {
                "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
            "orientation": {"type": "object", "properties": {
                "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}}}},
        },
    },
}, ensure_ascii=False)
TF_SCHEMA = json.dumps({
    "title": "foxglove.FrameTransform",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
        "parent_frame_id": {"type": "string"},
        "child_frame_id": {"type": "string"},
        "translation": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
        "rotation": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}}},
    },
}, ensure_ascii=False)


def ts_ns_to_dict(t_ns: int) -> dict:
    sec, nsec = divmod(int(t_ns), 10**9)
    return {"sec": sec, "nsec": nsec}


def full_extradata_annexb(extradata: bytes) -> bytes:
    """avcC (sps/pps 集合) → annex-b。"""
    out = b""
    i = 5
    while i < len(extradata):
        cnt = extradata[i]; i += 1
        for _ in range(cnt):
            if i + 2 > len(extradata):
                return out
            ln = struct.unpack(">H", extradata[i:i + 2])[0]
            out += b"\x00\x00\x00\x01" + extradata[i + 2:i + 2 + ln]
            i += 2 + ln
    return out


def packet_annexb(pkt_bytes: bytes) -> bytes:
    """mp4 内 h264 packet (4 字节长度前缀) → annex-b。"""
    out = b""
    i = 0
    while i + 4 <= len(pkt_bytes):
        ln = struct.unpack(">I", pkt_bytes[i:i + 4])[0]
        if ln == 0 or i + 4 + ln > len(pkt_bytes):
            break
        out += b"\x00\x00\x00\x01" + pkt_bytes[i + 4:i + 4 + ln]
        i += 4 + ln
    return out or pkt_bytes


def _annexb_from_av(pkt) -> bytes:
    b = bytes(pkt)
    if len(b) > 4 and b[0] == 0 and b[1] == 0 and b[2] == 0 and b[3] == 1:
        return b
    if len(b) > 3 and b[0] == 0 and b[1] == 0 and b[2] == 1:
        return b
    return packet_annexb(b)


def encode_video_topic(w: Writer, chan, mp4_path: Path, frame_ts: list[int], t0_ns: int):
    """重编码 x264 baseline (无B帧, 逐帧完整包, Foxglove 兼容) 并写入。

    frame_ts: 每个视频帧的 wall-clock ns（与 state/action 对齐）。
    数量与实际帧数不符时在首末之间均匀铺开。
    """
    inp = av.open(str(mp4_path))
    st = inp.streams.video[0]
    out = av.open(io.BytesIO(), "w", format="h264")
    ost = out.add_stream("libx264", rate=VIDEO_FPS_FALLBACK)
    ost.width, ost.height = st.codec_context.width, st.codec_context.height
    ost.pix_fmt = "yuv420p"
    ost.options = {
        "profile": "baseline",
        "tune": "zerolatency",
        "x264-params": "repeat-headers=1:sync-lookahead=0:bframes=0",
        "crf": "20",
    }
    frames = list(inp.decode(st))
    n_frames = len(frames)
    if len(frame_ts) == n_frames:
        ts_list = frame_ts
    else:
        t_a = frame_ts[0] if frame_ts else t0_ns
        t_b = frame_ts[-1] if len(frame_ts) > 1 else t_a + n_frames * int(1e9 / VIDEO_FPS_FALLBACK)
        step = (t_b - t_a) / max(n_frames - 1, 1)
        ts_list = [int(t_a + i * step) for i in range(n_frames)]
    first = True
    n = 0
    for frame in frames:
        for pkt in ost.encode(frame):
            data = _annexb_from_av(pkt)
            if first:
                data = full_extradata_annexb(ost.codec_context.extradata or b"") + data
                first = False
            if pkt.size == 0 or not data:
                continue
            t_ns = ts_list[n]
            msg = json.dumps({
                "timestamp": ts_ns_to_dict(t_ns),
                "frame_id": chan[1].split("/")[2],
                "format": "h264",
                "data": base64.b64encode(data).decode(),
            }, ensure_ascii=False)
            w.add_message(chan[0], t_ns, msg.encode(), t_ns)
            n += 1
    for pkt in ost.encode(None):
        data = _annexb_from_av(pkt)
        if pkt.size == 0 or not data:
            continue
        t_ns = ts_list[n]
        msg = json.dumps({
            "timestamp": ts_ns_to_dict(t_ns),
            "frame_id": chan[1].split("/")[2],
            "format": "h264",
            "data": base64.b64encode(data).decode(),
        }, ensure_ascii=False)
        w.add_message(chan[0], t_ns, msg.encode(), t_ns)
        n += 1
    return n


def pose_msg(t_ns: int, frame_id: str, pose7: list) -> bytes:
    # eval 侧 pose7 = [x, y, z, qw, qx, qy, qz] (transforms3d wxyz 约定)
    x, y, z, qw, qx, qy, qz = [float(v) for v in pose7[:7]]
    return json.dumps({
        "timestamp": ts_ns_to_dict(t_ns),
        "frame_id": frame_id,
        "pose": {
            "position": {"x": x, "y": y, "z": z},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
    }, ensure_ascii=False).encode()


def tf_msg(t_ns: int, child: str, pose7: list) -> bytes:
    x, y, z, qw, qx, qy, qz = [float(v) for v in pose7[:7]]
    return json.dumps({
        "timestamp": ts_ns_to_dict(t_ns),
        "parent_frame_id": BASE_FRAME,
        "child_frame_id": child,
        "translation": {"x": x, "y": y, "z": z},
        "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
    }, ensure_ascii=False).encode()


def joint_msg(t_ns: int, position: float) -> bytes:
    return json.dumps({
        "timestamp": ts_ns_to_dict(t_ns),
        "joints": [{"name": "gripper", "position": float(position)}],
    }, ensure_ascii=False).encode()


def convert_episode(ep_dir: Path, out_mcap: Path, extra_meta: dict, files: dict | None = None):
    """files 为 None 时按"一集一目录"读取; 否则显式给 {"jsonl": Path|None, "videos": {cam: Path}}。"""
    if files is None:
        jsonl = sorted(ep_dir.glob("episode_*_data.jsonl"))
        videos = {cam: (vids[0] if (vids := sorted(ep_dir.glob(f"*_{cam}_*.mp4"))) else None) for cam in CAM_MAP}
    else:
        jsonl = [files["jsonl"]] if files.get("jsonl") else []
        videos = files["videos"]
    steps = []
    meta = {}
    if jsonl:
        with open(jsonl[0], encoding="utf-8") as fp:
            for line in fp:
                d = json.loads(line)
                if "meta" in d:
                    meta = d["meta"]
                else:
                    steps.append(d)
    has_data = bool(steps)
    instruction = meta.get("instruction", "") or extra_meta.get("instruction", "")

    t0_ns = steps[0]["t"] if has_data else time.time_ns()

    with open(out_mcap, "wb") as f:
        w = Writer(f)
        w.start()
        v_schema = w.register_schema("foxglove.CompressedVideo", "jsonschema", VIDEO_SCHEMA.encode())
        t_schema = w.register_schema("custom.TextMap", "jsonschema", TEXT_SCHEMA.encode())
        j_schema = w.register_schema("foxglove.JointStates", "jsonschema", JOINT_SCHEMA.encode())
        p_schema = w.register_schema("foxglove.PoseInFrame", "jsonschema", POSE_SCHEMA.encode())
        tf_schema = w.register_schema("foxglove.FrameTransform", "jsonschema", TF_SCHEMA.encode())

        instr_ch = w.register_channel("observation/instruction", "json", t_schema)
        st_joints = {i: w.register_channel(f"observation/state/effector_{i}/joint_state", "json", j_schema) for i in (0, 1)}
        st_pose = {i: w.register_channel(f"observation/state/end_{i}/pose", "json", p_schema) for i in (0, 1)}
        ac_joints = {i: w.register_channel(f"action/effector_{i}/joint_state", "json", j_schema) for i in (0, 1)}
        ac_pose = {i: w.register_channel(f"action/end_{i}/pose", "json", p_schema) for i in (0, 1)}
        tf_ch = {i: w.register_channel("/tf", "json", tf_schema) for i in (0, 1)}

        fields = [
            {"key": "instruction", "value": instruction},
            {"key": "language", "value": "en"},
            {"key": "task", "value": str(meta.get("task", extra_meta.get("task", "")))},
            {"key": "model", "value": str(extra_meta.get("model", ""))},
            {"key": "horizon", "value": str(extra_meta.get("horizon", ""))},
            {"key": "layout", "value": str(meta.get("layout", extra_meta.get("layout", "")))},
            {"key": "success", "value": str(meta.get("success", extra_meta.get("success", "")))},
            {"key": "score", "value": str(meta.get("score", extra_meta.get("score", "")))},
            {"key": "source", "value": "eval_rollout"},
        ]
        msg = json.dumps({"timestamp": ts_ns_to_dict(t0_ns), "fields": fields}, ensure_ascii=False)
        w.add_message(instr_ch, t0_ns, msg.encode(), t0_ns)

        try:
            video_ts = [s["t"] for s in steps if s.get("video")] if has_data else []
            n_frames_total = None
            for cam, topic in CAM_MAP.items():
                vid = videos.get(cam)
                if vid is None:
                    continue
                n = encode_video_topic(w, (w.register_channel(topic, "json", v_schema), topic), vid, video_ts, t0_ns)
                n_frames_total = n if n_frames_total is None else n_frames_total

            if has_data:
                n_written = 0
                for s in steps:
                    t_ns = int(s["t"])
                    st, ac = s.get("state") or {}, s.get("action") or {}
                    for key, val in st.items():
                        arm = "left" if key.startswith("left") else ("right" if key.startswith("right") else "")
                        idx = ARM_IDX[arm]
                        if key.endswith("_ee_pose") and len(val) >= 7:
                            m = pose_msg(t_ns, BASE_FRAME, val)
                            w.add_message(st_pose[idx], t_ns, m, t_ns)
                            w.add_message(tf_ch[idx], t_ns, tf_msg(t_ns, f"end_{idx}", val), t_ns)
                        elif key.endswith("_ee_joint_state") and len(val) >= 1:
                            w.add_message(st_joints[idx], t_ns, joint_msg(t_ns, val[0]), t_ns)
                    for key, val in ac.items():
                        arm = "left" if key.startswith("left") else ("right" if key.startswith("right") else "")
                        idx = ARM_IDX[arm]
                        if key.endswith("_ee_pose") and len(val) >= 7:
                            w.add_message(ac_pose[idx], t_ns, pose_msg(t_ns, BASE_FRAME, val), t_ns)
                        elif key.endswith("_ee_joint_state") and len(val) >= 1:
                            w.add_message(ac_joints[idx], t_ns, joint_msg(t_ns, val[0]), t_ns)
                    n_written += 1
                print(f"{out_mcap}: {n_written} steps (full schema), {n_frames_total or 0} frames/cam")
            else:
                print(f"{out_mcap}: video-only (no jsonl), {n_frames_total or 0} frames/cam")
        finally:
            w.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", nargs="?", help="episode 目录 (含 episode_* mp4 / *_data.jsonl)")
    ap.add_argument("out_mcap", nargs="?", help="输出 mcap 路径 (单集模式)")
    ap.add_argument("--scan", dest="scan_dir", help="批量模式: 扫描该目录下所有 episode 目录")
    ap.add_argument("--out", dest="out_dir", help="批量模式: mcap 输出目录")
    ap.add_argument("--meta", default="{}", help="metadata json (instruction/task/model/success/score)")
    args = ap.parse_args()
    extra_meta = json.loads(args.meta) if args.meta else {}

    if args.scan_dir:
        import re
        scan = Path(args.scan_dir)
        out_root = Path(args.out_dir) if args.out_dir else scan
        out_root.mkdir(parents=True, exist_ok=True)
        groups: dict[str, dict] = {}
        for p in sorted(scan.iterdir()):
            m = re.match(r"(episode_\d+)", p.name)
            if not m:
                continue
            g = groups.setdefault(m.group(1), {"jsonl": None, "videos": {}})
            if p.name.endswith("_data.jsonl"):
                g["jsonl"] = p
            elif "_cam_" in p.name and p.suffix == ".mp4":
                cam = "cam_" + p.name.split("_cam_")[1].rsplit("_", 1)[0]
                g["videos"][cam] = p
        if not groups:
            print(f"no episodes under {scan}", file=sys.stderr)
            sys.exit(1)
        # 逐集成功/得分: 从同目录 _result.json details[idx]={layout_id,success,score}
        details = {}
        rj = scan / "_result.json"
        if rj.exists():
            try:
                details = json.loads(rj.read_text()).get("details", {})
            except Exception as e:
                print(f"WARN: fail to read _result.json: {e}", file=sys.stderr)
        ok = 0
        for name, g in sorted(groups.items()):
            meta = dict(extra_meta)
            suffix = ""
            try:
                idx = int(name.split("_")[1])
                d = details.get(idx) or details.get(str(idx)) or {}
                if d:
                    meta["success"] = bool(d.get("success"))
                    meta["score"] = d.get("score", "")
                    meta["layout"] = d.get("layout_id", "")
                    layout_tag = f"_layout{int(d['layout_id']):03d}" if d.get("layout_id") is not None else ""
                    suffix = f"{layout_tag}_{'success' if d.get('success') else 'fail'}"
            except Exception:
                pass
            try:
                convert_episode(scan, out_root / f"{name}{suffix}.mcap", meta, files=g)
                ok += 1
            except Exception as e:
                print(f"FAIL {name}: {e}", file=sys.stderr)
        print(f"converted {ok}/{len(groups)} episodes -> {out_root}")
        return

    if not args.src_dir or not args.out_mcap:
        ap.error("需要 <src_dir> <out_mcap> 或 --scan/--out")
    convert_episode(Path(args.src_dir), Path(args.out_mcap), extra_meta)


if __name__ == "__main__":
    main()
