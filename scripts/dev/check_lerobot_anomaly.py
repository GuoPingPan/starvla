#!/usr/bin/env python3
"""RoboDojo lerobot v3.0 数据集异常数值检测。

检查项:
 1. NaN/Inf           — state/action 全列扫描
 2. 四元数范数        — |q| 偏离 1 超过 1e-3 计异常
 3. 夹爪范围          — 超出 [0,1] 计异常
 4. 位置工作空间      — |x|>1m / |y|>0.6m / z∉[0.3,1.6] 计异常
 5. 逐帧跳变          — 单步位移 >10cm 或旋转 >30° 计异常帧
 6. 时间戳            — 非单调 / 帧间隔偏离 40ms±8ms
 7. action 对齐度     — |action[t]-state[t+1]| 位置/旋转分布 (官方语义应 ~0)
 8. 呆滞轨迹          — 全程位移 <2cm 或全程夹爪无动作
 9. 帧数一致性        — episodes meta length vs parquet 行数
10. 重复帧            — 相邻帧完全相同占比

用法: python check_lerobot_anomaly.py <dataset_root> [--max-eps N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def ang_err_deg(a, b):
    dot = np.abs(np.sum(a * b, axis=-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--max-eps", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)

    df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
    eps = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    st = np.stack(df["observation.state"].map(np.asarray))
    ac = np.stack(df["action"].map(np.asarray))
    n = len(df)
    problems = []

    # 1. NaN/Inf
    bad = (~np.isfinite(st)).any(1) | (~np.isfinite(ac)).any(1)
    if bad.any():
        problems.append(f"NaN/Inf: {bad.sum()} 帧 (episode {sorted(set(df.episode_index[bad]))[:5]})")
    print(f"[1] NaN/Inf: {bad.sum()} 帧")

    # 2. quat 范数
    for name, arr, sl in [("state.L", st, (3, 7)), ("state.R", st, (11, 15)),
                          ("action.L", ac, (3, 7)), ("action.R", ac, (11, 15))]:
        qn = np.linalg.norm(arr[:, sl[0]:sl[1]], axis=1)
        off = np.abs(qn - 1) > 1e-3
        if off.any():
            problems.append(f"四元数范数 {name}: {off.sum()} 帧偏离 1 (min={qn.min():.4f} max={qn.max():.4f})")
        print(f"[2] |q| {name}: mean={qn.mean():.5f} 极端 [{qn.min():.4f},{qn.max():.4f}] 异常 {off.sum()}")

    # 3. 夹爪
    for name, col in [("grip.L", 7), ("grip.R", 15)]:
        out = ((st[:, col] < -1e-4) | (st[:, col] > 1 + 1e-4))
        if out.any():
            problems.append(f"夹爪{name} 越界: {out.sum()} 帧 [{st[:,col].min():.3f},{st[:,col].max():.3f}]")
        print(f"[3] {name} range [{st[:, col].min():.3f},{st[:, col].max():.3f}] 越界 {out.sum()}")

    # 4. 位置工作空间
    for name, sl in [("L", slice(0, 3)), ("R", slice(8, 11))]:
        p = st[:, sl]
        out = (np.abs(p[:, 0]) > 1.0) | (np.abs(p[:, 1]) > 0.6) | (p[:, 2] < 0.3) | (p[:, 2] > 1.6)
        if out.any():
            problems.append(f"位置{name} 超工作空间: {out.sum()} 帧")
        print(f"[4] 位置{name} x[{p[:,0].min():.2f},{p[:,0].max():.2f}] y[{p[:,1].min():.2f},{p[:,1].max():.2f}] z[{p[:,2].min():.2f},{p[:,2].max():.2f}] 异常 {out.sum()}")

    # 5. 逐帧跳变 (逐 episode)
    jump_eps = {}
    for eid, idx in df.groupby("episode_index").indices.items():
        s = st[idx]
        if len(s) < 2:
            continue
        d_pos = np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1) + np.linalg.norm(np.diff(s[:, 8:11], axis=0), axis=1)
        d_rot = np.maximum(ang_err_deg(s[1:, 3:7], s[:-1, 3:7]), ang_err_deg(s[1:, 11:15], s[:-1, 11:15]))
        j = (d_pos > 0.10).sum() + (d_rot > 30).sum()
        if j:
            jump_eps[int(eid)] = int(j)
    if jump_eps:
        problems.append(f"跳变: {len(jump_eps)} 个 episode 共 {sum(jump_eps.values())} 帧")
    print(f"[5] 逐帧跳变: {len(jump_eps)} eps / {sum(jump_eps.values())} 帧")

    # 6. 时间戳
    ts_bad = 0
    for eid, idx in df.groupby("episode_index").indices.items():
        t = df["timestamp"].values[idx]
        d = np.diff(t)
        ts_bad += int(((d <= 0) | (np.abs(d - 0.04) > 0.008)).sum())
    if ts_bad:
        problems.append(f"时间戳异常: {ts_bad} 帧间隔")
    print(f"[6] 时间戳异常间隔: {ts_bad}")

    # 7. action 对齐度
    pos_err, rot_err, grp_err = [], [], []
    for eid, idx in list(df.groupby("episode_index").indices.items())[:200]:
        s, a = st[idx], ac[idx]
        if len(s) < 2:
            continue
        pos_err.append(np.abs(a[:-1, :3] - s[1:, :3]).mean() * 1000)
        pos_err.append(np.abs(a[:-1, 8:11] - s[1:, 8:11]).mean() * 1000)
        rot_err.append(ang_err_deg(a[:-1, 3:7], s[1:, 3:7]).mean())
        rot_err.append(ang_err_deg(a[:-1, 11:15], s[1:, 11:15]).mean())
        grp_err.append(np.abs(a[:-1, 7] - s[1:, 7]).mean())
        grp_err.append(np.abs(a[:-1, 15] - s[1:, 15]).mean())
    print(f"[7] action[t] vs state[t+1]: 位置 {np.mean(pos_err):.3f}mm (p99 {np.percentile(pos_err,99):.2f}) | "
          f"旋转 {np.mean(rot_err):.3f}° (p99 {np.percentile(rot_err,99):.2f}) | 夹爪 {np.mean(grp_err):.4f}")

    # 8. 呆滞轨迹
    dead = []
    for eid, idx in df.groupby("episode_index").indices.items():
        s = st[idx]
        travel = np.linalg.norm(s[:, :3].max(0) - s[:, :3].min(0)) + np.linalg.norm(s[:, 8:11].max(0) - s[:, 8:11].min(0))
        grip_static = np.all(np.abs(np.diff(s[:, [7, 15]], axis=0)).max() < 1e-4)
        if travel < 0.02 and grip_static:
            dead.append(int(eid))
    if dead:
        problems.append(f"呆滞轨迹: {len(dead)} eps {dead[:5]}")
    print(f"[8] 呆滞轨迹: {len(dead)} eps")

    # 9. 帧数一致性
    len_bad = 0
    for _, r in eps.iterrows():
        rows = int((df.episode_index == r.episode_index).sum())
        if rows != int(r.length):
            len_bad += 1
    if len_bad:
        problems.append(f"帧数不一致: {len_bad} eps")
    print(f"[9] 帧数不一致 eps: {len_bad}")

    # 10. 重复帧
    dup = 0
    for eid, idx in list(df.groupby("episode_index").indices.items())[:200]:
        s = st[idx]
        dup += int((np.abs(np.diff(s, axis=0)).max(1) == 0).sum())
    print(f"[10] 完全重复帧 (前200eps): {dup}")

    print(f"\n总帧数 {n} / episodes {len(eps)}")
    if problems:
        print("\n===== 发现问题 =====")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\n✅ 全部检查通过, 无异常")


if __name__ == "__main__":
    main()
