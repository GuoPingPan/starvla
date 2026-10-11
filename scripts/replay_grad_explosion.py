#!/usr/bin/env python
"""PI05 梯度爆炸复现与取证 (针对 0825_172537 一类"训练 1w+ 步后发散")。

背景
----
0825_172537 run: 0-15800 步健康 (grad 0.4-10), 15900 步起缓慢发散
(5→1e2→1e3→1e5), 16300 步破 1e8, 34250 步 loss=nan。checkpoint 只存权重
(无优化器/RNG 状态), 加上 4 卡 DeepSpeed 梯度平均 + GPU 非确定性 +
torch.compile, **bit 级复现不可能**; 但该发散是数据×LR×权重状态的慢动力学,
从最后一个干净 checkpoint (steps_15000) 单卡重放可在邻近步区间复现同类爆炸
—— 数据顺序与原 run 不同 (从 epoch 头开始), 发散步数会漂移, 失稳模式应重现。

三种模式
--------
replay (默认)
    单卡重放训练: fp32 权重 + autocast bf16 + AdamW + clip(1.0), LR scheduler
    快进到 --ckpt_steps 使 LR 与原 run 对齐。每步记录 loss / 总梯度范数 /
    分模块梯度范数 (action_head / vlm.lm / vlm.vit 逐层); 首次超 --watch_thresh
    即 dump 案发现场: batch.pt / weights.pt / grad_ranking.txt / flow_debug.pt /
    meta.json, 之后继续训练直到 --max_dumps 次 dump。
bisect
    对已 dump 的"罪犯 batch" 二分定位肇事样本 (固定 flow time/noise 控制变量),
    最终输出单样本梯度范数排行 + 肇事样本逐 key 张量统计。
cross
    罪犯 batch × {干净权重 (--checkpoint), 病变权重 (--weights_pt)} 交叉测梯度,
    区分"坏数据"还是"坏权重"还是组合效应。

用法
----
# 1) 重放抓现场 (原 run 在 15.9k-16.3k 步区间发散, 从 15000 重放 4000 步覆盖)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/replay_grad_explosion.py \
    --config_yaml /root/data/model_outputs/umi_all_nohjerk_noproblem_smooth_oss_80000step/0825_172537/config.full.yaml \
    --checkpoint   /root/data/model_outputs/umi_all_nohjerk_noproblem_smooth_oss_80000step/0825_172537/checkpoints/steps_15000_pytorch_model.pt \
    --ckpt_steps 15000 --max_steps 4000 --batch_size 4 --watch_thresh 50 \
    --dump_dir /root/data/model_outputs/grad_forensics

# 2) 二分肇事样本 (用 replay 抓到的 batch.pt; --weights_pt 不填则用 --checkpoint)
python scripts/replay_grad_explosion.py --mode bisect --config_yaml <同上> \
    --checkpoint <干净或病变权重> --batch_pt <dump_dir>/step_*/batch.pt

# 3) 交叉测试 (需要 --weights_pt, 通常用 replay dump 的 weights.pt)
python scripts/replay_grad_explosion.py --mode cross --config_yaml <同上> \
    --checkpoint <干净 ckpt> --weights_pt <dump_dir>/step_*/weights.pt \
    --batch_pt <dump_dir>/step_*/batch.pt --num_normal_batches 3

判读
----
- replay: 看 replay_log.jsonl 的 group_* 字段 —— 哪个模块/层最先持续增长,
  就是爆炸源; flow_debug.pt 的 per_sample_loss / time 可看是否有离群样本。
- bisect: 单样本梯度若比其余样本大 1-2 个数量级 → 该样本是数据侧触发器。
- cross: 罪犯 batch 在干净权重上也炸 = 数据驱动; 正常 batch 在病变权重上也炸
  = 权重进入脆弱区 (edge of stability, 与 LR 过高一致); 只有组合才炸 = 交互。
"""

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path

# 必须在 import starVLA 之前设置: 关 torch.compile (便于定位 + 避免 dynamo 干扰)
os.environ.setdefault("STARVLA_COMPILE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from omegaconf import OmegaConf
from accelerate.utils import set_seed

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, setup_optimizer_and_scheduler

_LAYER_RE = re.compile(r"layers\.(\d+)")


def group_of(name: str) -> str:
    """把参数名归到粗粒度模块 (action expert / VLM LM / ViT, 各自逐层)。"""
    if name.startswith("action_head"):
        m = _LAYER_RE.search(name)
        return f"action_head.L{int(m.group(1)):02d}" if m else "action_head.misc"
    if "vision_tower" in name:
        m = _LAYER_RE.search(name)
        return f"vlm.vit.L{int(m.group(1)):02d}" if m else "vlm.vit.misc"
    if "language_model" in name:
        m = _LAYER_RE.search(name)
        return f"vlm.lm.L{int(m.group(1)):02d}" if m else "vlm.lm.misc"
    return "other"


def total_grad_norm(model) -> float:
    """总梯度 L2 范数; 任一梯度非有限时返回 nan (clip 对 nan 无效, 必须显式判)。"""
    total, bad = 0.0, False
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            bad = True
            continue
        total += float(g.norm()) ** 2
    return float("nan") if bad else math.sqrt(total)


def group_grad_norms(model) -> dict:
    """按 group_of 聚合的梯度范数 (sqrt of sum of squares), 降序。"""
    acc = {}
    for name, p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n = float("nan") if not torch.isfinite(g).all() else float(g.norm())
        grp = group_of(name)
        s = acc.setdefault(grp, [0.0, False])
        if n != n:
            s[1] = True
        else:
            s[0] += n * n
    return {k: (float("nan") if v[1] else math.sqrt(v[0])) for k, v in
            sorted(acc.items(), key=lambda kv: -kv[1][0])}


def load_state_dict_loose(model, path) -> None:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"📦 loaded {path}: {len(sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    for k in list(missing)[:3] + list(unexpected)[:3]:
        print(f"   e.g. {k}")


def forward_with_debug(model, examples, time=None, noise=None):
    """复刻 PI0.forward 的训练前向, 但拿到 return_debug 的 (x_t, u_t, time, noise)。"""
    batch = model._prepare_examples(examples, include_actions=True)
    observation = model._build_observation_from_batch(batch)
    device = model.action_head.action_in_proj.weight.device
    actions = batch["action"].to(device=device, dtype=torch.float32)
    out = model.forward_from_observation(
        observation,
        actions,
        noise=noise,
        time=time,
        return_debug=True,
        action_dim_actual=batch.get("action_dim_actual", actions.shape[-1]),
        action_pad=batch.get("action_pad"),
    )
    return out, batch


def per_sample_loss(out, batch) -> torch.Tensor:
    """(B,) 每 sample 的 masked MSE —— 找离群样本 (只裁特征维, horizon 维近似)。"""
    el = (out["v_t"] - out["u_t"]).pow(2)
    d = batch.get("action_dim_actual", None)
    if d is not None:
        if torch.is_tensor(d):
            d = int(d.reshape(-1)[0].item())
        elif isinstance(d, (list, tuple)):
            d = int(d[0])
        el = el[..., : int(d)]
    return el.mean(dim=tuple(range(1, el.ndim)))


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def build_world(args, need_data=True, need_optimizer=True):
    cfg = OmegaConf.load(args.config_yaml)
    cfg = apply_config_compat(cfg)
    cfg.output_dir = str(args.dump_dir)
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size:
        cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.datasets.vla_data.num_workers = args.num_workers

    set_seed(int(cfg.get("seed", 42)))
    model = build_framework(cfg)
    TrainerUtils.load_pretrained_backbones(model, cfg.trainer.get("pretrained_checkpoint", None))
    if args.checkpoint:
        load_state_dict_loose(model, args.checkpoint)
    TrainerUtils.freeze_backbones(model, freeze_modules=cfg.trainer.get("freeze_modules", "") or "")
    model = model.to(args.device)
    model.train()

    loader = optimizer = scheduler = None
    if need_data:
        loader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, mode="train")
    if need_optimizer:
        optimizer, scheduler = setup_optimizer_and_scheduler(model, cfg)
        if args.ckpt_steps > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(args.ckpt_steps):
                    scheduler.step()
            print(f"⏩ LR scheduler fast-forwarded {args.ckpt_steps} steps → lr={scheduler.get_last_lr()}")
    return cfg, model, loader, optimizer, scheduler


def dump_scene(args, model, examples, out, batch, step, raw_gn, groups, loss):
    d = args.dump_dir / f"step_{args.ckpt_steps + step}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(to_cpu(examples), d / "batch.pt")
    if args.save_weights and not (d.parent / f"{d.name}.weights_saved"):
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, d / "weights.pt")
        (d.parent / f"{d.name}.weights_saved").touch()
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        rows.append((name, float("nan") if not torch.isfinite(g).all() else float(g.norm())))
    rows.sort(key=lambda kv: (kv[1] == kv[1], -(kv[1] if kv[1] == kv[1] else 0.0)))
    with open(d / "grad_ranking.txt", "w") as f:
        for name, n in rows[:500]:
            f.write(f"{n:.6e}  {name}\n")
    torch.save(
        {
            "time": to_cpu(out["time"]),
            "per_sample_loss": to_cpu(per_sample_loss(out, batch)),
            "u_t_absmax": float(out["u_t"].abs().max()),
            "x_t_absmax": float(out["x_t"].abs().max()),
            "v_t_absmax": float(out["v_t"].abs().max()),
        },
        d / "flow_debug.pt",
    )
    meta = {
        "global_step": args.ckpt_steps + step,
        "loss": float(loss.item()),
        "grad_norm": raw_gn,
        "watch_thresh": args.watch_thresh,
        "groups": groups,
    }
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"🚨 DUMPED scene → {d}")


def run_replay(args):
    cfg, model, loader, optimizer, scheduler = build_world(args)
    log_path = args.dump_dir / "replay_log.jsonl"
    log_f = open(log_path, "w")
    dumps, step = 0, 0
    print(f">>> replay: ckpt_steps={args.ckpt_steps} max_steps={args.max_steps} "
          f"batch={cfg.datasets.vla_data.per_device_batch_size} thresh={args.watch_thresh}")
    for examples in loader:
        step += 1
        if step > args.max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, batch = forward_with_debug(model, examples)
            loss = out["action_loss"]
        loss.backward()
        raw_gn = total_grad_norm(model)
        groups = group_grad_norms(model)

        if step % args.log_every == 0:
            top = " ".join(f"{k}={v:.3g}" for k, v in list(groups.items())[:5])
            print(f"step {args.ckpt_steps + step} loss={float(loss.item()):.4f} "
                  f"grad={raw_gn:.4g} | {top}")
            log_f.write(json.dumps({
                "global_step": args.ckpt_steps + step,
                "loss": float(loss.item()),
                "grad_norm": raw_gn if math.isfinite(raw_gn) else None,
                "groups": groups,
            }) + "\n")
            log_f.flush()

        if (raw_gn != raw_gn) or raw_gn > args.watch_thresh:
            dump_scene(args, model, examples, out, batch, step, raw_gn, groups, loss)
            dumps += 1
            if dumps >= args.max_dumps:
                print(f"达到 max_dumps={args.max_dumps}, 停止重放")
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() != loss.item():
            print("loss 已 NaN (权重被毒化), 停止重放")
            break
    log_f.close()
    print(f"<<< replay 结束: {step} steps, {dumps} dumps, log → {log_path}")


def make_grad_probe(model):
    """返回 gn(examples, time, noise): 冻结 flow 采样后的子集梯度范数。"""

    def gn(examples, time=None, noise=None):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = forward_with_debug(model, examples, time=time, noise=noise)
        out["action_loss"].backward()
        return total_grad_norm(model), out

    return gn


def describe_example(ex, prefix=""):
    """逐 key 打印样本张量统计 —— 找离群值 / 非有限值 / 尺度异常。"""
    stats = {}
    for k, v in ex.items():
        if torch.is_tensor(v) and v.is_floating_point():
            finite = torch.isfinite(v)
            if finite.all():
                stats[k] = {"shape": list(v.shape), "min": float(v.min()), "max": float(v.max()),
                            "mean": float(v.mean()), "absmax": float(v.abs().max())}
            else:
                stats[k] = {"shape": list(v.shape), "nonfinite": int((~finite).sum())}
        elif torch.is_tensor(v):
            stats[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        else:
            stats[k] = {"value": str(v)[:200]}
    return stats


def run_bisect(args):
    _, model, _, _, _ = build_world(args, need_data=False, need_optimizer=False)
    if args.weights_pt:
        load_state_dict_loose(model, args.weights_pt)
    examples = torch.load(args.batch_pt, weights_only=False)
    n = len(examples)
    print(f">>> bisect: {n} examples, 阈值 {args.watch_thresh}")

    # 固定全 batch 的 flow time/noise, 子集复用对应切片 —— 控制变量, 只有样本变。
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full_out, _ = forward_with_debug(model, examples)
    time_full, noise_full = full_out["time"], full_out["noise"]
    gn = make_grad_probe(model)

    # 单样本整表 (2n-1 次 fwd/bwd 太浪费, 直接 1 次/样本; OOM 时用 --bisect_chunk)
    chunk = max(1, args.bisect_chunk)
    leaf_norms = []
    for s in range(0, n, chunk):
        idxs = list(range(s, min(s + chunk, n)))
        g, out = gn([examples[i] for i in idxs], time_full[idxs], noise_full[idxs])
        psl = per_sample_loss(out, {"action_dim_actual": args.action_dim_actual}).tolist()
        for j, i in enumerate(idxs):
            leaf_norms.append((float(g) if math.isfinite(g) else float("nan"), i, psl[j] if j < len(psl) else None))
        print(f"  chunk[{s}:{s + len(idxs)}] grad={g:.4g}")
    leaf_norms.sort(key=lambda t: (t[0] == t[0], -(t[0] if t[0] == t[0] else 0.0)))
    print("\n=== 单样本梯度排行 (top 10) ===")
    for g, i, psl in leaf_norms[:10]:
        print(f"  sample[{i}] grad={g:.4g} per_sample_loss={psl}")
    worst = leaf_norms[0][1]
    stats = describe_example(examples[worst])
    with open(args.dump_dir / "bisect_worst_sample.json", "w") as f:
        json.dump({"worst_index": worst, "grad_norm": leaf_norms[0][0], "keys": stats}, f, indent=2)
    print(f"肇事候选 sample[{worst}] → {args.dump_dir / 'bisect_worst_sample.json'}")


def run_cross(args):
    if not args.weights_pt:
        raise SystemExit("cross 模式需要 --weights_pt (病变权重, 通常取 dump 的 weights.pt)")
    _, model, loader, _, _ = build_world(args, need_data=True, need_optimizer=False)
    criminal = torch.load(args.batch_pt, weights_only=False)

    normals = []
    for i, examples in enumerate(loader):
        if i >= args.num_normal_batches:
            break
        normals.append(examples)

    def probe(examples, tag):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = forward_with_debug(model, examples)
        gn = make_grad_probe(model)
        g, _ = gn(examples, out["time"], out["noise"])
        print(f"    {tag}: grad={g:.4g}")
        return g

    print("=== A) 干净权重 (--checkpoint) ===")
    load_state_dict_loose(model, args.checkpoint)
    c_clean = probe(criminal, "criminal")
    n_clean = max(probe(nb, f"normal[{i}]") for i, nb in enumerate(normals))

    print("=== B) 病变权重 (--weights_pt) ===")
    load_state_dict_loose(model, args.weights_pt)
    c_sick = probe(criminal, "criminal")
    n_sick = max(probe(nb, f"normal[{i}]") for i, nb in enumerate(normals))

    t = args.watch_thresh
    print(f"\n=== 交叉表 (阈值 {t}) ===")
    print(f"            criminal_batch   normal_batches")
    print(f"clean_w     {c_clean:>12.4g}   {n_clean:>12.4g}")
    print(f"sick_w      {c_sick:>12.4g}   {n_sick:>12.4g}")
    if c_clean > t:
        print("结论: 数据驱动 —— 罪犯 batch 在干净权重上就炸 → 查该 batch 样本 (跑 bisect)。")
    elif n_sick > t:
        print("结论: 权重驱动 —— 病变权重对正常数据也炸 → edge-of-stability, 降 action_model LR / 提前 warmup。")
    elif c_sick > t:
        print("结论: 交互效应 —— 只有罪犯 batch × 病变权重组合才炸 → 两者都处理。")
    else:
        print("结论: 单卡未复现 (多卡平均/优化器状态贡献) → 用 replay 模式在训练动力学里抓。")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["replay", "bisect", "cross"], default="replay")
    p.add_argument("--config_yaml", required=True, help="失败 run 的 config.full.yaml")
    p.add_argument("--checkpoint", default=None, help="干净权重 (如 steps_15000_pytorch_model.pt)")
    p.add_argument("--weights_pt", default=None, help="病变权重 (dump 的 weights.pt; bisect/cross 用)")
    p.add_argument("--batch_pt", default=None, help="罪犯 batch (dump 的 batch.pt; bisect/cross 用)")
    p.add_argument("--ckpt_steps", type=int, default=0, help="scheduler 快进步数 (= ckpt 的全局步数)")
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=4, help="单卡重放批量 (显存所限, 默认 4)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--watch_thresh", type=float, default=50.0)
    p.add_argument("--max_dumps", type=int, default=3)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--bisect_chunk", type=int, default=8, help="bisect 单次前向样本数 (OOM 则调小)")
    p.add_argument("--action_dim_actual", type=int, default=10, help="per-sample loss 只统计前 N 维 (配置 action_dim_actual)")
    p.add_argument("--num_normal_batches", type=int, default=3)
    p.add_argument("--save_weights", action="store_true", default=True)
    p.add_argument("--no_save_weights", dest="save_weights", action="store_false")
    p.add_argument("--dump_dir", type=Path, default=Path("/root/data/model_outputs/grad_forensics"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.mode in ("bisect", "cross") and not args.batch_pt:
        p.error(f"{args.mode} 模式需要 --batch_pt")
    if args.mode == "replay":
        run_replay(args)
    elif args.mode == "bisect":
        run_bisect(args)
    else:
        run_cross(args)


if __name__ == "__main__":
    main()
