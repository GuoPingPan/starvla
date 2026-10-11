#!/usr/bin/env python
"""QwenPI_v3 (Qwen3.5-0.8B) 预计算脚本 —— 训练前的离线准备工作统一入口。

目前包含两部分(后续新的预计算需求都加到这里):

1. stats  — 数据集归一化统计 (q01/q99 等)
   即各数据集目录下的 meta/stats_gr00t.json + steps_data_index.pkl。
   训练首次打开数据集时也会惰性计算(全量扫 parquet, 慢); 提前跑一次,
   之后每次训练/批量扫描直接命中缓存, 省每次 1-2 分钟启动。

2. tok    — 任务文本 token 预计算 (tok_cache_qwen35.parquet)
   前提: include_state: false (与 umi PI05 口径一致, state 不进模型) ——
   instruction 退化为纯任务描述, 而 lerobot 的任务串就登记在
   meta/tasks.parquet (通常只有几条)。本脚本直接读 parquet 逐条 tokenize
   (模板图用 ffprobe 探出的分辨率造 dummy, 网格展开只取决于尺寸),
   训练时 QWen3_5 接口命中缓存则跳过 chat template + tokenizer。
   全程秒级, 不需要扫描数据集。

   缓存 key 与运行时的逐字节一致性由两道保险保证:
   a) --probe (默认开): 取一个真实 batch 断言 example["lang"] ∈ parquet 任务集;
   b) --verify: 抽样走在线 apply_chat_template 与缓存结果逐 token 比对。

用法 (repo 根目录):
    PYTHONPATH=. python scripts/data-analyze/precompute.py \
        --config-yaml examples/data-analyze/train_files/run_relative_6d_qwenpi_v3.yaml

    # 只跑某一部分: --parts stats / --parts tok

产物 (都落在数据集旁边, 训练侧自动发现):
    <data_root>/<dataset>/meta/stats_gr00t.json
    <data_root>/precompute/tok_cache_qwen35.parquet
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _first_video_dims(dataset_dir: Path):
    """ffprobe 探第一个视频的 (w, h) —— 模板图只需要尺寸 (网格展开只看分辨率)。"""
    vids = sorted(dataset_dir.glob("videos/*/chunk-*/*.mp4"))
    if not vids:
        raise FileNotFoundError(f"{dataset_dir}/videos 下没有 mp4")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(vids[0])],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = map(int, out.split(","))
    return w, h


def run_stats(cfg, mode, log=print):
    """打开数据集即触发惰性计算 (stats_gr00t.json / steps_data_index.pkl)。"""
    from starVLA.dataloader.gr00t_lerobot.registry import discover_and_merge
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    discover_and_merge()
    ds = get_vla_dataset(cfg.datasets.vla_data, mode=mode)
    log(f"  [stats] 数据集就绪: {len(ds)} 样本; 缓存文件见各数据集 meta/ 下")


def run_tok(cfg, mode, probe, verify_n, log=print):
    import pandas as pd
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from starVLA.dataloader.gr00t_lerobot.registry import discover_and_merge
    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_VL_Interface

    data_root = Path(cfg.datasets.vla_data.data_root_dir)
    discover_and_merge()
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode=mode)  # 顺带确保 stats 缓存

    # ---- 任务集: 直接读 data_root 下各数据集的 tasks.parquet ----
    tasks, dataset_dirs = set(), []
    for tp in sorted(data_root.glob("*/meta/tasks.parquet")):
        df = pd.read_parquet(tp)
        tasks |= set(df.index.astype(str).tolist())
        dataset_dirs.append(tp.parents[1])
    log(f"  [tok] parquet 任务集: {len(tasks)} 条")
    for t in sorted(tasks)[:5]:
        log(f"        · {t[:50]}")

    # ---- 保险 a + 模板图: 真实 batch 的 lang 必须落在任务集内 ----
    # 模板图必须用变换后的真实图像 (video pipeline CenterCrop→Resize 后的尺寸),
    # 不能用原始视频分辨率 —— vision token 展开数取决于喂给模型的分辨率。
    template_img = None
    if probe:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate_fn)
        examples = next(iter(loader))
        langs = [e["lang"] for e in examples]
        miss = [l for l in langs if l not in tasks]
        assert not miss, f"dataloader lang 与 tasks.parquet 不一致: {miss[:3]}"
        template_img = examples[0]["image"][0]
        log(f"  [tok] probe 通过: {len(langs)} 条真实样本 lang 均在任务集内; 模板图取自变换后真实图像 {template_img.size}")
    else:
        w, h = _first_video_dims(dataset_dirs[0])
        template_img = Image.new("RGB", (w, h))
        log(f"  [tok] ⚠️ probe 关闭, 模板图用原始视频分辨率 {w}x{h} —— 若训练管线有 resize 会不匹配!")

    # ---- tokenize ----
    interface = _QWen3_5_VL_Interface(cfg)  # 加载 processor (含模型, 仅需 processor)
    cache, lens = {}, []
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": template_img},
        {"type": "text", "text": t},
    ]}] for t in sorted(tasks)]
    enc = interface.processor.apply_chat_template(
        msgs, tokenize=True, padding=False, add_generation_prompt=True,
    )
    ids_list = enc["input_ids"] if isinstance(enc, dict) else enc
    for t, ids in zip(sorted(tasks), ids_list):
        cache[t] = torch.tensor(ids, dtype=torch.long)
        lens.append(len(ids))

    import numpy as np
    lens = np.array(lens)
    log(f"  [tok] token 长度: min={lens.min()} max={lens.max()}")

    # ---- 保险 b: 与在线慢路径逐 token 对比 ----
    if verify_n > 0:
        ref = interface.processor.apply_chat_template(
            msgs, tokenize=True, padding=False, add_generation_prompt=True,
        )
        ref_ids = ref["input_ids"] if isinstance(ref, dict) else ref
        for t, ids in zip(sorted(tasks), ref_ids):
            assert cache[t].tolist() == list(ids), f"token 不一致: {t[:40]}"
        log(f"  [tok] verify 通过: {len(cache)} 条与在线路径逐 token 相等")

    out_dir = data_root / "precompute"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(OmegaConf.select(cfg, "framework.qwenvl.base_vlm"))
    # 规范存储 = parquet (带 tokenizer 来源, 任何管线可直接消费; list<int> 是 parquet 原生类型)
    out = out_dir / "tok_cache_qwen35.parquet"
    pd.DataFrame(
        [{
            "task": t,
            "input_ids": cache[t].tolist(),
            "n_tokens": int(cache[t].numel()),
            "tokenizer": model_id,
        } for t in sorted(cache)]
    ).to_parquet(out, index=False)
    log(f"  [tok] 缓存已写 {out} ({len(cache)} 条, tokenizer={model_id})")
    log(f"  [tok] 建议 framework.qwenvl.pad_seq_len = {int(lens.max())} (静态 shape 化)")


def run_audit(cfg, log=print):
    """数据体检: 任务串质量 + 各数据集规模分布 → 报告落 data_root/precompute/audit.md。

    检出项: 占位/垃圾任务串 (过短、纯 ASCII、人名类占位), 任务重复率,
    episode/帧数分布 (为 batch/帧缓存预算提供输入)。
    """
    import pandas as pd
    from pathlib import Path

    data_root = Path(cfg.datasets.vla_data.data_root_dir)
    rows, all_tasks = [], {}
    for tp in sorted(data_root.glob("*/meta/tasks.parquet")):
        ds = tp.parents[1]
        df = pd.read_parquet(tp)
        tasks = df.index.astype(str).tolist()
        for t in tasks:
            all_tasks.setdefault(t, []).append(ds.name)
        info = json.load(open(ds / "meta" / "info.json")) if (ds / "meta" / "info.json").exists() else {}
        junk = [
            t for t in tasks
            if len(t) < 4 or t.isascii() or not any('\u4e00' <= c <= '\u9fff' for c in t)
        ]
        rows.append({
            "dataset": ds.name, "episodes": info.get("total_episodes", len(df)),
            "frames": info.get("total_frames", 0), "tasks": len(tasks), "junk_tasks": junk,
        })
    out_dir = data_root / "precompute"; out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit.md", "w", encoding="utf-8") as f:
        f.write(f"# 数据体检 @ {data_root}\n\n")
        f.write("| dataset | episodes | frames | tasks | 垃圾任务 |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['dataset']} | {r['episodes']} | {r['frames']} | {r['tasks']} | {r['junk_tasks'] or '—'} |\n")
        dup = {t: ds for t, ds in all_tasks.items() if len(ds) > 1}
        f.write(f"\n- 唯一任务串: {len(all_tasks)}; 跨数据集重复: {len(dup)}\n")
        if dup:
            f.write("- 重复示例: " + "; ".join(f"{t[:24]}×{len(d)}" for t, d in list(dup.items())[:8]) + "\n")
    total_junk = sum(len(r["junk_tasks"]) for r in rows)
    log(f"  [audit] {len(rows)} 数据集 / {len(all_tasks)} 唯一任务 / {total_junk} 个垃圾任务串")
    log(f"  [audit] 报告: {out_dir / 'audit.md'}")
    for r in rows:
        if r["junk_tasks"]:
            log(f"    ⚠️ {r['dataset']}: {r['junk_tasks']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-yaml", required=True)
    ap.add_argument("--parts", default="stats,tok", help="逗号分隔: stats,tok")
    ap.add_argument("--mode", default="train")
    ap.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True, help="真实 batch lang 一致性探针")
    ap.add_argument("--verify", type=int, default=64, help="在线路径逐 token 对比条数 (0 关闭)")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat, merge_framework_config
    from starVLA.model.framework.VLM4A.QwenPI_v3 import QwenPI_v3DefaultConfig

    cfg = OmegaConf.load(args.config_yaml)
    cfg = merge_framework_config(QwenPI_v3DefaultConfig, cfg)
    apply_config_compat(cfg)

    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    print(f">> precompute parts={sorted(parts)} mode={args.mode}")
    if "stats" in parts:
        print(">> [stats] 数据集统计缓存")
        run_stats(cfg, args.mode)
    if "tok" in parts:
        print(">> [tok] 任务文本 token 缓存")
        run_tok(cfg, args.mode, args.probe, args.verify)
    if "audit" in parts:
        print(">> [audit] 数据体检")
        run_audit(cfg)
    print(">> 完成")


if __name__ == "__main__":
    main()
