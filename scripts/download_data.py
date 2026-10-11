#!/usr/bin/env python3
"""download_data.py — 按 yaml + data_config 从对象存储拉数据集并解压 (双云)。

流程:
  1. 解析 <config.yaml> 取 datasets.vla_data.{data_root_dir, data_mix}
  2. 从 data_config.py 抽 DATASET_NAMED_MIXTURES[data_mix] (ast.literal_eval, 不 import starVLA)
  3. 对每个 (dataset_id, weight, robot_type, cloud_uri, ...) 元组:
       - 若带 cloud_uri (第4个元素): 按 scheme 分流下载 (oss:// 走阿里云, bos:// 走百度云)
         -> zip 解压到 <data_root_dir>/<dataset_id>/ , 目录 URI (尾 /) 递归同步
       - 否则: warn-skip (如 hefei/shenzhen_2 暂未配路径)
  4. 幂等: <目标>/meta/info.json 已存在则跳过 (zip); 目录 URI 按文件级同大小跳过

URI 重映射 (跨云镜像数据, 无需改 data_config 里数百条 bos:// 条目):
  环境变量 DATA_URI_REMAP="旧前缀=新前缀[,旧前缀2=新前缀2]"  (最长前缀优先)
  例: DATA_URI_REMAP="bos://robot-data-from-disk/=oss://algo-public0/robot-data-from-disk/"
  train.sh 在 OSS 环境下会默认注入一组重映射 (可用该变量覆盖)。

用法:
  python scripts/download_data.py <config.yaml> [--dry-run]
  python scripts/download_data.py <config.yaml> --data-config path/to/data_config.py
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_DATA_CONFIG = "examples/realRobots/zerith_pi05/train_files/data_registry/data_config.py"

# 复用同目录 cloud_io 的下载能力 (也走环境变量凭证 / DSW 凭证服务)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cloud_io  # noqa: E402

_scheme_clients: dict[str, object] = {}


def _client_for(scheme: str):
    if scheme not in _scheme_clients:
        _scheme_clients[scheme] = cloud_io.make_client(scheme)
    return _scheme_clients[scheme]


# --------------------------------------------------------------------------- #
# URI 重映射 (跨云镜像)
# --------------------------------------------------------------------------- #
def load_remap() -> list[tuple[str, str]]:
    """解析 DATA_URI_REMAP (逗号分隔 old=new), 按旧前缀长度降序 (最长前缀优先)。"""
    raw = os.environ.get("DATA_URI_REMAP", "").strip()
    if not raw:
        return []
    pairs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            print(f"  ⚠ DATA_URI_REMAP 条目缺少 '=': {chunk!r}, 忽略", file=sys.stderr)
            continue
        old, new = chunk.split("=", 1)
        pairs.append((old.strip(), new.strip()))
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def remap_uri(uri: str, remap: list[tuple[str, str]]) -> tuple[str, bool]:
    """返回 (重映射后的 uri, 是否命中)。最长前缀优先。"""
    for old, new in remap:
        if uri.startswith(old):
            return new + uri[len(old) :], True
    return uri, False


# --------------------------------------------------------------------------- #
def load_yaml_fields(config_yaml: str) -> tuple[str, str]:
    """返回 (data_root_dir, data_mix)。优先 OmegaConf, 回退 PyYAML。"""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_yaml)
        return cfg.datasets.vla_data.data_root_dir, cfg.datasets.vla_data.data_mix
    except Exception:
        import yaml

        with open(config_yaml) as f:
            d = yaml.safe_load(f)
        vd = d["datasets"]["vla_data"]
        return vd["data_root_dir"], vd["data_mix"]


def extract_named_mixtures(data_config_path: str) -> dict:
    """用 ast 安全抽取 data_config.py 里的 DATASET_NAMED_MIXTURES (不执行 import)。"""
    tree = ast.parse(Path(data_config_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "DATASET_NAMED_MIXTURES":
                    return ast.literal_eval(node.value)
    raise SystemExit(f"ERROR: {data_config_path} 里找不到 DATASET_NAMED_MIXTURES")


# --------------------------------------------------------------------------- #
def flatten_if_wrapped(target: Path):
    """若解压后 target 下只有一个子目录 (zip 自带一层包装), 把它的内容提上来。"""
    children = [p for p in target.iterdir() if not p.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir() and not (target / "meta" / "info.json").exists():
        wrapper = children[0]
        if (wrapper / "meta" / "info.json").exists():
            for item in wrapper.iterdir():
                shutil.move(str(item), str(target / item.name))
            wrapper.rmdir()


def download_one_dataset(
    dataset_id: str,
    cloud_uri: str,
    data_root_dir: str,
    dry_run: bool = False,
    remap: list[tuple[str, str]] | None = None,
) -> bool:
    target = Path(data_root_dir) / dataset_id
    remap = remap or []
    uri, hit = remap_uri(cloud_uri, remap)
    scheme, bucket, key = cloud_io.parse_cloud_uri(uri)
    is_dir = uri.endswith("/")
    # 幂等: zip 是原子下载, 已有 info.json 视为完整, 跳过;
    # 目录是逐文件续传 (download_dir 自己按同大小跳过/补缺), 绝不能只看 info.json 就整体跳过
    # —— 否则中断后无法续传 (info.json 通常很早就下完, 但大视频还没下)。
    if not is_dir and (target / "meta" / "info.json").exists():
        print(f"  ✓ 已存在, 跳过: {target}")
        return True

    print(
        f"  {'[dry-run] ' if dry_run else ''}{dataset_id}: {uri}"
        f"{'  (重映射自 ' + cloud_uri + ')' if hit else ''}"
        f" {'(目录, 递归)' if is_dir else '(zip)'} -> {target}"
    )
    if dry_run:
        return True

    client = _client_for(scheme)
    target.mkdir(parents=True, exist_ok=True)
    if is_dir:
        # 目录 URI (尾 /): cloud_io.download_dir 递归拉全部对象到 target/, 幂等(同大小跳过、缺的补下)。
        # 适合大数据集 (RoboDojo ~100GB, 单 zip 不现实), 且中断后重跑自动续传。prefix 必须含尾 /。
        cloud_io.download_dir(client, bucket, key, target)
    else:
        # zip URI: 下载 zip -> 解压 -> (若 zip 自带一层包装则提上来)。
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / f"{dataset_id}.zip"
            client.download_file(bucket, key, zip_path)
            print(f"    解压 {zip_path.name} ({zip_path.stat().st_size // (1024*1024)} MB) ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
        flatten_if_wrapped(target)
    ok = (target / "meta" / "info.json").exists()
    print(f"    {'✓' if ok else '⚠ 下载后未见 meta/info.json, 请检查远端目录/zip 结构'}: {target}")
    return ok


# --------------------------------------------------------------------------- #
def resolve_data_config(config_yaml: str, explicit: str | None) -> str:
    """data_config.py 解析: 显式 > yaml 同级 data_registry > DEFAULT(zerith_pi05)。

    约定: 每个 benchmark 的 data_config 在 <yaml 所在目录>/data_registry/data_config.py
    (zerith_pi05 / RoboDojo / LIBERO 等都遵循)。这样 train.sh 无需为每个 benchmark 显式传
    --data-config。
    """
    if explicit:
        return explicit
    sibling = Path(config_yaml).resolve().parent / "data_registry" / "data_config.py"
    if sibling.is_file():
        return str(sibling)
    return DEFAULT_DATA_CONFIG


def main():
    ap = argparse.ArgumentParser(description="按 yaml+data_config 从对象存储拉数据集 (oss:// | bos://)")
    ap.add_argument("config_yaml", help="训练 yaml (含 datasets.vla_data.data_mix / data_root_dir)")
    ap.add_argument("--data-config", default=None, help="data_config.py 路径 (默认按 yaml 同级 data_registry 自动发现)")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不下载")
    args = ap.parse_args()

    data_root_dir, data_mix = load_yaml_fields(args.config_yaml)
    data_config_path = resolve_data_config(args.config_yaml, args.data_config)
    print(f"data_config  = {data_config_path}")
    mixtures = extract_named_mixtures(data_config_path)
    if data_mix not in mixtures:
        raise SystemExit(f"ERROR: data_mix={data_mix!r} 不在 DATASET_NAMED_MIXTURES。" f" 可用: {list(mixtures)}")

    remap = load_remap()
    if remap:
        print("URI 重映射 (DATA_URI_REMAP):")
        for old, new in remap:
            print(f"    {old} -> {new}")

    print(f"data_mix      = {data_mix}")
    print(f"data_root_dir = {data_root_dir}")
    print("目标布局     = <data_root_dir>/<dataset_id>/ (LeRobot v3.0)")

    entries = mixtures[data_mix]
    n_ok = n_skip = 0
    for entry in entries:
        dataset_id = entry[0]
        cloud_uri = entry[3] if len(entry) >= 4 else None
        if not cloud_uri:
            print(f"  ⚠ {dataset_id}: 无云端路径 (元组第4元素未配), 跳过 —— 训练会缺数据")
            n_skip += 1
            continue
        ok = download_one_dataset(dataset_id, cloud_uri, data_root_dir, dry_run=args.dry_run, remap=remap)
        n_ok += int(ok)

    print(f"\n完成: {n_ok} 拉取/已存在, {n_skip} 跳过(缺路径)")
    if n_skip:
        print("提示: 缺路径的数据集需在 data_config.py 的元组第4位补 oss:// 或 bos:// URI")


if __name__ == "__main__":
    main()
