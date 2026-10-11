#!/usr/bin/env python3
"""DDL 导出包 (zip 内含 *.parquet 清单) → zerith_pi05 data_config.py 的 mixture 条目。

DDL job 导出的 zip 里是一个 parquet 清单, 每行 = 一个采集数据集, 含
``collect_data_id`` (数据集 id)、``robot_type``、``task_name``、``bos_path``
(BOS 上该数据集的 zip URI) 等列。本脚本把全部行打包成 **单个** mixture
(默认名 ``ddl_job_<job_id>``) 并写回 ``data_config.py``。

生成条目格式与手工条目完全一致 (4 元组, 第 4 元素 = BOS URI):
    ("20260719185726_lerobot_0f437b52", 1.0, "umi_2gripper_relative_6d",
     "bos://robot-data-from-disk/.../....zip"),

注: 训练 (lerobot_datasets.py) 用 ``*_`` 忽略第 4 元素; 仅 train.sh 的
download_data.py 读它来拉数据。

幂等: 同一 job 重复跑会替换旧块 (用哨兵注释定位), 不会重复堆积。

用法:
    python scripts/ddl_to_data_config.py /path/to/ddl-job-4715dbbe.zip
    python scripts/ddl_to_data_config.py ddl-job-4715dbbe.zip --mixture-name my_mix
    python scripts/ddl_to_data_config.py ddl-job-4715dbbe.zip --dry-run
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import zipfile
from pathlib import Path

# parquet 读取: 优先 pyarrow, 回退 pandas
try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
# scripts/ 在 repo 根下; 默认写到 zerith_pi05 的 data_config.py (本脚本只懂 UMI robot_type)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CONFIG = (
    REPO_ROOT
    / "examples"
    / "realRobots"
    / "zerith_pi05"
    / "train_files"
    / "data_registry"
    / "data_config.py"
)

# parquet robot_type → data_config.py 里 ROBOT_TYPE_CONFIG_MAP 的 key
ROBOT_TYPE_MAP = {
    "UMI_2gripper": "umi_2gripper_relative_6d",
}

SENTINEL_BEGIN_FMT = "    # <<<< AUTO-GENERATED from {job} >>>>"
SENTINEL_END_FMT = "    # <<<< END AUTO-GENERATED from {job} >>>>"


# ---------------------------------------------------------------------------
# 读 DDL zip / parquet
# ---------------------------------------------------------------------------
def _read_parquet_from_zip(zip_path: Path):
    """打开 zip, 找到第一个 *.parquet, 返回 pandas DataFrame。"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        parquets = [n for n in zf.namelist() if n.lower().endswith(".parquet")]
        if not parquets:
            sys.exit(f"[ERROR] zip 里没找到 .parquet 文件: {zip_path}")
        if len(parquets) > 1:
            print(f"[WARN] zip 里有 {len(parquets)} 个 parquet, 只用第一个: {parquets[0]}")
        with zf.open(parquets[0]) as fh:
            if pq is not None:
                return pq.read_table(fh).to_pandas()
            import pandas as pd  # 回退
            return pd.read_parquet(fh)


def _required_columns(df):
    need = ["collect_data_id", "robot_type", "bos_path"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] parquet 缺少必要列 {missing}; 实际列: {list(df.columns)}")
    if "task_name" not in df.columns:
        df["task_name"] = ""


def _resolve_robot_types(df, override):
    """返回 {row_idx: config_key}。遇到未知且无 override 时报错退出。"""
    if override:
        return {i: override for i in df.index}

    mapping, unknown = {}, set()
    for i, rt in df["robot_type"].items():
        rt = (rt or "").strip()
        cfg = ROBOT_TYPE_MAP.get(rt)
        if cfg is None:
            unknown.add(rt)
        else:
            mapping[i] = cfg
    if unknown:
        sys.exit(
            f"[ERROR] parquet 里出现未登记的 robot_type {sorted(unknown)}。\n"
            f"        已知映射: {ROBOT_TYPE_MAP}\n"
            f"        用 --robot-type <key> 强制覆盖 (key 须在 ROBOT_TYPE_CONFIG_MAP 里)。"
        )
    return mapping


def _job_id_from_zip(zip_path: Path) -> str:
    """ddl-job-4715dbbe.zip → 4715dbbe; 兜底用去后缀的 stem。"""
    m = re.search(r"ddl-job-([0-9a-zA-Z]+)", zip_path.name)
    return m.group(1) if m else zip_path.stem


# ---------------------------------------------------------------------------
# 生成 mixture 条目文本 (dict 内层, 不含外层花括号)
# ---------------------------------------------------------------------------
def _fmt_tuple(dataset_id, weight, robot_type, bos_uri, task_name):
    """格式化成 4 元组两行, 与手工条目一致的缩进 (8 / 9 空格)。

    注: robot_type 后必须有逗号, 否则 "rt" + 换行 + "bos" 会被 Python 当成
    隐式字符串拼接 → 静默得到错误的 3 元组 (无 BOS URI)。
    """
    cmt = f"  # {task_name}" if task_name else ""
    return [
        f'        ("{dataset_id}", {weight}, "{robot_type}",{cmt}',
        f'         "{bos_uri or ""}"),',
    ]


def build_block_lines(df, robot_type_map, mixture_name, weight):
    """把所有行打包成一个 mixture, 返回源文本行列表。"""
    lines = [f'    "{mixture_name}": [']
    for i in df.index:
        row = df.loc[i]
        lines += _fmt_tuple(
            row["collect_data_id"],
            weight,
            robot_type_map[i],
            row["bos_path"],
            row["task_name"],
        )
    lines.append("    ],")
    return lines


# ---------------------------------------------------------------------------
# 写回 data_config.py (幂等: 哨兵注释定位 + 替换)
# ---------------------------------------------------------------------------
def _existing_mixture_keys(config_path: Path) -> set:
    """ast 解析 data_config.py, 返回 DATASET_NAMED_MIXTURES 里已有的 key 集合。"""
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DATASET_NAMED_MIXTURES":
                    if isinstance(node.value, ast.Dict):
                        return {
                            k.value for k in node.value.keys if isinstance(k, ast.Constant)
                        }
    return set()


def _auto_block_keys(src_lines):
    """扫所有历史自动块里的 mixture key (形如 ``    "name": [``), 供碰撞豁免。"""
    keys, in_block = set(), False
    for ln in src_lines:
        if ln.startswith("    # <<<< AUTO-GENERATED from "):
            in_block = True
            continue
        if ln.startswith("    # <<<< END AUTO-GENERATED from "):
            in_block = False
            continue
        if in_block:
            m = re.match(r'\s+"([^"]+)":\s*\[', ln)
            if m:
                keys.add(m.group(1))
    return keys


def _dict_end_line(src_text: str) -> int:
    """DATASET_NAMED_MIXTURES 字典闭合 ``}`` 的行号 (1-indexed)。"""
    tree = ast.parse(src_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DATASET_NAMED_MIXTURES":
                    return node.end_lineno
    raise RuntimeError("在 data_config.py 里没找到 DATASET_NAMED_MIXTURES 赋值。")


def edit_data_config(config_path, job_label, mixture_name, new_lines, dry_run, force):
    """把 new_lines 插到 DATASET_NAMED_MIXTURES 闭合括号前 (替换同 job 旧块)。"""
    begin = SENTINEL_BEGIN_FMT.format(job=job_label)
    end = SENTINEL_END_FMT.format(job=job_label)

    # 1) 碰撞检查: 要写的 key 不能撞上手工条目 (非自动块里的同名 key)
    src_lines = config_path.read_text(encoding="utf-8").splitlines()
    existing = _existing_mixture_keys(config_path)
    auto_keys = _auto_block_keys(src_lines)
    if mixture_name in existing and mixture_name not in auto_keys and not force:
        sys.exit(
            f"[ERROR] mixture 名 {mixture_name!r} 与已有手工条目冲突。\n"
            f"        改 --mixture-name, 或加 --force 覆盖。"
        )

    # 2) 删掉同 job 的旧自动块 (幂等)
    filtered, in_block = [], False
    for ln in src_lines:
        if ln.startswith(begin):
            in_block = True
            continue
        if in_block and ln.startswith(end):
            in_block = False
            continue
        if in_block:
            continue
        filtered.append(ln)

    # 3) 重新定位闭合行号 (删过块, 行号变了)
    end_line = _dict_end_line("\n".join(filtered))

    # 4) 拼新块 (前导空行 + 哨兵)
    block = ["", begin, f"    # {mixture_name}: {sum(1 for l in new_lines if l.strip().startswith('('))} 个数据集"]
    block += new_lines
    block += [end]

    out = filtered[: end_line - 1] + block + filtered[end_line - 1:]

    if dry_run:
        print("\n===== --dry-run: 将写入以下内容 (插到 DATASET_NAMED_MIXTURES 末尾) =====")
        print("\n".join(block))
        return

    config_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[OK] 已写入 mixture {mixture_name!r} ({sum(1 for l in new_lines if l.strip().startswith('('))} 个数据集) 到 {config_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="DDL zip (parquet 清单) → zerith_pi05 data_config.py 的单个 mixture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("zip_path", type=Path, help="DDL 导出的 zip 文件路径")
    ap.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help=f"目标 data_config.py (默认: {DEFAULT_DATA_CONFIG})",
    )
    ap.add_argument(
        "--mixture-name",
        default=None,
        help="mixture 名 (默认: ddl_job_<job_id>)",
    )
    ap.add_argument("--robot-type", default=None, help="强制 robot_type (覆盖自动映射)")
    ap.add_argument("--weight", type=float, default=1.0, help="每个数据集的权重 (默认 1.0)")
    ap.add_argument("--job-id", default=None, help="覆盖从 zip 名解析出的 job id")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要写入的内容, 不改文件")
    ap.add_argument("--force", action="store_true", help="同名 mixture 与手工条目冲突时强制覆盖")
    args = ap.parse_args()

    if not args.zip_path.is_file():
        sys.exit(f"[ERROR] zip 不存在: {args.zip_path}")
    if not args.data_config.is_file():
        sys.exit(f"[ERROR] data_config.py 不存在: {args.data_config}")

    job_id = args.job_id or _job_id_from_zip(args.zip_path)
    job_label = f"ddl-job-{job_id}"
    mixture_name = args.mixture_name or f"ddl_job_{job_id}"

    df = _read_parquet_from_zip(args.zip_path)
    _required_columns(df)
    df = df.dropna(subset=["collect_data_id"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["collect_data_id"]).reset_index(drop=True)
    robot_type_map = _resolve_robot_types(df, args.robot_type)

    lines = build_block_lines(df, robot_type_map, mixture_name, args.weight)

    # 概览
    print(f"DDL job: {job_label}")
    print(f"数据集数: {len(df)}  → mixture: {mixture_name!r}")
    print(f"robot_type 映射: {sorted(set(robot_type_map.values()))}")
    for i in df.index:
        print(
            f"  - {df.at[i, 'collect_data_id']}  "
            f"[{robot_type_map[i]}]  {str(df.at[i, 'task_name'])[:40]}"
        )

    edit_data_config(args.data_config, job_label, mixture_name, lines, args.dry_run, args.force)


if __name__ == "__main__":
    main()
