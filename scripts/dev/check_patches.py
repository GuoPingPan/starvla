#!/usr/bin/env python3
"""
scripts/dev/check_patches.py — 校验本分支改动的「上游既有文件」是否都登记在 upstream_patches.md

用途（agent 改完代码后自检）:
  对每个改动文件, 若它在上游 github/starVLA_dev 中存在 (= 对上游文件的修改, 非新增),
  则必须在 docs/zerith_docs/workflow/upstream_patches.md 的登记表里。否则报错提醒登记。

用法:
  python scripts/dev/check_patches.py                 # 检查 dev/main...HEAD 的改动
  python scripts/dev/check_patches.py --base <ref>     # 指定基线 (默认 dev/main)
  python scripts/dev/check_patches.py --upstream <ref> # 指定上游 ref (默认 github/starVLA_dev)

退出码: 0=全部已登记(或无上游改动); 1=有未登记的上游文件。

详见 docs/zerith_docs/workflow/upstream_patches.md。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHES_DOC = REPO_ROOT / "docs" / "zerith_docs" / "workflow" / "upstream_patches.md"
# 登记表行: | P0xx | `path` | ...
TABLE_ROW = re.compile(r"^\|\s*P\d{3}\s*\|\s*`([^`]+)`")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def changed_files(base: str) -> list[str]:
    """本分支相对 base 的改动文件 (含已提交 + 工作区未提交)。"""
    out = git("diff", "--name-only", f"{base}...HEAD")
    files = [f for f in out.splitlines() if f]
    # 加上工作区/暂存区未提交改动 (相对 HEAD)
    out2 = git("diff", "--name-only") + "\n" + git("diff", "--name-only", "--cached")
    for f in out2.splitlines():
        if f and f not in files:
            files.append(f)
    return files


def upstream_exists(upstream: str, path: str) -> bool:
    """该文件是否在上游 ref 中存在 (= 对上游文件的修改, 而非新增)。"""
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{upstream}:{path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def registered_files() -> set[str]:
    """从 upstream_patches.md 登记表解析已登记的上游文件路径。"""
    if not PATCHES_DOC.exists():
        return set()
    found: set[str] = set()
    for line in PATCHES_DOC.read_text(encoding="utf-8").splitlines():
        m = TABLE_ROW.match(line)
        if m:
            found.add(m.group(1).strip())
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="dev/main", help="集成分支基线 (默认 dev/main)")
    ap.add_argument("--upstream", default="github/starVLA_dev", help="上游 ref")
    args = ap.parse_args()

    registered = registered_files()
    changed = changed_files(args.base)

    unregistered: list[str] = []
    for f in changed:
        if not f or f.endswith("/"):
            continue
        if upstream_exists(args.upstream, f) and f not in registered:
            unregistered.append(f)

    if not unregistered:
        print(f"✅ 检查通过: {len(changed)} 个改动文件中, 无未登记的上游文件。")
        return 0

    print(f"❌ 发现 {len(unregistered)} 个未登记的上游文件 (修改了上游既有文件但不在登记表):", file=sys.stderr)
    for f in unregistered:
        print(f"   - {f}", file=sys.stderr)
    print(
        "\n请在 docs/zerith_docs/workflow/upstream_patches.md 的「修改的上游文件」表里登记, 并在代码里加 "
        "ZERITH PATCH 标记区块。\n详见该文档「标记区块约定」。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
