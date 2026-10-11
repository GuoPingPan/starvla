#!/usr/bin/env python3
"""bos_io.py — 统一的 BOS 读写助手 (boto3 / S3 兼容接口)。

凭证全部从环境变量读, 不写进脚本/仓库:
    BOS_ACCESS_KEY_ID      BOS Access Key ID
    BOS_SECRET_ACCESS_KEY  BOS Secret Access Key
    BOS_ENDPOINT           S3 兼容 endpoint, 默认 https://s3.bj.bcebos.com
    BOS_REGION             区域, 默认 bj

同一个 endpoint+凭证可访问同区域多个桶 (如 umi-gripper / robot-data-from-disk),
所以跨桶操作直接用完整 bos:// URI 即可, 无需切换凭证。

CLI 子命令 (供 train.sh 调用):
    ls        <bos_uri_or_bucket> [prefix]      列对象/子目录
    get       <bos_uri> <local_path>            下载单个对象
    get-dir   <bos_uri_prefix> <local_dir>      递归下载目录 (--exclude 可多次)
    put-dir   <local_dir> <bos_uri_prefix>      递归上传目录 (--exclude 可多次)
    src-zip   <sha|latest>                      按 commit sha 取源码 zip (starvla_res/, 人无关共享区)

URI 形如 bos://<bucket>/<key>; "prefix" 用 bos://<bucket>/<prefix>/ 表示目录。
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


# --------------------------------------------------------------------------- #
# 客户端 / URI
# --------------------------------------------------------------------------- #
def _missing_creds() -> list[str]:
    miss = [n for n in ("BOS_ACCESS_KEY_ID", "BOS_SECRET_ACCESS_KEY") if not os.environ.get(n)]
    return miss


def make_client():
    """从环境变量构造 boto3 S3 兼容 client (virtual-hosted, 与 bcecmd 一致)。"""
    miss = _missing_creds()
    if miss:
        print(f"ERROR: 缺少 BOS 凭证环境变量: {', '.join(miss)}\n"
              "请先 export BOS_ACCESS_KEY_ID / BOS_SECRET_ACCESS_KEY "
              "(以及可选 BOS_ENDPOINT / BOS_REGION)。", file=sys.stderr)
        sys.exit(2)
    try:
        import boto3
        from botocore.client import Config
    except ImportError:
        print("ERROR: 未安装 boto3。uv pip install boto3 / pip install boto3", file=sys.stderr)
        sys.exit(2)

    endpoint = os.environ.get("BOS_ENDPOINT", "https://s3.bj.bcebos.com")
    if endpoint and not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["BOS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BOS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("BOS_REGION", "bj"),
        config=Config(
            s3={"addressing_style": "virtual"},
            connect_timeout=30,  # 建连超时 (默认 60, 显式给值便于排障)
            read_timeout=60,     # 单次 socket 读超时; 超时后交给 standard retry 自动恢复
            # standard 模式对 ReadTimeout / ConnectionClosed 这类"卡住型"错误重试可靠;
            # legacy (默认) 有漏重试行为, 是 s3transfer future 无限等待的常见诱因。
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def parse_bos_uri(uri: str) -> tuple[str, str]:
    """bos://<bucket>/<key> -> (bucket, key)。key 可能为 '' (桶根)。"""
    if not uri.startswith("bos://"):
        raise ValueError(f"不是 bos:// URI: {uri}")
    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket:
        raise ValueError(f"URI 缺少 bucket: {uri}")
    return bucket, key


# --------------------------------------------------------------------------- #
# 列举
# --------------------------------------------------------------------------- #
def list_objects(client, bucket: str, prefix: str = "", delimiter: str = "/"):
    """返回 (common_prefixes, [(key, size), ...])。delimiter='' 时递归列全部对象。"""
    paginator = client.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    if delimiter:
        kwargs["Delimiter"] = delimiter
    dirs, files = [], []
    for page in paginator.paginate(**kwargs):
        for c in page.get("CommonPrefixes", []) or []:
            dirs.append(c["Prefix"])
        for o in page.get("Contents", []) or []:
            files.append((o["Key"], o.get("Size", 0)))
    return dirs, files


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
class _ReadTimeout(Exception):
    """看门狗触发的读超时 (对抗 boto3 read_timeout 在 HTTPS streaming body 上不生效)。"""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _ReadTimeout()


def _read_chunk_with_watchdog(body, n_bytes: int, timeout: float):
    """带硬超时的 body.read。

    用 SIGALRM 在 timeout 秒后强制打断阻塞的 socket recv —— 这是对抗"socket 半死
    (连接没断但数据停滞; HTTPS 下 SSL read 往往不响应 boto3 的 read_timeout)"的
    唯一可靠手段。超时 → raise _ReadTimeout → 由 download_file 捕获重试。
    仅主线程 + 有 SIGALRM 的平台 (Linux/Mac) 可用; 否则降级为裸 read。
    """
    import threading
    if not (hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()):
        return body.read(n_bytes)
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return body.read(n_bytes)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _bcecmd_download(bucket: str, key: str, local_path: Path) -> bool:
    """用百度官方 CLI bcecmd 下载单个对象。成功返回 True; 不可用/失败返回 False (调用方 fallback)。

    为什么优先它: 实测同机 boto3 下载 BOS 卡在 socket recv (HTTPS streaming body
    read_timeout 不生效), 而 bcecmd 秒下 (5MB 文件 0.13s)。bcecmd 自带断点续传/重试/并发,
    凭证走它自己的配置 (与 boto3 的 BOS_* 环境变量无关)。
    """
    import shutil, subprocess
    bce = shutil.which("bcecmd")
    if not bce:
        return False
    print(f"    ↓ [bcecmd] {local_path.name}", flush=True)
    try:
        r = subprocess.run(
            [bce, "bos", "cp", f"bos://{bucket}/{key}", str(local_path), "--yes", "--quiet"],
            capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("    ⚠ bcecmd 超时, fallback boto3", flush=True)
        return False
    if r.returncode != 0:
        print(f"    ⚠ bcecmd 失败 (rc={r.returncode}): {r.stderr.strip()[:200]} → fallback boto3",
              flush=True)
        return False
    return True


def download_file(client, bucket: str, key: str, local_path: str | Path,
                  max_retries: int = 6, chunk_mb: int = 8):
    """下载单个对象: 优先 bcecmd (百度官方 CLI, 稳), 不可用时 fallback 到 boto3 流式 + SIGALRM 看门狗。

    bcecmd 路径: subprocess 调 `bcecmd bos cp`, 自带断点续传/重试/并发 (本机实测比 boto3 快且不卡)。
    boto3 路径 (fallback): get_object 流式 + 断点续传 + SIGALRM 硬超时 + 指数退避重试。
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if _bcecmd_download(bucket, key, local_path):
        return
    tmp_path = local_path.with_name(local_path.name + ".part")
    name = Path(key).name or key

    for attempt in range(1, max_retries + 1):
        try:
            # 断点续传: .part 已存在则从其大小续下 (Range 请求), 不重头
            offset = tmp_path.stat().st_size if tmp_path.exists() else 0
            kwargs = {"Bucket": bucket, "Key": key}
            if offset:
                kwargs["Range"] = f"bytes={offset}-"
            resp = client.get_object(**kwargs)
            # ContentRange 形如 "bytes 100-999/5000" → 末段是真实总大小
            cr = resp.get("ContentRange", "") or ""
            total = int(cr.rsplit("/", 1)[-1]) if "/" in cr else (offset + int(resp.get("ContentLength", 0) or 0))

            seen = offset
            step = 10 * 1024 * 1024  # 每 10MB 打一次进度 (小文件也能看到动)
            nxt = (seen // step + 1) * step
            with resp["Body"] as body, open(tmp_path, "ab" if offset else "wb") as f:
                while True:
                    # 30s 硬看门狗: boto3 read_timeout 在 HTTPS streaming body 上不可靠,
                    # 用 SIGALRM 强制打断卡死的 socket recv, 超时则抛 _ReadTimeout 触发重试。
                    chunk = _read_chunk_with_watchdog(body, chunk_mb * 1024 * 1024, 30.0)
                    if not chunk:
                        break
                    f.write(chunk)
                    seen += len(chunk)
                    while seen >= nxt:
                        pct = f" ({seen * 100 // total}%)" if total else ""
                        print(f"    ↓ {name}: {seen // (1024 * 1024)} MB{pct}", flush=True)
                        nxt += step

            if total and tmp_path.stat().st_size < total:
                raise IOError(f"不完整 {tmp_path.stat().st_size}/{total}")
            os.replace(tmp_path, local_path)  # 原子重命名, 防半成品
            return
        except Exception as e:
            if attempt >= max_retries:
                raise
            wait = min(2 ** attempt, 30)
            print(f"    ⚠ 下载中断 [{type(e).__name__}: {e}], {wait}s 后从断点续传 "
                  f"({attempt}/{max_retries})", flush=True)
            time.sleep(wait)


def _excluded(rel: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(rel, p) or fnmatch("/" + rel, p) for p in patterns)


def download_dir(client, bucket: str, prefix: str, local_dir: str | Path,
                 exclude: list[str] | None = None, quiet: bool = False):
    """递归下载 <bucket>/<prefix>/ 下全部对象到 <local_dir>/, 重建子目录。"""
    exclude = exclude or []
    local_dir = Path(local_dir)
    _, files = list_objects(client, bucket, prefix=prefix, delimiter="")
    # prefix 形如 "Orion/pretrained/" -> 本地相对路径 = key 去掉 prefix
    n = 0
    for key, size in files:
        rel = key[len(prefix):] if key.startswith(prefix) else key
        if not rel or rel.endswith("/"):
            continue
        if _excluded(rel, exclude):
            continue
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == size:
            continue  # 已存在且同大小, 跳过 (幂等, 不重复下, 如 16GB 预训练)
        if not quiet:
            print(f"  ↓ {key} ({size} B) -> {dst}")
        download_file(client, bucket, key, dst)
        n += 1
    if not quiet:
        print(f"download_dir: {n} 个文件 -> {local_dir}")
    return n


# --------------------------------------------------------------------------- #
# 上传
# --------------------------------------------------------------------------- #
def upload_file(client, local_path: str | Path, bucket: str, key: str):
    client.upload_file(str(local_path), bucket, key)


def upload_dir(client, local_dir: str | Path, bucket: str, prefix: str,
               exclude: list[str] | None = None, quiet: bool = False):
    """递归上传 <local_dir>/ 下全部文件到 <bucket>/<prefix>/。"""
    from fnmatch import fnmatch
    exclude = exclude or []
    local_dir = Path(local_dir)
    prefix = prefix.rstrip("/") + "/"
    n = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        if any(fnmatch(rel, p) for p in exclude):
            continue
        key = prefix + rel
        if not quiet:
            print(f"  ↑ {path} -> bos://{bucket}/{key}")
        upload_file(client, path, bucket, key)
        n += 1
    if not quiet:
        print(f"upload_dir: {n} 个文件 -> bos://{bucket}/{prefix}")
    return n


# --------------------------------------------------------------------------- #
# 源码 zip (按 commit sha)
# --------------------------------------------------------------------------- #
def fetch_src_zip(client, sha: str, bucket: str = "umi-gripper",
                  prefix: str = "starvla_res/", dest_dir: str | Path = ".",
                  pkg_prefix: str = "starvla") -> Path:
    """在 <bucket>/<prefix> 下按 commit sha 找源码 zip; sha=='latest' 取最新。

    sha 匹配为 git 风格前缀: zip 名末段 <sha> (CI_COMMIT_SHORT_SHA) 以输入为
    前缀即可命中, 故传任意位数 (短前缀 / 完整短 sha / 完整长 sha) 都行; 大小写
    不敏感。前缀歧义 (命中多个不同 commit) 会报错并列出可选项。
    同一 commit 的多次 CI 产物, 取时间戳最新的一份。

    返回下载到本地的路径 (dest_dir/<zip 名>)。"""
    _, files = list_objects(client, bucket, prefix=prefix, delimiter="")
    cands = [(k, s) for k, s in files if k.endswith(".zip") and Path(k).name.startswith(pkg_prefix)]
    if not cands:
        raise SystemExit(f"ERROR: {bucket}/{prefix} 下没有 {pkg_prefix}_*.zip (先跑 CI 上传产物)")
    if sha and sha.lower() != "latest":
        q = sha.lower()
        # zip 命名: <pkg>_<ref>_<ts>_<sha>.zip -> stored sha = stem 按下划线切最后一段
        matched = []  # [(key, size, stored_sha), ...]
        for k, s in cands:
            stored = Path(k).stem.rsplit("_", 1)[-1].lower()
            # 双向前缀: 传短(前缀) 或 传长(完整 sha) 都能命中
            if stored.startswith(q) or q.startswith(stored):
                matched.append((k, s, stored))
        if not matched:
            avail = "\n  ".join(Path(k).name for k, _ in cands[-10:])
            raise SystemExit(f"ERROR: 找不到 commit {sha} 的 zip。最近可用:\n  {avail}")
        uniq = sorted({st for _, _, st in matched})
        if len(uniq) > 1:
            raise SystemExit(
                f"ERROR: commit 前缀 '{sha}' 有歧义, 命中 {len(uniq)} 个 commit: "
                f"{', '.join(uniq)}; 请多传几位 (或用 latest)。"
            )
        cands = [(k, s) for k, s, _ in matched]
    # 最新 = 列表最后一个 (按 key 排序, 时间戳在文件名里递增)
    key, _size = sorted(cands)[-1]
    name = Path(key).name
    dst = Path(dest_dir) / name
    print(f">> 源码产物: bos://{bucket}/{key} -> {dst}")
    download_file(client, bucket, key, dst)
    return dst


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _arg_exclude(s):
    return s.split(",") if s else []


def main():
    ap = argparse.ArgumentParser(description="BOS 读写助手 (boto3)", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="列对象/子目录")
    p_ls.add_argument("uri", help="bos://bucket/[prefix] 或 bucket")
    p_ls.add_argument("prefix", nargs="?", default="", help="可选额外 prefix")
    p_ls.add_argument("-r", "--recursive", action="store_true", help="递归列全部对象")

    p_get = sub.add_parser("get", help="下载单个对象")
    p_get.add_argument("uri", help="bos://bucket/key")
    p_get.add_argument("local", help="本地保存路径")

    p_getd = sub.add_parser("get-dir", help="递归下载目录")
    p_getd.add_argument("uri", help="bos://bucket/prefix/")
    p_getd.add_argument("local_dir", help="本地目录")
    p_getd.add_argument("--exclude", action="append", default=[], help="排除 glob (可多次)")

    p_putd = sub.add_parser("put-dir", help="递归上传目录")
    p_putd.add_argument("local_dir", help="本地目录")
    p_putd.add_argument("uri", help="bos://bucket/prefix/")
    p_putd.add_argument("--exclude", action="append", default=[], help="排除 glob (可多次)")

    p_put = sub.add_parser("put", help="上传单个文件")
    p_put.add_argument("local", help="本地文件")
    p_put.add_argument("uri", help="bos://bucket/prefix/  (key = prefix/<local 文件名>)")

    p_src = sub.add_parser("src-zip", help="按 commit sha 取源码 zip")
    p_src.add_argument("sha", help="commit sha (任意位数前缀均可, 如 1ff / 1ffcb1c / 1ffcb1c1...) 或 'latest'")
    p_src.add_argument("--bucket", default="umi-gripper")
    p_src.add_argument("--prefix", default="starvla_res/")
    p_src.add_argument("--dest-dir", default=".")
    p_src.add_argument("--pkg-prefix", default="starvla")

    args = ap.parse_args()
    client = make_client()

    if args.cmd == "ls":
        bucket, key = parse_bos_uri(args.uri) if args.uri.startswith("bos://") else (args.uri, "")
        prefix = key or args.prefix
        delim = "" if args.recursive else "/"
        dirs, files = list_objects(client, bucket, prefix=prefix, delimiter=delim)
        for d in dirs:
            print(f"  PRE  {d}")
        for k, sz in files:
            if k.endswith("/"):
                continue
            print(f"  {sz:>12}  {k}")

    elif args.cmd == "get":
        bucket, key = parse_bos_uri(args.uri)
        download_file(client, bucket, key, args.local)
        print(f"✓ bos://{bucket}/{key} -> {args.local}")

    elif args.cmd == "get-dir":
        bucket, prefix = parse_bos_uri(args.uri)
        download_dir(client, bucket, prefix, args.local_dir, exclude=args.exclude)

    elif args.cmd == "put-dir":
        bucket, prefix = parse_bos_uri(args.uri)
        upload_dir(client, args.local_dir, bucket, prefix, exclude=args.exclude)

    elif args.cmd == "put":
        bucket, prefix = parse_bos_uri(args.uri)
        key = f"{prefix.rstrip('/')}/{Path(args.local).name}"
        upload_file(client, args.local, bucket, key)
        print(f"✓ {args.local} -> bos://{bucket}/{key}")

    elif args.cmd == "src-zip":
        out = fetch_src_zip(client, args.sha, bucket=args.bucket, prefix=args.prefix,
                            dest_dir=args.dest_dir, pkg_prefix=args.pkg_prefix)
        print(f"✓ {out}")


if __name__ == "__main__":
    main()
