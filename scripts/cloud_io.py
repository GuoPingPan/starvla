#!/usr/bin/env python3
"""cloud_io.py — 统一的双云对象存储读写助手 (阿里云 OSS + 百度云 BOS)。

支持两种 URI scheme, 按 scheme 自动选择后端:
    oss://<bucket>/<key>   阿里云 OSS  (oss2 SDK)
    bos://<bucket>/<key>   百度云 BOS  (boto3 S3 兼容接口, 下载优先 bcecmd)

凭证全部从环境变量读, 不写进脚本/仓库:
  阿里云 OSS:
    OSS_ACCESS_KEY_ID      AccessKey ID     (缺省时自动探测 PAI DSW 凭证服务)
    OSS_ACCESS_KEY_SECRET  AccessKey Secret
    OSS_SECURITY_TOKEN     可选, STS 临时凭证 token
    OSS_CREDENTIALS_URI    可选, 凭证服务地址 (默认 http://localhost:7002/api/v1/credentials/0)
    OSS_ENDPOINT           默认 https://oss-cn-beijing.aliyuncs.com (华北2)
  百度云 BOS:
    BOS_ACCESS_KEY_ID / BOS_SECRET_ACCESS_KEY / BOS_ENDPOINT (默认 s3.bj.bcebos.com)

CLI 子命令 (供 train.sh / bos_ckpt.sh 调用):
    ls        <uri> [prefix] [-r]          列对象/子目录 (mtime 用 UTC)
    get       <uri> <local_path>            下载单个对象
    get-dir   <uri_prefix> <local_dir>      递归下载目录 (--exclude 可多次)
    put-dir   <local_dir> <uri_prefix>      递归上传目录 (--exclude 可多次)
    put       <local> <uri_prefix>/         上传单个文件
    rm        <uri>                         删除单个对象
    src-zip   <sha|latest> [--cloud auto|oss|bos]   按 commit sha 取源码 zip

URI 形如 oss://<bucket>/<key> 或 bos://<bucket>/<key>; 目录前缀以 / 结尾。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

SCHEMES = ("oss", "bos")

# 各云的默认产物桶 (源码 zip / ckpt / 预训练所在)
DEFAULT_BUCKET = {"oss": "algo-public0", "bos": "umi-gripper"}


# --------------------------------------------------------------------------- #
# URI
# --------------------------------------------------------------------------- #
def parse_cloud_uri(uri: str) -> tuple[str, str, str]:
    """(oss|bos)://<bucket>/<key> -> (scheme, bucket, key)。key 可能为 '' (桶根)。"""
    scheme = urlparse(uri).scheme
    if scheme not in SCHEMES:
        raise ValueError(f"不是受支持的云 URI ({'/'.join(s + '://' for s in SCHEMES)}): {uri}")
    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    if not bucket:
        raise ValueError(f"URI 缺少 bucket: {uri}")
    return scheme, bucket, key


# 兼容旧名 (bos_io.parse_bos_uri): 忽略 scheme 校验, 返回 (bucket, key)
def parse_bos_uri(uri: str) -> tuple[str, str]:
    return parse_cloud_uri(uri)[1:]


# --------------------------------------------------------------------------- #
# 凭证
# --------------------------------------------------------------------------- #
def _oss_env_creds() -> tuple[str, str, str | None] | None:
    ak, sk = os.environ.get("OSS_ACCESS_KEY_ID"), os.environ.get("OSS_ACCESS_KEY_SECRET")
    if ak and sk:
        return ak, sk, os.environ.get("OSS_SECURITY_TOKEN")
    return None


class _DswCredState:
    """PAI DSW 凭证服务的临时 STS 凭证, 带过期刷新。"""

    URI_DEFAULT = "http://localhost:7002/api/v1/credentials/0"

    def __init__(self, uri: str | None = None):
        self.uri = uri or os.environ.get("OSS_CREDENTIALS_URI") or self.URI_DEFAULT
        self._expiry_ts = 0.0
        self._cred: dict | None = None

    def fetch(self, force: bool = False) -> dict | None:
        """取凭证; 提前 5 分钟过期。服务不可达返回 None (由调用方降级/报错)。"""
        if not force and self._cred and time.time() < self._expiry_ts - 300:
            return self._cred
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(self.uri, timeout=3) as r:
                d = json.load(r)
            cred = d.get("Data") or d
            if not (cred.get("AccessKeyId") and cred.get("AccessKeySecret")):
                return None
            exp = cred.get("Expiration")
            try:
                dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                self._expiry_ts = dt.timestamp()
            except Exception:
                self._expiry_ts = time.time() + 600  # 解析失败按 10 分钟算
            self._cred = cred
            return cred
        except Exception:
            return None


_DSW_CREDS = _DswCredState()


def oss_creds_available() -> bool:
    """OSS 凭证是否可得 (环境变量 或 DSW 凭证服务)。"""
    if _oss_env_creds():
        return True
    if os.environ.get("OSS_ALLOW_DSW_CREDS", "1") != "0":
        return _DSW_CREDS.fetch() is not None
    return False


def bos_creds_available() -> bool:
    return bool(os.environ.get("BOS_ACCESS_KEY_ID") and os.environ.get("BOS_SECRET_ACCESS_KEY"))


def pick_cloud(prefer: str = "auto") -> str:
    """auto: OSS 凭证可得优先 OSS, 否则 BOS。"""
    if prefer in SCHEMES:
        return prefer
    return "oss" if oss_creds_available() else "bos"


# --------------------------------------------------------------------------- #
# 后端: 阿里云 OSS (oss2)
# --------------------------------------------------------------------------- #
def _fmt_ts(ts) -> str:
    """unix 秒 / datetime -> 'YYYY-MM-DD HH:MM:SS' (本地时间, 与 ls 习惯一致)。"""
    dt = ts if isinstance(ts, datetime) else datetime.fromtimestamp(int(ts))
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class OssClient:
    scheme = "oss"

    def __init__(self):
        try:
            import oss2
        except ImportError:
            print("ERROR: 未安装 oss2。pip install oss2 / uv pip install oss2", file=sys.stderr)
            sys.exit(2)
        self._oss2 = oss2
        endpoint = os.environ.get("OSS_ENDPOINT", "https://oss-cn-beijing.aliyuncs.com")
        if endpoint and not endpoint.startswith("http"):
            endpoint = f"https://{endpoint}"
        self.endpoint = endpoint
        self._auth = None
        self._auth_is_sts = False
        self._buckets: dict[str, object] = {}
        if not self._refresh_auth():
            print(
                "ERROR: 缺少 OSS 凭证: 请 export OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET "
                "(或写入 /root/.ossenv 由脚本 source), 或确保 PAI DSW 凭证服务可达。\n"
                f"(探测地址: {_DSW_CREDS.uri})",
                file=sys.stderr,
            )
            sys.exit(2)

    def _refresh_auth(self, force: bool = False) -> bool:
        env = _oss_env_creds()
        if env:
            ak, sk, token = env
            if token:
                self._auth = self._oss2.StsAuth(ak, sk, token)
                self._auth_is_sts = True
            else:
                self._auth = self._oss2.Auth(ak, sk)
                self._auth_is_sts = False
            self._buckets.clear()
            return True
        cred = _DSW_CREDS.fetch(force=force) if os.environ.get("OSS_ALLOW_DSW_CREDS", "1") != "0" else None
        if cred:
            self._auth = self._oss2.StsAuth(cred["AccessKeyId"], cred["AccessKeySecret"], cred.get("SecurityToken"))
            self._auth_is_sts = True
            self._buckets.clear()
            return True
        return False

    def _maybe_refresh(self):
        """STS 临时凭证过期前自动续期 (AK/SK 模式无操作)。"""
        if self._auth_is_sts and not _oss_env_creds():
            _DSW_CREDS.fetch()  # fetch 内部自带"提前 5 分钟过期"判断

    def bucket(self, bucket: str):
        self._maybe_refresh()
        if bucket not in self._buckets:
            self._buckets[bucket] = self._oss2.Bucket(self._auth, self.endpoint, bucket, connect_timeout=30)
        return self._buckets[bucket]

    # ---- 列举 ----
    def list_objects(self, bucket: str, prefix: str = "", delimiter: str = "/"):
        """返回 (common_prefixes, [(key, size, mtime_str), ...])。delimiter='' 递归。"""
        b = self.bucket(bucket)
        dirs, files, marker = [], [], ""
        while True:
            kw = {"prefix": prefix, "marker": marker, "max_keys": 1000}
            if delimiter:
                kw["delimiter"] = delimiter
            r = b.list_objects(**kw)
            dirs += list(r.prefix_list)
            files += [(o.key, o.size, _fmt_ts(o.last_modified)) for o in r.object_list]
            if not r.is_truncated:
                break
            marker = r.next_marker
        return dirs, files

    # ---- 下载 ----
    def download_file(self, bucket: str, key: str, local_path: str | Path, max_retries: int = 6):
        """断点续传下载 (oss2 resumable_download 自带分片+重试), .part 完成后原子重命名。"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_name(local_path.name + ".part")
        name = Path(key).name or key
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                self._maybe_refresh()
                step = 50 * 1024 * 1024  # 每 50MB 打一次进度
                state = {"seen": 0, "nxt": step}

                def _cb(consumed, total, state=state, step=step):
                    if consumed >= state["nxt"] and total:
                        pct = f" ({consumed * 100 // total}%)" if total else ""
                        print(f"    ↓ {name}: {consumed // (1024 * 1024)} MB{pct}", flush=True)
                        state["nxt"] = ((consumed // step) + 1) * step

                self._oss2.resumable_download(
                    self.bucket(bucket),
                    key,
                    str(tmp_path),
                    multiget_threshold=10 * 1024 * 1024,
                    part_size=10 * 1024 * 1024,
                    progress_callback=_cb,
                    num_threads=4,
                )
                os.replace(tmp_path, local_path)  # 原子重命名, 防半成品
                return
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    break
                wait = min(2**attempt, 30)
                print(
                    f"    ⚠ 下载中断 [{type(e).__name__}: {e}], {wait}s 后重试 " f"({attempt}/{max_retries})", flush=True
                )
                time.sleep(wait)
        raise last_err if last_err else RuntimeError(f"download failed: oss://{bucket}/{key}")

    # ---- 上传 ----
    def upload_file(self, local_path: str | Path, bucket: str, key: str):
        self._maybe_refresh()
        self._oss2.resumable_upload(
            self.bucket(bucket),
            key,
            str(local_path),
            multipart_threshold=10 * 1024 * 1024,
            part_size=10 * 1024 * 1024,
            num_threads=4,
        )

    # ---- 删除 ----
    def delete_object(self, bucket: str, key: str):
        self.bucket(bucket).delete_object(key)


# --------------------------------------------------------------------------- #
# 后端: 百度云 BOS (boto3 S3 兼容; 下载优先 bcecmd)
# --------------------------------------------------------------------------- #
def _boto3_client():
    miss = [n for n in ("BOS_ACCESS_KEY_ID", "BOS_SECRET_ACCESS_KEY") if not os.environ.get(n)]
    if miss:
        print(
            f"ERROR: 缺少 BOS 凭证环境变量: {', '.join(miss)}\n"
            "请先 export BOS_ACCESS_KEY_ID / BOS_SECRET_ACCESS_KEY "
            "(以及可选 BOS_ENDPOINT / BOS_REGION)。",
            file=sys.stderr,
        )
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
            connect_timeout=30,
            read_timeout=60,
            # standard 模式对 ReadTimeout / ConnectionClosed 这类"卡住型"错误重试可靠;
            # legacy (默认) 有漏重试行为, 是 s3transfer future 无限等待的常见诱因。
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


class _ReadTimeout(Exception):
    """看门狗触发的读超时 (对抗 boto3 read_timeout 在 HTTPS streaming body 上不生效)。"""


def _alarm_handler(signum, frame):
    raise _ReadTimeout()


def _read_chunk_with_watchdog(body, n_bytes: int, timeout: float):
    """带硬超时的 body.read (SIGALRM 强制打断卡死的 socket recv)。仅主线程可用。"""
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
    """用百度官方 CLI bcecmd 下载单个对象。成功 True; 不可用/失败 False (调用方 fallback)。

    实测同机 boto3 下载 BOS 卡在 socket recv (HTTPS streaming body read_timeout 不生效),
    而 bcecmd 秒下。bcecmd 自带断点续传/重试/并发, 凭证走它自己的配置。
    """
    import shutil
    import subprocess

    bce = shutil.which("bcecmd")
    if not bce:
        return False
    print(f"    ↓ [bcecmd] {local_path.name}", flush=True)
    try:
        r = subprocess.run(
            [bce, "bos", "cp", f"bos://{bucket}/{key}", str(local_path), "--yes", "--quiet"],
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("    ⚠ bcecmd 超时, fallback boto3", flush=True)
        return False
    if r.returncode != 0:
        print(f"    ⚠ bcecmd 失败 (rc={r.returncode}): {r.stderr.strip()[:200]} → fallback boto3", flush=True)
        return False
    return True


class BosClient:
    scheme = "bos"

    def __init__(self):
        self._client = _boto3_client()

    @property
    def raw(self):
        return self._client

    # ---- 列举 ----
    def list_objects(self, bucket: str, prefix: str = "", delimiter: str = "/"):
        """返回 (common_prefixes, [(key, size, mtime_str), ...])。delimiter='' 递归。"""
        paginator = self._client.get_paginator("list_objects_v2")
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if delimiter:
            kwargs["Delimiter"] = delimiter
        dirs, files = [], []
        for page in paginator.paginate(**kwargs):
            for c in page.get("CommonPrefixes", []) or []:
                dirs.append(c["Prefix"])
            for o in page.get("Contents", []) or []:
                mt = o.get("LastModified")
                files.append((o["Key"], o.get("Size", 0), _fmt_ts(mt) if mt else ""))
        return dirs, files

    # ---- 下载 ----
    def download_file(self, bucket: str, key: str, local_path: str | Path, max_retries: int = 6, chunk_mb: int = 8):
        """下载单个对象: 优先 bcecmd, 不可用时 boto3 流式 + SIGALRM 看门狗 + 断点续传。"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if _bcecmd_download(bucket, key, local_path):
            return
        tmp_path = local_path.with_name(local_path.name + ".part")
        name = Path(key).name or key
        client = self._client

        for attempt in range(1, max_retries + 1):
            try:
                # 断点续传: .part 已存在则从其大小续下 (Range 请求), 不重头
                offset = tmp_path.stat().st_size if tmp_path.exists() else 0
                kwargs = {"Bucket": bucket, "Key": key}
                if offset:
                    kwargs["Range"] = f"bytes={offset}-"
                resp = client.get_object(**kwargs)
                cr = resp.get("ContentRange", "") or ""
                total = int(cr.rsplit("/", 1)[-1]) if "/" in cr else (offset + int(resp.get("ContentLength", 0) or 0))

                seen = offset
                step = 10 * 1024 * 1024
                nxt = (seen // step + 1) * step
                with resp["Body"] as body, open(tmp_path, "ab" if offset else "wb") as f:
                    while True:
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
                os.replace(tmp_path, local_path)
                return
            except Exception as e:
                if attempt >= max_retries:
                    raise
                wait = min(2**attempt, 30)
                print(
                    f"    ⚠ 下载中断 [{type(e).__name__}: {e}], {wait}s 后从断点续传 " f"({attempt}/{max_retries})",
                    flush=True,
                )
                time.sleep(wait)

    # ---- 上传 ----
    def upload_file(self, local_path: str | Path, bucket: str, key: str):
        self._client.upload_file(str(local_path), bucket, key)

    # ---- 删除 ----
    def delete_object(self, bucket: str, key: str):
        self._client.delete_object(Bucket=bucket, Key=key)


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
_clients: dict[str, object] = {}


def make_client(scheme: str = "auto"):
    """按 scheme 取 (缓存的) 后端 client。auto = OSS 凭证可得优先 OSS。"""
    if scheme == "auto":
        scheme = pick_cloud()
    if scheme not in SCHEMES:
        raise ValueError(f"未知 cloud scheme: {scheme}")
    if scheme not in _clients:
        _clients[scheme] = OssClient() if scheme == "oss" else BosClient()
    return _clients[scheme]


def list_objects(client, bucket: str, prefix: str = "", delimiter: str = "/"):
    return client.list_objects(bucket, prefix, delimiter)


def download_file(client, bucket: str, key: str, local_path: str | Path):
    return client.download_file(bucket, key, local_path)


def upload_file(client, local_path: str | Path, bucket: str, key: str):
    return client.upload_file(local_path, bucket, key)


def delete_object(client, bucket: str, key: str):
    return client.delete_object(bucket, key)


# --------------------------------------------------------------------------- #
# 目录级操作 (双云通用)
# --------------------------------------------------------------------------- #
def _excluded(rel: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(rel, p) or fnmatch("/" + rel, p) for p in patterns)


def download_dir(
    client, bucket: str, prefix: str, local_dir: str | Path, exclude: list[str] | None = None, quiet: bool = False
):
    """递归下载 <bucket>/<prefix>/ 下全部对象到 <local_dir>/, 重建子目录。幂等(同大小跳过)。"""
    exclude = exclude or []
    local_dir = Path(local_dir)
    _, files = client.list_objects(bucket, prefix=prefix, delimiter="")
    n = 0
    for item in files:
        key, size = item[0], item[1]
        rel = key[len(prefix) :] if key.startswith(prefix) else key
        if not rel or rel.endswith("/"):
            continue
        if _excluded(rel, exclude):
            continue
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == size:
            continue  # 已存在且同大小, 跳过 (幂等, 如 16GB 预训练)
        if not quiet:
            print(f"  ↓ {key} ({size} B) -> {dst}")
        client.download_file(bucket, key, dst)
        n += 1
    if not quiet:
        print(f"download_dir: {n} 个文件 -> {local_dir}")
    return n


def upload_dir(
    client, local_dir: str | Path, bucket: str, prefix: str, exclude: list[str] | None = None, quiet: bool = False
):
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
            print(f"  ↑ {path} -> {client.scheme}://{bucket}/{key}")
        client.upload_file(path, bucket, key)
        n += 1
    if not quiet:
        print(f"upload_dir: {n} 个文件 -> {client.scheme}://{bucket}/{prefix}")
    return n


# --------------------------------------------------------------------------- #
# 源码 zip (按 commit sha)
# --------------------------------------------------------------------------- #
def fetch_src_zip(
    client,
    sha: str,
    bucket: str | None = None,
    prefix: str = "starvla_res/",
    dest_dir: str | Path = ".",
    pkg_prefix: str = "starvla",
) -> Path:
    """在 <bucket>/<prefix> 下按 commit sha 找源码 zip; sha=='latest' 取最新。

    sha 匹配为 git 风格双向前缀 (传短/长 sha 均可); 前缀歧义会报错并列出可选项。
    同一 commit 的多次 CI 产物, 取时间戳最新的一份。返回下载到本地的路径。"""
    bucket = bucket or DEFAULT_BUCKET[client.scheme]
    _, files = client.list_objects(bucket, prefix=prefix, delimiter="")
    cands = [(k, s) for k, s, *_ in files if k.endswith(".zip") and Path(k).name.startswith(pkg_prefix)]
    if not cands:
        raise SystemExit(f"ERROR: {client.scheme}://{bucket}/{prefix} 下没有 {pkg_prefix}_*.zip (先跑 CI 上传产物)")
    if sha and sha.lower() != "latest":
        q = sha.lower()
        matched = []  # [(key, size, stored_sha), ...]
        for k, s in cands:
            stored = Path(k).stem.rsplit("_", 1)[-1].lower()
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
    print(f">> 源码产物: {client.scheme}://{bucket}/{key} -> {dst}")
    client.download_file(bucket, key, dst)
    return dst


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _arg_exclude(s):
    return s.split(",") if s else []


def main():
    ap = argparse.ArgumentParser(
        description="双云对象存储读写助手 (OSS oss2 / BOS boto3)", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("ls", help="列对象/子目录")
    p_ls.add_argument("uri", help="(oss|bos)://bucket/[prefix] 或 bucket")
    p_ls.add_argument("prefix", nargs="?", default="", help="可选额外 prefix")
    p_ls.add_argument("-r", "--recursive", action="store_true", help="递归列全部对象")

    p_get = sub.add_parser("get", help="下载单个对象")
    p_get.add_argument("uri", help="(oss|bos)://bucket/key")
    p_get.add_argument("local", help="本地保存路径")

    p_getd = sub.add_parser("get-dir", help="递归下载目录")
    p_getd.add_argument("uri", help="(oss|bos)://bucket/prefix/")
    p_getd.add_argument("local_dir", help="本地目录")
    p_getd.add_argument("--exclude", action="append", default=[], help="排除 glob (可多次)")

    p_putd = sub.add_parser("put-dir", help="递归上传目录")
    p_putd.add_argument("local_dir", help="本地目录")
    p_putd.add_argument("uri", help="(oss|bos)://bucket/prefix/")
    p_putd.add_argument("--exclude", action="append", default=[], help="排除 glob (可多次)")

    p_put = sub.add_parser("put", help="上传单个文件")
    p_put.add_argument("local", help="本地文件")
    p_put.add_argument("uri", help="(oss|bos)://bucket/prefix/  (key = prefix/<local 文件名>)")

    p_rm = sub.add_parser("rm", help="删除单个对象")
    p_rm.add_argument("uri", help="(oss|bos)://bucket/key")

    p_src = sub.add_parser("src-zip", help="按 commit sha 取源码 zip")
    p_src.add_argument("sha", help="commit sha (任意位数前缀) 或 'latest'")
    p_src.add_argument(
        "--cloud", default="auto", choices=["auto", "oss", "bos"], help="用哪个云的产物桶 (auto: OSS 凭证可得优先)"
    )
    p_src.add_argument("--bucket", default=None, help="默认: oss→OSS_BUCKET/boto默认, bos→umi-gripper")
    p_src.add_argument("--prefix", default="starvla_res/")
    p_src.add_argument("--dest-dir", default=".")
    p_src.add_argument("--pkg-prefix", default="starvla")

    args = ap.parse_args()

    if args.cmd == "ls":
        if "://" in args.uri:
            scheme, bucket, key = parse_cloud_uri(args.uri)
        else:
            scheme, bucket, key = pick_cloud(), args.uri, ""
        client = make_client(scheme)
        prefix = key or args.prefix
        delim = "" if args.recursive else "/"
        dirs, files = client.list_objects(bucket, prefix=prefix, delimiter=delim)
        for d in dirs:
            print(f"  PRE  {d}")
        for item in files:
            if item[0].endswith("/"):
                continue
            print(f"  {item[1]:>12}  {item[2]}  {item[0]}")

    elif args.cmd == "get":
        scheme, bucket, key = parse_cloud_uri(args.uri)
        make_client(scheme).download_file(bucket, key, args.local)
        print(f"✓ {scheme}://{bucket}/{key} -> {args.local}")

    elif args.cmd == "get-dir":
        scheme, bucket, prefix = parse_cloud_uri(args.uri)
        download_dir(make_client(scheme), bucket, prefix, args.local_dir, exclude=args.exclude)

    elif args.cmd == "put-dir":
        scheme, bucket, prefix = parse_cloud_uri(args.uri)
        upload_dir(make_client(scheme), args.local_dir, bucket, prefix, exclude=args.exclude)

    elif args.cmd == "put":
        scheme, bucket, prefix = parse_cloud_uri(args.uri)
        key = f"{prefix.rstrip('/')}/{Path(args.local).name}"
        make_client(scheme).upload_file(args.local, bucket, key)
        print(f"✓ {args.local} -> {scheme}://{bucket}/{key}")

    elif args.cmd == "rm":
        scheme, bucket, key = parse_cloud_uri(args.uri)
        make_client(scheme).delete_object(bucket, key)
        print(f"✓ 已删除 {scheme}://{bucket}/{key}")

    elif args.cmd == "src-zip":
        scheme = pick_cloud(args.cloud)
        bucket = args.bucket or os.environ.get("OSS_BUCKET" if scheme == "oss" else "BOS_BUCKET", DEFAULT_BUCKET[scheme])
        out = fetch_src_zip(
            make_client(scheme),
            args.sha,
            bucket=bucket,
            prefix=args.prefix,
            dest_dir=args.dest_dir,
            pkg_prefix=args.pkg_prefix,
        )
        print(f"✓ {out}")


if __name__ == "__main__":
    main()
