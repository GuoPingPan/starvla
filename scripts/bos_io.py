#!/usr/bin/env python3
"""bos_io.py — 兼容 shim: 已迁移到 cloud_io.py (双云: 阿里云 OSS + 百度云 BOS)。

历史接口 (make_client / parse_bos_uri / list_objects / download_file / download_dir /
upload_file / upload_dir / fetch_src_zip / CLI) 全部保留, 行为不变;
新增 oss:// URI 支持 —— 按 scheme 自动分流后端 (OSS 用 oss2, BOS 用 boto3/bcecmd)。

新代码请直接 import cloud_io。
"""

from cloud_io import (  # noqa: F401
    BosClient,
    OssClient,
    delete_object,
    download_dir,
    download_file,
    fetch_src_zip,
    list_objects,
    make_client,
    parse_bos_uri,
    parse_cloud_uri,
    pick_cloud,
    upload_dir,
    upload_file,
)

if __name__ == "__main__":
    from cloud_io import main

    main()
