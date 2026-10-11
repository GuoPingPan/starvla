# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License (the "License");
# """data-analyze 项目数据注册表。

# 与原版 zerith_pi05 解耦: 本项目 (data-analyze) 的 mixture 注册在此; robot_type 的
# DataConfig / transform / 数据集类 直接复用 zerith_pi05 的实现 (SE(3) 相对位姿 +
# rotation_6d + 逐时间步 q99 归一化), 仅 import, 不重复定义。

# 注意: ``scripts/download_data.py`` 用 ast.literal_eval 读本文件的
# DATASET_NAMED_MIXTURES —— 必须是**字面量** dict, 不能动态拼接;
# 训练侧 registry.py 则 import 本模块并把条目合并进全局注册表。
# """

# ---- 复用 zerith_pi05 的 robot_type 实现 (含 umi_2gripper_relative_6d DataConfig) ----
# registry.py 扫描本目录时会把下面的 map 与全局合并; 不重定义, 单一真值源。
from examples.realRobots.zerith_pi05.train_files.data_registry.data_config import (
    ROBOT_TYPE_CONFIG_MAP as _UMI_ROBOT_TYPES,
)

try:  # 同目录包内引用 (h1 数据集类 + robot_type; 迁自 realRobots/H1_ddl_d18480a6)
    from .h1_dataset import ROBOT_TYPE_CONFIG_MAP as _H1_ROBOT_TYPES
except ImportError:  # registry.py 按文件路径 import 时的兜底
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("h1_dataset", __file__.replace("data_config.py", "h1_dataset.py"))
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    _H1_ROBOT_TYPES = _mod.ROBOT_TYPE_CONFIG_MAP

ROBOT_TYPE_CONFIG_MAP = {**_UMI_ROBOT_TYPES, **_H1_ROBOT_TYPES}

# ---- 本项目自己的数据 mixture (字面量, download_data.py ast 兼容) ----
# 元组格式: (dataset_id, weight, robot_type, cloud_uri)
# 第4元素 = 对象存储 zip/dir URI; download_data.py 读它拉数据; 训练忽略。
DATASET_NAMED_MIXTURES = {
    "relative_6d_741552b7": [
        # 采集批次 PR-0011/BIZ-STORE-001_214
        ("20260909140758_lerobot_741552b7", 1.0, "umi_2gripper_relative_6d",
         "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_214/PR-0011_BIZ-STORE-001_214_20260909140758_lerobot_741552b7_20260909140758_lerobot_741552b7.zip"),
    ],
    "h1_ddl_d18480a6": [
        (
            "20260909155441_lerobot_f61b924b",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_214/PR-0011_BIZ-STORE-001_214_20260909155441_lerobot_f61b924b_20260909155441_lerobot_f61b924b.zip",
        ),
        (
            "20260909155646_lerobot_165e92fd",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_218/PR-0011_BIZ-STORE-001_218_20260909155646_lerobot_165e92fd_20260909155646_lerobot_165e92fd.zip",
        ),
        (
            "20260909155823_lerobot_fc8b7a1f",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_217/PR-0011_BIZ-STORE-001_217_20260909155823_lerobot_fc8b7a1f_20260909155823_lerobot_fc8b7a1f.zip",
        ),
        (
            "20260909155821_lerobot_2a9ebb61",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_212/PR-0011_BIZ-STORE-001_212_20260909155821_lerobot_2a9ebb61_20260909155821_lerobot_2a9ebb61.zip",
        ),
        (
            "20260909160736_lerobot_74634506",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_213/PR-0011_BIZ-STORE-001_213_20260909160736_lerobot_74634506_20260909160736_lerobot_74634506.zip",
        ),
        (
            "20260909161139_lerobot_d6c6ab11",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_216/PR-0011_BIZ-STORE-001_216_20260909161139_lerobot_d6c6ab11_20260909161139_lerobot_d6c6ab11.zip",
        ),
        (
            "20260909161129_lerobot_1669a254",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_215/PR-0011_BIZ-STORE-001_215_20260909161129_lerobot_1669a254_20260909161129_lerobot_1669a254.zip",
        ),
        (
            "20260909162216_lerobot_e76f38a1",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_211/PR-0011_BIZ-STORE-001_211_20260909162216_lerobot_e76f38a1_20260909162216_lerobot_e76f38a1.zip",
        ),
        (
            "20260909175028_lerobot_208b0c3d",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_218/PR-0011_BIZ-STORE-001_218_20260909175028_lerobot_208b0c3d_20260909175028_lerobot_208b0c3d.zip",
        ),
        (
            "20260909175125_lerobot_a7993ccb",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_213/PR-0011_BIZ-STORE-001_213_20260909175125_lerobot_a7993ccb_20260909175125_lerobot_a7993ccb.zip",
        ),
        (
            "20260909175323_lerobot_813ddbae",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_217/PR-0011_BIZ-STORE-001_217_20260909175323_lerobot_813ddbae_20260909175323_lerobot_813ddbae.zip",
        ),
        (
            "20260909175456_lerobot_74867465",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_216/PR-0011_BIZ-STORE-001_216_20260909175456_lerobot_74867465_20260909175456_lerobot_74867465.zip",
        ),
        (
            "20260909175643_lerobot_0fd80cdc",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_212/PR-0011_BIZ-STORE-001_212_20260909175643_lerobot_0fd80cdc_20260909175643_lerobot_0fd80cdc.zip",
        ),
        (
            "20260909180326_lerobot_7af49f52",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_215/PR-0011_BIZ-STORE-001_215_20260909180326_lerobot_7af49f52_20260909180326_lerobot_7af49f52.zip",
        ),
        (
            "20260909181641_lerobot_47cd03c3",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_211/PR-0011_BIZ-STORE-001_211_20260909181641_lerobot_47cd03c3_20260909181641_lerobot_47cd03c3.zip",
        ),
        (
            "20260909191822_lerobot_044a14a3",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_217/PR-0011_BIZ-STORE-001_217_20260909191822_lerobot_044a14a3_20260909191822_lerobot_044a14a3.zip",
        ),
        (
            "20260909191824_lerobot_ff43c3e3",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data2/disk2lakehouse/PR-0011/BIZ-STORE-001_211/PR-0011_BIZ-STORE-001_211_20260909191824_lerobot_ff43c3e3_20260909191824_lerobot_ff43c3e3.zip",
        ),
        (
            "20260909191853_lerobot_5f599eee",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_219/PR-0011_BIZ-STORE-001_219_20260909191853_lerobot_5f599eee_20260909191853_lerobot_5f599eee.zip",
        ),
        (
            "20260909192101_lerobot_d5ea6c17",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data1/disk2lakehouse/PR-0011/BIZ-STORE-001_213/PR-0011_BIZ-STORE-001_213_20260909192101_lerobot_d5ea6c17_20260909192101_lerobot_d5ea6c17.zip",
        ),
        (
            "20260909192142_lerobot_5e1d0916",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_216/PR-0011_BIZ-STORE-001_216_20260909192142_lerobot_5e1d0916_20260909192142_lerobot_5e1d0916.zip",
        ),
        (
            "20260909192220_lerobot_142fd6bb",
            1.0,
            "h1_manipulation",
            "bos://robot-data-from-disk/disk2lakehouse/PR-0011/BIZ-STORE-001_215/PR-0011_BIZ-STORE-001_215_20260909192220_lerobot_142fd6bb_20260909192220_lerobot_142fd6bb.zip",
        ),
        (
            "20260909192307_lerobot_97018056",
            1.0,
            "h1_manipulation",
            "bos://lakehouse-data3/disk2lakehouse/PR-0011/BIZ-STORE-001_212/PR-0011_BIZ-STORE-001_212_20260909192307_lerobot_97018056_20260909192307_lerobot_97018056.zip",
        ),
    ],
}
