# SPDX-FileCopyrightText: Copyright (c) 2025 starVLA community
# SPDX-License-Identifier: MIT

"""rerun_logger 维度无关性的单元测试 (进程内合成, 无需 GPU / 真实数据 / 读 .rrd)。

覆盖 log_episode_to_rerun 的通用化改动:
- 默认 dim_names=None → UMI 8 维 + 1 相机 (向后兼容): 8 个 TimeSeriesView + 1 个 obs_image 视图, grid 3 列
- dim_names=16 (RoboDojo 双臂 wxyz) + 3 相机: 16 个 TimeSeriesView + 3 个 obs_image 视图, grid 4 列,
  且第 4 个 dim = left_quat_w (验证 wxyz 顺序, w 在前; 切勿重排成 qw 在后)
- 维度不匹配 (dim_names 长度 != npz raw 维度) → ValueError
- 显式 grid_columns 覆盖自动派生; 无 obs_images → 0 个 obs_image 视图

rerun 0.34.1 无离线 .rrd 读取 API (dataframe / load_recording 均不存在), 故 monkeypatch
rr / rrb 为 MagicMock, 断言 blueprint 组装 (TimeSeriesView / Spatial2DView / Grid 的调用)
而非读取产物。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from starVLA.eval.viz import rerun_logger
from starVLA.eval.viz.rerun_logger import log_episode_to_rerun

# RoboDojo 双臂 16 维 raw (wxyz, w 在前) —— 与
# examples/simBenchmarks/RoboDojo/eval_files/stage2_visualize.py:ROBODOJO_RAW_DIM_NAMES 保持一致。
_ROBODOJO_16 = [
    "left_pos_x", "left_pos_y", "left_pos_z",
    "left_quat_w", "left_quat_x", "left_quat_y", "left_quat_z",
    "left_gripper",
    "right_pos_x", "right_pos_y", "right_pos_z",
    "right_quat_w", "right_quat_x", "right_quat_y", "right_quat_z",
    "right_gripper",
]


def _make_results(n_samples: int = 4, dim: int = 8, horizon: int = 6) -> dict:
    rng = np.random.default_rng(0)
    return {
        "anchors": np.arange(n_samples, dtype=np.int64),
        "state_at_anchor": rng.random((n_samples, dim)),
        "action_gt": rng.random((n_samples, dim)),
        "action_pred_chunks": rng.random((n_samples, horizon, dim)),
        "latency_seconds": rng.random(n_samples),
    }


def _make_obs_images(n_samples: int = 4, cams=("cam_wrist_0",)) -> dict:
    return {f"video.{c}": np.zeros((n_samples, 8, 8, 3), dtype=np.uint8) for c in cams}


class LogEpisodeToRerunDimAgnosticTest(unittest.TestCase):
    """mock rr / rrb 跑 log_episode_to_rerun, 断言 blueprint 组装。"""

    def _run(self, results, obs_images, dim_names=None, grid_columns=None):
        with patch.object(rerun_logger, "rr", new=MagicMock()) as rr_mock, \
                patch.object(rerun_logger, "rrb", new=MagicMock()) as rrb_mock:
            out = Path(tempfile.mkdtemp()) / "episode.rrd"
            log_episode_to_rerun(
                results, {"ep": 1}, out,
                obs_images=obs_images, dim_names=dim_names, grid_columns=grid_columns,
            )
            return rr_mock, rrb_mock

    @staticmethod
    def _kw_origins(call_args_list):
        return [str(c.kwargs.get("origin", "")) for c in call_args_list]

    def test_default_8dim_1cam_umi_compatible(self):
        _, rrb = self._run(_make_results(dim=8), _make_obs_images(cams=("cam_wrist_0",)))
        ts = self._kw_origins(rrb.TimeSeriesView.call_args_list)
        self.assertEqual(len(ts), 8)
        self.assertEqual(ts[0], "/pos_x")          # 默认 UMI 8 维名
        self.assertEqual(ts[-1], "/gripper")
        s2d = self._kw_origins(rrb.Spatial2DView.call_args_list)
        self.assertEqual(len(s2d), 1)
        self.assertEqual(s2d[0], "/obs_image/cam_wrist_0")
        self.assertEqual(rrb.Grid.call_args.kwargs["grid_columns"], 3)   # 8<=9 → 3 (UMI 不变)

    def test_robodojo_16dim_3cam_wxyz(self):
        _, rrb = self._run(
            _make_results(dim=16),
            _make_obs_images(cams=("cam_high", "cam_left_wrist", "cam_right_wrist")),
            dim_names=_ROBODOJO_16,
        )
        ts = self._kw_origins(rrb.TimeSeriesView.call_args_list)
        self.assertEqual(len(ts), 16)
        # wxyz 顺序: 第 4 个 dim (idx 3) = left_quat_w —— w 在前, 非 qw 在后
        self.assertEqual(ts[3], "/left_quat_w")
        self.assertEqual(ts[7], "/left_gripper")
        self.assertEqual(ts[8], "/right_pos_x")
        self.assertEqual(ts[-1], "/right_gripper")
        s2d = self._kw_origins(rrb.Spatial2DView.call_args_list)
        self.assertEqual(len(s2d), 3)
        self.assertEqual(
            set(s2d),
            {"/obs_image/cam_high", "/obs_image/cam_left_wrist", "/obs_image/cam_right_wrist"},
        )
        self.assertEqual(rrb.Grid.call_args.kwargs["grid_columns"], 4)   # 16>9 → 4

    def test_dim_mismatch_raises(self):
        # dim_names 长度 (2) != npz raw 维度 (8) → ValueError, 杜绝静默截断 / 越界
        with self.assertRaises(ValueError):
            self._run(_make_results(dim=8), _make_obs_images(), dim_names=["a", "b"])

    def test_explicit_grid_columns(self):
        _, rrb = self._run(_make_results(dim=8), _make_obs_images(), grid_columns=2)
        self.assertEqual(rrb.Grid.call_args.kwargs["grid_columns"], 2)

    def test_no_obs_images(self):
        _, rrb = self._run(_make_results(dim=8), None)
        self.assertEqual(len(rrb.Spatial2DView.call_args_list), 0)


if __name__ == "__main__":
    unittest.main()
