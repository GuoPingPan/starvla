# SPDX-FileCopyrightText: Copyright (c) 2025 starVLA community
# SPDX-License-Identifier: MIT

"""QuaternionTo6DRotation 共享转换的单元测试。

覆盖:
- 几何正确性: xyzw 四元数 -> 6D 与 (matrix 路径) 一致
- 两种输入约定 (xyzw / wxyz)
- 可逆性: apply -> unapply 恢复等价旋转 (6D 对比, 规避 q/-q canonicalize)
- flat 字典 + 边界: 缺失 key no-op; np.ndarray 输入 -> torch.Tensor; 非 4D 不动
"""

import unittest

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.transform.quaternion_to_6d import QuaternionTo6DRotation
from starVLA.dataloader.gr00t_lerobot.transform.rotation_utils import (
    matrix_to_rotation_6d,
    quaternion_to_matrix,
)

_ATOL = 1e-4
_RTOL = 1e-4


def _expected_6d_from_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
    """与转换内部 jit 路径无关的几何参考: xyzw -> wxyz -> matrix -> 6D。"""
    q_wxyz = q_xyzw[..., [3, 0, 1, 2]]
    return matrix_to_rotation_6d(quaternion_to_matrix(q_wxyz))


def _random_unit_quaternions(n: int, last_dim_extra: tuple = ()) -> torch.Tensor:
    """生成 (n, 4[, *extra]) 的随机单位四元数 (xyzw 存储)。"""
    shape = (n, 4, *last_dim_extra) if last_dim_extra else (n, 4)
    q = torch.randn(*shape, dtype=torch.float32)
    q = q / q.norm(dim=1, keepdim=True)
    return q


class QuaternionTo6DRotationTest(unittest.TestCase):
    def test_apply_xyzw_geometric_correctness(self):
        """xyzw 四元数经 apply 得到的 6D 与 matrix 参考路径逐元素一致。"""
        q_xyzw = _random_unit_quaternions(64)
        t = QuaternionTo6DRotation(apply_to=["state.end_0_rotation"], input_convention="xyzw")
        out = t.apply({"state.end_0_rotation": q_xyzw})["state.end_0_rotation"]
        expected = _expected_6d_from_xyzw(q_xyzw)
        self.assertEqual(out.shape, (64, 6))
        self.assertTrue(out.dtype == torch.float32)
        self.assertTrue(torch.allclose(out, expected, atol=_ATOL, rtol=_RTOL))

    def test_apply_identity_both_conventions(self):
        """单位旋转在 xyzw / wxyz 两种约定下都应得到 [1,0,0,0,1,0]。"""
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        # identity: w=1, x=y=z=0 -> xyzw=[0,0,0,1], wxyz=[1,0,0,0]
        for convention, q in (
            ("xyzw", torch.tensor([0.0, 0.0, 0.0, 1.0])),
            ("wxyz", torch.tensor([1.0, 0.0, 0.0, 0.0])),
        ):
            t = QuaternionTo6DRotation(apply_to=["k"], input_convention=convention)
            out = t.apply({"k": q})["k"]
            self.assertTrue(
                torch.allclose(out, identity_6d, atol=_ATOL, rtol=_RTOL),
                msg=f"convention={convention} -> {out}",
            )

    def test_roundtrip_recovers_equivalent_rotation(self):
        """apply -> unapply 后四元数被 canonicalize (w>=0), 但表示同一旋转。"""
        q_xyzw = _random_unit_quaternions(32)
        t = QuaternionTo6DRotation(apply_to=["k"], input_convention="xyzw")

        six = t.apply({"k": q_xyzw.clone()})["k"]
        rec = t.unapply({"k": six})["k"]  # 恢复的 xyzw (canonicalized)
        self.assertEqual(rec.shape[-1], 4)

        # 恢复值再转 6D 应与原 6D 一致 (q/-q 对 6D 不敏感)
        rec_again = t.apply({"k": rec.clone()})["k"]
        self.assertTrue(torch.allclose(rec_again, six, atol=_ATOL, rtol=_RTOL))
        # 恢复值与原值表示同一旋转 (经 matrix 参考路径比较)
        self.assertTrue(
            torch.allclose(_expected_6d_from_xyzw(rec), _expected_6d_from_xyzw(q_xyzw), atol=_ATOL, rtol=_RTOL)
        )

    def test_flat_dict_missing_key_is_noop(self):
        """apply_to 含字典中不存在的 key 时不报错、不影响其它 key。"""
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        t = QuaternionTo6DRotation(apply_to=["state.end_0_rotation", "not.present"], input_convention="xyzw")
        data = {"state.end_0_rotation": q, "other": torch.zeros(3)}
        out = t.apply(data)
        self.assertIn("state.end_0_rotation", out)
        self.assertEqual(out["state.end_0_rotation"].shape[-1], 6)  # 被转换
        self.assertIn("other", out)  # 未列入 apply_to, 原样保留

    def test_numpy_input_converted_to_tensor(self):
        """np.ndarray 输入应被转为 torch.Tensor 并完成 4D->6D 转换。"""
        q_np = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        t = QuaternionTo6DRotation(apply_to=["k"], input_convention="xyzw")
        out = t.apply({"k": q_np})["k"]
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[-1], 6)

    def test_non_quaternion_input_unchanged(self):
        """最后一维不是 4 的输入 (如 3D 位置) 应原样保留, 不被转换。"""
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        t = QuaternionTo6DRotation(apply_to=["k"], input_convention="xyzw")
        out = t.apply({"k": pos})["k"]
        self.assertTrue(torch.equal(out, pos))


if __name__ == "__main__":
    unittest.main()
