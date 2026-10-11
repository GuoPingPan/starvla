"""Geometric checks for RelativePoseTransform's quaternion conventions."""

import math
import unittest

import torch

from starVLA.dataloader.gr00t_lerobot.transform.relative_pose import RelativePoseTransform


_SPEC = {
    "action.pose": {
        "position": "action.position",
        "rotation": "action.rotation",
        "state_position": "state.position",
        "state_rotation": "state.rotation",
    }
}


class RelativePoseQuaternionConventionTest(unittest.TestCase):
    def _sample(self, convention: str):
        # A 90° Z base followed by a 90° X relative rotation. The target
        # quaternion has four nonzero components, so a wrong component order
        # changes both the relative rotation and the rotated translation.
        root_half = math.sqrt(0.5)
        base_wxyz = torch.tensor([[root_half, 0.0, 0.0, root_half]])
        target_wxyz = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        if convention == "xyzw":
            base = base_wxyz[:, [1, 2, 3, 0]]
            target = target_wxyz[:, [1, 2, 3, 0]]
            expected_rotation = torch.tensor([[root_half, 0.0, 0.0, root_half]])
        else:
            base = base_wxyz
            target = target_wxyz
            expected_rotation = torch.tensor([[root_half, root_half, 0.0, 0.0]])
        data = {
            "state.position": torch.tensor([[0.0, 0.0, 0.0]]),
            "state.rotation": base,
            "action.position": torch.tensor([[1.0, 0.0, 0.0]]),
            "action.rotation": target,
        }
        return data, expected_rotation

    def test_wxyz_fast_and_slow_geometry_and_roundtrip(self):
        for fast in (True, False):
            with self.subTest(fast=fast):
                transform = RelativePoseTransform(
                    apply_to=["action.pose"], split_pose_keys=_SPEC,
                    input_quaternion_convention="wxyz", fast_forward=fast,
                )
                original, expected_rotation = self._sample("wxyz")
                relative = transform.apply(original)
                self.assertTrue(torch.allclose(
                    relative["action.position"], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5,
                ))
                self.assertTrue(torch.allclose(
                    relative["action.rotation"], expected_rotation, atol=1e-5,
                ))
                restored = transform.unapply(relative)
                self.assertTrue(torch.allclose(
                    restored["action.position"], original["action.position"], atol=1e-5,
                ))
                self.assertTrue(torch.allclose(
                    restored["action.rotation"], original["action.rotation"], atol=1e-5,
                ))

    def test_xyzw_default_preserves_existing_geometry(self):
        transform = RelativePoseTransform(
            apply_to=["action.pose"], split_pose_keys=_SPEC, fast_forward=True,
        )
        original, expected_rotation = self._sample("xyzw")
        relative = transform.apply(original)
        self.assertTrue(torch.allclose(
            relative["action.position"], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            relative["action.rotation"], expected_rotation, atol=1e-5,
        ))


if __name__ == "__main__":
    unittest.main()
