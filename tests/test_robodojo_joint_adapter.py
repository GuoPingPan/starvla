"""Mapping checks for RoboDojo's absolute bimanual joint controller."""

import unittest

import numpy as np

from starVLA.deploy.robodojo_pi05_joint.mapping import (
    action_at,
    observation_to_topics,
    validate_action_chunk,
    validate_training_keys,
)


class RoboDojoJointMappingTest(unittest.TestCase):
    def setUp(self):
        self.obs_state = {
            "left_arm_joint_state": np.arange(6, dtype=np.float32),
            "left_ee_joint_state": [0.25],
            "right_arm_joint_state": np.arange(6, 12, dtype=np.float32),
            "right_ee_joint_state": [0.75],
        }
        self.actions = {
            "left_arm_joint": np.arange(12, dtype=np.float32).reshape(2, 6),
            "left_gripper": np.array([[1.2], [0.4]], dtype=np.float32),
            "right_arm_joint": np.arange(12, 24, dtype=np.float32).reshape(2, 6),
            "right_gripper": np.array([[-0.2], [0.8]], dtype=np.float32),
        }

    def test_state_and_action_follow_training_order(self):
        validate_training_keys(
            ["state.left_arm_joint", "state.left_gripper", "state.right_arm_joint", "state.right_gripper"],
            ["action.left_arm_joint", "action.left_gripper", "action.right_arm_joint", "action.right_gripper"],
        )
        topics = observation_to_topics(self.obs_state)
        np.testing.assert_array_equal(
            np.concatenate(list(topics.values())),
            np.array([0, 1, 2, 3, 4, 5, 0.25, 6, 7, 8, 9, 10, 11, 0.75], dtype=np.float32),
        )
        validate_action_chunk(self.actions, min_horizon=2)
        step = action_at(self.actions, 0)
        self.assertEqual(
            list(step),
            ["left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"],
        )
        np.testing.assert_array_equal(step["left_arm_joint_state"], np.arange(6, dtype=np.float32))
        np.testing.assert_array_equal(step["right_arm_joint_state"], np.arange(12, 18, dtype=np.float32))
        self.assertEqual(step["left_ee_joint_state"].tolist(), [1.0])
        self.assertEqual(step["right_ee_joint_state"].tolist(), [0.0])

    def test_bad_checkpoint_or_observation_fails_before_control(self):
        with self.assertRaises(ValueError):
            validate_training_keys(
                ["state.left_arm_joint", "state.right_arm_joint", "state.left_gripper", "state.right_gripper"],
                ["action.left_arm_joint", "action.left_gripper", "action.right_arm_joint", "action.right_gripper"],
            )
        with self.assertRaises(KeyError):
            observation_to_topics({"left_arm_joint_state": self.obs_state["left_arm_joint_state"]})
        bad = dict(self.obs_state, right_arm_joint_state=[float("nan")] * 6)
        with self.assertRaises(ValueError):
            observation_to_topics(bad)

    def test_bad_decoded_chunk_fails_before_control(self):
        bad = dict(self.actions, right_arm_joint=np.zeros((1, 6), dtype=np.float32))
        with self.assertRaises(ValueError):
            validate_action_chunk(bad, min_horizon=2)
        bad = dict(self.actions, left_arm_joint=np.zeros((2, 5), dtype=np.float32))
        with self.assertRaises(ValueError):
            validate_action_chunk(bad, min_horizon=2)


if __name__ == "__main__":
    unittest.main()
