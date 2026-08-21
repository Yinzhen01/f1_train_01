import ast
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoid.motion_reference import load_joint_motion_csv


class MotionReferenceTest(unittest.TestCase):
    @staticmethod
    def _nested_class_assignments(source, *class_names):
        node = ast.parse(source)
        body = node.body
        for class_name in class_names:
            node = next(
                item
                for item in body
                if isinstance(item, ast.ClassDef) and item.name == class_name
            )
            body = node.body

        assignments = {}
        for item in body:
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
            ):
                assignments[item.targets[0].id] = ast.literal_eval(item.value)
        return assignments

    def test_repository_asset_contains_only_timestamp_and_12_controlled_joints(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "motions"
            / "x1"
            / "walk_12dof.csv"
        )
        controlled = (
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_pitch_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        )
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames or ()), ("timestamp",) + controlled)
            self.assertEqual(sum(1 for _ in reader), 415)

    def test_loader_maps_by_name_and_ignores_other_joints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motion.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "unused_joint", "joint_b", "joint_a"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "timestamp": 0.0,
                            "unused_joint": 99.0,
                            "joint_b": 2.0,
                            "joint_a": 1.0,
                        },
                        {
                            "timestamp": 0.1,
                            "unused_joint": 98.0,
                            "joint_b": 4.0,
                            "joint_a": 3.0,
                        },
                    ]
                )

            table = load_joint_motion_csv(path, ["joint_a", "joint_b"])

        self.assertEqual(table.joint_names, ("joint_a", "joint_b"))
        np.testing.assert_allclose(table.positions, [[1.0, 2.0], [3.0, 4.0]])

    def test_trimmed_clip_can_be_closed_for_periodic_sampling(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "motions"
            / "x1"
            / "walk_12dof.csv"
        )
        names = (
            "left_hip_pitch_joint",
            "left_knee_pitch_joint",
            "right_hip_pitch_joint",
            "right_knee_pitch_joint",
        )
        table = load_joint_motion_csv(
            source,
            names,
            start_time=0.6,
            end_time=5.533333333333333,
            close_loop=True,
        )

        self.assertEqual(table.positions.shape[1], len(names))
        self.assertAlmostEqual(table.duration, 4.933333333333333, places=6)
        np.testing.assert_allclose(table.positions[0], table.positions[-1])
        self.assertTrue(np.all(np.diff(table.timestamps) > 0.0))

    def test_missing_controlled_joint_is_rejected(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "motions"
            / "x1"
            / "walk_12dof.csv"
        )
        with self.assertRaisesRegex(ValueError, "missing columns"):
            load_joint_motion_csv(source, ["not_a_robot_joint"])

    def test_retarget_reward_profile_is_reference_first(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "humanoid"
            / "envs"
            / "x1"
            / "x1_dh_stand_retarget_walk_config.py"
        ).read_text(encoding="utf-8")
        rewards = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkCfg", "rewards"
        )
        scales = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkCfg", "rewards", "scales"
        )

        self.assertEqual(rewards["tracking_sigma"], 30.0)
        self.assertEqual(scales["ref_joint_pos"], 6.0)
        self.assertEqual(scales["tracking_lin_vel"], 4.0)
        self.assertEqual(scales["low_speed"], 2.0)
        for disabled_term in (
            "default_joint_pos",
            "feet_contact_number",
            "feet_air_time",
            "feet_clearance",
            "track_vel_hard",
        ):
            self.assertEqual(scales[disabled_term], 0.0)


if __name__ == "__main__":
    unittest.main()
