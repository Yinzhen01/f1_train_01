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
        self.assertEqual(scales["ref_feet_contact"], 2.0)
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

    def test_retarget_profile_uses_contact_consistent_period_and_speed(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "humanoid"
            / "envs"
            / "x1"
            / "x1_dh_stand_retarget_walk_config.py"
        ).read_text(encoding="utf-8")
        motion = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkCfg", "motion_reference"
        )
        ranges = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkCfg", "commands", "ranges"
        )
        rewards = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkCfg", "rewards"
        )

        self.assertIn("walk_12dof_contact_consistent.csv", motion["file"])
        self.assertEqual(motion["contact_columns"], ("left_contact", "right_contact"))
        self.assertAlmostEqual(motion["phase_offset"], 26.0 / 146.0)
        self.assertAlmostEqual(motion["end_time"], 5.466666666666667)
        self.assertEqual(ranges["lin_vel_x"], [0.124, 0.124])
        self.assertAlmostEqual(rewards["cycle_time"], 4.866666666666667)

    def test_vx045_profile_time_scales_reference_and_command_together(self):
        config = (
            Path(__file__).resolve().parents[1]
            / "humanoid"
            / "envs"
            / "x1"
            / "x1_dh_stand_retarget_walk_config.py"
        ).read_text(encoding="utf-8")
        ranges = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkVx045Cfg", "commands", "ranges"
        )
        rewards = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkVx045Cfg", "rewards"
        )
        motion = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkVx045Cfg", "motion_reference"
        )
        scales = self._nested_class_assignments(
            config, "X1DHStandRetargetWalkVx045Cfg", "rewards", "scales"
        )

        self.assertEqual(ranges["lin_vel_x"], [0.45, 0.45])
        self.assertAlmostEqual(rewards["cycle_time"], 1.3393364741639295)
        self.assertEqual(motion["clearance_scale"], 1.05)
        self.assertEqual(motion["clearance_lift_offset"], 0.005)
        self.assertEqual(motion["clearance_max"], 0.18)
        self.assertEqual(rewards["ref_feet_clearance_sigma"], 400.0)
        self.assertEqual(rewards["ref_feet_clearance_low_penalty"], 0.5)
        self.assertEqual(scales["ref_feet_clearance"], 3.0)

    def test_foot_clearance_reference_matches_selected_motion(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "motions"
            / "x1"
            / "walk_foot_clearance.csv"
        )
        table = load_joint_motion_csv(
            source,
            ("left_foot_clearance", "right_foot_clearance"),
            start_time=0.6,
            end_time=5.466666666666667,
            close_loop=True,
        )
        self.assertEqual(table.frame_count, 147)
        self.assertAlmostEqual(table.duration, 4.866666666666667)
        self.assertGreater(float(table.positions[:, 0].max()), 0.15)
        self.assertGreater(float(table.positions[:, 1].max()), 0.11)

    def test_reference_contacts_are_periodic_and_phase_aligned(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "motions"
            / "x1"
            / "walk_12dof_contact_consistent.csv"
        )
        table = load_joint_motion_csv(
            source,
            ("left_contact", "right_contact"),
            start_time=0.6,
            end_time=5.466666666666667,
            close_loop=True,
        )
        contacts = table.positions[:-1] >= 0.5
        self.assertEqual(table.frame_count, 147)
        np.testing.assert_allclose(table.positions[0], table.positions[-1])
        np.testing.assert_allclose(contacts.mean(axis=0), [0.5410959, 0.5410959])
        self.assertEqual(int(np.sum(np.all(contacts, axis=1))), 12)

        frame_count = contacts.shape[0]
        phase = np.arange(frame_count, dtype=np.float64) / frame_count
        sample_index = (
            np.arange(frame_count) + 26
        ) % frame_count
        sin_phase = np.sin(2.0 * np.pi * phase)
        stance = np.column_stack((sin_phase >= 0.0, sin_phase < 0.0))
        stance[np.abs(sin_phase) < 0.1] = True
        agreement = np.mean(stance == contacts[sample_index])
        self.assertGreater(agreement, 0.99)


if __name__ == "__main__":
    unittest.main()
