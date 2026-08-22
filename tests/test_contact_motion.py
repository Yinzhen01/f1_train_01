import csv
import json
import unittest
from pathlib import Path

import numpy as np

from humanoid.contact_motion import estimate_cycle_frames, periodic_contact_schedule
from humanoid.motion_kinematics import (
    chain_to_link,
    evaluate_chain,
    parse_urdf,
    sole_center_from_mesh,
)


class ContactMotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.urdf = cls.root / "resources" / "robots" / "x1" / "urdf" / "X1_12DOF.urdf"

    def test_mesh_derived_sole_points_use_mirrored_local_y_axis(self):
        by_child, limits = parse_urdf(self.urdf)
        points = []
        for side in ("left", "right"):
            link = f"{side}_ankle_roll_link"
            chain = chain_to_link(by_child, link)
            zero_pose = evaluate_chain(chain, {}, limits)
            points.append(sole_center_from_mesh(self.urdf, link, zero_pose))

        self.assertAlmostEqual(points[0][1], -0.0408, places=5)
        self.assertAlmostEqual(points[1][1], 0.0408, places=5)
        self.assertAlmostEqual(points[0][2], 0.005014, places=5)
        self.assertAlmostEqual(points[1][2], 0.005015, places=5)
        # The old [0, 0, -0.041] point moved along foot length, not downward.
        self.assertGreater(abs(points[0][1]), abs(points[0][2]))

    def test_repository_walk_has_146_frame_full_body_cycle(self):
        source = self.root / "resources" / "motions" / "x1" / "walk_12dof.csv"
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        timestamps = np.asarray([float(row["timestamp"]) for row in rows])
        joint_names = [name for name in rows[0] if name != "timestamp"]
        positions = np.asarray(
            [[float(row[name]) for name in joint_names] for row in rows]
        )
        self.assertEqual(estimate_cycle_frames(timestamps, positions), 146)

    def test_periodic_schedule_ignores_an_extra_transient_crossing(self):
        frames = np.arange(360)
        relative_height = np.sin(2.0 * np.pi * (frames - 20) / 120.0)
        relative_height[145:150] *= -1.0
        schedule = periodic_contact_schedule(relative_height, 120, smoothing_frames=11)
        self.assertEqual(schedule.half_cycle_frames, 60)
        self.assertTrue(np.all(np.diff(schedule.switch_frames) == 60))
        actual_switches = (
            np.flatnonzero(schedule.support_foot[1:] != schedule.support_foot[:-1]) + 1
        )
        np.testing.assert_array_equal(actual_switches, schedule.switch_frames)

    def test_generated_asset_contains_only_root_12_joints_and_contacts(self):
        source = (
            self.root
            / "resources"
            / "motions"
            / "x1"
            / "walk_12dof_contact_consistent.csv"
        )
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
        self.assertEqual(len(rows), 415)
        self.assertEqual(len(fields), 22)
        self.assertEqual(fields[-2:], ("left_contact", "right_contact"))
        self.assertFalse(any("shoulder" in name or "lumbar" in name for name in fields))

    def test_real_motion_diagnostics_show_large_slip_reduction(self):
        path = (
            self.root
            / "resources"
            / "motions"
            / "x1"
            / "walk_contact_diagnostics.json"
        )
        metrics = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["cycle_frames"], 146)
        self.assertGreater(metrics["mean_forward_step_length_m"], 0.28)
        self.assertLess(metrics["mean_forward_step_length_m"], 0.33)
        self.assertLess(metrics["reconstructed"]["stance_slip_p95_mps"], 0.01)
        self.assertGreater(metrics["slip_rms_reduction_fraction"], 0.60)


if __name__ == "__main__":
    unittest.main()
