"""Offline evidence calculations, independent of cloud training implementation."""
import importlib.util
from pathlib import Path
import unittest
import hashlib
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from humanoid.amp.scaled_experiment import ScaledExperiment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_rollout", ROOT/"tools/amp/inspect_rollout.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
torch.set_num_threads(1)


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = ScaledExperiment(ROOT, ROOT/"configs/amp/lafan_walk02_recovery.json")

    def test_prefix_rejects_resets_and_missing_ticks(self):
        arrays = {"standing_valid": np.array([[True], [False], [True]])}
        with self.assertRaisesRegex(ValueError, "restarted"):
            audit.episode(arrays, "standing", 0)
        arrays = {"standing_valid": np.array([[True], [True], [False]]),
                  "standing_time": np.array([.01, .03, .04])}
        with self.assertRaisesRegex(ValueError, "ticks"):
            audit.episode(arrays, "standing", 0)

    def test_swing_does_not_count_truncated_or_short_runs(self):
        self.assertEqual(audit.swing_runs(np.zeros(20, dtype=bool)), 0)
        self.assertEqual(audit.swing_runs([True]*2+[False]*8+[True]*2), 1)
        self.assertEqual(audit.swing_runs([True]*2+[False]*7+[True]*2), 0)
        self.assertEqual(audit.swing_runs([False]*8+[True]*2+[False]*8), 0)

    def test_mesh_rotation_and_rigid_velocity(self):
        state = np.zeros((1, 2, 13))
        state[..., 6] = 1
        state[..., 2] = .1
        state[..., 7] = .3
        state[..., 11] = 2
        vertices = {name: np.array([[0., 0., -.1], [.1, 0., 0.]]) for name in ("L", "R")}
        height, speed = audit.foot_geometry({"foot_state": state}, vertices, ("L", "R"))
        np.testing.assert_allclose(height, 0)
        np.testing.assert_allclose(speed, .1)  # vx + omega_y * sole_z

    def test_shared_features_reconstruct_exact_demonstration(self):
        clip = self.e.clip
        root = np.zeros((clip.frames, 13), dtype=np.float32)
        root[:, :3] = clip.arrays["qpos"][:, :3]
        root[:, 3:7] = clip.arrays["qpos"][:, [4, 5, 6, 3]]
        rotation = Rotation.from_quat(root[:, 3:7]).as_matrix()
        keys = np.einsum("nij,nkj->nki", rotation, clip.key_positions_b)+root[:, None, :3]
        all_data = dict(root_state=root, dof_pos=clip.joint_pos, key_positions_w=keys)
        data = {key: value[1:] for key, value in all_data.items()}
        data["initial"] = {key: value[0] for key, value in all_data.items()}
        result = audit.features_from_episode(data, self.e)
        torch.testing.assert_close(result, clip.features[1:], atol=3e-5, rtol=1e-5)

    def test_nearest_distance_self_and_displacement(self):
        demo = torch.zeros((2, 10, 39))
        mean, std = torch.zeros(39), torch.ones(39)
        self.assertEqual(audit.nearest_window_distance(demo, demo, mean, std)["mean"], 0.)
        self.assertAlmostEqual(audit.nearest_window_distance(demo+2, demo, mean, std)["mean"], 2., places=5)
        self.assertIsNone(audit.nearest_window_distance(demo[:0], demo, mean, std))

    def test_initial_failure_cannot_survive_or_claim_post_transient_metrics(self):
        data = dict(time=np.zeros(0), initial={"failure": True}, failure=np.zeros(0, dtype=bool))
        report, geometry = audit.analyze_episode(data, None, None, None, 20)
        self.assertTrue(report["failure"])
        self.assertFalse(report["survived"])
        self.assertIsNone(report["post_2s"])
        self.assertIsNone(geometry)

    def test_heading_wrap_and_spectral_bands(self):
        root = np.zeros((101, 13))
        root[:, 3:7] = Rotation.from_euler("z", np.deg2rad(np.linspace(170, 190, 101))).as_quat()
        self.assertAlmostEqual(audit.heading_metrics(root)["heading_change_deg"], 20.)
        t = np.arange(2000)*.01
        slow = audit.velocity_spectrum(np.sin(2*np.pi*2*t)[:, None])
        fast = audit.velocity_spectrum(np.sin(2*np.pi*20*t)[:, None])
        self.assertLess(slow["power_fraction_above_cutoff"], 1e-6)
        self.assertGreater(fast["power_fraction_above_cutoff"], .999999)
        self.assertEqual(audit.velocity_spectrum(np.zeros((2000, 12)))["power_fraction_above_cutoff"], 0.)

    def test_nearest_group_error_uses_same_whole_window(self):
        demo = torch.zeros((2, 10, 39)); demo[1] = 100
        query = torch.zeros((1, 10, 39)); query[..., 15:27] = 2
        out = audit.nearest_window_distance(query, demo, torch.zeros(39), torch.ones(39))
        self.assertEqual(out["nearest_window_group_mse"]["joint_velocity"], 4.)
        self.assertEqual(out["nearest_window_group_mse"]["joint_position"], 0.)

    def test_cloud_log_markers_survive_sdk_line_interleaving(self):
        from tools.amp.verify_refinement_smoke import parse_updates
        text = '[SDK] upload[f1-amp-update] {"iteration":1001}[SDK] done\n[f1-amp-update] {"iteration":1002}\n'
        self.assertEqual([r['iteration'] for r in parse_updates(text)], [1001, 1002])

    def test_comparison_keeps_failed_episodes_and_exposes_missing_metrics(self):
        from tools.amp.compare_rollout_reports import summarize
        rows = [dict(survived=True, post_2s={'vx_mean': .45}),
                dict(survived=False, post_2s={'vx_mean': .15}),
                dict(survived=False, post_2s=None)]
        all_observed, survivors = summarize(rows), summarize(rows, True)
        self.assertEqual(all_observed['vx']['n'], 2)
        self.assertAlmostEqual(all_observed['vx']['mean'], .3)
        self.assertEqual(survivors['vx']['n'], 1)
        self.assertAlmostEqual(survivors['vx']['mean'], .45)
        self.assertEqual(all_observed['slip']['n'], 0)
        self.assertIsNone(all_observed['slip']['mean'])

    def test_mounted_lookup_requires_exact_hash(self):
        from tools.amp.evaluate_mounted import locate_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"model_1000_test.pt"
            path.write_bytes(b"owned-test-fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(locate_checkpoint([root, root], digest), path.resolve())
            with self.assertRaises(FileNotFoundError):
                locate_checkpoint([root], "0"*64)


if __name__ == "__main__":
    unittest.main()
