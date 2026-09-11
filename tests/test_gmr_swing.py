"""Reward math, geometry, inherited contracts and strict 7000-update lineage."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from humanoid import gmr_swing as swing
from humanoid import gmr_swing_finetune as warm
from humanoid.gmr_motion import GMRMotion
from test_gmr_smooth import config_scope, plain, network
from test_gmr_environment_math import quat_apply, quat_mul

ROOT = Path(__file__).resolve().parents[1]


def settings_scope():
    scope = config_scope()
    path = ROOT / "humanoid/envs/x1/x1_gmr_swing_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def geometry():
    return swing.load_geometry(ROOT / "resources/motions/gmr_kit317_swing_geometry.json",
        warm.SOURCE_REFERENCE_SHA256, ROOT / "resources/robots/x1/urdf/X1_12DOF.urdf",
        ["left_ankle_roll_link", "right_ankle_roll_link"], "cpu")


class SwingConfigTest(unittest.TestCase):
    def test_only_two_new_costs_and_their_settings(self):
        scope = settings_scope()
        old = plain(scope["X1GMRSmoothCfg"]())
        new = plain(scope["X1GMRSwingCfg"]())
        parameters = new.pop("swing")
        self.assertEqual(parameters["height_tolerance_m"], .005)
        self.assertEqual(new["rewards"]["scales"].pop("gmr_swing_clearance"), -1.)
        self.assertEqual(new["rewards"]["scales"].pop("gmr_swing_contact"), -1.)
        self.assertEqual(old, new)  # Every observation/action/dynamics/old reward field.

    def test_only_runner_name_and_budget_change_in_ppo(self):
        scope = settings_scope()
        old = plain(scope["X1GMRSmoothCfgPPO"]())
        new = plain(scope["X1GMRSwingCfgPPO"]())
        self.assertEqual(new["runner"]["max_iterations"], 1000)
        new["runner"]["max_iterations"] = old["runner"]["max_iterations"]
        new["runner"]["experiment_name"] = old["runner"]["experiment_name"]
        self.assertEqual(old, new)
        self.assertIn('register("x1_gmr_swing", X1GMRSwingEnv', (ROOT / "humanoid/envs/__init__.py").read_text())

    def test_reference_and_source_identity(self):
        self.assertEqual(warm.SOURCE_TASK, "TASK_20260911_116")
        self.assertEqual(warm.SOURCE_COMPLETED_UPDATES, 7000)
        self.assertEqual(warm.sha256(ROOT / "resources/motions/gmr_kit317_upright_12dof.npz"), warm.SOURCE_REFERENCE_SHA256)
        source = (ROOT / "humanoid/scripts/train_gmr_swing.py").read_text()
        self.assertIn('args.checkpoint != 7000', source)
        self.assertIn('1 <= additional <= 1000', source)
        self.assertLess(source.index('identity = warm_start_runner'), source.index('runner.learn('))


class SwingGeometryTest(unittest.TestCase):
    def test_hull_minimum_matches_full_mesh_for_random_rotations(self):
        sys.path.insert(0, str(ROOT / "humanoid/scripts"))
        from render_gmr_policy import foot_meshes
        points, meta = geometry()
        meshes = foot_meshes(ROOT / "resources/robots/x1/urdf/X1_12DOF.urdf", meta["foot_names"])
        rotation = Rotation.random(64, random_state=11)
        q = torch.tensor(rotation.as_quat(), dtype=torch.float32)[:, None].expand(-1, 2, -1)
        position = torch.ones(64, 2, 3)
        actual = swing.minimum_height(position, q, points).numpy()
        for foot, vertices in enumerate(meshes):
            centered = vertices - np.array(meta["sole_local"])[foot]
            expected = (rotation.as_matrix()[:, 2, :] @ centered.T).min(axis=1) + 1.
            np.testing.assert_allclose(actual[:, foot], expected, atol=2e-7, rtol=0)

    def test_geometry_identity_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference mismatch"):
            swing.load_geometry(ROOT / "resources/motions/gmr_kit317_swing_geometry.json", "wrong",
                ROOT / "resources/robots/x1/urdf/X1_12DOF.urdf",
                ["left_ankle_roll_link", "right_ankle_roll_link"], "cpu")

    def test_reference_standing_gate_is_zero_and_each_swing_has_active_samples(self):
        with np.load(ROOT / "resources/motions/gmr_kit317_upright_12dof.npz", allow_pickle=False) as data:
            contact = torch.tensor(data["foot_contact"])
        gate = swing.swing_envelope(contact, 30., .04, .06)
        self.assertTrue(torch.all(gate[contact > .1] == 0))
        self.assertTrue(torch.all(gate[:31] == 0))
        self.assertTrue(torch.all(gate[126:] == 0))
        samples = swing.sample_envelope(gate, torch.tensor([1.45, 2.04, 2.19]), 30.)
        self.assertGreater(samples[0, 1].item(), .9)
        self.assertGreater(samples[1, 0].item(), .9)
        self.assertGreater(samples[2, 0].item(), .9)

    def test_contact_edges_short_runs_and_clamped_sampling(self):
        contact = torch.ones(31, 2)
        contact[3:28, 0] = 0
        contact[5:7, 1] = 0
        gate = swing.swing_envelope(contact, 100., .04, .06)
        self.assertEqual(gate[3, 0], 0.)
        self.assertEqual(gate[27, 0], 0.)
        self.assertEqual(gate[:, 1].sum(), 0.)
        self.assertGreater(gate[:, 0].max(), .9)
        value = swing.sample_envelope(gate, torch.tensor([-2., 100.]), 100.)
        torch.testing.assert_close(value, torch.zeros(2, 2))


class SwingRewardTest(unittest.TestCase):
    def setUp(self):
        self.cfg = settings_scope()["X1GMRSwingCfg"]().swing

    def terms(self, actual, target=.05, contact=0., valid=True, phase=1.):
        return swing.swing_terms(torch.tensor([[actual, actual]]), torch.full((1, 2), target),
            torch.full((1, 2), phase), torch.full((1, 2), contact), torch.tensor([valid]), self.cfg)

    def test_clearance_deficit_is_supervised_even_without_contact(self):
        out = self.terms(.025)
        torch.testing.assert_close(out["clearance_cost"], torch.tensor([2.]))
        self.assertEqual(out["contact_cost"].item(), 0.)

    def test_exact_reference_tolerance_and_overshoot_have_no_new_cost(self):
        for height in (.05, .045, .2):
            self.assertLess(self.terms(height)["clearance_cost"].item(), 1e-10)

    def test_unexpected_contact_uses_existing_threshold(self):
        self.assertEqual(self.terms(.05, contact=5.)["contact_cost"].item(), 0.)
        self.assertEqual(self.terms(.05, contact=5.01)["contact_cost"].item(), 2.)

    def test_boundary_stance_near_floor_and_reset_disable_new_costs(self):
        for kwargs in (dict(phase=0.), dict(valid=False), dict(target=.004)):
            out = self.terms(0., contact=100., **kwargs)
            self.assertEqual(out["clearance_cost"].item(), 0.)
            self.assertEqual(out["contact_cost"].item(), 0.)

    def test_large_errors_bounded_and_height_gate_continuous(self):
        self.assertEqual(self.terms(-1.)["clearance_cost"].item(), 18.)
        out = self.terms(0., target=.01, contact=100.)
        self.assertAlmostEqual(out["contact_cost"].item(), 1., places=5)

    def test_world_frame_target_independent_of_actual_root_and_cache_precedes_sum(self):
        class Base:
            def compute_reward(self):
                self.sum_seen = self._reward_gmr_swing_clearance().clone()
            def get_command_tracking_debug(self):
                return {}
        path = ROOT / "humanoid/envs/x1/x1_gmr_swing_env.py"
        tree = ast.parse(path.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        scope = dict(X1GMRSmoothEnv=Base, torch=torch, quat_apply=quat_apply, quat_mul=quat_mul,
                     minimum_height=swing.minimum_height, sample_envelope=swing.sample_envelope,
                     swing_terms=swing.swing_terms)
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), scope)
        e = scope["X1GMRSwingEnv"].__new__(scope["X1GMRSwingEnv"])
        e.cfg = SimpleNamespace(swing=self.cfg)
        e.swing_hulls = torch.zeros(2, 4, 3)
        e.swing_envelope = torch.ones(2, 2)
        e.motion = SimpleNamespace(fps=30.)
        e.motion_time = lambda: torch.zeros(1)
        e.episode_length_buf = torch.tensor([3])
        e.env_origins = torch.tensor([[10., 20., 3.]])
        e.reference = dict(root_quat=torch.tensor([[0., 0., 0., 1.]]),
                           root_pos=torch.tensor([[0., 0., .6]]),
                           foot_pos_base=torch.tensor([[[0., 0., -.55], [0., 0., -.55]]]),
                           foot_quat_base=torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]]))
        e.rigid_state = torch.zeros(1, 2, 13); e.rigid_state[:, :, 6] = 1.
        e.feet_indices = [0, 1]
        e.contact_forces = torch.zeros(1, 2, 3)
        e._sole_positions = lambda: torch.tensor([[[10., 20., 3.025], [10., 20., 3.025]]])
        e._sole_velocities = lambda: torch.zeros(1, 2, 3)
        e.compute_reward()
        torch.testing.assert_close(e.swing_target_height, torch.full((1, 2), .05), atol=3e-7, rtol=0)
        self.assertAlmostEqual(e.sum_seen.item(), 2., places=4)
        self.assertTrue(all(torch.isfinite(v) for v in e.get_command_tracking_debug().values()))
        e.episode_length_buf.zero_(); e.compute_reward()
        self.assertEqual(e.sum_seen.item(), 0.)  # No stale per-environment costs across resets.


class SwingWarmStartTest(unittest.TestCase):
    def test_source_restores_all_weights_and_starts_at_7000(self):
        source, target = network(), network()
        optimizer = torch.optim.Adam(target.parameters(), lr=1e-5)
        estimator = torch.optim.Adam(target.state_estimator.parameters(), lr=1e-5)
        runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(actor_critic=target,
            optimizer=optimizer, state_estimator_optimizer=estimator))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_7000.pt"
            torch.save(dict(model_state_dict=source.state_dict(), iter=6999), path)
            with patch.object(warm, "SOURCE_SHA256", warm.sha256(path)):
                result = warm.warm_start_runner(runner, path)
        self.assertEqual(runner.current_learning_iteration, 7000)
        self.assertEqual(result["loaded_tensors"], 33)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)

    def test_missing_or_wrong_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_7000.pt"
            with self.assertRaises(FileNotFoundError):
                warm.locate_verified_checkpoint(path, [directory])
            torch.save({"not_source": torch.zeros(1)}, path)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                warm.locate_verified_checkpoint(path, [directory])


if __name__ == "__main__":
    unittest.main()
