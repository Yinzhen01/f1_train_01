"""CPU reward/identity/configuration gates; cloud smoke remains mandatory."""
import ast
import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from humanoid import gmr_finetune as warm

ROOT = Path(__file__).resolve().parents[1]


def config_scope():
    scope = dict(inspect=inspect, LEGGED_GYM_ROOT_DIR=str(ROOT))
    for filename in ("base/base_config.py", "base/legged_robot_config.py", "x1/x1_dh_stand_config.py",
                     "x1/x1_dh_stand_no_dr_config.py", "x1/x1_gmr_clip_config.py",
                     "x1/x1_gmr_upright_config.py", "x1/x1_gmr_smooth_config.py"):
        path = ROOT / "humanoid/envs" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), scope)
    return scope


def plain(obj):
    if isinstance(obj, (str, int, float, bool, dict, list, tuple)) or obj is None:
        return obj
    return {k: plain(getattr(obj, k)) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}


def network():
    scope = config_scope()
    cfg, ppo = scope["X1GMRSmoothCfg"](), scope["X1GMRSmoothCfgPPO"]()
    spec = importlib.util.spec_from_file_location("smooth_actor", ROOT / "humanoid/algo/ppo/actor_critic_dh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        net = module.ActorCriticDH(235, 47, 219, 12, **plain(ppo.policy))
    return net


def environment_class():
    class Base:
        def compute_reward(self):
            pass
    path = ROOT / "humanoid/envs/x1/x1_gmr_smooth_env.py"
    definition = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    scope = dict(torch=torch, json=json, X1GMRUprightEnv=Base)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), scope)
    return scope["X1GMRSmoothEnv"]


class SmoothConfigTest(unittest.TestCase):
    def test_only_two_environment_reward_changes(self):
        scope = config_scope()
        old, new = plain(scope["X1GMRUprightCfg"]()), plain(scope["X1GMRSmoothCfg"]())
        self.assertEqual(new["rewards"]["scales"].pop("gmr_torque_rate"), -5e-5)
        self.assertEqual(new["rewards"]["scales"]["action_smoothness"], -1.2)
        new["rewards"]["scales"]["action_smoothness"] = -.01
        self.assertEqual(old, new)  # Includes URDF, obs, defaults, PD, DR, contact and reference.

    def test_ppo_network_unchanged_fixed_low_learning_rate(self):
        scope = config_scope()
        old, new = scope["X1GMRUprightCfgPPO"](), scope["X1GMRSmoothCfgPPO"]()
        self.assertEqual(plain(old.policy), plain(new.policy))
        self.assertEqual(new.algorithm.learning_rate, 1e-5)
        self.assertEqual(new.algorithm.schedule, "fixed")
        self.assertEqual(new.algorithm.lin_vel_idx, 199)
        self.assertEqual(new.runner.max_iterations, 2000)

    def test_unchanged_reference_hash_and_registered_task(self):
        self.assertEqual(warm.sha256(ROOT / "resources/motions/gmr_kit317_upright_12dof.npz"), warm.SOURCE_REFERENCE_SHA256)
        self.assertIn('register("x1_gmr_smooth", X1GMRSmoothEnv', (ROOT / "humanoid/envs/__init__.py").read_text())


class SmoothRewardTest(unittest.TestCase):
    def setUp(self):
        cls = environment_class()
        self.e = e = cls.__new__(cls)
        e.episode_length_buf = torch.tensor([1, 2, 3])
        e.actions = torch.ones(3, 12)
        e.last_actions = torch.zeros(3, 12)
        e.last_last_actions = torch.zeros(3, 12)
        e.torques = torch.ones(3, 12) * 2.
        e.last_torques = torch.ones(3, 12)
        e.torque_limits = torch.ones(12) * 10.
        e.dt = .01
        e.smooth_ankle_ids = [4, 5, 10, 11]
        e.feet_indices = [0, 1]
        e.contact_forces = torch.ones(3, 2, 3)

    def test_reset_history_gates(self):
        torch.testing.assert_close(self.e._reward_action_smoothness(), torch.tensor([0., 1., 2.]))
        torch.testing.assert_close(self.e._reward_gmr_torque_rate(), torch.tensor([0., 100., 100.]))

    def test_constant_nonzero_action_has_no_l1_pose_bias(self):
        self.e.last_actions[:] = self.e.actions
        self.e.last_last_actions[:] = self.e.actions
        torch.testing.assert_close(self.e._reward_action_smoothness(), torch.zeros(3))

    def test_torque_normalization_and_time_units(self):
        self.e.torque_limits *= 2.
        torch.testing.assert_close(self.e._reward_gmr_torque_rate(), torch.tensor([0., 25., 25.]))
        self.e.dt *= 2.
        torch.testing.assert_close(self.e._reward_gmr_torque_rate(), torch.tensor([0., 6.25, 6.25]))

    def test_diagnostics_survive_history_overwrite(self):
        self.e.compute_reward()
        self.e.last_actions[:] = self.e.actions
        self.e.last_torques[:] = self.e.torques
        d = self.e.smoothness_diagnostics
        self.assertAlmostEqual(d["gmr_action_delta_rms"].item(), 1.)
        self.assertAlmostEqual(d["gmr_torque_delta_rms_nm"].item(), 1.)
        self.assertEqual(d["gmr_ankle_saturation_fraction"].item(), 0.)

    def test_empty_history_metrics_are_finite(self):
        self.e.episode_length_buf[:] = 1
        self.e.compute_reward()
        self.assertTrue(all(torch.isfinite(v) for v in self.e.smoothness_diagnostics.values()))
        self.assertEqual(self.e.smoothness_diagnostics["gmr_action_delta_rms"].item(), 0.)


class WarmStartTest(unittest.TestCase):
    def test_missing_and_wrong_checkpoint_never_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_5000.pt"
            with self.assertRaises(FileNotFoundError):
                warm.locate_verified_checkpoint(path, [directory])
            torch.save({"bad": torch.zeros(1)}, path)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                warm.locate_verified_checkpoint(path, [directory])

    def test_strict_warm_start_keeps_all_weights_and_counts_completed_updates(self):
        source, target = network(), network()
        source.std.data.fill_(.066)
        optimizer = torch.optim.Adam(target.parameters(), lr=1e-5)
        estimator = torch.optim.Adam(target.state_estimator.parameters(), lr=1e-5)
        runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(actor_critic=target, optimizer=optimizer,
                                 state_estimator_optimizer=estimator), current_learning_iteration=0, it=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_5000.pt"
            torch.save({"model_state_dict": source.state_dict(), "iter": 4999}, path)
            with patch.object(warm, "SOURCE_SHA256", warm.sha256(path)):
                result = warm.warm_start_runner(runner, path)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(runner.current_learning_iteration, 5000)
        self.assertEqual(result["stored_iteration"], 4999)
        self.assertFalse(optimizer.state)
        self.assertFalse(estimator.state)
        obs, critic = torch.randn(2, 3102), torch.randn(2, 219)
        action, value = target.act_inference(obs), target.evaluate(critic)
        self.assertEqual(action.shape, (2, 12))
        (action.square().mean() + value.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in target.parameters() if p.grad is not None))

    def test_wrong_iteration_and_nonfinite_weights_rejected(self):
        net = network()
        runner = SimpleNamespace(device="cpu")
        for iteration, nonfinite in ((5000, False), (4999, True)):
            state = net.state_dict()
            if nonfinite:
                state["std"][0] = float("nan")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model_5000.pt"
                torch.save({"model_state_dict": state, "iter": iteration}, path)
                with patch.object(warm, "SOURCE_SHA256", warm.sha256(path)):
                    with self.assertRaises(ValueError):
                        warm.warm_start_runner(runner, path)


if __name__ == "__main__":
    unittest.main()
