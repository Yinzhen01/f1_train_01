"""Single-variable sweep, diagnostics and strict model8000 source gates."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from humanoid import gmr_accel_finetune as warm
from test_gmr_swing import settings_scope
from test_gmr_smooth import plain, network

ROOT = Path(__file__).resolve().parents[1]


def accel_scope():
    scope = settings_scope()
    path = ROOT / "humanoid/envs/x1/x1_gmr_accel_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), "exec"), scope)
    return scope


class AccelConfigTest(unittest.TestCase):
    def test_every_environment_field_is_equal_except_dof_acc(self):
        scope = accel_scope()
        expected = plain(scope["X1GMRSwingCfg"]())
        for multiplier in (1, 3, 10):
            actual = plain(scope["X1GMRAccel%dCfg" % multiplier]())
            weight = actual["rewards"]["scales"]["dof_acc"]
            self.assertEqual(weight, warm.GROUP_WEIGHTS["x1_gmr_accel_%dx" % multiplier])
            actual["rewards"]["scales"]["dof_acc"] = -1e-7
            self.assertEqual(expected, actual)
        self.assertEqual(scope["X1GMRSwingCfg"]().rewards.scales.dof_acc, -1e-7)

    def test_all_ppo_fields_unchanged_except_experiment_name(self):
        scope = accel_scope()
        old, new = plain(scope["X1GMRSwingCfgPPO"]()), plain(scope["X1GMRAccelCfgPPO"]())
        new["runner"]["experiment_name"] = old["runner"]["experiment_name"]
        self.assertEqual(old, new)
        self.assertEqual(new["algorithm"]["learning_rate"], 1e-5)
        self.assertEqual(new["runner"]["max_iterations"], 1000)

    def test_registered_groups_and_source_gates(self):
        registration = (ROOT / "humanoid/envs/__init__.py").read_text()
        for task in warm.GROUP_WEIGHTS:
            self.assertIn('register("%s", X1GMRAccelEnv' % task, registration)
        entry = (ROOT / "humanoid/scripts/train_gmr_accel.py").read_text()
        self.assertIn('args.checkpoint != 8000', entry)
        self.assertIn('((512, 20), (4096, 1000))', entry)
        self.assertIn('commit != extra.expected_commit', entry)
        self.assertLess(entry.index('identity = warm_start_runner'), entry.index('runner.learn('))


class AccelDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        class Base:
            def compute_reward(self):
                self.original_reward_called = True
            def get_command_tracking_debug(self):
                return {"existing_metric": 1.}
        path = ROOT / "humanoid/envs/x1/x1_gmr_accel_env.py"
        definition = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        scope = dict(torch=torch, X1GMRSwingEnv=Base)
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), scope)
        cls = scope["X1GMRAccelEnv"]
        self.assertNotIn('_reward_dof_acc', cls.__dict__)
        self.assertNotIn('step', cls.__dict__)
        self.e = e = cls.__new__(cls)
        e.num_actions = 12
        e.dof_vel = torch.ones(3, 12)
        e.last_dof_vel = torch.zeros(3, 12)
        e.episode_length_buf = torch.tensor([1, 2, 3])
        e.smooth_ankle_ids = [4, 5, 10, 11]
        e.dt = .01
        e.reward_scales = {'dof_acc': -1e-7 * e.dt}

    def test_rms_units_and_unchanged_state(self):
        e = self.e
        before = e.dof_vel.clone(), e.last_dof_vel.clone()
        e.compute_reward()
        self.assertTrue(e.original_reward_called)
        self.assertAlmostEqual(e.accel_diagnostics['gmr_joint_acc_rms_rad_s2'].item(), 100.)
        self.assertAlmostEqual(e.accel_diagnostics['gmr_ankle_acc_rms_rad_s2'].item(), 100.)
        self.assertAlmostEqual(e.accel_diagnostics['gmr_acc_raw_cost'].item(), 120000.)
        self.assertAlmostEqual(e.accel_diagnostics['gmr_acc_weighted_cost_before_dt'].item(), -.012)
        torch.testing.assert_close(before[0], e.dof_vel)
        torch.testing.assert_close(before[1], e.last_dof_vel)
        self.assertIn('existing_metric', e.get_command_tracking_debug())

    def test_reset_only_diagnostics_are_finite_zero(self):
        self.e.episode_length_buf.zero_()
        self.e.compute_reward()
        for value in self.e.accel_diagnostics.values():
            self.assertTrue(torch.isfinite(value))
            self.assertEqual(value.item(), 0.)

    def test_constant_velocity_no_acceleration(self):
        self.e.last_dof_vel[:] = self.e.dof_vel
        self.e.compute_reward()
        self.assertEqual(self.e.accel_diagnostics['gmr_joint_acc_rms_rad_s2'].item(), 0.)


class AccelWarmStartTest(unittest.TestCase):
    def setUp(self):
        self.source, self.target = network(), network()
        self.runner = SimpleNamespace(device='cpu', alg=SimpleNamespace(actor_critic=self.target,
            optimizer=torch.optim.Adam(self.target.parameters(), lr=1e-5),
            state_estimator_optimizer=torch.optim.Adam(self.target.state_estimator.parameters(), lr=1e-5)))

    def test_all_33_tensors_restore_and_next_update_is_8000(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_8000.pt'
            torch.save(dict(model_state_dict=self.source.state_dict(), iter=7999), path)
            with patch.object(warm, 'SOURCE_SHA256', warm.sha256(path)):
                result = warm.warm_start_runner(self.runner, path)
        self.assertEqual(result['loaded_tensors'], 33)
        self.assertEqual(self.runner.current_learning_iteration, 8000)
        self.assertEqual(result['source_task'], 'TASK_20260911_171')
        for name, value in self.source.state_dict().items():
            torch.testing.assert_close(self.target.state_dict()[name], value, rtol=0, atol=0)

    def test_wrong_iteration_and_nonfinite_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_8000.pt'
            torch.save(dict(model_state_dict=self.source.state_dict(), iter=6999), path)
            with patch.object(warm, 'SOURCE_SHA256', warm.sha256(path)):
                with self.assertRaisesRegex(ValueError, 'iteration'):
                    warm.warm_start_runner(self.runner, path)
            state = self.source.state_dict()
            state['std'][0] = float('nan')
            torch.save(dict(model_state_dict=state, iter=7999), path)
            with patch.object(warm, 'SOURCE_SHA256', warm.sha256(path)):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    warm.warm_start_runner(self.runner, path)

    def test_missing_wrong_hash_and_existing_optimizer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model_8000.pt'
            with self.assertRaises(FileNotFoundError):
                warm.locate_verified_checkpoint(path, [directory])
            torch.save(dict(model_state_dict=self.source.state_dict(), iter=7999), path)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                warm.locate_verified_checkpoint(path, [directory])
            self.runner.alg.optimizer.state['nonempty'] = {}
            with patch.object(warm, 'SOURCE_SHA256', warm.sha256(path)):
                with self.assertRaisesRegex(ValueError, 'fresh optimizer'):
                    warm.warm_start_runner(self.runner, path)


if __name__ == '__main__':
    unittest.main()
