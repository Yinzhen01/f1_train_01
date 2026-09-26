"""Offline reward reconstruction against actual environment method bodies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from humanoid.amp.features import quat_rotate, quat_conj
from humanoid.amp.recovery import centered_velocity_reward
from humanoid.amp.refinement import robust_acceleration_cost, ground_force_weight
from humanoid.amp.discriminator import AMPDiscriminator
from tools.amp.audit_reward_balance import task_terms, summarize_window, differences, style_terms

ROOT = Path(__file__).resolve().parents[1]


def method(file, name):
    tree = ast.parse((ROOT/file).read_text(encoding='utf-8'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1: raise AssertionError('Method not unique')
    module = ast.Module(body=nodes, type_ignores=[])
    scope = dict(torch=torch, centered_velocity_reward=centered_velocity_reward,
                 robust_acceleration_cost=robust_acceleration_cost, ground_force_weight=ground_force_weight,
                 quat_rotate=quat_rotate, quat_conj=quat_conj)
    exec(compile(ast.fix_missing_locations(module), str(file), 'exec'), scope)
    return scope[name]


class RewardBalanceTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(7); n = 320
        data = {name: rng.normal(size=(n, 12)).astype(np.float32)
                for name in ('dof_pos', 'dof_vel', 'action', 'torque')}
        data['time'] = np.arange(1, n+1)*.01
        data['initial'] = {k: np.zeros(12, dtype=np.float32) for k in data if k != 'time'}
        data['root_state'] = np.zeros((n, 13), dtype=np.float32)
        data['root_state'][:, 3:7] = Rotation.from_euler('xyz', rng.normal(size=(n, 3))*.1).as_quat()
        for k in ('base_lin_vel', 'base_ang_vel'): data[k] = rng.normal(size=(n, 3)).astype(np.float32)*.2
        data['foot_state'] = rng.normal(size=(n, 2, 13)).astype(np.float32)*.2
        data['foot_state'][:, :, 3:7] = Rotation.from_euler('xyz', rng.normal(size=(n*2, 3))*.2).as_quat().reshape(n, 2, 4)
        data['foot_force'] = rng.normal(size=(n, 2, 3)).astype(np.float32)*600
        data['failure'] = np.zeros(n, dtype=bool); data['failure'][-1] = True
        vertices = {name: np.array([[-.1, -.04, -.02], [.1, -.04, -.02], [.1, .04, -.02],
                                    [-.1, .04, -.02], [0., 0., .05]]) for name in ('left', 'right')}
        rewards = dict(max_contact_force=700., soft_dof_pos_limit=.9, soft_dof_vel_limit=.95, soft_torque_limit=.85)
        rewards['scales'] = dict(recovery_progress=2., refine_heading=-1.5, recovery_tilt=-.5,
            recovery_yaw_rate=-1., recovery_vertical_velocity=-.1, recovery_smoothness=-.02,
            refine_acceleration=-.5, refine_slip=-2., feet_contact_forces=-.01, contact_force_tail=-.0025,
            dof_acc=-1e-7, dof_vel=-2e-8, torques=-8e-9, dof_pos_limits=-10., dof_vel_limits=-1.,
            dof_torque_limits=-.1, termination=-50., collision=-1.)
        props = dict(lower=[-.5]*12, upper=[1.]*12, velocity=[1.5]*12, effort=[2.]*12)
        manifest = dict(environment=dict(rewards=rewards, safety=dict(pos_limit=.95, vel_limit=.8, torque_limit=.9)),
                        runtime=dict(dof_properties=props), foot_names=['left', 'right'])
        return data, manifest, vertices

    def test_reconstructed_components_match_actual_reward_methods(self):
        data, manifest, vertices = self.fixture()
        terms, missing = task_terms(data, manifest, vertices)
        self.assertEqual(missing, {'collision': -1.})
        n = len(data['time']); tensor = lambda a: torch.tensor(a, dtype=torch.float32)
        env = SimpleNamespace(dt=.01, num_envs=n, device='cpu', episode_length_buf=torch.arange(1, n+1))
        for key in ('dof_pos', 'dof_vel', 'base_lin_vel', 'base_ang_vel'): setattr(env, key, tensor(data[key]))
        env.actions = tensor(data['action']); env.last_actions = env.actions-tensor(differences(data, 'action'))
        env.last_last_actions = torch.cat((env.last_actions[:1], env.last_actions[:-1]))
        env.last_dof_vel = env.dof_vel-tensor(differences(data, 'dof_vel'))
        env.torques = tensor(data['torque']); env.contact_forces = tensor(data['foot_force']); env.feet_indices = [0, 1]
        env.rigid_state = tensor(data['foot_state']); env.env_origins = torch.zeros(n, 3)
        env.base_euler_xyz = tensor(Rotation.from_quat(data['root_state'][:, 3:7]).as_euler('xyz'))
        env.projected_gravity = -tensor(Rotation.from_quat(data['root_state'][:, 3:7]).as_matrix()[:, 2, :])
        env.commands = torch.tensor([.45, 0., 0.]).expand(n, 3)
        env.refine_sole_local = tensor(np.stack([v[v[:, 2] == v[:, 2].min()].mean(0) for v in vertices.values()]))
        env.refine_collision_hulls = [tensor(v) for v in vertices.values()]
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(**manifest['environment']['rewards']))
        lo, hi = -.5*.95, 1.*.95
        env.dof_pos_limits = torch.tensor([(lo+hi)/2-(hi-lo)*.9/2, (lo+hi)/2+(hi-lo)*.9/2]).expand(12, 2)
        env.dof_vel_limits = torch.full((12,), 1.5*.8); env.torque_limits = torch.full((12,), 2.*.9)
        files = dict(recovery='humanoid/envs/x1/x1_amp_recovery_env.py',
                     refine='humanoid/envs/x1/x1_amp_refine_env.py',
                     base='humanoid/envs/x1/x1_dh_stand_env.py')
        for key in terms:
            if key in ('contact_force_tail', 'termination'): continue
            family = 'recovery' if key.startswith('recovery_') else 'refine' if key.startswith('refine_') else 'base'
            with self.subTest(reward=key):
                actual = method(files[family], '_reward_'+key)(env).numpy()
                actual *= manifest['environment']['rewards']['scales'][key]*.01
                np.testing.assert_allclose(terms[key][2:], actual[2:], rtol=2e-5, atol=2e-7)
        np.testing.assert_array_equal(terms['recovery_smoothness'][:2], 0.)
        np.testing.assert_array_equal(terms['refine_acceleration'][:2], 0.)
        self.assertAlmostEqual(terms['termination'][-1], -.5)

    def test_window_uses_full_half_open_interval_and_no_filled_history(self):
        data, _, _ = self.fixture(); terms = {'one': np.ones(len(data['time']))}
        result = summarize_window(data, terms, 1., 3.)
        self.assertEqual(result['samples'], 200)
        self.assertEqual(result['rewards']['partial_total_no_collision'], 1.)
        self.assertIsNone(summarize_window(data, terms, 2., 4.))
        terms['one'][110] = np.nan
        with self.assertRaises(ValueError): summarize_window(data, terms, 1., 3.)

    def test_reject_unavailable_rewards_and_world_frame(self):
        data, manifest, vertices = self.fixture()
        manifest['environment']['rewards']['direction_world_fraction'] = .15
        with self.assertRaises(ValueError): task_terms(data, manifest, vertices)
        manifest['environment']['rewards']['direction_world_fraction'] = 0.
        manifest['environment']['rewards']['scales']['unknown'] = 1.
        with self.assertRaises(ValueError): task_terms(data, manifest, vertices)

    def test_style_matches_training_formula_and_excludes_nine_missing_ticks(self):
        features = torch.linspace(-.2, .2, 50*39).reshape(50, 39)
        spec = SimpleNamespace(dim=39, history=10, fps=100)
        discriminator = AMPDiscriminator(spec, torch.zeros(39), torch.ones(39))
        for p in discriminator.parameters(): p.data.fill_(.01)
        experiment = SimpleNamespace(cfg={'reward': dict(style_scale=5., style_weight=1.)})
        data = dict(time=np.arange(1, 51)*.01)
        with patch('tools.amp.audit_reward_balance.features_from_episode', return_value=features):
            actual = style_terms(data, experiment, discriminator)
        windows = features.unfold(0, 10, 1).permute(0, 2, 1).contiguous()
        expected = discriminator.style_reward(windows, scale=5., dt=.01).numpy()
        self.assertTrue(np.isnan(actual[:9]).all())
        np.testing.assert_allclose(actual[9:], expected, rtol=1e-6, atol=1e-9)
        self.assertGreater(expected.max(), 0.)


if __name__ == '__main__': unittest.main()
