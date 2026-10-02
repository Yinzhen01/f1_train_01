"""Read-only real-class capture tests without requiring the Isaac Gym package."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def environment_classes():
    class Parent:
        def _begin_physics_step(self): pass
        def _after_physics_substep(self, substep): pass
        def reset_direction_diagnostics(self): pass
    scope = dict(torch=torch, X1AMPDirectionEnv=Parent)
    for filename in ('x1_amp_jitter_env.py', 'x1_amp_physics_diagnostic_env.py'):
        path = ROOT/'humanoid/envs/x1'/filename
        tree = ast.parse(path.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        exec(compile(tree, str(path), 'exec'), scope)
    return scope['X1AMPJitterEnv'], scope['X1AMPPhysicsDiagnosticEnv']


class FakeGym:
    def __init__(self):
        self.force_refreshes = 0
        self.body_refreshes = 0
        self.resolved_names = []
        self.body_ids = dict(base_link=2, left_ankle_roll_link=4,
                             right_ankle_roll_link=7)

    def find_actor_rigid_body_handle(self, env, actor, name):
        self.resolved_names.append(name)
        return self.body_ids.get(name, -1)

    def refresh_net_contact_force_tensor(self, sim):
        self.force_refreshes += 1

    def refresh_rigid_body_state_tensor(self, sim):
        # The parent must first refresh contact force in this same substep.
        if self.force_refreshes != self.body_refreshes+1:
            raise AssertionError('Raw body capture preceded parent force capture')
        self.body_refreshes += 1


def make_environment(cls):
    env = cls()
    env.num_envs = 2
    env.device = 'cpu'
    env.sim = object()
    env.sim_params = SimpleNamespace(dt=.001)
    env.cfg = SimpleNamespace(control=SimpleNamespace(decimation=10))
    env.dof_vel = torch.zeros(2, 12)
    env.torques = torch.zeros(2, 12)
    env.contact_forces = torch.zeros(2, 8, 3)
    env.rigid_state = torch.zeros(2, 8, 13)
    env.env_origins = torch.tensor([[10., 20., 30.], [-10., -20., -30.]])
    env.feet_indices = [4, 7]
    env.episode_length_buf = torch.tensor([10, 0])
    env.envs = ['env0', 'env1']
    env.actor_handles = ['actor0', 'actor1']
    env.gym = FakeGym()
    return env


def set_substep_state(env, step):
    env.dof_vel[:] = torch.arange(12)+step
    env.torques[:] = torch.arange(12)*2-step
    env.contact_forces[:] = torch.arange(48).reshape(2, 8, 3)+step
    env.rigid_state[:] = torch.arange(208).reshape(2, 8, 13)+step
    env.rigid_state[0, :, :3] += env.env_origins[0]


class PhysicsDiagnosticEnvironmentTests(unittest.TestCase):
    def test_ten_ordered_substeps_body_frames_clones_and_parent_parity(self):
        parent_cls, raw_cls = environment_classes()
        baseline, diagnostic = make_environment(parent_cls), make_environment(raw_cls)
        for env in (baseline, diagnostic):
            env.dof_vel[:] = 3.
            env.torques[:] = -4.
            env._begin_physics_step()
        expected_states = []
        for step in range(10):
            for env in (baseline, diagnostic):
                set_substep_state(env, step)
                before = [value.clone() for value in (env.dof_vel, env.torques,
                                                      env.contact_forces, env.rigid_state)]
                env._after_physics_substep(step)
                for actual, expected in zip((env.dof_vel, env.torques,
                                             env.contact_forces, env.rigid_state), before):
                    torch.testing.assert_close(actual, expected)
            expected = diagnostic.rigid_state[0, [2, 4, 7]].clone()
            expected[:, :3] -= diagnostic.env_origins[0]
            expected_states.append(expected)
        base, raw = baseline.jitter_capture(), diagnostic.jitter_capture()
        for name, expected in base.items():
            torch.testing.assert_close(raw[name], expected)
        torch.testing.assert_close(diagnostic.jitter_sums, baseline.jitter_sums)
        self.assertEqual(diagnostic.jitter_physics_steps, 10)
        self.assertEqual(diagnostic.gym.body_refreshes, 10)
        self.assertEqual(diagnostic.gym.resolved_names,
                         ['base_link', 'left_ankle_roll_link', 'right_ankle_roll_link'])
        self.assertEqual(tuple(raw['physics_raw_velocity'].shape), (10, 12))
        self.assertEqual(tuple(raw['physics_raw_body_force'].shape), (10, 3, 3))
        self.assertEqual(tuple(raw['physics_raw_body_state'].shape), (10, 3, 13))
        torch.testing.assert_close(raw['physics_raw_body_state'], torch.stack(expected_states))
        torch.testing.assert_close(raw['physics_interval_initial_velocity'], torch.full((12,), 3.))
        torch.testing.assert_close(raw['physics_interval_initial_torque'], torch.full((12,), -4.))
        torch.testing.assert_close(raw['physics_raw_velocity'][0], torch.arange(12).float())
        torch.testing.assert_close(raw['physics_raw_velocity'][-1], torch.arange(12).float()+9)
        frozen = {key: value.clone() for key, value in raw.items() if 'raw_' in key or 'interval_initial_' in key}
        diagnostic.dof_vel.fill_(999.)
        diagnostic.torques.fill_(999.)
        diagnostic.contact_forces.fill_(999.)
        diagnostic.rigid_state.fill_(999.)
        for name, expected in frozen.items():
            torch.testing.assert_close(diagnostic.jitter_capture()[name], expected)

    def test_new_interval_clears_samples_and_retains_new_initial_values(self):
        _, raw_cls = environment_classes()
        env = make_environment(raw_cls)
        env._begin_physics_step()
        for step in range(10):
            set_substep_state(env, step)
            env._after_physics_substep(step)
        old = env.jitter_capture()
        env.dof_vel.fill_(21.)
        env.torques.fill_(34.)
        env._begin_physics_step()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            env.jitter_capture()
        for step in range(10):
            set_substep_state(env, step+100)
            env._after_physics_substep(step)
        new = env.jitter_capture()
        self.assertEqual(tuple(new['physics_raw_velocity'].shape), (10, 12))
        torch.testing.assert_close(new['physics_interval_initial_velocity'], torch.full((12,), 21.))
        torch.testing.assert_close(new['physics_interval_initial_torque'], torch.full((12,), 34.))
        self.assertEqual(new['physics_raw_velocity'][0, 0].item(), 100.)
        self.assertEqual(old['physics_raw_velocity'][0, 0].item(), 0.)
        self.assertEqual(len(env.gym.resolved_names), 3)

    def test_missing_body_and_out_of_order_capture_fail_closed(self):
        _, raw_cls = environment_classes()
        env = make_environment(raw_cls)
        env.gym.body_ids.pop('base_link')
        with self.assertRaisesRegex(ValueError, 'collision bodies'):
            env._begin_physics_step()
        env = make_environment(raw_cls)
        env._begin_physics_step()
        with self.assertRaisesRegex(ValueError, 'once and in order'):
            env._after_physics_substep(1)


if __name__ == '__main__':
    unittest.main()
