"""Inherit the full tensor restore, cloud gate, budget and no-DR regressions."""
import ast
import copy
from types import SimpleNamespace
import torch

import test_f1_amp_jitter as jitter_tests
from test_f1_amp_jitter import ROOT, jitter_report
from humanoid.amp.jitter import SUBSTEP_GROUPS, jitter_contract, validate_jitter_diagnostics
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.refinement import robust_acceleration_cost


class SubstepTests(jitter_tests.JitterTests):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_jitter_'+g+'.json'))
                           for g in SUBSTEP_GROUPS}

    def test_exact_substep_calls_weights_cost_selection(self):
        for g in self.experiments:
            report = jitter_report(g)
            for key, val in (('substep_acceleration_calls', 239), ('substep_torque_calls', 239),
                ('acceleration_scale', -.5), ('torque_scale', -.2), ('used_acceleration_cost_sum', 1.),
                ('substep_valid_intervals', 0), ('substep_acceleration_cost_sum', float('nan'))):
                bad = copy.deepcopy(report); bad[key] = val
                with self.assertRaises(ValueError): validate_jitter_diagnostics(bad, g, 240, 32)

    def test_cost_before_history_advance_cancellation_and_reset(self):
        class Parent:
            def _begin_physics_step(self): pass
            def _after_physics_substep(self, substep): pass
            def reset_direction_diagnostics(self): pass
        scope = dict(torch=torch, X1AMPDirectionEnv=Parent, robust_acceleration_cost=robust_acceleration_cost)
        for file in ('x1_amp_jitter_env.py', 'x1_amp_substep_env.py'):
            path = ROOT/'humanoid/envs/x1'/file
            tree = ast.parse(path.read_text())
            tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
            exec(compile(tree, str(path), 'exec'), scope)
        for g in SUBSTEP_GROUPS:
            env = scope['X1AMPSubstepEnv']()
            env.num_envs = 2; env.device = 'cpu'; env.dt = .01
            env.dof_vel = torch.zeros(2, 12); env.torques = torch.zeros(2, 12)
            env.last_dof_vel = torch.zeros(2, 12); env.episode_length_buf = torch.tensor([10, 0])
            env.cfg = SimpleNamespace(control=SimpleNamespace(decimation=10), substep_penalty=jitter_contract(g)['substep'])
            env.sim_params = SimpleNamespace(dt=.001); env.sim = None
            env.feet_indices = [0, 1]; env.contact_forces = torch.zeros(2, 2, 3)
            env.gym = SimpleNamespace(refresh_net_contact_force_tensor=lambda sim: None)
            env._begin_physics_step()
            for k in range(10):
                env.dof_vel[:] = 1. if k % 2 == 0 else 0.
                env.torques[:] = 2. if k % 2 == 0 else 0.
                before = env.dof_vel.clone(), env.torques.clone()
                env._after_physics_substep(k)
                for actual, expected in zip((env.dof_vel, env.torques), before):
                    self.assertTrue(torch.equal(actual, expected))
            physical = 2*((1+10**2)**.5-1)
            self.assertAlmostEqual(env.substep_acceleration_cost[0].item(), physical, places=4)
            used = env._reward_refine_acceleration()
            self.assertAlmostEqual(used[0].item(), physical if g == 'substep_accel' else 0., places=4)
            self.assertEqual(used[1].item(), 0.)
            torque = env._reward_substep_torque()
            self.assertAlmostEqual(torque[0].item(), 2*(2**.5-1), places=5)
            self.assertEqual(torque[1].item(), 0.)
            self.assertEqual(env.substep_sums[4].item(), 1.)
            env._begin_physics_step()
            self.assertEqual(env.substep_acceleration_cost.sum().item(), 0.)
            self.assertEqual(env.substep_torque_cost.sum().item(), 0.)

    def test_robust_of_mean_square_is_not_used(self):
        # Nonlinear cost must be applied to each substep before time averaging.
        sample = torch.zeros(10, 1, 12); sample[0] = 1000.
        actual = robust_acceleration_cost(sample).mean(0)
        wrong = robust_acceleration_cost(sample.square().mean(0).sqrt())
        self.assertGreater((wrong-actual).item(), 1.)
