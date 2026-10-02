"""Controller recurrence plus inherited restoration, no-DR and cloud gates."""
import ast
import copy
from types import SimpleNamespace
import numpy as np
import torch

import test_f1_amp_jitter as jitter_tests
from test_f1_amp_jitter import ROOT, jitter_report
from humanoid.amp.jitter import FILTER_GROUPS, jitter_contract, validate_jitter_diagnostics
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.target_filter import filter_alpha, filter_target
from tools.amp.jitter_audit import validate_filter_arrays


class TargetFilterTests(jitter_tests.JitterTests):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_jitter_'+g+'.json'))
                           for g in FILTER_GROUPS}

    def test_invalid_filter_and_steady_state(self):
        for cutoff, dt in ((5, 0), (5, -1), (50, .01), (0, .01), (float('nan'), .01), (5, float('inf'))):
            with self.assertRaises(ValueError): filter_alpha(cutoff, dt)
        for g in FILTER_GROUPS:
            alpha = filter_alpha(jitter_contract(g)['target_filter']['cutoff_hz'], .01)
            result = np.zeros(12)
            for _ in range(200): result = filter_target(np.ones(12), result, alpha)
            np.testing.assert_allclose(result, 1.)
            # High-frequency attenuation, but walking fundamental is retained.
            gain = lambda f: alpha/abs(1-(1-alpha)*np.exp(-2j*np.pi*f*.01))
            self.assertGreater(gain(.62), .99)
            self.assertLess(gain(10), .65)

    def test_prefailure_diagnostic_excludes_terminal_spike_and_short_prefix(self):
        from tools.amp.audit_filter_prefixes import metrics
        signal = np.zeros((320, 12)); signal[310:, :] = 100.
        d = dict(time=np.arange(1, 321)*.01, action=signal, torque=signal, dof_vel=signal,
                 base_lin_vel=np.zeros((320, 3)), initial={k:np.zeros(12) for k in ('action', 'torque', 'dof_vel')})
        self.assertIsNone(metrics(d, 2.5))
        self.assertEqual(metrics(d, 3.)['samples'], 101)
        self.assertEqual(metrics(d, 3.)['acceleration_rms'], 0.)
        self.assertGreater(metrics(d, 3.2)['acceleration_rms'], 0.)

    def test_filter_diagnostics_fail_closed(self):
        for g in FILTER_GROUPS:
            for key, value in (('filter_calls', 239), ('filter_reset_count', 0), ('filter_alpha', .5),
                ('filter_valid_intervals', 9000), ('filter_input_delta_squared_sum', float('nan')),
                ('filter_applied_delta_squared_sum', 0.)):
                report = jitter_report(g); report[key] = value
                with self.assertRaises(ValueError): validate_jitter_diagnostics(report, g, 240, 32)

    def test_filter_precedes_parent_pd_without_mutating_ppo_action_and_resets(self):
        class Parent:
            def reset_direction_diagnostics(self): pass
            def _begin_physics_step(self): self.parent_seen_action = self.actions.clone()
            def reset_idx(self, ids): self.actions[ids] = 0.
            def jitter_capture(self): return {}
        path = ROOT/'humanoid/envs/x1/x1_amp_target_filter_env.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        scope = dict(torch=torch, X1AMPJitterEnv=Parent, filter_alpha=filter_alpha, filter_target=filter_target)
        exec(compile(tree, str(path), 'exec'), scope)
        env = scope['X1AMPTargetFilterEnv'](); env.device = 'cpu'; env.dt = .01
        env.cfg = SimpleNamespace(target_filter=jitter_contract('target5')['target_filter'])
        env.actions = torch.zeros(2, 12); env.episode_length_buf = torch.tensor([10, 0])
        env.reset_idx(torch.tensor([0, 1]))  # Base constructor can reset before buffers exist.
        raw = torch.ones(2, 12); env.actions = raw
        env._begin_physics_step(); alpha = filter_alpha(5., .01)
        torch.testing.assert_close(raw, torch.ones_like(raw))
        torch.testing.assert_close(env.actions, torch.full_like(raw, alpha))
        torch.testing.assert_close(env.parent_seen_action, env.actions)
        env.actions = raw; env._begin_physics_step()
        torch.testing.assert_close(env.actions, torch.full_like(raw, alpha*(2-alpha)))
        self.assertEqual(env.filter_sums[3].item(), 2.)  # Reset guard excludes env1.
        env.reset_idx(torch.tensor([0]))
        for name in ('target_filter_state', 'target_previous_raw', 'target_raw', 'target_filter_before'):
            self.assertEqual(getattr(env, name)[0].abs().sum().item(), 0.)
        env.actions = raw; env._begin_physics_step()
        torch.testing.assert_close(env.actions[0], torch.full((12,), alpha))
        self.assertEqual(env.filter_reset_count, 1)

    def test_recorded_filter_recurrence_raw_and_continuity_audit(self):
        for g in FILTER_GROUPS:
            spec = jitter_contract(g)['target_filter']; alpha = filter_alpha(spec['cutoff_hz'], .01)
            raw = np.ones((10, 2, 12), dtype=np.float32); raw[5:] = -1.
            action = np.zeros_like(raw); before = np.zeros_like(raw)
            for t in range(10):
                if t: before[t] = action[t-1]
                action[t] = filter_target(raw[t], before[t], alpha)
            arrays = dict(standing_action=action, standing_control_raw_action=raw,
                standing_control_filter_before=before, standing_valid=np.ones((10, 2), dtype=bool),
                standing_initial_action=np.zeros((2, 12), dtype=np.float32))
            manifest = dict(target_filter=spec, modes=['standing'], environment=dict(normalization=dict(clip_actions=18.)))
            self.assertTrue(validate_filter_arrays(manifest, arrays, g))
            for field in ('standing_action', 'standing_control_raw_action', 'standing_control_filter_before'):
                bad = copy.deepcopy(arrays); bad[field][3, 0, 2] += .1
                with self.assertRaises(ValueError): validate_filter_arrays(manifest, bad, g)
