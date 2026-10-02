"""Check exact ankle-only cost, source continuation and reward-call gate."""
import ast
import copy
from types import SimpleNamespace
import unittest

import torch

from test_f1_amp_jitter import ROOT, config_type, jitter_report, plain
from humanoid.amp.jitter import (apply_jitter_config, jitter_contract,
                                  validate_jitter, validate_jitter_diagnostics)
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.learnability import assert_no_domain_randomization
from tools.amp.jitter_audit import compare_environment
from humanoid.amp.sustain import apply_sustain_config


class AnkleSlewTests(unittest.TestCase):
    def test_contract_and_only_ankle_reward_change(self):
        experiment = ScaledExperiment(ROOT, ROOT/'configs/amp/lafan_walk02_jitter_ankle_slew.json')
        self.assertEqual(experiment.cfg['jitter'], jitter_contract('ankle_slew'))
        source = validate_jitter(experiment)
        for key in ('motion_sha256', 'urdf_lf_sha256', 'feature_fingerprint'):
            self.assertEqual(source.identity()[key], experiment.identity()[key])
        source_cfg = plain(apply_sustain_config(config_type()(), 'control'))
        target_cfg = apply_jitter_config(config_type()(), 'ankle_slew')
        assert_no_domain_randomization(target_cfg)
        self.assertTrue(compare_environment(source_cfg, plain(target_cfg), 'ankle_slew'))
        bad = copy.deepcopy(plain(target_cfg))
        bad['control']['decimation'] = 5
        with self.assertRaises(ValueError):
            compare_environment(source_cfg, bad, 'ankle_slew')

    def test_reward_uses_only_ankle_pitch_and_masks_reset_history(self):
        class Parent:
            def reset_direction_diagnostics(self): pass
        path = ROOT/'humanoid/envs/x1/x1_amp_jitter_env.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        scope = dict(torch=torch, X1AMPDirectionEnv=Parent)
        exec(compile(tree, str(path), 'exec'), scope)
        env = scope['X1AMPJitterEnv']()
        env.device = 'cpu'
        env.cfg = SimpleNamespace(ankle_slew=jitter_contract('ankle_slew')['ankle_slew'])
        env.dof_names = ['joint_%d' % i for i in range(12)]
        env.dof_names[4] = 'left_ankle_pitch_joint'
        env.dof_names[10] = 'right_ankle_pitch_joint'
        env.actions = torch.zeros(2, 12)
        env.actions[0, 4] = 1.
        env.actions[0, 10] = -2.
        env.actions[0, 0] = 100.  # Hip action must not change the cost.
        env.actions[1, 4] = 3.
        env.last_actions = torch.zeros(2, 12)
        env.last_last_actions = torch.zeros(2, 12)
        env.episode_length_buf = torch.tensor([3, 2])
        env.reset_direction_diagnostics()
        result = env._reward_ankle_slew()
        torch.testing.assert_close(result, torch.tensor([10., 0.]))
        self.assertEqual(env.ankle_slew_calls, 1)
        self.assertEqual(env.ankle_slew_cost_sum.item(), 10.)
        env.last_actions[0, 4] = 1.
        env.last_actions[0, 10] = -1.
        result = env._reward_ankle_slew()
        torch.testing.assert_close(result, torch.tensor([2., 0.]))

    def test_diagnostics_reject_missing_or_wrong_reward(self):
        report = jitter_report('ankle_slew')
        self.assertTrue(validate_jitter_diagnostics(report, 'ankle_slew', 240, 32))
        for key, bad_value in (('ankle_slew_calls', 239), ('ankle_slew_cost_sum', 0.),
                               ('ankle_slew_scale', -.0002), ('ankle_slew', {})):
            bad = dict(report); bad[key] = bad_value
            with self.assertRaises(ValueError):
                validate_jitter_diagnostics(bad, 'ankle_slew', 240, 32)


if __name__ == '__main__': unittest.main()
