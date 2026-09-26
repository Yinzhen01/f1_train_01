import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.sustain import (apply_sustain_config, sustain_contract, sustain_source,
    validate_sustain, validate_sustain_certificate, validate_sustain_diagnostics, warm_start_sustain)
from humanoid.amp.direction import apply_direction_config
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.verify_signal_formal import assert_equal
from tools.amp.sustain_audit import compare_environment, validate_sustain_updates
from test_f1_amp_direction import direction_report
from test_f1_amp_contact import reward_report
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def sustain_report(group):
    result = direction_report('heading')
    result['progress_scale'] = .01*sustain_contract(group)['progress_scale']
    return result


class SustainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_sustain_'+g+'.json'))
                           for g in ('control', 'progress3')}

    def test_contract_limits_source_budget_and_amp(self):
        for g, e in self.experiments.items():
            source = validate_sustain(e)
            self.assertEqual(e.cfg['reward'], source.cfg['reward'])
            self.assertEqual(e.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(e.cfg['smoke'], dict(num_envs=32, updates=10))
            self.assertEqual(e.cfg['evaluation_updates'], [2500])
            for mutate in (lambda c: c['formal'].update(updates=500),
                lambda c: c['sustain'].update(source_completed_updates=2260),
                lambda c: c['sustain'].update(progress_scale=4.),
                lambda c: c['reward'].update(style_weight=0), lambda c: c.update(evaluation_duration_s=20)):
                bad = copy.copy(e); bad.cfg = copy.deepcopy(e.cfg); mutate(bad.cfg)
                with self.assertRaises(ValueError): validate_sustain(bad)

    def test_only_progress_weight_changes_no_dr_and_no_class_leak(self):
        scope = config_scope(); path = ROOT/'humanoid/envs/x1/x1_amp_refine_config.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), 'exec'), scope)
        baseline = plain(apply_direction_config(scope['X1AMPRefineCfg'](), 'heading'))
        for g in self.experiments:
            cfg = apply_sustain_config(scope['X1AMPRefineCfg'](), g)
            assert_no_domain_randomization(cfg)
            self.assertTrue(compare_environment(baseline, plain(cfg), g))
        self.assertEqual(scope['X1AMPRefineCfg']().rewards.scales.recovery_progress, 2.)
        self.assertEqual(scope['X1AMPRefineCfg']().env.episode_length_s, 6.)
        self.assertFalse(hasattr(scope['X1AMPRefineCfg']().rewards, 'direction_world_fraction'))

    def test_actual_diagnostics_require_weight_and_unchanged_body_heading(self):
        for g in self.experiments:
            report = sustain_report(g); before = copy.deepcopy(report)
            self.assertTrue(validate_sustain_diagnostics(report, g, 240, 32))
            self.assertEqual(report, before)
            for field, value in (('calls', 239), ('heading_calls', 239), ('progress_scale', 2.),
                ('world_fraction', .15), ('heading_scale', -.005), ('selected_sum', 155.),
                ('body_sum', float('nan')), ('heading_cost_sum', 0.)):
                bad = dict(report); bad[field] = value
                with self.assertRaises(ValueError): validate_sustain_diagnostics(bad, g, 240, 32)

    def test_cloud_gate_needs_full_source_and_physics_reward_evidence(self):
        for g, e in self.experiments.items():
            cert = dict(identity=e.identity(), implementation_fingerprint='fixture', num_envs=32, updates=10,
                physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
                nonzero_style_reward=True, actor_updated=True, discriminator_updated=True,
                all_finite=True, complete=True, valid_windows=1, no_dr_configuration_verified=True,
                smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
                continuation=dict(**sustain_source(e.cfg), actor_and_discriminator_restored=True,
                    optimizers_restored=True, replay_windows=50000),
                horizon_diagnostics=dict(smoke_injected_resets=True, configured_episode_length_s=60.,
                    control_steps=240, captured_transitions=7680, forced_physical_reset_verified=True, forced_timeout_reset_verified=True),
                contact_diagnostics=reward_report('tail'), direction_diagnostics=sustain_report(g))
            self.assertTrue(validate_cloud_smoke(e, cert, 'fixture'))
            for field in ('contact_diagnostics', 'horizon_diagnostics', 'continuation', 'direction_diagnostics'):
                bad = copy.deepcopy(cert); bad.pop(field)
                with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'fixture')
            bad = copy.deepcopy(cert['continuation']); bad['source_completed_updates'] = 2260
            with self.assertRaises(ValueError): validate_sustain_certificate(e, bad)

    def test_environment_audit_rejects_other_changes(self):
        source = dict(env=dict(episode_length_s=60., num_envs=4096),
            rewards=dict(direction_world_fraction=0., scales=dict(recovery_progress=2., refine_heading=-1.5)),
            control=dict(decimation=10))
        for g in self.experiments:
            target = copy.deepcopy(source); target['rewards']['scales']['recovery_progress'] = sustain_contract(g)['progress_scale']
            self.assertTrue(compare_environment(source, target, g))
            smoke = copy.deepcopy(target); smoke['env']['num_envs'] = 32
            self.assertTrue(compare_environment(source, smoke, g, smoke=True))
            for mutate in (lambda c: c['control'].update(decimation=5),
                lambda c: c['rewards']['scales'].update(refine_heading=-.5),
                lambda c: c['env'].update(episode_length_s=20)):
                bad = copy.deepcopy(target); mutate(bad)
                with self.assertRaises(ValueError): compare_environment(source, bad, g)

    def test_exact_update_range_and_actual_amp_log(self):
        for formal, end in ((False, 2260), (True, 2500)):
            rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
                style_negative_fraction=0, discriminator_bridge_gradient_penalty=.1) for i in range(2251, end+1)]
            self.assertTrue(validate_sustain_updates(rows, formal))
            for bad in (rows[:-1], rows+[rows[-1]], [dict(r, iteration=r['iteration']-250) for r in rows],
                        [dict(r, weighted_style_reward=0) for r in rows]):
                with self.assertRaises(ValueError): validate_sustain_updates(bad, formal)

    def test_restore_every_real_heading2250_learning_tensor(self):
        checkpoint = ROOT.parent/'f1-amp-direction-refine/outputs/amp-direction/TASK_20260926_089/model_8802250.pt'
        if not checkpoint.is_file(): self.skipTest('Real source absent')
        cfg = config_scope()['X1AMPRecoveryCfgPPO']()
        actor_type, ppo_type = cpu_ppo_classes(ROOT)
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        for g, e in self.experiments.items():
            with self.subTest(group=g), contextlib.redirect_stdout(io.StringIO()):
                actor = actor_type(235, 47, 219, 12, **plain(cfg.policy))
                ppo = ppo_type(actor, device='cpu', **plain(cfg.algorithm))
                d = AMPDiscriminator(e.spec, e.mean, e.std)
                trainer = DiscriminatorTrainer(d, bridge_gradient_penalty=1.)
                bridge = AMPBridge(e.spec, trainer, 2, e.cfg['reward'], 'cpu', e.cfg['recovery'])
                alg = SimpleNamespace(actor_critic=actor, optimizer=ppo.optimizer,
                    state_estimator_optimizer=ppo.state_estimator_optimizer, ppo=ppo, bridge=bridge)
                runner = SimpleNamespace(device='cpu', alg=alg, amp_trainer=trainer)
                proof = warm_start_sustain(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 2250)
                self.assertEqual(runner.alg.updates, 2250)
                self.assertEqual(proof['replay_windows'], 50000)
                for actual, key in ((actor.state_dict(), 'model_state_dict'), (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (d.state_dict(), 'amp_discriminator_state_dict'), (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    assert_equal(actual, state[key])


if __name__ == '__main__': unittest.main()
