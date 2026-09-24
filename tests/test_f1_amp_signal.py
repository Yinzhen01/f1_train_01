import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.signal import signal_contract, validate_signal, signal_source, warm_start_signal
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.verify_signal_smoke import validate_signal_updates
from test_f1_amp_recovery import config_scope, plain
import test_f1_amp_interrupted as interrupted_tests

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_signal_'+g+'.json'))
                           for g in ('signed', 'bridge')}

    def test_isolated_changes_and_fixed_budget(self):
        for group, e in self.experiments.items():
            source = validate_signal(e)
            self.assertEqual(e.cfg['signal'], signal_contract(group))
            self.assertEqual(e.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(e.cfg['reward'], source.cfg['reward'])
            self.assertEqual(e.identity()['motion_sha256'], source.identity()['motion_sha256'])
            self.assertEqual(e.identity()['feature_fingerprint'], source.identity()['feature_fingerprint'])
            for mutate in (lambda c: c['formal'].update(updates=500),
                           lambda c: c['reward'].update(style_weight=2),
                           lambda c: c['signal'].update(source_completed_updates=1500)):
                bad = copy.copy(e); bad.cfg = copy.deepcopy(e.cfg); mutate(bad.cfg)
                with self.assertRaises(ValueError): validate_signal(bad)

    def test_signed_reward_preserves_positive_part_and_bounds_negative(self):
        e = self.experiments['signed']
        d = AMPDiscriminator(e.spec, e.mean, e.std, [8], style_floor=-1)
        old = AMPDiscriminator(e.spec, e.mean, e.std, [8])
        for p in d.parameters(): p.data.zero_()
        x = e.windows[:4]
        for score in (-10., -2., -1.2, -1., -.8, 0., 1., 2., 10.):
            with torch.no_grad(): d.network[-1].bias.fill_(score)
            old.load_state_dict(d.state_dict())
            reward = d.style_reward(x, scale=5)
            expected = .05*max(-1., 1-.25*(score-1)**2)
            torch.testing.assert_close(reward, torch.full((4,), expected), atol=1e-7, rtol=0)
            self.assertFalse(reward.requires_grad)
            self.assertTrue(bool((reward >= -.05).all() and (reward <= .05).all()))
            if 1-.25*(score-1)**2 >= 0:
                torch.testing.assert_close(reward, old.style_reward(x, scale=5), atol=0, rtol=0)

    def test_bridge_regularizer_is_detached_finite_and_rng_neutral(self):
        e = self.experiments['bridge']
        d = AMPDiscriminator(e.spec, e.mean, e.std, [8])
        demo = e.windows[:8].clone().requires_grad_(True)
        policy = (demo.detach()+.01*torch.randn_like(demo)).requires_grad_(True)
        before = torch.get_rng_state().clone()
        base = d.losses(demo, policy)
        zero = d.losses(demo, policy, bridge_gradient_penalty=0)
        for key in base: torch.testing.assert_close(base[key], zero[key], atol=0, rtol=0)
        losses = d.losses(demo, policy, bridge_gradient_penalty=1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertGreater(float(losses['bridge_gradient_penalty']), 0)
        torch.testing.assert_close(losses['total'], base['total']+losses['bridge_gradient_penalty'])
        losses['total'].backward()
        self.assertIsNone(demo.grad); self.assertIsNone(policy.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in d.parameters()))

    def test_old_or_other_intervention_certificate_rejected(self):
        e = self.experiments['signed']
        cert = dict(identity=e.identity(), implementation_fingerprint='test', num_envs=32, updates=10,
                    physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
                    nonzero_style_reward=True, actor_updated=True, discriminator_updated=True,
                    all_finite=True, complete=True, valid_windows=1, no_dr_configuration_verified=True,
                    smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
                    continuation=dict(**signal_source(e.cfg), actor_and_discriminator_restored=True,
                                      optimizers_restored=True, replay_windows=50000))
        self.assertTrue(validate_cloud_smoke(e, cert, 'test'))
        for field, value in (('kind', 'interrupted_matched_run'), ('style_floor', 0), ('source_task', 'TASK_20260924_084')):
            bad = copy.deepcopy(cert); bad['continuation'][field] = value
            with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'test')

    def test_smoke_requires_actual_intervention_and_ten_finite_updates(self):
        rows = [dict(iteration=i, style_reward=-.001, weighted_style_reward=-.001,
                     style_negative_fraction=.1, style_zero_fraction=0.) for i in range(1251, 1261)]
        self.assertTrue(validate_signal_updates(rows, 'signed'))
        for bad in (rows[:-1], rows+[rows[-1]], [dict(row, style_negative_fraction=0.) for row in rows],
                    [dict(row, style_reward=float('nan')) for row in rows],
                    [dict(row, discriminator_bridge_gradient_penalty=.1) for row in rows]):
            with self.assertRaises(ValueError): validate_signal_updates(bad, 'signed')
        bridge = [dict(row, style_reward=.01, weighted_style_reward=.01,
                       style_negative_fraction=0., discriminator_bridge_gradient_penalty=.1) for row in rows]
        self.assertTrue(validate_signal_updates(bridge, 'bridge'))
        with self.assertRaises(ValueError):
            validate_signal_updates([dict(row, discriminator_bridge_gradient_penalty=0) for row in bridge], 'bridge')

    def test_both_variants_restore_every_actual_source_tensor(self):
        checkpoint = ROOT.parent/'f1-amp-dynamic-refine/outputs/amp-refine/TASK_20260924_072/model_8801250.pt'
        if not checkpoint.is_file(): self.skipTest('Real source artifact absent')
        cfg = config_scope()['X1AMPRecoveryCfgPPO']()
        actor_type, ppo_type = cpu_ppo_classes(ROOT)
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        for group, e in self.experiments.items():
            with self.subTest(group=group), contextlib.redirect_stdout(io.StringIO()):
                actor = actor_type(235, 47, 219, 12, **plain(cfg.policy))
                ppo = ppo_type(actor, device='cpu', **plain(cfg.algorithm))
                d = AMPDiscriminator(e.spec, e.mean, e.std, style_floor=e.cfg['signal']['style_floor'])
                trainer = DiscriminatorTrainer(d, bridge_gradient_penalty=e.cfg['signal']['bridge_gradient_penalty'])
                bridge = AMPBridge(e.spec, trainer, 2, e.cfg['reward'], 'cpu', e.cfg['recovery'])
                alg = SimpleNamespace(actor_critic=actor, optimizer=ppo.optimizer,
                    state_estimator_optimizer=ppo.state_estimator_optimizer, ppo=ppo, bridge=bridge)
                runner = SimpleNamespace(device='cpu', alg=alg, amp_trainer=trainer)
                proof = warm_start_signal(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 1250)
                self.assertEqual(runner.alg.updates, 1250)
                self.assertEqual(proof['replay_windows'], 50000)
                checker = interrupted_tests.InterruptedTests()
                for actual, key in ((actor.state_dict(), 'model_state_dict'), (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (d.state_dict(), 'amp_discriminator_state_dict'), (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    checker.assert_nested_equal(actual, state[key])
                # Actual D update remains finite and leaves normalizer frozen.
                mean, std = d.mean.clone(), d.std.clone()
                stats = trainer.step(e.windows[:32], bridge.replay.buffer[:32])
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))
                torch.testing.assert_close(d.mean, mean, atol=0, rtol=0)
                torch.testing.assert_close(d.std, std, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
