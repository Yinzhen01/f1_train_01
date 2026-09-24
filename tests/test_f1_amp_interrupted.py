"""Interrupted model1250 recovery: identity, all optimizer/replay state and budget."""
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.interrupted import interrupted_source, restart_budget, recover_interrupted, validate_recovery_certificate
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class InterruptedTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_budget_stops_at_original1500_and_sources_are_distinct(self):
        seen = set()
        for group in ('control', 'smooth', 'noamp'):
            e = ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_refine_'+group+'.json'))
            self.assertEqual(restart_budget(e.cfg, 'formal'), {'num_envs': 4096, 'updates': 250})
            self.assertEqual(restart_budget(e.cfg, 'smoke'), {'num_envs': 32, 'updates': 10})
            source = interrupted_source(e.cfg)
            seen.add(source['source_sha256'])
            proof = dict(**source, actor_and_discriminator_restored=True, optimizers_restored=True, replay_windows=50000)
            self.assertTrue(validate_recovery_certificate(e.cfg, proof))
            proof['source_completed_updates'] = 1301
            with self.assertRaises(ValueError):
                validate_recovery_certificate(e.cfg, proof)
            with self.assertRaises(ValueError):
                restart_budget(e.cfg, 'unbounded')
        self.assertEqual(len(seen), 3)

    def test_restart_certificate_cannot_authorize_original_or_other_group(self):
        e = ScaledExperiment(ROOT, ROOT/'configs/amp/lafan_walk02_refine_smooth.json')
        certificate = dict(identity=e.identity(), implementation_fingerprint='test-only', num_envs=32, updates=10,
            physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
            nonzero_style_reward=True, actor_updated=True, discriminator_updated=True, all_finite=True,
            complete=True, valid_windows=100, no_dr_configuration_verified=True,
            smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
            continuation=dict(**interrupted_source(e.cfg), actor_and_discriminator_restored=True,
                              optimizers_restored=True, replay_windows=50000))
        self.assertTrue(validate_cloud_smoke(e, certificate, 'test-only', interrupted=True))
        with self.assertRaises(ValueError):
            validate_cloud_smoke(e, certificate, 'test-only')
        certificate['continuation']['source_task'] = 'TASK_20260924_071'
        with self.assertRaises(ValueError):
            validate_cloud_smoke(e, certificate, 'test-only', interrupted=True)

    def test_all_three_actual_checkpoints_restore_every_training_tensor(self):
        for group, task in (('control', '071'), ('smooth', '072'), ('noamp', '073')):
            with self.subTest(group=group):
                e = ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_refine_'+group+'.json'))
                checkpoint = ROOT.parent/('f1-amp-dynamic-refine/outputs/amp-refine/TASK_20260924_'+task+'/model_8801250.pt')
                if not checkpoint.is_file():
                    self.skipTest('Local real artifact absent; requires cloud proof')
                cfg = config_scope()['X1AMPRecoveryCfgPPO']()
                actor_type, ppo_type = cpu_ppo_classes(ROOT)
                with contextlib.redirect_stdout(io.StringIO()):
                    actor = actor_type(235, 47, 219, 12, **plain(cfg.policy))
                ppo = ppo_type(actor, device='cpu', **plain(cfg.algorithm))
                trainer = DiscriminatorTrainer(AMPDiscriminator(e.spec, e.mean, e.std))
                bridge = AMPBridge(e.spec, trainer, 2, e.cfg['reward'], 'cpu', e.cfg['recovery'])
                alg = SimpleNamespace(actor_critic=actor, optimizer=ppo.optimizer,
                    state_estimator_optimizer=ppo.state_estimator_optimizer, ppo=ppo, bridge=bridge)
                runner = SimpleNamespace(device='cpu', alg=alg, amp_trainer=trainer)
                proof = recover_interrupted(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 1250)
                self.assertEqual(runner.it, 1249)
                self.assertEqual(runner.alg.updates, 1250)
                self.assertEqual(proof['replay_windows'], 50000)
                state = torch.load(checkpoint, weights_only=True, map_location='cpu')
                for actual, key in ((actor.state_dict(), 'model_state_dict'),
                    (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (trainer.discriminator.state_dict(), 'amp_discriminator_state_dict'),
                    (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    self.assert_nested_equal(actual, state[key])


if __name__ == '__main__':
    unittest.main()
