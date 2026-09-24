import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.horizon import (horizon_contract, validate_horizon, horizon_source, warm_start_horizon,
    HorizonProbe, validate_horizon_probe, apply_horizon_config)
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.verify_horizon_smoke import validate_horizon_updates
from tools.amp.verify_signal_formal import assert_equal
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class FakeEnv:
    def __init__(self, duration):
        self.cfg = SimpleNamespace(env=SimpleNamespace(episode_length_s=duration))
        self.num_envs, self.device, self.dt = 2, 'cpu', .01
        self.max_episode_length = round(duration/.01)
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.reset_buf = torch.zeros(2, dtype=torch.bool)
        self.time_out_buf = self.reset_buf.clone()

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf.clone()
        if getattr(self, 'amp_force_reset_step', -1) == self.common_step_counter:
            self.reset_buf[0] = True; self.time_out_buf[0] = False

    def _reward_termination(self):
        return (self.reset_buf & ~self.time_out_buf).float()


class HorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_horizon_'+g+'.json'))
                           for g in ('short', 'long')}

    def test_only_horizon_and_continuation_identity_change(self):
        for group, e in self.experiments.items():
            source = validate_horizon(e)
            self.assertEqual(e.cfg['horizon'], horizon_contract(group))
            self.assertEqual(e.cfg['reward'], source.cfg['reward'])
            self.assertEqual(e.cfg['formal'], dict(num_envs=4096, updates=250))
            for mutate in (lambda c: c['formal'].update(updates=500), lambda c: c['horizon'].update(episode_length_s=20),
                           lambda c: c['reward'].update(style_weight=0), lambda c: c.update(evaluation_duration_s=20)):
                bad = copy.copy(e); bad.cfg = copy.deepcopy(e.cfg); mutate(bad.cfg)
                with self.assertRaises(ValueError): validate_horizon(bad)

    def test_nominal_environment_changes_only_episode_length(self):
        scope = config_scope()
        path = ROOT/'humanoid/envs/x1/x1_amp_refine_config.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), 'exec'), scope)
        baseline = plain(scope['X1AMPRefineCfg']())
        for group in ('short', 'long'):
            cfg = scope['X1AMPRefineCfg']()
            apply_horizon_config(cfg, group)
            after = plain(cfg)
            self.assertEqual(after['env']['episode_length_s'], 6. if group == 'short' else 60.)
            after['env']['episode_length_s'] = 6.
            self.assertEqual(after, baseline)
        self.assertEqual(scope['X1AMPRefineCfg']().env.episode_length_s, 6.)

    def test_smoke_checks_timeout_and_physical_termination_without_lowering_horizon(self):
        for seconds in (6., 60.):
            env = FakeEnv(seconds); probe = HorizonProbe(env, smoke=True)
            for step in range(1, 241):
                env.common_step_counter = step; env.episode_length_buf += 1
                env.check_termination()
                env.episode_length_buf[env.reset_buf] = 0
            record = probe.report()
            self.assertTrue(validate_horizon_probe(record, seconds, True))
            self.assertEqual(record['physical_failures'], 1)
            self.assertEqual(record['timeouts'], 1)
            self.assertEqual(env.max_episode_length, round(seconds*100))
            self.assertEqual(record['captured_transitions'], 480)
            bad = dict(record, forced_timeout_reset_verified=False)
            with self.assertRaises(ValueError): validate_horizon_probe(bad, seconds, True)

    def test_formal_probe_is_read_only_and_exposes_long_coverage(self):
        env = FakeEnv(60.); probe = HorizonProbe(env, smoke=False)
        for step, age in ((37, 37), (61, 61), (100, 602), (101, 4002)):
            env.common_step_counter = step; env.episode_length_buf[:] = age
            before = env.episode_length_buf.clone(); env.check_termination()
            self.assertTrue(torch.equal(before, env.episode_length_buf))
            self.assertFalse(env.reset_buf.any())
        r = probe.report()
        self.assertTrue(validate_horizon_probe(r, 60., False))
        self.assertEqual(r['beyond_6s_reset_horizon'], 4)
        self.assertEqual(r['beyond_40s_reset_horizon'], 2)
        self.assertFalse(hasattr(env, 'amp_force_reset_step'))
        env = FakeEnv(6.); probe = HorizonProbe(env)
        env.episode_length_buf[:] = 601; env.check_termination()
        self.assertTrue(validate_horizon_probe(probe.report(), 6., False))
        self.assertEqual(probe.report()['beyond_6s_reset_horizon'], 0)

    def test_old_source_or_missing_timeout_proof_cannot_authorize_formal(self):
        e = self.experiments['long']
        cert = dict(identity=e.identity(), implementation_fingerprint='fixture', num_envs=32, updates=10,
            physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
            nonzero_style_reward=True, actor_updated=True, discriminator_updated=True,
            all_finite=True, complete=True, valid_windows=1, no_dr_configuration_verified=True,
            smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
            continuation=dict(**horizon_source(e.cfg), actor_and_discriminator_restored=True,
                              optimizers_restored=True, replay_windows=50000),
            horizon_diagnostics=dict(smoke_injected_resets=True, configured_episode_length_s=60.,
                control_steps=240, captured_transitions=7680, forced_physical_reset_verified=True, forced_timeout_reset_verified=True))
        self.assertTrue(validate_cloud_smoke(e, cert, 'fixture'))
        for field in ('source_task', 'source_completed_updates'):
            bad = copy.deepcopy(cert); bad['continuation'][field] = 'wrong'
            with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'fixture')
        bad = copy.deepcopy(cert); bad.pop('horizon_diagnostics')
        with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'fixture')

    def test_ten_actual_updates_keep_bridge_style_active(self):
        rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
            style_negative_fraction=0, discriminator_bridge_gradient_penalty=.1) for i in range(1501, 1511)]
        self.assertTrue(validate_horizon_updates(rows))
        for bad in (rows[:-1], [dict(r, discriminator_bridge_gradient_penalty=0) for r in rows],
                    [dict(r, weighted_style_reward=0) for r in rows]):
            with self.assertRaises(ValueError): validate_horizon_updates(bad)

    def test_both_restore_every_actual_bridge1500_learning_tensor(self):
        checkpoint = ROOT.parent/'f1-amp-style-signal/outputs/amp-signal/TASK_20260924_094/model_8801500.pt'
        if not checkpoint.is_file(): self.skipTest('Real source absent')
        cfg = config_scope()['X1AMPRecoveryCfgPPO']()
        actor_type, ppo_type = cpu_ppo_classes(ROOT)
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        for group, e in self.experiments.items():
            with self.subTest(group=group), contextlib.redirect_stdout(io.StringIO()):
                actor = actor_type(235, 47, 219, 12, **plain(cfg.policy))
                ppo = ppo_type(actor, device='cpu', **plain(cfg.algorithm))
                d = AMPDiscriminator(e.spec, e.mean, e.std)
                trainer = DiscriminatorTrainer(d, bridge_gradient_penalty=1.)
                bridge = AMPBridge(e.spec, trainer, 2, e.cfg['reward'], 'cpu', e.cfg['recovery'])
                alg = SimpleNamespace(actor_critic=actor, optimizer=ppo.optimizer,
                    state_estimator_optimizer=ppo.state_estimator_optimizer, ppo=ppo, bridge=bridge)
                runner = SimpleNamespace(device='cpu', alg=alg, amp_trainer=trainer)
                proof = warm_start_horizon(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 1500)
                self.assertEqual(runner.alg.updates, 1500)
                self.assertEqual(proof['replay_windows'], 50000)
                for actual, key in ((actor.state_dict(), 'model_state_dict'), (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (d.state_dict(), 'amp_discriminator_state_dict'), (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    assert_equal(actual, state[key])


if __name__ == '__main__': unittest.main()
