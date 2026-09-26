import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.progress import (apply_progress_config, progress_contract, progress_rewards,
    progress_source, validate_progress, validate_progress_certificate, validate_progress_diagnostics,
    warm_start_progress)
from humanoid.amp.contact import apply_contact_config
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.verify_signal_formal import assert_equal
from tools.amp.progress_audit import compare_environment, validate_progress_updates
from test_f1_amp_contact import reward_report
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def progress_report(group, steps=240, num_envs=32):
    return dict(velocity_frame=group, calls=steps, env_steps=steps*num_envs, runtime_scale=.02,
        body_sum=2., world_sum=-1., selected_sum=2. if group == 'body' else -1.,
        abs_body_world_difference_sum=3.)


class ProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_progress_'+g+'.json'))
                           for g in ('body', 'world')}

    def test_exact_contract_and_bounded_budget(self):
        for g, e in self.experiments.items():
            source = validate_progress(e)
            self.assertEqual(e.cfg['reward'], source.cfg['reward'])
            self.assertEqual(e.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(e.cfg['smoke'], dict(num_envs=32, updates=10))
            for mutate in (lambda c: c['formal'].update(updates=500),
                lambda c: c['progress'].update(source_completed_updates=2010),
                lambda c: c['progress'].update(velocity_frame='heading'),
                lambda c: c['reward'].update(style_weight=0), lambda c: c.update(evaluation_duration_s=20)):
                bad = copy.copy(e); bad.cfg = copy.deepcopy(e.cfg); mutate(bad.cfg)
                with self.assertRaises(ValueError): validate_progress(bad)

    def test_only_frame_changes_source_config_and_no_dr(self):
        scope = config_scope(); path = ROOT/'humanoid/envs/x1/x1_amp_refine_config.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), 'exec'), scope)
        baseline = plain(apply_contact_config(scope['X1AMPRefineCfg'](), 'tail'))
        for g in self.experiments:
            cfg = apply_progress_config(scope['X1AMPRefineCfg'](), g)
            assert_no_domain_randomization(cfg)
            self.assertTrue(compare_environment(baseline, plain(cfg), g))
        self.assertFalse(hasattr(scope['X1AMPRefineCfg']().rewards, 'progress_velocity_frame'))
        self.assertEqual(scope['X1AMPRefineCfg']().env.episode_length_s, 6.)

    def test_rotation_changes_world_not_body_reward(self):
        yaw = torch.deg2rad(torch.tensor([0., 50., 90.]))
        body = torch.tensor([[.45, 0.]]).repeat(3, 1)
        world = torch.stack([.45*torch.cos(yaw), .45*torch.sin(yaw)], -1)
        for g in self.experiments:
            selected, b, w = progress_rewards(body, world, body, g)
            torch.testing.assert_close(b, torch.ones(3))
            self.assertAlmostEqual(float(w[0]), 1.)
            self.assertTrue(bool((w[1:] < 0).all()))
            self.assertTrue(torch.equal(selected, b if g == 'body' else w))
        with self.assertRaises(ValueError): progress_rewards(body, world, body, 'invalid')

    def test_real_environment_reward_override_and_counters(self):
        path = ROOT/'humanoid/envs/x1/x1_amp_progress_env.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        scope = dict(X1AMPContactEnv=object, torch=torch, progress_rewards=progress_rewards)
        exec(compile(tree, str(path), 'exec'), scope)
        for g in self.experiments:
            env = scope['X1AMPProgressEnv'](); env.device = 'cpu'; env.num_envs = 2
            env.cfg = SimpleNamespace(rewards=SimpleNamespace(progress_velocity_frame=g))
            env.reward_scales = dict(recovery_progress=.02)
            env.base_lin_vel = torch.tensor([[.45, 0., 0.], [.45, 0., 0.]])
            env.root_states = torch.zeros(2, 13); env.root_states[:, 8] = .45
            env.commands = env.base_lin_vel.clone()
            result = env._reward_recovery_progress()
            self.assertTrue(bool((result > 0).all()) if g == 'body' else bool((result < 0).all()))
            self.assertTrue(validate_progress_diagnostics(env.progress_diagnostics(), g, 1, 2))
            env.reset_progress_diagnostics(); self.assertEqual(env.progress_diagnostics()['calls'], 0)

    def test_diagnostics_reject_wrong_frame_or_no_actual_calls(self):
        for g in self.experiments:
            self.assertTrue(validate_progress_diagnostics(progress_report(g), g, 240, 32))
            for field, value in (('calls', 239), ('runtime_scale', 2.), ('env_steps', 1),
                ('velocity_frame', 'invalid'), ('selected_sum', 0.), ('body_sum', float('nan')),
                ('abs_body_world_difference_sum', 0.)):
                bad = progress_report(g); bad[field] = value
                with self.assertRaises(ValueError): validate_progress_diagnostics(bad, g, 240, 32)

    def test_cloud_gate_requires_progress_contact_and_source2000(self):
        for g, e in self.experiments.items():
            cert = dict(identity=e.identity(), implementation_fingerprint='fixture', num_envs=32, updates=10,
                physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
                nonzero_style_reward=True, actor_updated=True, discriminator_updated=True,
                all_finite=True, complete=True, valid_windows=1, no_dr_configuration_verified=True,
                smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
                continuation=dict(**progress_source(e.cfg), actor_and_discriminator_restored=True,
                    optimizers_restored=True, replay_windows=50000),
                horizon_diagnostics=dict(smoke_injected_resets=True, configured_episode_length_s=60.,
                    control_steps=240, captured_transitions=7680, forced_physical_reset_verified=True, forced_timeout_reset_verified=True),
                contact_diagnostics=reward_report('tail'), progress_diagnostics=progress_report(g))
            self.assertTrue(validate_cloud_smoke(e, cert, 'fixture'))
            for field in ('contact_diagnostics', 'horizon_diagnostics', 'continuation', 'progress_diagnostics'):
                bad = copy.deepcopy(cert); bad.pop(field)
                with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'fixture')
            bad = copy.deepcopy(cert['continuation']); bad['source_completed_updates'] = 2010
            with self.assertRaises(ValueError): validate_progress_certificate(e, bad)

    def test_audit_rejects_other_reward_physics_or_budget_changes(self):
        source = dict(env=dict(episode_length_s=60., num_envs=4096), rewards=dict(scales=dict(recovery_progress=2., contact_force_tail=-.0025)), control=dict(decimation=10))
        for g in self.experiments:
            target = copy.deepcopy(source); target['rewards']['progress_velocity_frame'] = g
            self.assertTrue(compare_environment(source, target, g))
            smoke = copy.deepcopy(target); smoke['env']['num_envs'] = 32
            self.assertTrue(compare_environment(source, smoke, g, smoke=True))
            target['control']['decimation'] = 5
            with self.assertRaises(ValueError): compare_environment(source, target, g)
        for formal, end in ((False, 2010), (True, 2250)):
            rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
                style_negative_fraction=0, discriminator_bridge_gradient_penalty=.1) for i in range(2001, end+1)]
            self.assertTrue(validate_progress_updates(rows, formal))
            for bad in (rows[:-1], rows+[rows[-1]], [dict(r, weighted_style_reward=0) for r in rows]):
                with self.assertRaises(ValueError): validate_progress_updates(bad, formal)

    def test_restore_every_real_tail2000_learning_tensor(self):
        checkpoint = ROOT.parent/'f1-amp-contact-refine/outputs/amp-contact/TASK_20260926_071/model_8802000.pt'
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
                proof = warm_start_progress(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 2000)
                self.assertEqual(runner.alg.updates, 2000)
                self.assertEqual(proof['replay_windows'], 50000)
                for actual, key in ((actor.state_dict(), 'model_state_dict'), (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (d.state_dict(), 'amp_discriminator_state_dict'), (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    assert_equal(actual, state[key])


if __name__ == '__main__': unittest.main()
