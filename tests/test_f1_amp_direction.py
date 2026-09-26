import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.direction import (apply_direction_config, direction_contract, direction_rewards,
    direction_source, validate_direction, validate_direction_certificate, validate_direction_diagnostics,
    warm_start_direction)
from humanoid.amp.contact import apply_contact_config
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.verify_signal_formal import assert_equal
from tools.amp.direction_audit import compare_environment, validate_direction_updates
from test_f1_amp_contact import reward_report
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def direction_report(group, steps=240, num_envs=32):
    c = direction_contract(group)
    return dict(world_fraction=c['world_fraction'], calls=steps, heading_calls=steps,
        env_steps=steps*num_envs, progress_scale=.02, heading_scale=.01*c['heading_scale'],
        body_sum=200., world_sum=-100., selected_sum=155. if group == 'mix15' else 200.,
        abs_body_world_difference_sum=300., heading_cost_sum=20.)


class DirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_direction_'+g+'.json'))
                           for g in ('mix15', 'heading')}

    def test_exact_contract_and_bounded_budget(self):
        for g, e in self.experiments.items():
            source = validate_direction(e)
            self.assertEqual(e.cfg['reward'], source.cfg['reward'])
            self.assertEqual(e.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(e.cfg['smoke'], dict(num_envs=32, updates=10))
            for mutate in (lambda c: c['formal'].update(updates=500),
                lambda c: c['direction'].update(source_completed_updates=2010),
                lambda c: c['direction'].update(world_fraction=1.),
                lambda c: c['reward'].update(style_weight=0), lambda c: c.update(evaluation_duration_s=20)):
                bad = copy.copy(e); bad.cfg = copy.deepcopy(e.cfg); mutate(bad.cfg)
                with self.assertRaises(ValueError): validate_direction(bad)

    def test_only_frame_changes_source_config_and_no_dr(self):
        scope = config_scope(); path = ROOT/'humanoid/envs/x1/x1_amp_refine_config.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), 'exec'), scope)
        baseline = plain(apply_contact_config(scope['X1AMPRefineCfg'](), 'tail'))
        for g in self.experiments:
            cfg = apply_direction_config(scope['X1AMPRefineCfg'](), g)
            assert_no_domain_randomization(cfg)
            self.assertTrue(compare_environment(baseline, plain(cfg), g))
        self.assertFalse(hasattr(scope['X1AMPRefineCfg']().rewards, 'direction_world_fraction'))
        self.assertEqual(scope['X1AMPRefineCfg']().env.episode_length_s, 6.)

    def test_rotation_and_only_bounded_reward_shift(self):
        yaw = torch.deg2rad(torch.tensor([0., 50., 90.]))
        body = torch.tensor([[.45, 0.]]).repeat(3, 1)
        world = torch.stack([.45*torch.cos(yaw), .45*torch.sin(yaw)], -1)
        for g in self.experiments:
            fraction = direction_contract(g)['world_fraction']
            selected, b, w = direction_rewards(body, world, body, fraction)
            torch.testing.assert_close(b, torch.ones(3))
            torch.testing.assert_close(selected, (1-fraction)*b+fraction*w)
            self.assertTrue(bool((selected > 0).all()))
            self.assertTrue(bool((w[1:] < 0).all()))
        with self.assertRaises(ValueError): direction_rewards(body, world, body, 1.)

    def test_real_environment_reward_override_and_counters(self):
        path = ROOT/'humanoid/envs/x1/x1_amp_direction_env.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        parent = type('Parent', (), {'_reward_refine_heading': lambda self: 1-torch.cos(self.base_euler_xyz[:, 2])})
        scope = dict(X1AMPContactEnv=parent, torch=torch, direction_rewards=direction_rewards)
        exec(compile(tree, str(path), 'exec'), scope)
        for g in self.experiments:
            c = direction_contract(g)
            env = scope['X1AMPDirectionEnv'](); env.device = 'cpu'; env.num_envs = 2
            env.cfg = SimpleNamespace(rewards=SimpleNamespace(direction_world_fraction=c['world_fraction']))
            env.reward_scales = dict(recovery_progress=.02, refine_heading=.01*c['heading_scale'])
            env.base_lin_vel = torch.tensor([[.45, 0., 0.], [.45, 0., 0.]])
            env.root_states = torch.zeros(2, 13); env.root_states[:, 8] = .45
            env.commands = env.base_lin_vel.clone(); env.base_euler_xyz = torch.tensor([[0., 0., .3], [0., 0., -.5]])
            result = env._reward_recovery_progress()
            expected = direction_rewards(env.base_lin_vel[:, :2], env.root_states[:, 7:9],
                                         env.commands[:, :2], c['world_fraction'])[0]
            self.assertTrue(torch.equal(result, expected))
            torch.testing.assert_close(env._reward_refine_heading(), 1-torch.cos(env.base_euler_xyz[:, 2]))
            self.assertTrue(validate_direction_diagnostics(env.direction_diagnostics(), g, 1, 2))
            env.reset_direction_diagnostics(); self.assertEqual(env.direction_diagnostics()['calls'], 0)

    def test_diagnostics_reject_wrong_mixture_heading_or_no_calls(self):
        for g in self.experiments:
            self.assertTrue(validate_direction_diagnostics(direction_report(g), g, 240, 32))
            for field, value in (('calls', 239), ('heading_calls', 239), ('progress_scale', 2.), ('env_steps', 1),
                ('world_fraction', 1.), ('heading_scale', -5.), ('selected_sum', 0.), ('body_sum', float('nan')),
                ('abs_body_world_difference_sum', 0.), ('heading_cost_sum', 0.)):
                bad = direction_report(g); bad[field] = value
                with self.assertRaises(ValueError): validate_direction_diagnostics(bad, g, 240, 32)

    def test_cloud_gate_requires_direction_contact_and_source2000(self):
        for g, e in self.experiments.items():
            cert = dict(identity=e.identity(), implementation_fingerprint='fixture', num_envs=32, updates=10,
                physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
                nonzero_style_reward=True, actor_updated=True, discriminator_updated=True,
                all_finite=True, complete=True, valid_windows=1, no_dr_configuration_verified=True,
                smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
                continuation=dict(**direction_source(e.cfg), actor_and_discriminator_restored=True,
                    optimizers_restored=True, replay_windows=50000),
                horizon_diagnostics=dict(smoke_injected_resets=True, configured_episode_length_s=60.,
                    control_steps=240, captured_transitions=7680, forced_physical_reset_verified=True, forced_timeout_reset_verified=True),
                contact_diagnostics=reward_report('tail'), direction_diagnostics=direction_report(g))
            self.assertTrue(validate_cloud_smoke(e, cert, 'fixture'))
            for field in ('contact_diagnostics', 'horizon_diagnostics', 'continuation', 'direction_diagnostics'):
                bad = copy.deepcopy(cert); bad.pop(field)
                with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'fixture')
            bad = copy.deepcopy(cert['continuation']); bad['source_completed_updates'] = 2010
            with self.assertRaises(ValueError): validate_direction_certificate(e, bad)

    def test_audit_rejects_other_reward_physics_or_budget_changes(self):
        source = dict(env=dict(episode_length_s=60., num_envs=4096), rewards=dict(scales=dict(recovery_progress=2., contact_force_tail=-.0025, refine_heading=-.5)), control=dict(decimation=10))
        for g in self.experiments:
            target = copy.deepcopy(source); target['rewards']['direction_world_fraction'] = direction_contract(g)['world_fraction']; target['rewards']['scales']['refine_heading'] = direction_contract(g)['heading_scale']
            self.assertTrue(compare_environment(source, target, g))
            smoke = copy.deepcopy(target); smoke['env']['num_envs'] = 32
            self.assertTrue(compare_environment(source, smoke, g, smoke=True))
            target['control']['decimation'] = 5
            with self.assertRaises(ValueError): compare_environment(source, target, g)
        for formal, end in ((False, 2010), (True, 2250)):
            rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
                style_negative_fraction=0, discriminator_bridge_gradient_penalty=.1) for i in range(2001, end+1)]
            self.assertTrue(validate_direction_updates(rows, formal))
            for bad in (rows[:-1], rows+[rows[-1]], [dict(r, weighted_style_reward=0) for r in rows]):
                with self.assertRaises(ValueError): validate_direction_updates(bad, formal)

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
                proof = warm_start_direction(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 2000)
                self.assertEqual(runner.alg.updates, 2000)
                self.assertEqual(proof['replay_windows'], 50000)
                for actual, key in ((actor.state_dict(), 'model_state_dict'), (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (d.state_dict(), 'amp_discriminator_state_dict'), (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    assert_equal(actual, state[key])


if __name__ == '__main__': unittest.main()
