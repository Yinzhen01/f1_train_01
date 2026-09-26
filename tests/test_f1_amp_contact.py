import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.contact import (apply_contact_config, contact_contract, contact_source,
    force_tail_cost, validate_contact, validate_contact_certificate, validate_contact_diagnostics,
    warm_start_contact)
from humanoid.amp.horizon import apply_horizon_config
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.verify_signal_formal import assert_equal
from tools.amp.contact_audit import compare_environment, validate_contact_updates
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def reward_report(group, steps=240):
    c = contact_contract(group)
    return dict(tail_calls=steps if group == 'tail' else 0,
        tail_cost_sum=400. if group == 'tail' else 0., tail_positive_env_steps=1 if group == 'tail' else 0,
        runtime_scales=dict(refine_slip=c['refine_slip_scale']*.01,
            feet_contact_forces=-.0001, contact_force_tail=c['contact_force_tail_scale']*.01))


class ContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_contact_'+g+'.json'))
                           for g in ('control', 'slip', 'tail')}

    def test_exact_matched_groups_and_short_budget(self):
        for g, e in self.experiments.items():
            source = validate_contact(e)
            self.assertEqual(e.cfg['contact_refinement'], contact_contract(g))
            self.assertEqual(e.cfg['reward'], source.cfg['reward'])
            self.assertEqual(e.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(e.cfg['smoke'], dict(num_envs=32, updates=10))
            for mutate in (lambda c: c['formal'].update(updates=500),
                lambda c: c['contact_refinement'].update(episode_length_s=6),
                lambda c: c['contact_refinement'].update(source_completed_updates=1760),
                lambda c: c['reward'].update(style_weight=0), lambda c: c.update(evaluation_duration_s=20)):
                bad = copy.copy(e); bad.cfg = copy.deepcopy(e.cfg); mutate(bad.cfg)
                with self.assertRaises(ValueError): validate_contact(bad)

    def test_only_selected_contact_scale_changes_and_no_shared_mutation(self):
        scope = config_scope()
        path = ROOT/'humanoid/envs/x1/x1_amp_refine_config.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), 'exec'), scope)
        baseline = plain(apply_horizon_config(scope['X1AMPRefineCfg'](), 'long'))
        for g in self.experiments:
            cfg = apply_contact_config(scope['X1AMPRefineCfg'](), g)
            assert_no_domain_randomization(cfg)
            after = plain(cfg)
            self.assertEqual(after['rewards']['scales'].pop('contact_force_tail'), -.0025 if g == 'tail' else 0.)
            self.assertEqual(after['rewards']['scales']['refine_slip'], -6. if g == 'slip' else -2.)
            after['rewards']['scales']['refine_slip'] = -2.
            self.assertEqual(after, baseline)
        self.assertEqual(scope['X1AMPRefineCfg']().env.episode_length_s, 6.)
        self.assertFalse(hasattr(scope['X1AMPRefineCfg']().rewards.scales, 'contact_force_tail'))

    def test_force_tail_continuity_slope_and_vector_norm(self):
        force = torch.zeros(6, 2, 3, dtype=torch.float64)
        force[:, 0, 2] = torch.tensor([0., 700., 1100., 1101., 1500., 5000.])
        expected = torch.tensor([0., 0., 0., 1., 400., 3900.], dtype=torch.float64)
        self.assertTrue(torch.equal(force_tail_cost(force), expected))
        original = (force.norm(dim=-1)-700).clamp(0., 400.).sum(-1)*-.01*.01
        total = original+force_tail_cost(force)*-.0025*.01
        self.assertAlmostEqual(float(total[4]), -.05)
        self.assertAlmostEqual(float(total[3]-total[2]), -.000025)
        rotated = force.clone(); rotated[:, :, 0] = force[:, :, 2]; rotated[:, :, 2] = 0
        self.assertTrue(torch.equal(force_tail_cost(rotated), expected))
        rotated[:, 1] = rotated[:, 0]
        self.assertTrue(torch.equal(force_tail_cost(rotated), expected*2))
        with self.assertRaises(ValueError): force_tail_cost(force, threshold=0)
        with self.assertRaises(ValueError): force_tail_cost(force[:, :1])

    def test_runtime_reward_calls_and_counter_reset(self):
        path = ROOT/'humanoid/envs/x1/x1_amp_contact_env.py'
        tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        scope = dict(X1AMPRefineEnv=object, torch=torch, force_tail_cost=force_tail_cost)
        exec(compile(tree, str(path), 'exec'), scope)
        env = scope['X1AMPContactEnv'](); env.device = 'cpu'
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(max_contact_force=700.))
        env.feet_indices = [1, 3]; env.contact_forces = torch.zeros(2, 4, 3)
        env.contact_forces[0, 1, 2] = 1500.
        env.reward_scales = reward_report('tail')['runtime_scales']
        value = env._reward_contact_force_tail()
        self.assertEqual(value.tolist(), [400., 0.])
        self.assertTrue(validate_contact_diagnostics(env.contact_diagnostics(), 'tail', 1, 2))
        env.reset_contact_diagnostics()
        self.assertEqual(env.contact_diagnostics()['tail_calls'], 0)
        self.assertEqual(env.contact_diagnostics()['tail_cost_sum'], 0.)

    def test_real_activation_and_dt_scaled_weights_required(self):
        for g in self.experiments:
            self.assertTrue(validate_contact_diagnostics(reward_report(g), g, 240, 32))
            bad = reward_report(g); bad['runtime_scales']['refine_slip'] *= 100
            with self.assertRaises(ValueError): validate_contact_diagnostics(bad, g, 240, 32)
        for field, value in (('tail_calls', 239), ('tail_cost_sum', 0.), ('tail_cost_sum', float('nan')),
                              ('tail_positive_env_steps', 0), ('tail_positive_env_steps', 7681)):
            bad = reward_report('tail'); bad[field] = value
            with self.assertRaises(ValueError): validate_contact_diagnostics(bad, 'tail', 240, 32)
        bad = reward_report('control'); bad['tail_calls'] = 240
        with self.assertRaises(ValueError): validate_contact_diagnostics(bad, 'control', 240, 32)

    def test_cloud_gate_requires_contact_proof_and_source1750(self):
        for g, e in self.experiments.items():
            cert = dict(identity=e.identity(), implementation_fingerprint='fixture', num_envs=32, updates=10,
                physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
                nonzero_style_reward=True, actor_updated=True, discriminator_updated=True,
                all_finite=True, complete=True, valid_windows=1, no_dr_configuration_verified=True,
                smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
                continuation=dict(**contact_source(e.cfg), actor_and_discriminator_restored=True,
                    optimizers_restored=True, replay_windows=50000),
                horizon_diagnostics=dict(smoke_injected_resets=True, configured_episode_length_s=60.,
                    control_steps=240, captured_transitions=7680, forced_physical_reset_verified=True, forced_timeout_reset_verified=True),
                contact_diagnostics=reward_report(g))
            self.assertTrue(validate_cloud_smoke(e, cert, 'fixture'))
            for field in ('contact_diagnostics', 'horizon_diagnostics', 'continuation'):
                bad = copy.deepcopy(cert); bad.pop(field)
                with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'fixture')
            bad = copy.deepcopy(cert['continuation']); bad['source_completed_updates'] = 1760
            with self.assertRaises(ValueError): validate_contact_certificate(e, bad)

    def test_artifact_audit_rejects_other_reward_or_physics_changes(self):
        source = dict(env=dict(episode_length_s=60., num_envs=4096),
            rewards=dict(scales=dict(refine_slip=-2., feet_contact_forces=-.01)), control=dict(decimation=10))
        for g in self.experiments:
            target = copy.deepcopy(source); c = contact_contract(g)
            target['rewards']['scales'].update(refine_slip=c['refine_slip_scale'], contact_force_tail=c['contact_force_tail_scale'])
            self.assertTrue(compare_environment(source, target, g))
            smoke = copy.deepcopy(target); smoke['env']['num_envs'] = 32
            self.assertTrue(compare_environment(source, smoke, g, smoke=True))
            for mutate in (lambda x: x['control'].update(decimation=5),
                lambda x: x['rewards']['scales'].update(feet_contact_forces=-.1),
                lambda x: x['env'].update(episode_length_s=6.)):
                bad = copy.deepcopy(target); mutate(bad)
                with self.assertRaises(ValueError): compare_environment(source, bad, g)

    def test_log_requires_exact_bounded_actual_updates(self):
        for formal, end in ((False, 1760), (True, 2000)):
            rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
                style_negative_fraction=0, discriminator_bridge_gradient_penalty=.1) for i in range(1751, end+1)]
            self.assertTrue(validate_contact_updates(rows, formal))
            for bad in (rows[:-1], rows+[rows[-1]], [dict(r, weighted_style_reward=0) for r in rows],
                [dict(r, discriminator_bridge_gradient_penalty=0) for r in rows]):
                with self.assertRaises(ValueError): validate_contact_updates(bad, formal)

    def test_restore_every_real_long1750_learning_tensor(self):
        checkpoint = ROOT.parent/'f1-amp-horizon-coverage/outputs/amp-horizon/TASK_20260924_113/model_8801750.pt'
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
                proof = warm_start_contact(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 1750)
                self.assertEqual(runner.alg.updates, 1750)
                self.assertEqual(proof['replay_windows'], 50000)
                for actual, key in ((actor.state_dict(), 'model_state_dict'), (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (d.state_dict(), 'amp_discriminator_state_dict'), (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    assert_equal(actual, state[key])


if __name__ == '__main__': unittest.main()
