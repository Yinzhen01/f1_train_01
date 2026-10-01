import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.jitter import (apply_jitter_config, jitter_contract, jitter_source,
    validate_jitter, validate_jitter_certificate, validate_jitter_diagnostics, warm_start_jitter)
from humanoid.amp.sustain import apply_sustain_config
from humanoid.amp.scaled_experiment import ScaledExperiment, validate_cloud_smoke
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.integration import AMPBridge
from humanoid.amp.dry_run import cpu_ppo_classes
from tools.amp.verify_signal_formal import assert_equal
from tools.amp.jitter_audit import compare_environment, validate_jitter_updates
from test_f1_amp_sustain import sustain_report
from test_f1_amp_contact import reward_report
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def jitter_report(group):
    result = sustain_report('control')
    result.update(smoothness_calls=240, physics_substeps=2400,
        smoothness_scale=.01*jitter_contract(group)['smoothness_scale'],
        smoothness_cost_sum=17., physics_accel_squared_sum=240., physics_torque_delta_squared_sum=70.)
    if group.startswith('substep_'):
        spec = jitter_contract(group)['substep']
        result.update(substep_penalty=spec, substep_acceleration_calls=240,
            substep_torque_calls=240 if spec['torque_scale'] else 0,
            acceleration_scale=-.005, torque_scale=.01*spec['torque_scale'],
            substep_acceleration_cost_sum=20., endpoint_acceleration_cost_sum=4.,
            substep_torque_cost_sum=8., used_acceleration_cost_sum=20. if spec['acceleration'] else 4.,
            substep_valid_intervals=7400)
    return result


def config_type():
    scope = config_scope(); path = ROOT/'humanoid/envs/x1/x1_amp_refine_config.py'
    tree = ast.parse(path.read_text()); tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), 'exec'), scope)
    return scope['X1AMPRefineCfg']


class JitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {g: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_jitter_'+g+'.json'))
                           for g in ('control', 'smooth')}

    def test_contract_source_budget_and_unchanged_amp(self):
        for g, e in self.experiments.items():
            source = validate_jitter(e)
            self.assertEqual(e.cfg['reward'], source.cfg['reward'])
            self.assertEqual(e.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(e.cfg['smoke'], dict(num_envs=32, updates=10))
            self.assertEqual(e.cfg['evaluation_updates'], [2750])
            for mutate in (lambda c: c['formal'].update(updates=500),
                lambda c: c['jitter'].update(source_completed_updates=2510),
                lambda c: c['jitter'].update(smoothness_scale=-2.),
                lambda c: c['reward'].update(style_weight=0), lambda c: c.update(evaluation_duration_s=20)):
                bad = copy.copy(e); bad.cfg = copy.deepcopy(e.cfg); mutate(bad.cfg)
                with self.assertRaises(ValueError): validate_jitter(bad)

    def test_only_smoothness_weight_changes_no_dr_and_no_class_leak(self):
        cls = config_type()
        baseline = plain(apply_sustain_config(cls(), 'control'))
        for g in self.experiments:
            cfg = apply_jitter_config(cls(), g)
            assert_no_domain_randomization(cfg)
            self.assertTrue(compare_environment(baseline, plain(cfg), g))
        self.assertEqual(cls().rewards.scales.recovery_smoothness, -.02)
        self.assertEqual(cls().env.episode_length_s, 6.)

    def test_actual_diagnostics_fail_on_missing_calls_or_telemetry(self):
        for g in self.experiments:
            report = jitter_report(g)
            self.assertTrue(validate_jitter_diagnostics(report, g, 240, 32))
            for field, value in (('smoothness_calls', 239), ('physics_substeps', 2399),
                ('smoothness_scale', -.2), ('physics_accel_squared_sum', float('nan')),
                ('physics_torque_delta_squared_sum', 0.), ('selected_sum', 155.)):
                bad = dict(report); bad[field] = value
                with self.assertRaises(ValueError): validate_jitter_diagnostics(bad, g, 240, 32)

    def test_cloud_gate_requires_reset_source_and_substep_evidence(self):
        for g, e in self.experiments.items():
            cert = dict(identity=e.identity(), implementation_fingerprint='fixture', num_envs=32, updates=10,
                physics_dt_verified=True, body_frames_verified=True, reset_history_verified=True,
                nonzero_style_reward=True, actor_updated=True, discriminator_updated=True,
                all_finite=True, complete=True, valid_windows=1, no_dr_configuration_verified=True,
                smoke_evaluation_verified=True, rsi_reset_count=32, replay_window_count=50000,
                continuation=dict(**jitter_source(e.cfg), actor_and_discriminator_restored=True,
                    optimizers_restored=True, replay_windows=50000),
                horizon_diagnostics=dict(smoke_injected_resets=True, configured_episode_length_s=60.,
                    control_steps=240, captured_transitions=7680, forced_physical_reset_verified=True, forced_timeout_reset_verified=True),
                contact_diagnostics=reward_report('tail'), direction_diagnostics=jitter_report(g))
            self.assertTrue(validate_cloud_smoke(e, cert, 'fixture'))
            for field in ('contact_diagnostics', 'horizon_diagnostics', 'continuation', 'direction_diagnostics'):
                bad = copy.deepcopy(cert); bad.pop(field)
                with self.assertRaises(ValueError): validate_cloud_smoke(e, bad, 'fixture')
            bad = copy.deepcopy(cert['continuation']); bad['source_completed_updates'] = 2510
            with self.assertRaises(ValueError): validate_jitter_certificate(e, bad)

    def test_environment_audit_rejects_other_changes(self):
        source = plain(apply_sustain_config(config_type()(), 'control'))
        for g in self.experiments:
            target = plain(apply_jitter_config(config_type()(), g))
            self.assertTrue(compare_environment(source, target, g))
            for mutate in (lambda c: c['control'].update(decimation=5),
                lambda c: c['rewards']['scales'].update(refine_heading=-.5),
                lambda c: c['env'].update(episode_length_s=20)):
                bad = copy.deepcopy(target); mutate(bad)
                with self.assertRaises(ValueError): compare_environment(source, bad, g)

    def test_exact_update_range_and_amp_log(self):
        for formal, end in ((False, 2510), (True, 2750)):
            rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
                style_negative_fraction=0, discriminator_bridge_gradient_penalty=.1) for i in range(2501, end+1)]
            self.assertTrue(validate_jitter_updates(rows, formal))
            for bad in (rows[:-1], rows+[rows[-1]], [dict(r, iteration=r['iteration']-250) for r in rows],
                        [dict(r, weighted_style_reward=0) for r in rows]):
                with self.assertRaises(ValueError): validate_jitter_updates(bad, formal)

    def test_restore_every_real_control2500_learning_tensor(self):
        checkpoint = ROOT.parent/'f1-amp-sustain/outputs/amp-sustain/TASK_20260926_110/model_8802500.pt'
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
                proof = warm_start_jitter(runner, e, checkpoint)
                self.assertEqual(runner.current_learning_iteration, 2500)
                self.assertEqual(runner.alg.updates, 2500)
                self.assertEqual(proof['replay_windows'], 50000)
                for actual, key in ((actor.state_dict(), 'model_state_dict'), (ppo.optimizer.state_dict(), 'optimizer_state_dict'),
                    (ppo.state_estimator_optimizer.state_dict(), 'es_optimizer_state_dict'),
                    (d.state_dict(), 'amp_discriminator_state_dict'), (trainer.optimizer.state_dict(), 'amp_optimizer_state_dict'),
                    (bridge.replay.state_dict(), 'amp_replay_state')):
                    assert_equal(actual, state[key])

    def test_substep_cancellation_readonly_and_reset_guard(self):
        class Parent:
            def _begin_physics_step(self): pass
            def _after_physics_substep(self, substep): pass
            def reset_direction_diagnostics(self): pass
        path = ROOT/'humanoid/envs/x1/x1_amp_jitter_env.py'
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        scope = dict(torch=torch, X1AMPDirectionEnv=Parent)
        exec(compile(tree, str(path), 'exec'), scope)
        env = scope['X1AMPJitterEnv']()
        env.num_envs = 2; env.device = 'cpu'
        env.dof_vel = torch.zeros(2, 12); env.torques = torch.zeros(2, 12)
        env.episode_length_buf = torch.tensor([10, 0])
        env.cfg = SimpleNamespace(control=SimpleNamespace(decimation=10))
        env.sim_params = SimpleNamespace(dt=.001); env.sim = None
        env.feet_indices = [0, 1]; env.contact_forces = torch.zeros(2, 2, 3)
        env.gym = SimpleNamespace(refresh_net_contact_force_tensor=lambda sim: None)
        env._begin_physics_step()
        for k in range(10):
            env.dof_vel[:] = 1. if k % 2 == 0 else 0.
            env.torques[:] = 2. if k % 2 == 0 else 0.
            env.contact_forces[:, :, 2] = 1000. if k == 3 else 0.
            before = env.dof_vel.clone(), env.torques.clone(), env.contact_forces.clone()
            env._after_physics_substep(k)
            for actual, expected in zip((env.dof_vel, env.torques, env.contact_forces), before):
                self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(env.dof_vel, torch.zeros_like(env.dof_vel)))
        torch.testing.assert_close(env.jitter_accel_peak, torch.full((2, 12), 1000.))
        self.assertTrue(torch.equal(env.jitter_force_peak, torch.full((2, 2), 1000.)))
        self.assertAlmostEqual(float(env.jitter_sums[1])/12e6, 1., places=6)
        self.assertAlmostEqual(float(env.jitter_sums[2]), 48., places=4)
        self.assertEqual(env.jitter_physics_steps, 10)


if __name__ == '__main__': unittest.main()
