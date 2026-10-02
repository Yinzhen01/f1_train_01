"""CPU contract/restoration checks, not a simulator or motion-quality result."""
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from humanoid.amp.adapter import AMPAlgorithmAdapter
from humanoid.amp.discriminator import AMPDiscriminator, DiscriminatorTrainer
from humanoid.amp.dry_run import cpu_ppo_classes
from humanoid.amp.integration import AMPBridge
from humanoid.amp.learnability import assert_no_domain_randomization
from humanoid.amp.mu_temporal import (ACCELERATION_SCALES, apply_mu_config, mu_contract,
    mu_source, state_fingerprint, validate_mu, validate_mu_certificate,
    validate_mu_diagnostics, validate_mu_loss_report, warm_start_mu)
from humanoid.amp.scaled_experiment import ScaledExperiment
from humanoid.amp.sustain import apply_sustain_config
from test_f1_amp_jitter import config_type, jitter_report
from test_f1_amp_recovery import config_scope, plain

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class MuContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {group: ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_mu_'+group+'.json'))
                           for group in ('anchor', 'temporal')}

    def test_same_source_budget_and_only_auxiliary_contract(self):
        for group, experiment in self.experiments.items():
            source = validate_mu(experiment)
            self.assertEqual(experiment.cfg['reward'], source.cfg['reward'])
            self.assertEqual(experiment.cfg['formal'], dict(num_envs=4096, updates=250))
            self.assertEqual(experiment.cfg['smoke'], dict(num_envs=32, updates=10))
            self.assertEqual(experiment.cfg['evaluation_updates'], [2750])
            self.assertEqual(experiment.cfg['mu_temporal']['anchor_coef'], 1.)
            self.assertEqual(experiment.cfg['mu_temporal']['temporal_coef'], .01 if group == 'temporal' else 0.)
            self.assertEqual(len(ACCELERATION_SCALES), 12)
            for mutate in (lambda c: c['mu_temporal'].update(anchor_coef=2.),
                           lambda c: c['mu_temporal'].update(startup_steps=3),
                           lambda c: c['formal'].update(updates=500),
                           lambda c: c['reward'].update(style_weight=0),
                           lambda c: c['mu_temporal']['acceleration_scales_rad_s2'].update(left_knee_pitch_joint=100.)):
                changed = copy.copy(experiment)
                changed.cfg = copy.deepcopy(experiment.cfg)
                mutate(changed.cfg)
                with self.assertRaises(ValueError): validate_mu(changed)

    def test_exact_environment_no_filter_no_dr_no_reward_change(self):
        cfg_type = config_type()
        source = plain(apply_sustain_config(cfg_type(), 'control'))
        for group in self.experiments:
            cfg = apply_mu_config(cfg_type(), group)
            assert_no_domain_randomization(cfg)
            self.assertEqual(plain(cfg), source)
            self.assertFalse(hasattr(cfg, 'target_filter'))
            self.assertFalse(hasattr(cfg, 'ankle_slew'))
            self.assertFalse(hasattr(cfg, 'substep_penalty'))
            self.assertTrue(validate_mu_diagnostics(jitter_report('control'), group, 240, 32))

    def test_loss_evidence_cannot_admit_disabled_temporal_or_changed_teacher(self):
        for group in self.experiments:
            spec = mu_contract(group)
            report = dict(updates=10, teacher_unchanged=True, metadata_observed=True,
                valid_triplets=10, all_finite=True, anchor_coef=1., temporal_coef=spec['temporal_coef'],
                anchor_loss=.001, temporal_loss=.2, weighted_anchor_loss=.001,
                weighted_temporal_loss=.002 if group == 'temporal' else 0., aux_grad_norm=.01)
            self.assertTrue(validate_mu_loss_report(report, spec, 10))
            for key, value in (('updates', 250), ('teacher_unchanged', False),
                               ('valid_triplets', 0), ('metadata_observed', False),
                               ('temporal_loss', float('nan')), ('temporal_coef', 1.)):
                bad = dict(report, **{key: value})
                with self.assertRaises(ValueError): validate_mu_loss_report(bad, spec, 10)
            bad = dict(report, weighted_temporal_loss=0. if group == 'temporal' else .01)
            with self.assertRaises(ValueError): validate_mu_loss_report(bad, spec, 10)

    def test_real_control2500_restores_student_teacher_and_every_learning_state(self):
        checkpoint = ROOT.parent/'f1-amp-sustain/outputs/amp-sustain/TASK_20260926_110/model_8802500.pt'
        if not checkpoint.is_file(): self.skipTest('Actual source absent')
        actor_type, ppo_type = cpu_ppo_classes(ROOT)
        cfg = config_scope()['X1AMPRecoveryCfgPPO']()
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        for group, experiment in self.experiments.items():
            with self.subTest(group=group), contextlib.redirect_stdout(io.StringIO()):
                actor = actor_type(235, 47, 219, 12, **plain(cfg.policy))
                ppo = ppo_type(actor, device='cpu', **plain(cfg.algorithm))
                ppo.init_storage(2, 24, [3102], [219], [12])
                discriminator = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std)
                trainer = DiscriminatorTrainer(discriminator, bridge_gradient_penalty=1.)
                bridge = AMPBridge(experiment.spec, trainer, 2, experiment.cfg['reward'], 'cpu', experiment.cfg['recovery'])
                adapter = AMPAlgorithmAdapter(ppo, bridge, None, experiment)
                runner = SimpleNamespace(device='cpu', alg=adapter, amp_trainer=trainer)
                proof = warm_start_mu(runner, experiment, checkpoint)
                self.assertTrue(validate_mu_certificate(experiment, proof))
                self.assertEqual(proof['replay_windows'], 50000)
                self.assertEqual(runner.current_learning_iteration, 2500)
                self.assertEqual(adapter.updates, 2500)
                self.assertEqual(state_fingerprint(actor.state_dict()), state_fingerprint(state['model_state_dict']))
                self.assertEqual(state_fingerprint(ppo.mu_regularizer.teacher.state_dict()), runner.mu_initial_teacher_hash)
                self.assertTrue(ppo.mu_regularizer.teacher_unchanged())
                student_ids = {id(parameter) for group in ppo.optimizer.param_groups for parameter in group['params']}
                self.assertTrue(all(id(p) not in student_ids for p in ppo.mu_regularizer.teacher.parameters()))
                for key, value in state['amp_discriminator_state_dict'].items():
                    self.assertTrue(torch.equal(discriminator.state_dict()[key], value))
                for field in ('actor_and_discriminator_restored', 'optimizers_restored', 'teacher_frozen'):
                    self.assertIs(proof[field], True)
                bad = dict(proof, teacher_source_model_sha256='a'*64)
                with self.assertRaises(ValueError): validate_mu_certificate(experiment, bad)

    def test_source_state_fingerprint_detects_signed_zero_and_shape(self):
        source = {'a': torch.tensor([0., 1.])}
        self.assertNotEqual(state_fingerprint(source), state_fingerprint({'a': torch.tensor([-0., 1.])}))
        self.assertNotEqual(state_fingerprint(source), state_fingerprint({'a': source['a'].view(1, 2)}))


if __name__ == '__main__': unittest.main()
