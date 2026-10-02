"""Actual source-state CPU restoration/update tests, not simulator evidence."""
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
from humanoid.amp.feature_freeze import validate_gradient_report
from humanoid.amp.integration import AMPBridge
from humanoid.amp.mu_temporal import mu_contract, validate_mu, warm_start_mu, validate_mu_loss_report
from humanoid.amp.scaled_experiment import ScaledExperiment
from test_f1_amp_recovery import config_scope, plain
from test_amp_mu_storage import metadata
from tools.amp.mu_audit import validate_frozen_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'f1-amp-sustain/outputs/amp-sustain/TASK_20260926_110/model_8802500.pt'
torch.set_num_threads(1)


class FeatureContractTests(unittest.TestCase):
    def test_only_explicit_frozen_groups_acquire_learning_path_contract(self):
        for group in ('anchor', 'temporal'):
            old = mu_contract(group)
            new = mu_contract('freeze_'+group)
            self.assertNotIn('frozen_modules', old)
            for key, value in old.items():
                if key != 'group': self.assertEqual(new[key], value)
            self.assertEqual(new['frozen_modules'], ['long_history', 'state_estimator'])
            self.assertEqual(new['trainable_modules'], ['actor', 'critic', 'std'])
            experiment = ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_mu_freeze_'+group+'.json'))
            source = validate_mu(experiment)
            self.assertEqual(experiment.cfg['reward'], source.cfg['reward'])
            bad = copy.copy(experiment)
            bad.cfg = copy.deepcopy(experiment.cfg)
            bad.cfg['mu_temporal']['trainable_modules'] = ['actor']
            with self.assertRaises(ValueError): validate_mu(bad)

    def test_frozen_report_cannot_omit_real_gradients_or_optimizer_invariant(self):
        spec = mu_contract('freeze_temporal')
        report = dict(updates=10, teacher_unchanged=True, metadata_observed=True,
            valid_triplets=100, all_finite=True, anchor_coef=1., temporal_coef=.01,
            anchor_loss=.001, temporal_loss=.2, weighted_anchor_loss=.001,
            weighted_temporal_loss=.002, aux_grad_norm=.02)
        # Legacy mu evidence alone is insufficient for a frozen-feature run.
        with self.assertRaises(ValueError): validate_mu_loss_report(report, spec, 10)

    def test_full_source_adam_restores_then_freezes_and_updates_only_seventeen_slots(self):
        if not SOURCE.is_file(): self.skipTest('Actual immutable control2500 absent')
        actor_type, ppo_type = cpu_ppo_classes(ROOT)
        cfg = config_scope()['X1AMPRecoveryCfgPPO']()
        original = torch.load(SOURCE, weights_only=True, map_location='cpu')
        for group in ('anchor', 'temporal'):
            with self.subTest(group=group), contextlib.redirect_stdout(io.StringIO()):
                experiment = ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_mu_freeze_'+group+'.json'))
                actor = actor_type(235, 47, 219, 12, **plain(cfg.policy))
                ppo = ppo_type(actor, device='cpu', **plain(cfg.algorithm))
                ppo.init_storage(2, 24, [3102], [219], [12])
                discriminator = AMPDiscriminator(experiment.spec, experiment.mean, experiment.std)
                trainer = DiscriminatorTrainer(discriminator, bridge_gradient_penalty=1.)
                bridge = AMPBridge(experiment.spec, trainer, 2, experiment.cfg['reward'], 'cpu', experiment.cfg['recovery'])
                adapter = AMPAlgorithmAdapter(ppo, bridge, None, experiment)
                runner = SimpleNamespace(device='cpu', alg=adapter, amp_trainer=trainer)
                proof = warm_start_mu(runner, experiment, SOURCE)
                self.assertEqual(proof['frozen_parameter_count'], 16)
                self.assertEqual(proof['trainable_parameter_count'], 17)
                self.assertEqual(len(ppo.optimizer.param_groups[0]['params']), 33)
                before_actor = actor.actor[0].weight.detach().clone()
                actor.train()  # Native runner does this after restoration.
                torch.manual_seed(31)
                for tick in range(24):
                    ppo.set_mu_metadata(**metadata(2, tick))
                    obs, critic = torch.randn(2, 3102)*.1, torch.randn(2, 219)*.1
                    with torch.inference_mode():
                        ppo.act(obs, critic)
                        ppo.process_env_step(torch.ones(2)*.03, torch.zeros(2, dtype=torch.bool), {})
                with torch.inference_mode(): ppo.compute_returns(torch.zeros(2, 219))
                ppo.update()
                report = ppo.mu_latest
                validate_gradient_report(report)
                self.assertEqual(report['gradient_minibatches'], 8)
                self.assertEqual(report['es_grad_norm'], 0.)
                self.assertGreater(report['combined_grad_norm_preclip'], 0.)
                self.assertFalse(torch.equal(before_actor, actor.actor[0].weight))
                self.assertTrue(ppo.mu_regularizer.teacher_unchanged())
                actual = dict(model_state_dict=actor.state_dict(), optimizer_state_dict=ppo.optimizer.state_dict(),
                              es_optimizer_state_dict=ppo.state_estimator_optimizer.state_dict())
                self.assertTrue(validate_frozen_checkpoint(actual, SOURCE, report, updates=1))
                with self.assertRaises(ValueError): validate_frozen_checkpoint(actual, SOURCE, report, updates=2)
                names = list(actor.state_dict())
                slots = actual['optimizer_state_dict']['param_groups'][0]['params']
                for name, slot in zip(names, slots):
                    old = float(original['optimizer_state_dict']['state'][slot]['step'])
                    new = float(actual['optimizer_state_dict']['state'][slot]['step'])
                    frozen = name.startswith(('long_history.', 'state_estimator.'))
                    self.assertEqual(new, old if frozen else old+8)
                broken = copy.deepcopy(actual)
                broken['model_state_dict']['state_estimator.0.weight'][0, 0] += .001
                with self.assertRaises(ValueError): validate_frozen_checkpoint(broken, SOURCE, report)
                broken = copy.deepcopy(actual)
                broken['optimizer_state_dict']['param_groups'][0]['lr'] *= 2
                with self.assertRaises(ValueError): validate_frozen_checkpoint(broken, SOURCE, report)
                broken = copy.deepcopy(actual)
                slot = slots[names.index('long_history.0.weight')]
                broken['optimizer_state_dict']['state'][slot]['step'] += 1
                with self.assertRaises(ValueError): validate_frozen_checkpoint(broken, SOURCE, report)


if __name__ == '__main__': unittest.main()
