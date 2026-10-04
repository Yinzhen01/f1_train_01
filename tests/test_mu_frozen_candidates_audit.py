"""Offline audit routing tests; synthetic fixtures are not motion evidence."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humanoid.amp.mu_temporal import mu_contract
from humanoid.amp.scaled_experiment import ScaledExperiment
from tools.amp.audit_mu_candidates import candidate_checkpoint_proof, parse_arguments


ROOT = Path(__file__).resolve().parents[1]


def required_arguments():
    paths = ('source-endpoint', 'source-physics', 'source-checkpoint', 'anchor-bundle',
             'anchor-checkpoint', 'temporal-bundle', 'temporal-checkpoint', 'output')
    return [item for name in paths for item in ('--'+name, name+'.pt')]+[
        '--expected-commit', '0'*40]


class FrozenCandidateArgumentsTests(unittest.TestCase):
    def test_defaults_preserve_old_groups_and_case_argument_names(self):
        args = parse_arguments(required_arguments())
        self.assertEqual((args.anchor_group, args.temporal_group), ('anchor', 'temporal'))
        self.assertEqual(args.anchor_bundle, Path('anchor-bundle.pt'))
        self.assertEqual(args.temporal_bundle, Path('temporal-bundle.pt'))

    def test_frozen_groups_require_explicit_selection(self):
        args = parse_arguments(required_arguments()+[
            '--anchor-group', 'freeze_anchor', '--temporal-group', 'freeze_temporal'])
        self.assertEqual((args.anchor_group, args.temporal_group),
                         ('freeze_anchor', 'freeze_temporal'))

    def test_each_case_refuses_the_other_family_or_an_unknown_group(self):
        for flag, group in (('anchor-group', 'temporal'), ('anchor-group', 'freeze_temporal'),
                            ('temporal-group', 'anchor'), ('temporal-group', 'freeze_anchor'),
                            ('anchor-group', 'unknown')):
            with self.subTest(flag=flag, group=group), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_arguments(required_arguments()+['--'+flag, group])

    def test_registered_configuration_identities_are_not_interchangeable(self):
        identities = []
        for group in ('anchor', 'temporal', 'freeze_anchor', 'freeze_temporal'):
            experiment = ScaledExperiment(ROOT, ROOT/('configs/amp/lafan_walk02_mu_'+group+'.json'))
            self.assertEqual(experiment.cfg['mu_temporal'], mu_contract(group))
            identities.append(experiment.identity())
        self.assertEqual(len({item['config_sha256'] for item in identities}), 4)
        self.assertEqual(len({item['experiment'] for item in identities}), 4)


class FrozenCandidateCheckpointTests(unittest.TestCase):
    def fixture(self, group):
        experiment = SimpleNamespace(cfg={'mu_temporal': mu_contract(group)})
        return experiment, {'synthetic': 'loss-report'}, {'synthetic': 'actual-checkpoint'}

    def test_legacy_groups_keep_existing_loss_proof_without_frozen_load(self):
        for group in ('anchor', 'temporal'):
            experiment, report, actual = self.fixture(group)
            with self.subTest(group=group), \
                    patch('tools.amp.audit_mu_candidates.checkpoint_proof',
                          return_value=({'completed_updates': 2750}, report)) as proof, \
                    patch('tools.amp.audit_mu_candidates.validate_mu_loss_report') as loss, \
                    patch('tools.amp.audit_mu_candidates.torch.load') as load, \
                    patch('tools.amp.audit_mu_candidates.validate_frozen_checkpoint') as frozen:
                result = candidate_checkpoint_proof(Path('candidate.pt'), {}, experiment, Path('source.pt'))
                proof.assert_called_once_with(Path('candidate.pt'), {}, 2750)
                loss.assert_called_once_with(report, experiment.cfg['mu_temporal'], 250)
                load.assert_not_called()
                frozen.assert_not_called()
                self.assertEqual(result, {'completed_updates': 2750})

    def test_frozen_groups_check_actual_checkpoint_against_source_with_full_budget(self):
        for group in ('freeze_anchor', 'freeze_temporal'):
            experiment, report, actual = self.fixture(group)
            with self.subTest(group=group), \
                    patch('tools.amp.audit_mu_candidates.checkpoint_proof',
                          return_value=({'completed_updates': 2750}, report)), \
                    patch('tools.amp.audit_mu_candidates.validate_mu_loss_report') as loss, \
                    patch('tools.amp.audit_mu_candidates.torch.load', return_value=actual) as load, \
                    patch('tools.amp.audit_mu_candidates.validate_frozen_checkpoint') as frozen:
                result = candidate_checkpoint_proof(Path('candidate.pt'), {}, experiment, Path('source.pt'))
                loss.assert_called_once_with(report, experiment.cfg['mu_temporal'], 250)
                load.assert_called_once_with(Path('candidate.pt'), weights_only=True, map_location='cpu')
                frozen.assert_called_once_with(actual, Path('source.pt'), report, updates=250)
                self.assertIs(result['frozen_feature_checkpoint_verified'], True)

    def test_frozen_checkpoint_failure_cannot_be_reported_as_verified(self):
        experiment, report, actual = self.fixture('freeze_temporal')
        with patch('tools.amp.audit_mu_candidates.checkpoint_proof',
                   return_value=({'completed_updates': 2750}, report)), \
                patch('tools.amp.audit_mu_candidates.validate_mu_loss_report'), \
                patch('tools.amp.audit_mu_candidates.torch.load', return_value=actual), \
                patch('tools.amp.audit_mu_candidates.validate_frozen_checkpoint',
                      side_effect=ValueError('Frozen Adam step differs from source')):
            with self.assertRaisesRegex(ValueError, 'Frozen Adam'):
                candidate_checkpoint_proof(Path('candidate.pt'), {}, experiment, Path('source.pt'))

    def test_loss_contract_failure_is_rejected_before_frozen_checkpoint_load(self):
        experiment, report, actual = self.fixture('freeze_anchor')
        with patch('tools.amp.audit_mu_candidates.checkpoint_proof',
                   return_value=({'completed_updates': 2750}, report)), \
                patch('tools.amp.audit_mu_candidates.validate_mu_loss_report',
                      side_effect=ValueError('Invalid gradient evidence')), \
                patch('tools.amp.audit_mu_candidates.torch.load') as load, \
                patch('tools.amp.audit_mu_candidates.validate_frozen_checkpoint') as frozen:
            with self.assertRaisesRegex(ValueError, 'gradient'):
                candidate_checkpoint_proof(Path('candidate.pt'), {}, experiment, Path('source.pt'))
            load.assert_not_called()
            frozen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
