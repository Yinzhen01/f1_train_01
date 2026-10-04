"""Offline routing/guard tests with synthetic reports, never training evidence."""
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from humanoid.amp.feature_freeze import GRADIENT_FIELDS
from tools.amp.audit_mu_fixed_inputs import (
    FORMAL_COMMIT, GROUPS, LEGACY_GRADIENT_LIMITATION, SOURCE_SHA,
    build_parser, candidate_groups, gradient_evidence_limitations, mu_contract,
    validate_candidate_binding, validate_freeze_candidate,
)


FREEZE_COMMIT = '0ff3e61e36c34bb3fe9908481bae7a2a403c4223'


def binding_fixture(group, commit):
    spec = mu_contract(group)
    identity = {'experiment': 'synthetic_'+group, 'config_sha256': 'a'*64}
    full_hash = spec.get('source_model_state_sha256', 'c'*64)
    proof = dict(mu_contract=spec, source_sha256=SOURCE_SHA,
        source_completed_updates=2500, target_completed_updates=2750,
        teacher_frozen=True, teacher_source_model_sha256=full_hash,
        student_initial_model_sha256=full_hash)
    return identity, dict(identity=identity, code_commit=commit, continuation=proof), full_hash


def formal_fixture(group='freeze_anchor'):
    identity, training, full_hash = binding_fixture(group, FREEZE_COMMIT)
    spec = training['continuation']['mu_contract']
    proof = dict(frozen_features_unchanged=True, frozen_optimizer_unchanged=True,
        frozen_parameter_count=16, trainable_parameter_count=17,
        frozen_features_initial_sha256=spec['source_feature_state_sha256'],
        frozen_features_final_sha256=spec['source_feature_state_sha256'])
    training['continuation'].update(proof)
    report = dict(proof, updates=250, teacher_unchanged=True, metadata_observed=True,
        all_finite=True, anchor_coef=1., temporal_coef=spec['temporal_coef'],
        anchor_loss=.1, temporal_loss=.2, weighted_anchor_loss=.1,
        weighted_temporal_loss=.2*spec['temporal_coef'], aux_grad_norm=.1,
        gradient_minibatches=2000, teacher_final_model_sha256=full_hash)
    report.update({key: .1 for key in GRADIENT_FIELDS})
    report.update(es_grad_norm=0., actor_main_aux_cosine=.2,
        actor_cosine_valid_fraction=1., global_clip_factor=.5)
    # Synthetic continuity counts only: they exercise the exact 4096x24 budget.
    item = dict(metadata_observed=True, invalid_reason_counts_overlap=True,
        startup_steps=66, total_samples=4096*24, candidate_triplets=4096*22,
        valid_triplets=4096*22, invalid_triplets=4096*2,
        invalid_reasons=dict(rollout_boundary=4096*2, previous_done=0,
            episode_changed=0, tick_discontinuity=0, command_changed=0, startup=0))
    report.update(valid_triplets=250*item['valid_triplets'],
        rollout_diagnostics=[copy.deepcopy(item) for _ in range(250)])
    training.update(mode='formal', updates=250, num_envs=4096,
        ppo_config={'runner': {'num_steps_per_env': 24}})
    training['completion'] = dict(identity=identity, code_commit=FREEZE_COMMIT,
        updates=250, num_envs=4096, continuation=copy.deepcopy(training['continuation']),
        mu_loss_report=report)
    saved_report = copy.deepcopy(report)
    saved_report.pop('teacher_final_model_sha256')
    checkpoint = {'mu_loss_report': saved_report, 'sentinel': 'actual checkpoint argument'}
    experiment = SimpleNamespace(cfg={'mu_temporal': spec})
    return checkpoint, training, experiment


class FixedInputFreezeTests(unittest.TestCase):
    def test_default_arguments_keep_old_paths_groups_and_exact_commit(self):
        args = build_parser().parse_args(['--output', 'synthetic-output'])
        self.assertEqual(args.anchor_group, 'anchor')
        self.assertEqual(args.temporal_group, 'temporal')
        self.assertEqual(args.expected_commit, FORMAL_COMMIT)
        self.assertEqual(candidate_groups(), GROUPS)
        self.assertEqual(getattr(args, '081_checkpoint').parent.name, 'TASK_20261002_081')
        self.assertEqual(getattr(args, '082_checkpoint').parent.name, 'TASK_20261002_082')
        self.assertEqual(args.source_checkpoint.name, 'model_8802500.pt')

    def test_new_formal_paths_are_explicit_opt_in_without_relabelling_old_defaults(self):
        args = build_parser().parse_args(['--output', 'synthetic-output',
            '--anchor-group', 'freeze_anchor', '--temporal-group', 'freeze_temporal',
            '--expected-commit', FREEZE_COMMIT,
            '--081-checkpoint', 'TASK_20261002_117/model_8802750.pt',
            '--082-checkpoint', 'TASK_20261002_118/model_8802750.pt'])
        self.assertEqual(candidate_groups(args.anchor_group, args.temporal_group, args.expected_commit),
            {'081': 'freeze_anchor', '082': 'freeze_temporal'})
        self.assertEqual(getattr(args, '081_checkpoint').parent.name, 'TASK_20261002_117')
        self.assertEqual(getattr(args, '082_checkpoint').parent.name, 'TASK_20261002_118')

    def test_roles_unknown_groups_and_short_commit_are_rejected(self):
        for anchor, temporal, commit in (('freeze_temporal', 'temporal', FREEZE_COMMIT),
                ('anchor', 'freeze_anchor', FREEZE_COMMIT), ('unknown', 'temporal', FREEZE_COMMIT),
                ('freeze_anchor', 'freeze_temporal', FREEZE_COMMIT[:7]),
                ('freeze_anchor', 'freeze_temporal', FREEZE_COMMIT.upper())):
            with self.subTest(anchor=anchor, temporal=temporal, commit=commit), self.assertRaises(ValueError):
                candidate_groups(anchor, temporal, commit)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(['--output', 'synthetic-output', '--anchor-group', 'freeze_temporal'])

    def test_old_binding_defaults_remain_hard_bound_to_old_contract_commit_and_role(self):
        for run, group in GROUPS.items():
            identity, training, full_hash = binding_fixture(group, FORMAL_COMMIT)
            validate_candidate_binding(run, identity, training, identity.copy(), full_hash)
            bad = copy.deepcopy(training)
            bad['code_commit'] = FREEZE_COMMIT
            with self.subTest(run=run), self.assertRaises(ValueError):
                validate_candidate_binding(run, identity, bad, identity.copy(), full_hash)

    def test_freeze_binding_requires_explicit_group_and_actual_formal_commit(self):
        for run, group in (('081', 'freeze_anchor'), ('082', 'freeze_temporal')):
            identity, training, full_hash = binding_fixture(group, FREEZE_COMMIT)
            validate_candidate_binding(run, identity, training, identity.copy(), full_hash, group, FREEZE_COMMIT)
            with self.subTest(run=run), self.assertRaises(ValueError):
                validate_candidate_binding(run, identity, training, identity.copy(), full_hash)
            with self.subTest(run=run, commit='old'), self.assertRaises(ValueError):
                validate_candidate_binding(run, identity, training, identity.copy(), full_hash, group)

    def test_freeze_contract_and_full_teacher_source_cannot_be_substituted(self):
        identity, training, full_hash = binding_fixture('freeze_anchor', FREEZE_COMMIT)
        for changed in ('frozen_modules', 'coefficient', 'source_feature', 'teacher', 'initial', 'source'):
            bad = copy.deepcopy(training)
            proof = bad['continuation']
            if changed == 'frozen_modules': proof['mu_contract']['frozen_modules'] = ['state_estimator']
            elif changed == 'coefficient': proof['mu_contract']['anchor_coef'] = .1
            elif changed == 'source_feature': proof['mu_contract']['source_feature_state_sha256'] = 'b'*64
            elif changed == 'teacher': proof['teacher_source_model_sha256'] = 'b'*64
            elif changed == 'initial': proof['student_initial_model_sha256'] = 'b'*64
            else: proof['source_sha256'] = 'b'*64
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_candidate_binding('081', identity, bad, identity.copy(), full_hash,
                    'freeze_anchor', FREEZE_COMMIT)

    def test_both_freeze_groups_mandatorily_inspect_actual_checkpoint_and_250_adam_updates(self):
        for group in ('freeze_anchor', 'freeze_temporal'):
            checkpoint, training, experiment = formal_fixture(group)
            source_path = Path('synthetic-not-a-real-source.pt')
            with self.subTest(group=group), patch('tools.amp.mu_audit.validate_frozen_checkpoint',
                    return_value=True) as independent:
                proof = validate_freeze_candidate(checkpoint, training, source_path, experiment)
                independent.assert_called_once_with(checkpoint, source_path,
                    training['completion']['mu_loss_report'], updates=250)
                self.assertTrue(proof['frozen_feature_checkpoint_verified'])
                self.assertEqual(proof['teacher_final_model_sha256'],
                    experiment.cfg['mu_temporal']['source_model_state_sha256'])

    def test_independent_checkpoint_rejection_cannot_be_replaced_by_self_reported_true(self):
        checkpoint, training, experiment = formal_fixture()
        with patch('tools.amp.mu_audit.validate_frozen_checkpoint',
                side_effect=ValueError('Frozen Adam changed')) as independent:
            with self.assertRaisesRegex(ValueError, 'Frozen Adam changed'):
                validate_freeze_candidate(checkpoint, training, Path('synthetic-source.pt'), experiment)
            independent.assert_called_once()

    def test_bad_formal_budget_or_completion_identity_fails_before_tensor_audit(self):
        checkpoint, training, experiment = formal_fixture()
        for key, value in (('mode', 'smoke'), ('updates', 10), ('num_envs', 32),
                           ('completion', None)):
            bad = copy.deepcopy(training)
            bad[key] = value
            with self.subTest(key=key), patch('tools.amp.mu_audit.validate_frozen_checkpoint') as independent:
                with self.assertRaises(ValueError):
                    validate_freeze_candidate(checkpoint, bad, Path('synthetic-source.pt'), experiment)
                independent.assert_not_called()
        for key in ('identity', 'code_commit', 'continuation'):
            bad = copy.deepcopy(training)
            bad['completion'][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_freeze_candidate(checkpoint, bad, Path('synthetic-source.pt'), experiment)

    def test_final_teacher_aggregate_gradients_and_saved_report_must_be_exact(self):
        checkpoint, training, experiment = formal_fixture('freeze_temporal')
        changes = [('teacher_final_model_sha256', 'b'*64), ('gradient_minibatches', 1999),
            ('main_grad_norm', 0.), ('es_grad_norm', .1), ('aux_grad_norm_all_batches', None)]
        for key, value in changes:
            bad = copy.deepcopy(training)
            if value is None: bad['completion']['mu_loss_report'].pop(key)
            else: bad['completion']['mu_loss_report'][key] = value
            with self.subTest(key=key), patch('tools.amp.mu_audit.validate_frozen_checkpoint') as independent:
                with self.assertRaises(ValueError):
                    validate_freeze_candidate(checkpoint, bad, Path('synthetic-source.pt'), experiment)
                independent.assert_not_called()
        bad_checkpoint = copy.deepcopy(checkpoint)
        bad_checkpoint['mu_loss_report']['anchor_loss'] += .1
        with self.assertRaises(ValueError):
            validate_freeze_candidate(bad_checkpoint, training, Path('synthetic-source.pt'), experiment)

    def test_legacy_candidates_do_not_acquire_freeze_checkpoint_requirements(self):
        for group in ('anchor', 'temporal'):
            experiment = SimpleNamespace(cfg={'mu_temporal': mu_contract(group)})
            with self.subTest(group=group), patch('tools.amp.mu_audit.validate_frozen_checkpoint') as independent:
                self.assertIsNone(validate_freeze_candidate({}, {}, None, experiment))
                independent.assert_not_called()

    def test_gradient_descriptions_distinguish_old_first_batch_from_freeze_aggregates(self):
        self.assertEqual(gradient_evidence_limitations(GROUPS), [LEGACY_GRADIENT_LIMITATION])
        frozen = gradient_evidence_limitations(candidate_groups('freeze_anchor', 'freeze_temporal', FREEZE_COMMIT))
        self.assertEqual(len(frozen), 1)
        self.assertNotIn('main PPO gradients are not recorded', frozen[0])
        self.assertIn('every minibatch', frozen[0])
        self.assertIn('aggregates', frozen[0])
        self.assertIn('not individual gradient vectors', frozen[0])
        self.assertIn('aux_grad_norm_all_batches is the aggregate', frozen[0])
        self.assertIn('aux_grad_norm is still the legacy first-minibatch', frozen[0])
        mixed = gradient_evidence_limitations(candidate_groups('anchor', 'freeze_temporal', FREEZE_COMMIT))
        self.assertEqual(len(mixed), 2)
        self.assertIn('081', mixed[0])
        self.assertIn('082', mixed[1])


if __name__ == '__main__':
    unittest.main()
