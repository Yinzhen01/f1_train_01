"""Synthetic artifact/log tamper tests, not simulator or training evidence."""
import copy
import json
import math
import unittest

from humanoid.amp.feature_freeze import (COMPONENT_GRADIENT_FIELDS,
    COMPONENT_GRADIENT_SCHEMA, GRADIENT_FIELDS)
from humanoid.amp.mu_temporal import mu_contract
from test_amp_mu_audit import fixture as legacy_fixture
from tools.amp.mu_audit import (validate_mu_updates, validate_mu_loss_report,
    validate_mu_checkpoint_report, PPO_EPOCHS, PPO_MINIBATCHES, PPO_ROLLOUT_STEPS)


ALL_FIELDS = GRADIENT_FIELDS + COMPONENT_GRADIENT_FIELDS
GROUPS = ('freeze_split_temporal01', 'freeze_split_temporal05')


def record(update, minibatch, valid, batch_size, coefficient):
    """Finite scalar evidence from explicitly chosen toy gradient vectors."""
    main = (2. + (update-2501)*.001 + minibatch*.002, 1.)
    anchor = (.3 + minibatch*.001, .1)
    temporal = (.02*coefficient/.01, -.01*coefficient/.01) if valid else (0., 0.)
    auxiliary = tuple(a+b for a, b in zip(anchor, temporal))

    def norm(values):
        return math.sqrt(sum(value*value for value in values))

    def cosine(left, right):
        denominator = norm(left)*norm(right)
        return sum(a*b for a, b in zip(left, right))/denominator if denominator else 0.

    combined = math.sqrt(sum((a+b)**2 for a, b in zip(main, auxiliary))+1.)
    return dict(completed_update=update, minibatch_index=minibatch,
        batch_size=batch_size, valid_triplets=valid,
        main_grad_norm=math.sqrt(norm(main)**2+1.), aux_grad_norm_all_batches=norm(auxiliary),
        es_grad_norm=0., actor_main_grad_norm=norm(main), actor_aux_grad_norm=norm(auxiliary),
        actor_main_aux_cosine=cosine(main, auxiliary), actor_cosine_valid_fraction=1.,
        combined_grad_norm_preclip=combined, global_clip_factor=min(1., 1./(combined+1e-6)),
        actor_anchor_grad_norm=norm(anchor), actor_temporal_grad_norm=norm(temporal),
        actor_main_anchor_cosine=cosine(main, anchor),
        actor_main_temporal_cosine=cosine(main, temporal),
        actor_anchor_temporal_cosine=cosine(anchor, temporal),
        actor_main_anchor_cosine_valid_fraction=1.,
        actor_main_temporal_cosine_valid_fraction=float(bool(valid)),
        actor_anchor_temporal_cosine_valid_fraction=float(bool(valid)))


def set_means(report, records, prefix=''):
    for key in ALL_FIELDS:
        report[prefix+key] = sum(item[key] for item in records)/len(records)
    report[prefix+'gradient_minibatches'] = len(records)
    report[prefix+'component_gradient_schema'] = COMPONENT_GRADIENT_SCHEMA
    report[prefix+'component_gradient_records'] = copy.deepcopy(records)


def fixture(group=GROUPS[0], formal=False):
    spec, rows, report, continuation = legacy_fixture(group, formal)
    size = 4096 if formal else 32
    records = []
    for row, diagnostics in zip(rows, report['rollout_diagnostics']):
        triplets = row['mu_valid_triplets']
        actual = [record(row['iteration'], minibatch, triplets//4,
                         size*24//4, spec['temporal_coef']) for minibatch in range(1, 9)]
        set_means(row, actual, 'mu_')
        row.update(mu_frozen_features_unchanged=True, mu_frozen_optimizer_unchanged=True)
        diagnostics.update(total_samples=size*24, candidate_triplets=size*22,
            invalid_triplets=size*24-triplets)
        diagnostics['invalid_reasons'].update(rollout_boundary=size*2, startup=size*22-triplets)
        records.extend(actual)
    set_means(report, records)
    report.update(frozen_features_unchanged=True, frozen_optimizer_unchanged=True,
        frozen_parameter_count=16, trainable_parameter_count=17,
        frozen_features_initial_sha256=spec['source_feature_state_sha256'],
        frozen_features_final_sha256=spec['source_feature_state_sha256'])
    continuation.update(frozen_features_initial_sha256=spec['source_feature_state_sha256'])
    return spec, rows, report, continuation


class MuComponentAuditTests(unittest.TestCase):
    def test_exact_short_and_formal_records_and_all_means_are_accepted(self):
        for group in GROUPS:
            for formal in (False, True):
                with self.subTest(group=group, formal=formal):
                    spec, rows, report, continuation = fixture(group, formal)
                    self.assertTrue(validate_mu_updates(rows, group, formal, report))
                    self.assertTrue(validate_mu_loss_report(report, spec, len(rows), continuation,
                        rows, num_envs=4096 if formal else 32, rollout_steps=24))
                    self.assertEqual(len(report['component_gradient_records']), 2000 if formal else 80)
                    json.dumps(report, allow_nan=False)
                    self.assertFalse(report.get('effectiveness_verified', False))

    def test_every_actual_minibatch_field_is_required(self):
        _, rows, _, _ = fixture()
        for key in rows[3]['mu_component_gradient_records'][0]:
            bad = copy.deepcopy(rows)
            bad[3]['mu_component_gradient_records'][0].pop(key)
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])

    def test_missing_extra_malformed_or_reordered_records_are_rejected(self):
        _, rows, _, _ = fixture()
        variants = (None, {}, (), [], rows[3]['mu_component_gradient_records'][:-1],
                    rows[3]['mu_component_gradient_records']+[rows[3]['mu_component_gradient_records'][-1]])
        for value in variants:
            bad = copy.deepcopy(rows)
            bad[3]['mu_component_gradient_records'] = value
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])
        for mutate in (lambda records: records.reverse(),
                       lambda records: records[0].update(extra=1.),
                       lambda records: records.__setitem__(1, copy.deepcopy(records[0])),
                       lambda records: records.__setitem__(0, 'not a record')):
            bad = copy.deepcopy(rows)
            mutate(bad[3]['mu_component_gradient_records'])
            with self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])

    def test_update_index_minibatch_index_and_batch_metadata_are_exact(self):
        _, rows, _, _ = fixture()
        mutations = {'completed_update': (2501, 2504., True),
            'minibatch_index': (0, 2, 1., True), 'batch_size': (0, 191, 193, 192., True),
            'valid_triplets': (-1, 193, 25., True)}
        for key, values in mutations.items():
            for value in values:
                bad = copy.deepcopy(rows)
                bad[3]['mu_component_gradient_records'][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_mu_updates(bad, GROUPS[0])
        for value in (7, 8., True, None):
            bad = copy.deepcopy(rows)
            bad[3]['mu_gradient_minibatches'] = value
            with self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])

    def test_all_gradient_scalars_reject_nonfinite_or_non_numeric_values(self):
        _, rows, _, _ = fixture()
        for key in ALL_FIELDS:
            for value in (float('nan'), float('inf'), -float('inf'), True, '1', None):
                bad = copy.deepcopy(rows)
                bad[3]['mu_component_gradient_records'][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_mu_updates(bad, GROUPS[0])

    def test_actual_records_keep_pure_float_scalars_not_integer_substitutes(self):
        _, rows, _, _ = fixture()
        for key in ALL_FIELDS:
            bad = copy.deepcopy(rows)
            bad[3]['mu_component_gradient_records'][0][key] = 0
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])

    def test_cosine_validity_zero_gradient_and_frozen_es_consistency(self):
        _, rows, _, _ = fixture()
        mutations = (dict(actor_main_temporal_cosine=1.1),
            dict(actor_main_temporal_cosine_valid_fraction=.5),
            dict(actor_main_temporal_cosine_valid_fraction=0.),
            dict(actor_temporal_grad_norm=0.), dict(es_grad_norm=.0001))
        for changes in mutations:
            bad = copy.deepcopy(rows)
            bad[3]['mu_component_gradient_records'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])

    def test_every_per_update_and_run_gradient_mean_is_bound_to_records(self):
        _, rows, report, _ = fixture()
        for key in ALL_FIELDS:
            bad = copy.deepcopy(rows)
            bad[3]['mu_'+key] += .01
            with self.subTest(update_field=key), self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])
            bad_report = copy.deepcopy(report)
            bad_report[key] += .01
            with self.subTest(run_field=key), self.assertRaises(ValueError):
                validate_mu_updates(rows, GROUPS[0], report=bad_report)

    def test_component_schema_is_required_in_every_update_and_final_report(self):
        spec, rows, report, continuation = fixture()
        for value in (None, False, 'unknown', 'weighted_actor_split_v2'):
            bad = copy.deepcopy(rows)
            bad[3]['mu_component_gradient_schema'] = value
            with self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])
            bad_report = copy.deepcopy(report)
            bad_report['component_gradient_schema'] = value
            with self.assertRaises(ValueError):
                validate_mu_loss_report(bad_report, spec, 10, continuation, rows)

    def test_valid_triplets_reconcile_each_rollout_and_both_native_epochs(self):
        spec, rows, report, continuation = fixture()
        bad = copy.deepcopy(rows)
        bad[3]['mu_component_gradient_records'][0]['valid_triplets'] += 1
        with self.assertRaises(ValueError):
            validate_mu_updates(bad, GROUPS[0])
        bad_report = copy.deepcopy(report)
        bad_report['component_gradient_records'][3*8]['valid_triplets'] += 1
        with self.assertRaises(ValueError):
            validate_mu_loss_report(bad_report, spec, 10, continuation)
        # Same run total, but forged movement between updates must still fail.
        bad_report = copy.deepcopy(report)
        bad_report['component_gradient_records'][3*8]['valid_triplets'] += 1
        bad_report['component_gradient_records'][4*8]['valid_triplets'] -= 1
        with self.assertRaises(ValueError):
            validate_mu_loss_report(bad_report, spec, 10, continuation)

    def test_native_inherited_ppo_shape_not_the_eight_batch_product_alone(self):
        from test_f1_amp_recovery import config_scope
        actual = config_scope()['X1AMPRecoveryCfgPPO']()
        self.assertEqual((actual.algorithm.num_learning_epochs,
            actual.algorithm.num_mini_batches, actual.runner.num_steps_per_env),
            (PPO_EPOCHS, PPO_MINIBATCHES, PPO_ROLLOUT_STEPS))
        self.assertEqual((PPO_EPOCHS, PPO_MINIBATCHES, PPO_ROLLOUT_STEPS), (2, 4, 24))
        self.assertEqual(32*PPO_ROLLOUT_STEPS//PPO_MINIBATCHES, 192)
        self.assertEqual(4096*PPO_ROLLOUT_STEPS//PPO_MINIBATCHES, 24576)

    def test_wrong_four_epoch_two_minibatch_fixture_is_rejected(self):
        for formal in (False, True):
            for batch_factor, triplet_factor in ((2, 1), (1, 2), (2, 2)):
                spec, rows, report, continuation = fixture(formal=formal)
                for row in rows:
                    records = row['mu_component_gradient_records']
                    for item in records:
                        item['batch_size'] *= batch_factor
                        item['valid_triplets'] *= triplet_factor
                report['component_gradient_records'] = [copy.deepcopy(item)
                    for row in rows for item in row['mu_component_gradient_records']]
                with self.subTest(formal=formal, batch_factor=batch_factor,
                                  triplet_factor=triplet_factor), self.assertRaises(ValueError):
                    validate_mu_updates(rows, GROUPS[0], formal, report)
                with self.assertRaises(ValueError):
                    validate_mu_loss_report(report, spec, len(rows), continuation)

    def test_independent_contract_fixture_also_passes_native_audit(self):
        # Cross-check the separately maintained admission fixture; this suite
        # must not merely agree with its own chosen batch/epoch arithmetic.
        from test_f1_amp_mu_component import MuComponentContractTests
        contract = MuComponentContractTests()
        MuComponentContractTests.setUpClass()
        for group in GROUPS:
            experiment, certificate = contract.certificate(group)
            self.assertTrue(validate_mu_loss_report(certificate['mu_loss_report'],
                experiment.cfg['mu_temporal'], 10, certificate['continuation'],
                num_envs=32, rollout_steps=24))

    def test_final_records_exactly_equal_actual_ordered_log_records(self):
        _, rows, report, _ = fixture()
        bad_report = copy.deepcopy(report)
        # Swap two numeric records while preserving their update/mini-batch
        # labels and every scalar mean. Counts and aggregates alone cannot see it.
        left, right = bad_report['component_gradient_records'][3*8:3*8+2]
        for key in ALL_FIELDS:
            left[key], right[key] = right[key], left[key]
        with self.assertRaises(ValueError):
            validate_mu_updates(rows, GROUPS[0], report=bad_report)

    def test_saved_checkpoint_records_cannot_differ_from_final_certificate(self):
        spec, _, report, continuation = fixture()
        saved = copy.deepcopy(report)
        saved.pop('teacher_final_model_sha256')
        self.assertTrue(validate_mu_checkpoint_report(saved, report, spec, 10, continuation))
        left, right = saved['component_gradient_records'][3*8:3*8+2]
        for key in ALL_FIELDS:
            left[key], right[key] = right[key], left[key]
        with self.assertRaises(ValueError):
            validate_mu_checkpoint_report(saved, report, spec, 10, continuation)

    def test_legacy_mu_groups_without_component_schema_keep_previous_behavior(self):
        for group in ('anchor', 'temporal'):
            spec, rows, report, continuation = legacy_fixture(group)
            self.assertNotIn('component_gradient_schema', spec)
            self.assertTrue(validate_mu_updates(rows, group, report=report))
            self.assertTrue(validate_mu_loss_report(report, spec, 10, continuation, rows))

    def test_original_mu_gates_still_reject_teacher_continuity_or_amp_tampering(self):
        spec, rows, report, continuation = fixture()
        for changes in (dict(mu_teacher_unchanged=False), dict(mu_frozen_features_unchanged=False),
                dict(mu_frozen_optimizer_unchanged=False), dict(mu_temporal_coef=.5),
                dict(weighted_style_reward=0.), dict(discriminator_bridge_gradient_penalty=0.)):
            bad = copy.deepcopy(rows)
            bad[3].update(changes)
            with self.assertRaises(ValueError):
                validate_mu_updates(bad, GROUPS[0])
        with self.assertRaises(ValueError):
            validate_mu_loss_report(dict(report, teacher_final_model_sha256='b'*64),
                spec, 10, continuation, rows)


if __name__ == '__main__':
    unittest.main()
