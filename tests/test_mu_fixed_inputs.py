"""Synthetic math/protocol tests, not training or robot-quality evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.amp.audit_mu_fixed_inputs import (
    BASELINE_BUNDLE_SHA, FORMAL_COMMIT, SOURCE_SHA, SPANS, RUNS, action_parity,
    equal_initial_means, mu_contract, span_metrics, validate_candidate_binding,
    validate_complete_cohorts, validate_fixed_source_shas, write_report,
)


class FixedInputTests(unittest.TestCase):
    def test_matching_arbitrary_source_files_cannot_replace_fixed_source(self):
        validate_fixed_source_shas(BASELINE_BUNDLE_SHA, SOURCE_SHA)
        with self.assertRaises(ValueError):
            validate_fixed_source_shas('a'*64, SOURCE_SHA)
        with self.assertRaises(ValueError):
            validate_fixed_source_shas(BASELINE_BUNDLE_SHA, 'a'*64)

    def test_candidate_full_identity_group_and_formal_commit_are_bound(self):
        identity={'experiment': 'fixture', 'config_sha256': 'b'*64}
        source_model_hash='c'*64
        training=dict(identity=identity.copy(), code_commit=FORMAL_COMMIT,
            continuation=dict(mu_contract=mu_contract('anchor'), source_sha256=SOURCE_SHA,
                source_completed_updates=2500, target_completed_updates=2750, teacher_frozen=True,
                teacher_source_model_sha256=source_model_hash, student_initial_model_sha256=source_model_hash))
        validate_candidate_binding('081', identity, training, identity.copy(), source_model_hash)
        for changed in ('group', 'commit', 'identity', 'teacher'):
            invalid=copy.deepcopy(training)
            if changed == 'group':
                invalid['continuation']['mu_contract']=mu_contract('temporal')
            elif changed == 'commit':
                invalid['code_commit']='d'*40
            elif changed == 'identity':
                invalid['identity']['config_sha256']='e'*64
            else:
                invalid['continuation']['teacher_source_model_sha256']='f'*64
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_candidate_binding('081', identity, invalid, identity.copy(), source_model_hash)
        with self.assertRaises(ValueError):
            validate_candidate_binding('082', identity, training, identity.copy(), source_model_hash)

    def test_every_comparison_has_complete_identical32_unique_initial_states(self):
        rows=[dict(span=span, run=run, mode=mode, env=index, metrics=dict(complete_window=True))
            for span in SPANS for run in RUNS for mode in ('standing', 'reference') for index in range(16)]
        validate_complete_cohorts(rows)
        with self.assertRaises(ValueError):
            validate_complete_cohorts(rows[:-1])
        duplicate=copy.deepcopy(rows)
        duplicate[0]['env']=1
        with self.assertRaises(ValueError):
            validate_complete_cohorts(duplicate)
        incomplete=copy.deepcopy(rows)
        incomplete[0]['metrics']['complete_window']=False
        with self.assertRaises(ValueError):
            validate_complete_cohorts(incomplete)

    def test_stencil_rejects_both_outside_window_frames(self):
        time=np.arange(199, 205, dtype=np.float64)/100
        mu=(time**2)[:, None]
        mu[0] += 100
        mu[-1] -= 100
        original=mu.copy()
        result=span_metrics(time, mu, mu.copy(), np.array([4.]), 2., 2.04, 1000.)
        self.assertEqual(result['decision_frames'], 4)
        self.assertEqual(result['triplets'], 2)
        self.assertTrue(result['complete_window'])
        self.assertAlmostEqual(result['qmu_second_difference_rms_rad_s2'], 1., places=8)
        self.assertAlmostEqual(result['normalized_temporal_loss'], 1/16, places=8)
        self.assertAlmostEqual(result['qmu_second_difference_absolute_p95_per_joint_rad_s2'][0], 1., places=8)
        np.testing.assert_array_equal(mu, original)

    def test_parity_checks_applied_clipping_without_destroying_raw_mean(self):
        raw=np.array([[2., -2.], [1., -1.]])
        original=raw.copy()
        result=action_parity(raw, np.array([[1., -1.], [1., -1.]]), 1.)
        self.assertTrue(result['passed'])
        self.assertEqual(result['source_raw_mu_clip_count'], 2)
        self.assertEqual(result['source_raw_mu_boundary_count'], 4)
        np.testing.assert_array_equal(raw, original)
        with self.assertRaises(ValueError):
            action_parity(np.zeros((2, 1)), np.full((2, 1), .001), 1.)

    def test_metrics_use_unclipped_means_and_keep_clip_counts(self):
        time=np.arange(10, dtype=np.float64)/100
        raw=np.array([0., 4., 0., 4., 0., 4., 0., 4., 0., 4.])[:, None]
        result=span_metrics(time, raw, np.zeros_like(raw), np.array([1.]), 0., .1, 1.)
        self.assertEqual(result['raw_mu_clip_count'], 5)
        self.assertEqual(result['raw_mu_max_per_joint'], [4.])
        self.assertAlmostEqual(result['qmu_second_difference_rms_rad_s2'], 40000.)

    def test_missing_triplets_remain_null_and_equal_initial_means_are_not_sample_weighted(self):
        empty=span_metrics(np.array([.1, .11]), np.zeros((2, 1)), np.zeros((2, 1)), np.ones(1), .1, .12, 1.)
        self.assertIsNone(empty['normalized_temporal_loss'])
        rows=[]
        for env, value, count in ((0, 1., 1000), (1, 13., 3), (2, None, 0)):
            rows.append(dict(mode='standing', env=env, span='post2s', run='source',
                metrics=dict(normalized_temporal_loss=value, qmu_second_difference_rms_rad_s2=value,
                             anchor_pd_target_rms_rad=value, triplets=count)))
        result=equal_initial_means(rows)['post2s']['source']['normalized_temporal_loss']
        self.assertEqual(result['mean'], 7.)
        self.assertEqual(result['contributing_initial_states'], 2)
        self.assertIsNone(equal_initial_means(rows)['post2s']['081']['normalized_temporal_loss']['mean'])

    def test_nonconsecutive_history_time_is_rejected(self):
        with self.assertRaises(ValueError):
            span_metrics(np.array([0., .01, .03]), np.zeros((3, 1)), np.zeros((3, 1)), np.ones(1), 0., .04, 1.)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'diagnosis'
            write_report(output, {'sentinel': 1})
            before=(output/'report.json').read_bytes()
            with self.assertRaises(FileExistsError):
                write_report(output, {'sentinel': 2})
            self.assertEqual((output/'report.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
