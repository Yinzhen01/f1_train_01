"""CPU-only mathematical and population guards for the mu candidate audit.

These synthetic fixtures are not recorded robot evidence and do not establish
motion quality, cloud execution, or unclipped policy means at a clip boundary.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.amp.audit_mu_candidates import (
    paired_masks,
    paired_means,
    plot_cases,
    overview,
    rank_worst_initial_states,
    target_second_difference,
    validate_geometry_binding,
)
from tools.amp.audit_physics_diagnostic import common_prefix_samples


def manifest():
    return dict(
        dof_names=['joint_%d' % index for index in range(12)],
        environment=dict(control=dict(action_scale=.5),
                         normalization=dict(clip_actions=100.)),
        runtime=dict(control_dt=.01),
    )


def episode(count, failed=False, initial_failure=False):
    failure = np.zeros(count, dtype=bool)
    if failed and count:
        failure[-1] = True
    return dict(
        time=(np.arange(count, dtype=np.float64)+1)*.01,
        action=np.zeros((count, 12), dtype=np.float64),
        failure=failure,
        initial=dict(action=np.zeros(12, dtype=np.float64),
                     failure=initial_failure),
    )


def physics_arrays(count, mode='standing'):
    return {
        mode+'_physics_valid': np.ones((count, 1), dtype=bool),
        mode+'_physics_foot_force_peak': np.zeros((count, 1, 2), dtype=np.float64),
    }


def initial_state_rows():
    return [dict(mode=mode, env=index, failure=False, observed_s=60.,
                 overview=dict(substep_accel_rms=float(offset+index)))
            for mode, offset in (('standing', 0), ('reference', 16))
            for index in range(16)]


class CommonPrefixTests(unittest.TestCase):
    def test_surviving_pair_keeps_all_6000_ticks(self):
        self.assertEqual(common_prefix_samples(episode(6000), episode(6000)), 6000)

    def test_one_failure_excludes_exactly_its_terminal_second(self):
        self.assertEqual(common_prefix_samples(episode(6000), episode(529, failed=True)), 429)
        self.assertEqual(common_prefix_samples(episode(529, failed=True), episode(6000)), 429)

    def test_two_failures_crop_each_before_intersection_not_twice(self):
        self.assertEqual(common_prefix_samples(episode(529, failed=True), episode(517, failed=True)), 417)
        self.assertEqual(common_prefix_samples(episode(517, failed=True), episode(529, failed=True)), 417)

    def test_later_failure_does_not_crop_an_already_shorter_pair_twice(self):
        self.assertEqual(common_prefix_samples(episode(529, failed=True), episode(6000, failed=True)), 429)

    def test_empty_and_subsecond_failure_produce_no_common_samples(self):
        self.assertEqual(common_prefix_samples(episode(0, initial_failure=True), episode(6000)), 0)
        self.assertEqual(common_prefix_samples(episode(50, failed=True), episode(6000)), 0)

    def test_misaligned_physical_ticks_are_rejected(self):
        left, right = episode(200), episode(200)
        right['time'][50] += .001
        with self.assertRaises(ValueError):
            common_prefix_samples(left, right)


class PairedMasksTests(unittest.TestCase):
    def test_physical_mask_intersects_both_groups(self):
        left, right = episode(8), episode(8)
        a, b = physics_arrays(8), physics_arrays(8)
        a['standing_physics_valid'][1, 0] = False
        b['standing_physics_valid'][2, 0] = False
        expected = np.ones(8, dtype=bool)
        expected[[1, 2]] = False
        before_a, before_b = copy.deepcopy(a), copy.deepcopy(b)
        result = paired_masks(left, right, a, b, 'standing', 0)
        self.assertEqual(result['samples'], 8)
        self.assertEqual(result['physical'].dtype, np.bool_)
        np.testing.assert_array_equal(result['physical'], expected)
        np.testing.assert_array_equal(result['quiet'], expected)
        for key in a:
            np.testing.assert_array_equal(a[key], before_a[key])
            np.testing.assert_array_equal(b[key], before_b[key])

    def test_quiet_uses_both_groups_both_feet_and_inclusive_1100n(self):
        left, right = episode(8), episode(8)
        a, b = physics_arrays(8), physics_arrays(8)
        a['standing_physics_valid'][1, 0] = False
        b['standing_physics_valid'][2, 0] = False
        a['standing_physics_foot_force_peak'][3, 0, 0] = 1100.01
        b['standing_physics_foot_force_peak'][4, 0, 1] = 1100.01
        b['standing_physics_foot_force_peak'][5, 0] = 1100.
        expected = np.ones(8, dtype=bool)
        expected[[1, 2, 3, 4]] = False
        result = paired_masks(left, right, a, b, 'standing', 0)
        np.testing.assert_array_equal(result['quiet'], expected)
        self.assertTrue(result['quiet'][5])

    def test_masks_stop_at_the_common_nonterminal_prefix(self):
        left, right = episode(529, failed=True), episode(517, failed=True)
        a, b = physics_arrays(529), physics_arrays(517)
        a['standing_physics_valid'][416, 0] = False
        a['standing_physics_foot_force_peak'][418, 0, 0] = 5000.
        result = paired_masks(left, right, a, b, 'standing', 0)
        self.assertEqual(result['samples'], 417)
        self.assertEqual(result['physical'].shape, (417,))
        self.assertEqual(result['quiet'].shape, (417,))
        self.assertFalse(result['physical'][-1])
        self.assertEqual(int(result['quiet'].sum()), 416)

    def test_empty_prefix_is_empty_not_a_quiet_success(self):
        result = paired_masks(episode(0, initial_failure=True), episode(200),
                              physics_arrays(0), physics_arrays(200), 'standing', 0)
        self.assertEqual(result['samples'], 0)
        self.assertEqual(result['physical'].shape, (0,))
        self.assertEqual(result['quiet'].shape, (0,))

    def test_paired_masks_preserve_clock_rejection(self):
        left, right = episode(200), episode(200)
        right['time'][0] += .001
        with self.assertRaises(ValueError):
            paired_masks(left, right, physics_arrays(200), physics_arrays(200), 'standing', 0)


class TargetSecondDifferenceTests(unittest.TestCase):
    def test_quadratic_sequence_has_exact_scaled_second_difference_per_joint(self):
        data = episode(700)
        coefficient = 1e-5*np.arange(1, 13, dtype=np.float64)
        ticks = np.arange(1, 701, dtype=np.float64)
        data['action'] = ticks[:, None]**2*coefficient[None]
        before = data['action'].copy()
        result = target_second_difference(data, manifest(), 4., 6.)
        self.assertEqual(result['values'].shape, (698, 12))
        self.assertEqual(result['eligible'].dtype, np.bool_)
        np.testing.assert_array_equal(result['time'], data['time'][2:])
        expected = np.broadcast_to(.5*2*coefficient/.01**2, result['values'].shape)
        np.testing.assert_allclose(result['values'], expected, rtol=1e-8, atol=1e-9)
        self.assertEqual(result['samples'], 198)
        self.assertEqual(result['requested_samples'], 198)
        self.assertTrue(result['complete_window'])
        self.assertTrue(result['raw_mu_equals_recorded_action'])
        self.assertTrue(result['raw_mu_available'])
        self.assertEqual(result['clip_boundary_count'], 0)
        np.testing.assert_array_equal(data['action'], before)

    def test_all_three_target_times_must_be_inside_half_open_window(self):
        data = episode(700)
        result = target_second_difference(data, manifest(), 4., 6.)
        eligible_time = result['time'][result['eligible']]
        np.testing.assert_allclose(eligible_time, np.arange(402, 600)*.01, rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(eligible_time[0]), 4.02)
        self.assertAlmostEqual(float(eligible_time[-1]), 5.99)
        self.assertEqual(result['samples'], 198)

    def test_pre_window_and_end_boundary_spikes_do_not_leak_into_eligible_values(self):
        data = episode(700)
        data['action'][398, 0] = 50.  # 3.99 s: derivatives at 4.00/4.01 are ineligible.
        data['action'][599, 1] = 50.  # 6.00 s is outside [4, 6).
        result = target_second_difference(data, manifest(), 4., 6.)
        self.assertTrue(np.any(result['values'] != 0.))
        np.testing.assert_array_equal(result['values'][result['eligible']], np.zeros((198, 12)))

    def test_partial_failure_window_is_marked_incomplete_not_padded(self):
        data = episode(529, failed=True)
        result = target_second_difference(data, manifest(), 4., 6.)
        self.assertEqual(result['samples'], 128)
        self.assertEqual(result['requested_samples'], 198)
        self.assertFalse(result['complete_window'])
        self.assertAlmostEqual(float(result['time'][result['eligible']][-1]), 5.29)

    def test_empty_or_fewer_than_three_targets_have_no_fabricated_derivatives(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                result = target_second_difference(episode(count), manifest(), 4., 6.)
                self.assertEqual(result['values'].shape, (0, 12))
                self.assertEqual(result['time'].shape, (0,))
                self.assertEqual(result['eligible'].shape, (0,))
                self.assertEqual(result['samples'], 0)
                self.assertEqual(result['requested_samples'], 198)
                self.assertFalse(result['complete_window'])

    def test_clip_boundary_allows_applied_target_but_not_unclipped_mean_claim(self):
        for boundary in (-100., 100.):
            with self.subTest(boundary=boundary):
                data = episode(700)
                data['action'][450, 3] = boundary
                result = target_second_difference(data, manifest(), 4., 6.)
                self.assertEqual(result['clip_boundary_count'], 1)
                self.assertFalse(result['raw_mu_equals_recorded_action'])
                self.assertFalse(result['raw_mu_available'])
                self.assertTrue(np.isfinite(result['values']).all())
                self.assertTrue(np.any(result['values'][result['eligible']] != 0.))
                self.assertEqual(result['samples'], 198)

    def test_boundary_outside_requested_window_still_invalidates_full_prefix_raw_claim(self):
        data = episode(700)
        data['action'][0, 0] = 100.
        result = target_second_difference(data, manifest(), 4., 6.)
        self.assertEqual(result['clip_boundary_count'], 1)
        self.assertFalse(result['raw_mu_equals_recorded_action'])
        self.assertFalse(result['raw_mu_available'])
        np.testing.assert_array_equal(result['values'][result['eligible']], np.zeros((198, 12)))

    def test_super_clip_actions_are_rejected_instead_of_silently_clamped(self):
        for value in (-100.01, 100.01):
            with self.subTest(value=value):
                data = episode(10)
                data['action'][0, 0] = value
                with self.assertRaises(ValueError):
                    target_second_difference(data, manifest(), 0., .2)

    def test_control_scale_and_period_cannot_change_formula(self):
        for scale, dt in ((1., .01), (.5, .02)):
            with self.subTest(scale=scale, dt=dt):
                changed = manifest()
                changed['environment']['control']['action_scale'] = scale
                changed['runtime']['control_dt'] = dt
                with self.assertRaises(ValueError):
                    target_second_difference(episode(10), changed, 0., .2)


class WorstInitialStateRankingTests(unittest.TestCase):
    def test_all_32_unique_initial_conditions_are_ranked_and_not_mutated(self):
        rows = initial_state_rows()
        before = copy.deepcopy(rows)
        ranked = rank_worst_initial_states(rows)
        self.assertEqual(len(ranked), 32)
        self.assertEqual({(row['mode'], row['env']) for row in ranked},
                         {(row['mode'], row['env']) for row in rows})
        self.assertEqual([row['overview']['substep_accel_rms'] for row in ranked],
                         [float(index) for index in range(31, -1, -1)])
        self.assertEqual(rows, before)

    def test_failure_and_missing_are_retained_before_surviving_metric_leaders(self):
        rows = initial_state_rows()
        rows[7].update(failure=True, observed_s=.25)
        rows[7]['overview']['substep_accel_rms'] = None
        rows[26].update(failure=True, observed_s=5.17)
        rows[26]['overview']['substep_accel_rms'] = 2.
        rows[8]['overview']['substep_accel_rms'] = None
        ranked = rank_worst_initial_states(rows)
        self.assertEqual(len(ranked), 32)
        self.assertEqual({(row['mode'], row['env']) for row in ranked[:2]},
                         {('standing', 7), ('reference', 10)})
        self.assertTrue(all(row['failure'] for row in ranked[:2]))
        self.assertEqual((ranked[2]['mode'], ranked[2]['env']), ('standing', 8))
        self.assertIsNone(ranked[2]['overview']['substep_accel_rms'])
        self.assertEqual((ranked[3]['mode'], ranked[3]['env']), ('reference', 15))
        by_id = {(row['mode'], row['env']): row for row in ranked}
        self.assertIsNone(by_id[('standing', 7)]['overview']['substep_accel_rms'])
        self.assertEqual(by_id[('standing', 7)]['observed_s'], .25)
        self.assertTrue(by_id[('reference', 10)]['failure'])
        self.assertEqual(by_id[('reference', 10)]['observed_s'], 5.17)

    def test_all_missing_metrics_remain_32_missing_not_zeroes(self):
        rows = initial_state_rows()
        for row in rows:
            row['overview']['substep_accel_rms'] = None
        rows[0].update(failure=True, observed_s=0.)
        ranked = rank_worst_initial_states(rows)
        self.assertEqual(len(ranked), 32)
        self.assertTrue(ranked[0]['failure'])
        self.assertTrue(all(row['overview']['substep_accel_rms'] is None for row in ranked))

    def test_population_size_must_be_exactly_32(self):
        for rows in (initial_state_rows()[:-1], initial_state_rows()+[copy.deepcopy(initial_state_rows()[0])]):
            with self.subTest(size=len(rows)):
                with self.assertRaises(ValueError):
                    rank_worst_initial_states(rows)

    def test_duplicate_initial_condition_is_rejected_even_with_32_rows(self):
        rows = initial_state_rows()
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):
            rank_worst_initial_states(rows)

    def test_explicit_metric_is_ranked_without_substituting_default_metric(self):
        rows = initial_state_rows()
        for index, row in enumerate(rows):
            row['overview']['substep_torque_delta_rms_nm'] = float(31-index)
        ranked = rank_worst_initial_states(rows, metric='substep_torque_delta_rms_nm')
        self.assertEqual((ranked[0]['mode'], ranked[0]['env']), ('standing', 0))
        self.assertEqual((ranked[-1]['mode'], ranked[-1]['env']), ('reference', 15))


class PairedMeansTests(unittest.TestCase):
    def rows(self):
        rows = [dict(mode='reference', env=i,
                     left=dict(overview=overview(None)), right=dict(overview=overview(None))) for i in range(3)]
        for i, row in enumerate(rows):
            row['left']['overview']['substep_accel_rms'] = float(i+1)
            row['right']['overview']['substep_accel_rms'] = float((i+1)*10)
        return rows

    def test_bilateral_missing_values_use_identical_cohort(self):
        rows = self.rows()
        rows[0]['left']['overview']['substep_accel_rms'] = None
        rows[1]['right']['overview']['substep_accel_rms'] = None
        result = paired_means(rows)
        for side, expected in (('left', 3.), ('right', 30.)):
            metric = result[side]['substep_accel_rms']
            self.assertEqual(metric['mean'], expected)
            self.assertEqual(metric['contributing_initial_states'], 1)
            self.assertEqual(metric['common_initial_states'], [dict(mode='reference', env=2)])

    def test_disjoint_nonnull_cohorts_are_missing_not_separate_means(self):
        rows = self.rows()[:2]
        rows[0]['left']['overview']['substep_accel_rms'] = None
        rows[1]['right']['overview']['substep_accel_rms'] = None
        for side in paired_means(rows).values():
            self.assertIsNone(side['substep_accel_rms']['mean'])
            self.assertEqual(side['substep_accel_rms']['common_initial_states'], [])

    def test_duplicate_and_nonfinite_pairs_are_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            paired_means(rows+[copy.deepcopy(rows[0])])
        rows[1]['left']['overview']['substep_accel_rms'] = float('inf')
        with self.assertRaises(ValueError):
            paired_means(rows)


class GeometryBindingTests(unittest.TestCase):
    def source(self):
        return dict(body_names=['knee_l', 'knee_r', 'foot_l', 'foot_r'],
                    foot_names=['foot_l', 'foot_r'], urdf_lf_sha256='a'*64)

    def test_actual_original_order_and_urdf_are_required(self):
        source = self.source()
        validate_geometry_binding(copy.deepcopy(source), source, source['body_names'], 'a'*64)
        for field, value in (('foot_names', ['foot_r', 'foot_l']),
                             ('body_names', list(reversed(source['body_names']))),
                             ('urdf_lf_sha256', 'b'*64)):
            changed = copy.deepcopy(source)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_geometry_binding(changed, source, source['body_names'], 'a'*64)

    def test_missing_binding_or_wrong_local_geometry_is_rejected(self):
        source = self.source()
        for field in source:
            changed = copy.deepcopy(source)
            del changed[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_geometry_binding(changed, source, source['body_names'], 'a'*64)
        with self.assertRaises(ValueError):
            validate_geometry_binding(source, source, source['body_names'], 'b'*64)
        with self.assertRaises(ValueError):
            validate_geometry_binding(source, source, list(reversed(source['body_names'])), 'a'*64)


class PlotWindowTests(unittest.TestCase):
    def test_yaw_uses_window_mask_not_entire_episode(self):
        import matplotlib
        matplotlib.use('Agg')
        data = episode(700)
        data.update(base_lin_vel=np.zeros((700, 3)), foot_force=np.zeros((700, 2, 3)),
                    root_state=np.zeros((700, 13)))
        data['root_state'][:, 6] = 1.
        signals = dict(sole_height=np.zeros((700, 2)), sole_speed=np.zeros((700, 2)),
                       acceleration_native=np.zeros((700, 12)))
        arrays = physics_arrays(700)
        arrays.update(standing_physics_accel_squared=np.zeros((700, 1, 12)),
                      standing_physics_torque_delta_squared=np.zeros((700, 1, 12)))
        cases = {name:dict(arrays=arrays, manifest=manifest()) for name in ('original', 'anchor', 'temporal')}
        with tempfile.TemporaryDirectory() as folder, \
                patch('tools.amp.audit_mu_candidates.episode', return_value=data), \
                patch('tools.amp.audit_mu_candidates.fast_signals', return_value=signals):
            plot_cases(cases, None, Path(folder), 'standing', 0, 4., 6., 'window')
            self.assertGreater((Path(folder)/'window.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
