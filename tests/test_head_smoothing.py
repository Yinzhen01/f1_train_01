"""Synthetic mathematical fixtures only; no checkpoint/actor/PhysX/cloud proof."""
from dataclasses import replace
import copy
import unittest
from unittest.mock import patch

import numpy as np

from humanoid.amp.head_smoothing import (Episode, HEAD_BIAS_KEY, HEAD_WEIGHT_KEY,
    ACTION_SCALE, CONTROL_DT, DESIGN_WIDTH, HIDDEN_WIDTH, OUTPUT_WIDTH,
    _admission, _candidate_report, _limit_direction, _prepare_head_audit, _solve,
    _source_scales, curvature, episode_digest, evaluate_head_candidate,
    fit_head_smoothing, valid_hann, validate_head_only_change)


IDENTITY = 'a'*64


def episode(name, cohort, phase=0., samples=160, gain=1., seed=1):
    t = np.arange(samples)*CONTROL_DT
    x = np.zeros((samples, DESIGN_WIDTH), dtype=np.float64)
    x[:, 0] = gain*np.sin(2*np.pi*1.7*t+phase)
    x[:, 1] = gain*np.sin(2*np.pi*22*t+phase+.2)
    x[:, -1] = 1.
    return Episode(name, cohort, x, np.arange(65, 65+samples), seed, 0, IDENTITY)


def fixture():
    w = np.zeros((OUTPUT_WIDTH, HIDDEN_WIDTH), dtype=np.float64)
    w[:, 0], w[:, 1] = .4, .025
    b = np.linspace(-.03, .03, OUTPUT_WIDTH)
    return [episode('train1', 'new_train')], [episode('val1', 'new_val', .7, seed=2)], w, b


def fit(train=None, validation=None, w=None, b=None, **kwargs):
    base = fixture()
    args = [value if value is not None else default for value, default in zip((train, validation, w, b), base)]
    options = dict(source_identity=IDENTITY, action_clip=2., forbidden_cohort_ids={'original_audit_32'})
    options.update(kwargs)
    return fit_head_smoothing(*args, **options)


def stub_report(split='validation', **overrides):
    row = dict(episode_id='example', split=split, normalized_curvature_reduction=.10,
        output_change_peak_rad=.05, low_frequency_deviation_rms_per_joint_rad=[.01]*12,
        low_frequency_deviation_peak_per_joint_rad=[.025]*12,
        low_frequency_range_ratio=[1.]*12,
        source=dict(saturation_boundary_count=[0]*12, saturation_overflow_count=[0]*12),
        candidate=dict(saturation_boundary_count=[0]*12, saturation_overflow_count=[0]*12))
    row.update(overrides)
    return row


class HeadSmoothingMathematicsTests(unittest.TestCase):
    def test_float64_quadratic_fit_has_valid_output_shapes_and_does_not_mutate(self):
        train, val, w, b = fixture()
        old_x, old_w, old_b = train[0].design.copy(), w.copy(), b.copy()
        result = fit(train, val, w, b)
        self.assertEqual(result.candidate_weight.shape, (12, 128))
        self.assertEqual(result.candidate_bias.shape, (12,))
        self.assertEqual(result.candidate_weight.dtype, np.float64)
        self.assertTrue(np.isfinite(result.candidate_weight).all())
        self.assertTrue(np.array_equal(old_x, train[0].design))
        self.assertTrue(np.array_equal(old_w, w))
        self.assertTrue(np.array_equal(old_b, b))
        self.assertEqual(result.report['parameter_scalars'], 1548)
        self.assertEqual(result.report['ppo_updates_added'], 0)
        self.assertFalse(result.report['optimizer_state_reused'])
        self.assertFalse(result.report['closed_loop_verified'])
        self.assertLessEqual(result.report['train'][0]['output_change_peak_rad'], .05+1e-12)
        self.assertEqual(result.report['train'][0]['first_tick'], 65)
        self.assertEqual(result.report['train'][0]['seed'], 1)
        self.assertEqual(result.report['train'][0]['low_frequency_samples'], 140)
        self.assertTrue(result.report['admitted'])
        self.assertGreaterEqual(result.report['validation'][0]['normalized_curvature_reduction'], .10)

    def test_validation_never_changes_scales_solution_or_train_direction(self):
        first = fit()
        second = fit(validation=[episode('different', 'independent_val', 1.9, gain=100., seed=7)])
        self.assertTrue(np.array_equal(first.candidate_weight, second.candidate_weight))
        self.assertTrue(np.array_equal(first.candidate_bias, second.candidate_bias))
        for key in ('direction_scale', 'low_frequency_scales_rad', 'curvature_scales_rad_s2'):
            self.assertEqual(first.report[key], second.report[key])
        self.assertFalse(second.report['admitted'])
        self.assertTrue(any('output_change_exceeds' in reason for reason in second.report['rejection_reasons']))

    def test_hann_and_curvature_are_episode_local_no_cross_boundary_padding(self):
        one, two = np.ones((25, 2)), 100*np.ones((25, 2))
        self.assertEqual(valid_hann(one).shape, (5, 2))
        np.testing.assert_allclose(valid_hann(one), 1., atol=1e-14)
        np.testing.assert_allclose(valid_hann(two), 100., atol=1e-12)
        self.assertEqual(np.max(np.abs(curvature(one))), 0.)
        self.assertGreater(np.max(np.abs(curvature(np.vstack((one, two))))), 0.)
        quadratic = (np.arange(25)*CONTROL_DT)**2
        np.testing.assert_allclose(curvature(quadratic[:, None]), 2., atol=2e-12)

    def test_solved_normal_equation_has_small_residual(self):
        train, _, w, b = fixture()
        source = np.vstack((w.T, b))
        rows = [dict(design=train[0].design)]
        low, curve, _, _ = _source_scales(rows, source)
        solution = _solve(rows, source, low, curve)
        lx, cx = valid_hann(train[0].design), curvature(train[0].design)
        lg, cg = lx.T.dot(lx)/len(lx), cx.T.dot(cx)/len(cx)
        for joint in (0, 11):
            keep = .25*lg/low[joint]**2
            regularization = 1e-6*np.eye(129)
            normal = keep+.25*cg/curve[joint]**2+regularization
            rhs = (keep+regularization).dot(source[:, joint])
            np.testing.assert_allclose(normal.dot(solution[:, joint]), rhs, atol=1e-9, rtol=1e-9)

    def test_direction_is_single_global_train_bounded_and_identity_is_safe(self):
        train, _, w, b = fixture()
        rows, source = [dict(design=train[0].design)], np.vstack((w.T, b))
        candidate, scale, peak = _limit_direction(rows, source, source+100.)
        self.assertGreater(peak, .05)
        self.assertGreater(scale, 0.)
        self.assertLess(scale, 1.)
        np.testing.assert_allclose(candidate-source, scale*100., atol=1e-14)
        self.assertLessEqual(float(np.max(np.abs(.5*train[0].design.dot(candidate-source)))), .05+1e-12)
        identity, scale, peak = _limit_direction(rows, source, source.copy())
        self.assertTrue(np.array_equal(identity, source))
        self.assertEqual((scale, peak), (1., 0.))

    def test_episode_weight_not_long_episode_sample_weight(self):
        train, val, w, b = fixture()
        other = episode('train2', 'new_train', phase=.3, samples=320, gain=2.)
        first = fit(train+[other], val, w, b)
        permuted = fit([other]+train, val, w, b)
        np.testing.assert_allclose(first.candidate_weight, permuted.candidate_weight, atol=1e-12)
        source = np.vstack((w.T, b))
        expected = np.sqrt(np.mean([np.mean(curvature(.5*ep.design.dot(source))**2, axis=0)
                                   for ep in train+[other]], axis=0))
        np.testing.assert_allclose(first.report['curvature_scales_rad_s2'], expected, atol=1e-12)


class HeadSmoothingInputGuardsTests(unittest.TestCase):
    def test_bad_dt_history_ticks_length_bias_shape_nan_and_partial_rejected(self):
        train, _, _, _ = fixture()
        source = train[0]
        bad_x = source.design.copy()
        bad_x[0, 0] = np.nan
        bias_bad = source.design.copy()
        bias_bad[0, -1] = 0.
        cases = [replace(source, dt=0.), replace(source, dt=.02), replace(source, dt=np.nan),
            replace(source, history_frames=65), replace(source, history_frames=True),
            replace(source, history_frames=66.),
            replace(source, ticks=np.arange(len(source.design))),
            replace(source, ticks=source.ticks.astype(float)),
            replace(source, ticks=source.ticks+np.arange(len(source.design))),
            replace(source, design=source.design[:20], ticks=source.ticks[:20]),
            replace(source, design=source.design[:, :-1]), replace(source, design=bad_x),
            replace(source, design=bias_bad), replace(source, complete=False),
            replace(source, source_identity='b'*64), replace(source, seed=True), replace(source, env_id=-1)]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fit(train=[bad])
        for bad_ticks in (source.ticks[:-1], np.full(len(source.ticks), 65)):
            with self.assertRaises(ValueError):
                fit(train=[replace(source, ticks=bad_ticks)])
        wrapped = np.arange(len(source.ticks), dtype=np.uint64)
        wrapped[0] = np.iinfo(np.uint64).max
        with self.assertRaises(ValueError):
            fit(train=[replace(source, ticks=wrapped)])

    def test_public_operations_reject_nonfinite_and_overflowing_arrays(self):
        for op, bad in ((valid_hann, np.full((25, 2), np.nan)),
                (curvature, np.full((25, 2), np.inf)),
                (curvature, np.full((2, 2), 1.))):
            with self.subTest(op=op.__name__), self.assertRaises(ValueError):
                op(bad)
        huge = np.full((25, 2), 1e308)
        huge[1::2] *= -1
        with np.errstate(over='ignore', invalid='ignore'), self.assertRaises(ValueError):
            curvature(huge)

    def test_duplicates_and_cross_split_reuse_are_rejected_even_renamed(self):
        train, val, _, _ = fixture()
        cases = [dict(train=train*2), dict(train=train+[replace(train[0], episode_id='alias')]),
            dict(validation=[replace(train[0], cohort_id='new_val')]),
            dict(validation=[replace(train[0], episode_id='renamed', cohort_id='new_val', ticks=train[0].ticks+1)]),
            dict(validation=[replace(val[0], cohort_id='new_train')])]
        for case in cases:
            with self.subTest(case=case.keys()), self.assertRaises(ValueError):
                fit(**case)

    def test_protected_cohort_and_content_and_empty_protection_rejected(self):
        train, _, _, _ = fixture()
        with self.assertRaises(ValueError):
            fit(train=[replace(train[0], cohort_id='original_audit_32')])
        with self.assertRaises(ValueError):
            fit(forbidden_episode_digests=[episode_digest(train[0])])
        with self.assertRaises(ValueError):
            fit(forbidden_cohort_ids=[])
        with self.assertRaises(ValueError):
            fit(forbidden_episode_digests=['not-a-hash'])

    def test_nonfinite_head_wrong_shapes_and_parameter_search_are_refused(self):
        _, _, w, b = fixture()
        bad_w, bad_b = w.copy(), b.copy()
        bad_w[0, 0], bad_b[0] = np.inf, np.nan
        for options in (dict(w=bad_w), dict(b=bad_b), dict(w=w[:, :-1]), dict(b=b[:-1]),
                dict(w=w.astype(int)), dict(source_identity='invalid'), dict(action_clip=0.),
                dict(action_clip=np.nan), dict(temporal_weight=.5), dict(ridge=1e-5),
                dict(ridge=0.), dict(train=[]), dict(validation=[])):
            with self.subTest(options=options.keys()), self.assertRaises(ValueError):
                fit(**options)

    def test_constant_zero_training_curvature_rejected_and_zero_validation_cannot_pass(self):
        train, _, w, b = fixture()
        constant = train[0].design.copy()
        constant[:, :-1] = 0.
        with self.assertRaisesRegex(ValueError, 'zero-curvature'):
            fit(train=[replace(train[0], design=constant)])
        result = fit(validation=[replace(episode('constant', 'new_val'), design=constant)])
        self.assertFalse(result.report['admitted'])
        self.assertIn('validation/constant: zero_baseline_curvature_not_evaluable', result.report['rejection_reasons'])


class HeadSmoothingAdmissionTests(unittest.TestCase):
    def test_exact_gates_and_near_static_axes_have_no_division(self):
        self.assertEqual(_admission([], [stub_report(low_frequency_range_ratio=[.98, 1.02]+[None]*10)]), [])
        failures = [dict(normalized_curvature_reduction=.099), dict(normalized_curvature_reduction=None),
            dict(output_change_peak_rad=.0501), dict(low_frequency_deviation_rms_per_joint_rad=[.0101]*12),
            dict(low_frequency_deviation_peak_per_joint_rad=[.0251]*12),
            dict(low_frequency_range_ratio=[.979]*12), dict(low_frequency_range_ratio=[1.021]*12),
            dict(candidate=dict(saturation_boundary_count=[1]*12, saturation_overflow_count=[0]*12)),
            dict(candidate=dict(saturation_boundary_count=[0]*12, saturation_overflow_count=[1]*12))]
        for override in failures:
            with self.subTest(override=override.keys()):
                self.assertTrue(_admission([], [stub_report(**override)]))

    def test_one_good_episode_does_not_hide_another_bad_one_and_identity_fails_improvement(self):
        reasons = _admission([], [stub_report(episode_id='good', normalized_curvature_reduction=.7),
            stub_report(episode_id='identity', normalized_curvature_reduction=0.)])
        self.assertEqual(reasons, ['validation/identity: curvature_reduction_below_10_percent'])


class ExportedHeadAuditTests(unittest.TestCase):
    def audit(self, *, train=None, validation=None, source_weight=None, source_bias=None,
            candidate_weight=None, candidate_bias=None, **overrides):
        default_train, default_val, w, b = fixture()
        train = default_train if train is None else train
        validation = default_val if validation is None else validation
        w = w if source_weight is None else source_weight
        b = b if source_bias is None else source_bias
        if candidate_weight is None or candidate_bias is None:
            fitted = fit(train, validation, w, b)
            candidate_weight = fitted.candidate_weight.astype(np.float32)
            candidate_bias = fitted.candidate_bias.astype(np.float32)
        options = dict(source_identity=IDENTITY, candidate_identity=IDENTITY, action_clip=2.,
            forbidden_cohort_ids={'original_audit_32'})
        options.update(overrides)
        return evaluate_head_candidate(train, validation, w, b, candidate_weight, candidate_bias, **options)

    def test_readback_shares_exact_scales_metrics_gates_and_never_calls_solver_or_limiter(self):
        train, validation, w, b = fixture()
        fitted = fit(train, validation, w, b)
        cw, cb = fitted.candidate_weight.astype(np.float32), fitted.candidate_bias.astype(np.float32)
        before_w, before_b = cw.copy(), cb.copy()
        with patch('humanoid.amp.head_smoothing._solve', side_effect=AssertionError('no solve')), \
                patch('humanoid.amp.head_smoothing._limit_direction', side_effect=AssertionError('no scaling')):
            actual = self.audit(candidate_weight=cw, candidate_bias=cb)
        identity, clip, source, tr, va, scales = _prepare_head_audit(train, validation, w, b,
            IDENTITY, 2., {'original_audit_32'}, ())
        promoted = np.vstack((cw.astype(np.float64).T, cb.astype(np.float64)))
        expected = _candidate_report(identity, clip, source, tr, va, promoted, scales)
        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)
        self.assertTrue(actual['admitted'])
        self.assertFalse(actual['solver_run'])
        self.assertFalse(actual['candidate_adjusted'])
        self.assertEqual(actual['audit_origin'], 'exported_float32_head_readback')
        self.assertNotIn('direction_scale', actual)
        self.assertTrue(np.array_equal(cw, before_w))
        self.assertTrue(np.array_equal(cb, before_b))
        for key in ('low_frequency_scales_rad', 'curvature_scales_rad_s2', 'low_frequency_scale_floor_rad',
                'curvature_scale_floor_rad_s2', 'temporal_weight', 'ridge', 'action_clip'):
            self.assertEqual(actual[key], fitted.report[key])
        for split in ('train', 'validation'):
            self.assertEqual(actual[split][0]['source'], fitted.report[split][0]['source'])
            self.assertAlmostEqual(actual[split][0]['normalized_curvature_reduction'],
                                   fitted.report[split][0]['normalized_curvature_reduction'], places=6)

    def test_float64_fit_passes_but_float32_rounding_breaches_training_bound_without_repair(self):
        def alternating(name, cohort, count, offset):
            x = np.zeros((count, DESIGN_WIDTH), dtype=np.float64)
            x[:, 0] = (np.arange(count)+offset)/128.
            x[:, 1] = np.where(np.arange(count) % 2, -1., 1.)
            x[:, -1] = 1.
            return Episode(name, cohort, x, np.arange(65, 65+count), offset+1, 0, IDENTITY)
        train, validation = [alternating('train', 'new_train', 160, 0)], [alternating('val', 'new_val', 120, 1)]
        w, b = np.zeros((12, 128), dtype=np.float32), np.zeros(12, dtype=np.float32)
        w[:, 0], w[:, 1] = .5, .2
        fitted = fit(train, validation, w, b)
        self.assertTrue(fitted.report['admitted'], fitted.report['rejection_reasons'])
        self.assertLessEqual(fitted.report['train'][0]['output_change_peak_rad'], .05+1e-12)
        cw, cb = fitted.candidate_weight.astype(np.float32), fitted.candidate_bias.astype(np.float32)
        actual = self.audit(train=train, validation=validation, source_weight=w, source_bias=b,
                            candidate_weight=cw, candidate_bias=cb)
        self.assertFalse(actual['admitted'])
        self.assertGreater(actual['train'][0]['output_change_peak_rad'], .05+1e-12)
        self.assertIn('train/train: output_change_exceeds_0.05_rad', actual['rejection_reasons'])
        self.assertFalse(actual['candidate_adjusted'])

    def test_export_candidate_dtype_finite_shape_and_bound_source_identity_are_required(self):
        _, _, w, b = fixture()
        cw, cb = w.astype(np.float32), b.astype(np.float32)
        bad_w = cw.copy()
        bad_w[0, 0] = np.nan
        cases = [dict(candidate_identity='b'*64), dict(source_identity='b'*64, candidate_identity='b'*64),
            dict(candidate_weight=cw.astype(np.float64)), dict(candidate_bias=cb.astype(np.float64)),
            dict(candidate_weight=cw.astype(np.float16)), dict(candidate_weight=bad_w),
            dict(candidate_weight=cw[:, :-1]), dict(candidate_bias=cb[:-1])]
        for overrides in cases:
            args = dict(candidate_weight=cw, candidate_bias=cb)
            args.update(overrides)
            with self.subTest(overrides=overrides.keys()), self.assertRaises(ValueError):
                self.audit(**args)

    def test_readback_cannot_reuse_wrong_split_identity_or_protected_cohort(self):
        train, validation, w, b = fixture()
        candidate = dict(candidate_weight=w.astype(np.float32), candidate_bias=b.astype(np.float32))
        cases = [dict(train=[replace(train[0], source_identity='b'*64)]),
            dict(validation=[replace(validation[0], source_identity='b'*64)]),
            dict(validation=[replace(validation[0], cohort_id='new_train')]),
            dict(validation=[replace(train[0], episode_id='renamed', cohort_id='new_val')]),
            dict(train=[replace(train[0], cohort_id='original_audit_32')]),
            dict(forbidden_episode_digests=[episode_digest(validation[0])])]
        for overrides in cases:
            with self.subTest(overrides=overrides.keys()), self.assertRaises(ValueError):
                self.audit(**candidate, **overrides)

    def test_exported_identity_does_not_inherit_a_prior_fit_pass(self):
        _, _, w, b = fixture()
        self.assertTrue(fit().report['admitted'])
        actual = self.audit(candidate_weight=w.astype(np.float32), candidate_bias=b.astype(np.float32))
        self.assertFalse(actual['admitted'])
        self.assertIn('validation/val1: curvature_reduction_below_10_percent', actual['rejection_reasons'])


class FrozenHeadChangeTests(unittest.TestCase):
    def state(self):
        _, _, w, b = fixture()
        return {HEAD_WEIGHT_KEY: w, HEAD_BIAS_KEY: b, 'actor.0.weight': np.eye(3),
            'critic.0.bias': np.zeros(3), 'std': np.ones(12)}

    def test_only_two_head_tensors_may_change_and_identity_passes_exact_freeze(self):
        old = self.state()
        new = copy.deepcopy(old)
        new[HEAD_WEIGHT_KEY][0, 0] += .01
        new[HEAD_BIAS_KEY][0] += .01
        report = validate_head_only_change(old, new, IDENTITY, IDENTITY)
        self.assertEqual(set(report['changed_keys']), {HEAD_WEIGHT_KEY, HEAD_BIAS_KEY})
        self.assertTrue(report['frozen_exact'])
        self.assertEqual(report['frozen_tensor_count'], 3)
        self.assertEqual(validate_head_only_change(old, copy.deepcopy(old), IDENTITY, IDENTITY)['changed_keys'], [])

    def test_any_wrong_layer_identity_dtype_shape_key_or_nan_is_refused(self):
        old = self.state()
        mutations = []
        for key in ('actor.0.weight', 'critic.0.bias', 'std'):
            new = copy.deepcopy(old)
            new[key].flat[0] += .01
            mutations.append(new)
        new = copy.deepcopy(old)
        new[HEAD_BIAS_KEY] = new[HEAD_BIAS_KEY].astype(np.float32)
        mutations.append(new)
        new = copy.deepcopy(old)
        new[HEAD_WEIGHT_KEY] = new[HEAD_WEIGHT_KEY][:, :-1]
        mutations.append(new)
        new = copy.deepcopy(old)
        new[HEAD_BIAS_KEY][0] = np.nan
        mutations.append(new)
        new = copy.deepcopy(old)
        new.pop('std')
        mutations.append(new)
        for new in mutations:
            with self.subTest(keys=new.keys()), self.assertRaises(ValueError):
                validate_head_only_change(old, new, IDENTITY, IDENTITY)
        with self.assertRaises(ValueError):
            validate_head_only_change(old, copy.deepcopy(old), IDENTITY, 'b'*64)
        with self.assertRaises(ValueError):
            validate_head_only_change({key: old[key] for key in (HEAD_WEIGHT_KEY, HEAD_BIAS_KEY)},
                {key: old[key] for key in (HEAD_WEIGHT_KEY, HEAD_BIAS_KEY)}, IDENTITY, IDENTITY)


if __name__ == '__main__':
    unittest.main()
