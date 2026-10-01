import copy
import unittest

import numpy as np

from tools.amp.compare_jitter_rollouts import relative, numerical_gates, summarize_rows
from tools.amp.jitter_audit import validate_substep_arrays
from humanoid.amp.jitter import jitter_contract


class JitterComparisonTests(unittest.TestCase):
    def test_zero_denominator_is_not_automatic_improvement(self):
        self.assertIsNone(relative(0., 0.))
        self.assertIsNone(relative(None, 1.))
        self.assertEqual(relative(.8, 1.), .8)

    def test_quiet_but_stopped_or_failed_policy_cannot_pass(self):
        keys = ('action_delta_rms', 'action_spectrum_high_power_joint_mean', 'torque_delta_rms_nm',
            'acceleration_native_rms', 'torque_delta_step_rms_p95_nm', 'acceleration_native_step_rms_p95',
            'contact_proxy_slip_mean_m_s', 'heading_rms_to_world_x_deg', 'demo_nearest_window_rms_zscore')
        source = dict(total=32, survived=32, mean_metrics=dict(vx_mean=.4, **{k:1. for k in keys}))
        target = dict(total=32, survived=32, mean_metrics=dict(vx_mean=.37, **{k:.7 for k in keys}))
        self.assertTrue(all(numerical_gates(target, source).values()))
        stopped = copy.deepcopy(target); stopped['mean_metrics']['vx_mean'] = .1
        self.assertFalse(numerical_gates(stopped, source)['progress'])
        failed = dict(target, survived=31)
        self.assertFalse(numerical_gates(failed, source)['full_survival'])

    def test_failed_initialization_remains_in_denominator(self):
        rows = [dict(survived=False, post2s=None, quality=None) for _ in range(32)]
        result = summarize_rows(rows)
        self.assertEqual(result['total'], 32)
        self.assertEqual(result['survived'], 0)
        self.assertEqual(result['contributing_prefixes'], 0)
        self.assertTrue(all(value is None for value in result['mean_metrics'].values()))

    def test_substep_evidence_bounds_and_guard(self):
        manifest = dict(modes=['reference'], substep_telemetry=dict(physics_hz=1000,
            samples_per_control=10, reward_used=False))
        arrays = dict(reference_dof_vel=np.ones((5, 1, 12)), reference_initial_dof_vel=np.ones((1, 12)),
            reference_valid=np.ones((5, 1), dtype=bool), reference_physics_valid=np.ones((5, 1), dtype=bool),
            reference_physics_accel_squared=np.full((5, 1, 12), 100.),
            reference_physics_accel_peak=np.full((5, 1, 12), 10.),
            reference_physics_torque_delta_squared=np.full((5, 1, 12), 1.),
            reference_physics_foot_force_peak=np.full((5, 1, 2), 100.))
        self.assertTrue(validate_substep_arrays(manifest, arrays))
        bad = copy.deepcopy(arrays); bad['reference_dof_vel'][2] += 1.
        with self.assertRaises(ValueError): validate_substep_arrays(manifest, bad)
        bad = copy.deepcopy(arrays); bad['reference_physics_accel_peak'][:] = 1.
        with self.assertRaises(ValueError): validate_substep_arrays(manifest, bad)

    def test_actual_nonlinear_cost_bounds_and_identity(self):
        for group in ('substep_accel', 'substep_torque'):
            manifest = dict(identity=dict(experiment='f1_amp_walk02_jitter_'+group), modes=['reference'],
                substep_telemetry=dict(physics_hz=1000, samples_per_control=10, reward_used=True,
                                      penalty=jitter_contract(group)['substep']))
            arrays = dict(reference_dof_vel=np.zeros((5, 1, 12)), reference_initial_dof_vel=np.zeros((1, 12)),
                reference_torque=np.zeros((5, 1, 12)), reference_initial_torque=np.zeros((1, 12)),
                reference_valid=np.ones((5, 1), dtype=bool), reference_physics_valid=np.ones((5, 1), dtype=bool),
                reference_physics_accel_squared=np.full((5, 1, 12), 1e6),
                reference_physics_accel_peak=np.full((5, 1, 12), 1000.),
                reference_physics_torque_delta_squared=np.full((5, 1, 12), 4.),
                reference_physics_foot_force_peak=np.full((5, 1, 2), 100.),
                reference_physics_acceleration_cost=np.full((5, 1), 2*(101**.5-1)),
                reference_physics_torque_cost=np.full((5, 1), 2*(2**.5-1)))
            self.assertTrue(validate_substep_arrays(manifest, arrays))
            for key in ('acceleration_cost', 'torque_cost'):
                bad = copy.deepcopy(arrays); bad['reference_physics_'+key][:] = 100.
                with self.assertRaises(ValueError): validate_substep_arrays(manifest, bad)
            bad = copy.deepcopy(manifest); bad['substep_telemetry']['reward_used'] = False
            with self.assertRaises(ValueError): validate_substep_arrays(bad, arrays)


if __name__ == '__main__': unittest.main()
