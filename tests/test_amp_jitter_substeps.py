import unittest
import numpy as np
from tools.amp.audit_jitter_substeps import measured_summary


class SubstepAnalysisTests(unittest.TestCase):
    def test_cancellation_and_missed_force_peak_are_retained(self):
        result = measured_summary(np.zeros((2, 2)), np.full((2, 2), 100.),
            np.full((2, 2), 10.), np.array([[200., 0.], [0., 1200.]]),
            np.array([[1500., 0.], [0., 1200.]]), np.array([True, True]))
        self.assertEqual(result['endpoint_to_substep_energy_ratio'], 0.)
        self.assertEqual(result['substep_accel_rms'], 10.)
        self.assertEqual(result['burst_foot_intervals'], 2)
        self.assertEqual(result['burst_missed_by_endpoint'], 1)

    def test_reset_or_unpaired_tail_is_excluded_from_both_metrics(self):
        result = measured_summary(np.array([[2., 2.], [1000., 1000.]]),
            np.array([[4., 4.], [1e6, 1e6]]), np.array([[2., 2.], [1000., 1000.]]),
            np.zeros((2, 2)), np.zeros((2, 2)), np.array([True, False]))
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['substep_accel_rms'], 2.)
        self.assertEqual(result['endpoint_to_substep_energy_ratio'], 1.)

    def test_empty_prefix_has_no_fake_zero(self):
        self.assertIsNone(measured_summary(None, None, None, None, None, np.array([False])))

    def test_inconsistent_sampling_fails_closed(self):
        with self.assertRaises(ValueError):
            measured_summary(np.full((1, 2), 20.), np.ones((1, 2)), np.ones((1, 2)),
                np.zeros((1, 2)), np.zeros((1, 2)), np.array([True]))


if __name__ == '__main__':
    unittest.main()
