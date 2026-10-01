import unittest
import numpy as np
from tools.amp.analyze_jitter_events import backward_delta, command_geometry, high_frequency


class JitterAnalysisTests(unittest.TestCase):
    def test_backward_delta_matches_current_transition_without_future_leak(self):
        x = np.array([[2., 3.], [5., 1.], [4., 6.]])
        np.testing.assert_equal(backward_delta(x, [1., 2.]), [[1, 1], [3, -2], [-1, 5]])

    def test_target_excess_is_separate_from_actual_pose(self):
        target, excess = command_geometry([[2., -3.], [0., 0.]], [.1, -.2], .5, [-.4, -.4], [.35, .35])
        np.testing.assert_allclose(target, [[1.1, -1.7], [.1, -.2]])
        np.testing.assert_allclose(excess, [[.75, 1.3], [0., 0.]])

    def test_spectrum_distinguishes_stride_jitter_and_static(self):
        t = np.arange(1000)*.01
        x = np.c_[np.sin(2*np.pi*.6*t), np.sin(2*np.pi*15*t), np.zeros_like(t)]
        r = high_frequency(x)
        self.assertLess(r['fraction'][0], .001)
        self.assertGreater(r['fraction'][1], .999)
        self.assertEqual(r['stationary_columns'], [False, False, True])
        self.assertIsNone(high_frequency(x[:50]))


if __name__ == '__main__': unittest.main()
