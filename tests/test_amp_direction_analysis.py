"""Offline reward reconstruction must honor the recorded direction intervention."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.amp.diagnose_progress_frame import series, summarize


class DirectionAnalysisTests(unittest.TestCase):
    def fixture(self):
        root = np.zeros((3, 13), dtype=np.float32)
        root[:, 3:7] = Rotation.from_euler('z', [0., 45., 90.], degrees=True).as_quat()
        root[:, 8] = .45
        return dict(time=np.array([2., 2.01, 2.02]), root_state=root,
                    base_lin_vel=np.tile([.45, 0., 0.], (3, 1)).astype(np.float32))

    def test_mixture_is_of_rewards_not_of_velocities(self):
        data = self.fixture(); values = series(data)
        body = summarize(data, values)
        world = summarize(data, values, 'world')
        mixed = summarize(data, values, 'body', .15)
        self.assertAlmostEqual(mixed['selected_progress_reward_mean'],
            .85*body['selected_progress_reward_mean']+.15*world['selected_progress_reward_mean'], places=8)
        self.assertEqual(summarize(data, values, 'body', 0.)['selected_progress_reward_mean'],
                         body['selected_progress_reward_mean'])

    def test_heading_coefficient_and_dt_are_reconstructed(self):
        data = self.fixture()
        old = summarize(data, series(data))
        strong = summarize(data, series(data, -1.5))
        self.assertAlmostEqual(strong['heading_reward_mean'], 3*old['heading_reward_mean'])
        self.assertEqual(strong['selected_progress_reward_mean'], old['selected_progress_reward_mean'])
        self.assertAlmostEqual(series(data, -1.5)['heading'][-1], -.015)

    def test_unknown_and_conflicting_rewards_rejected(self):
        data = self.fixture(); values = series(data)
        with self.assertRaises(ValueError): summarize(data, values, 'world', .15)
        with self.assertRaises(ValueError): summarize(data, values, 'body', .5)
        with self.assertRaises(ValueError): series(data, -2.)


if __name__ == '__main__': unittest.main()
