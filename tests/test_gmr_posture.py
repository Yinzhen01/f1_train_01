import unittest

import torch

from humanoid.gmr_posture import tilt_error_squared, tilt_tracking_reward


class TiltRewardTest(unittest.TestCase):
    def gravity(self, pitch):
        angle = torch.deg2rad(torch.tensor(pitch, dtype=torch.float32))
        return torch.stack((torch.sin(angle), torch.zeros_like(angle), -torch.cos(angle)), dim=-1)

    def test_correct_reference_is_optimal_even_when_not_vertical(self):
        gravity = self.gravity([0., 5., -5.])
        torch.testing.assert_close(tilt_tracking_reward(gravity, gravity, .1), torch.ones(3))

    def test_upright_target_penalizes_backward_lean(self):
        actual = self.gravity([0., -5., -10.])
        target = self.gravity([0., 0., 0.])
        reward = tilt_tracking_reward(actual, target, .1)
        self.assertGreater(float(reward[0]), float(reward[1]))
        self.assertGreater(float(reward[1]), float(reward[2]))
        self.assertLess(float(reward[2]), .06)

    def test_forward_target_is_not_competing_with_vertical_target(self):
        actual = self.gravity([5., 0., -5.])
        target = self.gravity([5., 5., 5.])
        reward = tilt_tracking_reward(actual, target, .1)
        self.assertEqual(float(reward[0]), 1.)
        self.assertGreater(float(reward[1]), float(reward[2]))

    def test_normalization_and_invalid_sigma(self):
        g = self.gravity([4.])
        torch.testing.assert_close(tilt_error_squared(g, g * 9.81), torch.zeros(1), atol=1e-12, rtol=0.)
        with self.assertRaises(ValueError):
            tilt_tracking_reward(g, g, 0.)


if __name__ == '__main__':
    unittest.main()
