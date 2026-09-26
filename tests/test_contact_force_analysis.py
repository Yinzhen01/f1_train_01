import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tools.amp.diagnose_contact_tail import force_statistics, FORCE_FIELDS
from tools.amp.diagnose_progress_frame import series, summarize


class ContactForceAnalysisTests(unittest.TestCase):
    def test_no_post_transient_samples_are_missing_not_zero(self):
        result = force_statistics(np.empty((0, 2)))
        self.assertEqual(set(result), set(FORCE_FIELDS))
        self.assertTrue(all(v is None for v in result.values()))

    def test_configured_tail_separated_from_original_and_hypothetical(self):
        force = np.array([[1500., 0.]])
        old = force_statistics(force)
        tail = force_statistics(force, -.0025)
        self.assertAlmostEqual(old['actual_total_force_reward_mean'], -.04)
        self.assertAlmostEqual(tail['current_capped_reward_mean'], -.04)
        self.assertAlmostEqual(tail['actual_tail_reward_mean'], -.01)
        self.assertAlmostEqual(tail['actual_total_force_reward_mean'], -.05)
        self.assertAlmostEqual(tail['extra_uncapped_tail_reward_mean'], -.04)

    def test_nonfinite_and_unexpected_shape_rejected(self):
        for force in (np.zeros((3, 3)), np.array([[float('nan'), 0.]])):
            with self.assertRaises(ValueError): force_statistics(force)

    def test_body_reward_is_blind_to_global_rotation_but_world_reward_is_not(self):
        root = np.zeros((3, 13), dtype=np.float32)
        rotations = Rotation.from_euler('z', [0., 50., 90.], degrees=True)
        root[:, 3:7] = rotations.as_quat()
        body = np.tile([.45, 0., 0.], (3, 1)).astype(np.float32)
        root[:, 7:10] = rotations.apply(body)
        data = dict(root_state=root, base_lin_vel=body, time=np.array([1., 2., 3.]))
        v = series(data)
        np.testing.assert_allclose(v['body_progress'], .02, atol=1e-8)
        self.assertAlmostEqual(float(v['world_progress'][0]), .02)
        self.assertLess(v['world_progress'][1], 0.)
        self.assertLess(v['world_progress'][2], 0.)
        self.assertAlmostEqual(float(v['heading'][2]), -.005)
        self.assertGreater(summarize(data, v)['body_vx_mean'], summarize(data, v)['world_vx_mean'])

    def test_no_post2s_progress_is_not_zero(self):
        data = dict(time=np.array([1.]))
        self.assertTrue(all(v is None for v in summarize(data, {}).values()))

    def test_actual_frame_selects_corresponding_offline_reward(self):
        data = dict(time=np.array([2.]), base_lin_vel=np.zeros((1, 3)), root_state=np.zeros((1, 13)))
        v = dict(body_progress=np.array([.02]), world_progress=np.array([-.01]),
                 heading=np.zeros(1), yaw=np.zeros(1))
        self.assertEqual(summarize(data, v)['selected_progress_reward_mean'], .02)
        self.assertEqual(summarize(data, v, 'world')['selected_progress_reward_mean'], -.01)
        with self.assertRaises(ValueError): summarize(data, v, 'unknown')


if __name__ == '__main__': unittest.main()
