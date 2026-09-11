import unittest
import numpy as np
from humanoid.gmr_rollout_metrics import rotation_error, summarize_rollout, trunk_metrics


class RolloutMetricsTest(unittest.TestCase):
    def fixture(self):
        root = np.zeros((3, 13))
        root[:, 2] = .6
        root[:, 6] = 1.
        return dict(time=np.array([0, .01, .02]), root_state=root,
                    dof_pos=np.zeros((3, 12)), reference_dof_pos=np.zeros((3, 12)),
                    reference_root_pos=root[:, :3].copy(), reference_root_quat=root[:, 3:7].copy(),
                    foot_diagnostics_valid=[False, True, True], physical_failure=[False] * 3,
                    foot_force=np.tile([0., 0., 20.], (3, 2, 1)),
                    sole_velocity=np.zeros((3, 2, 3)), reference_contact=np.ones((3, 2)),
                    torque=np.zeros((3, 12)))

    def test_exact_completed_motion(self):
        result = summarize_rollout(self.fixture(), .02)
        self.assertTrue(result['completed_clip'])
        self.assertEqual(result['joint_rmse_rad'], 0.)
        self.assertEqual(result['contact_match_score'], 1.)

    def test_failure_at_clip_end_is_not_success(self):
        data = self.fixture()
        data['physical_failure'][-1] = True
        self.assertFalse(summarize_rollout(data, .02)['completed_clip'])

    def test_initial_and_noncontact_samples_excluded(self):
        data = self.fixture()
        data['sole_velocity'][0] = 100.
        data['sole_velocity'][1:, 0, 0] = .2
        data['sole_velocity'][1:, 1, 0] = 10.
        data['foot_force'][:, 1] = 0.
        self.assertAlmostEqual(summarize_rollout(data, .02)['contact_sole_speed_mean_m_s'], .2)

    def test_quaternion_sign_invariance(self):
        q = np.array([[0., 0., 0., 1.]])
        np.testing.assert_allclose(rotation_error(q, -q), 0.)

    def test_movable_chest_not_confused_with_pelvis(self):
        data = self.fixture()
        angle = np.deg2rad(10.) / 2
        data['trunk_quat'] = np.tile([0., np.sin(angle), 0., np.cos(angle)], (3, 1))
        data['reference_trunk_quat'] = data['reference_root_quat'].copy()
        result = summarize_rollout(data, .02)
        self.assertAlmostEqual(result['root_rotation_mean_deg'], 0.)
        self.assertAlmostEqual(result['trunk_tilt_error_mean_deg'], 10.)

    def test_smoothness_excludes_reset_sample(self):
        data = self.fixture()
        data['action'] = np.tile(np.array([100., 1., 3.])[:, None], (1, 12))
        data['torque'] = np.tile(np.array([100., 2., 5.])[:, None], (1, 12))
        result = summarize_rollout(data, .02)
        self.assertAlmostEqual(result['action_delta_rms'], 2.)
        self.assertAlmostEqual(result['torque_delta_rms_nm'], 3.)

    def test_trunk_pitch_and_error(self):
        angles = np.deg2rad(np.array([-10., 0., 5.]))
        q = np.column_stack((np.zeros(3), np.sin(angles / 2), np.zeros(3), np.cos(angles / 2)))
        target = np.tile([0., 0., 0., 1.], (3, 1))
        result = trunk_metrics(q, target)
        self.assertAlmostEqual(result['trunk_tilt_error_mean_deg'], 5.)
        self.assertAlmostEqual(result['actual_pitch_mean_deg'], -5. / 3.)
        self.assertAlmostEqual(result['actual_pitch_min_deg'], -10.)
        self.assertAlmostEqual(result['actual_pitch_max_deg'], 5.)

    def test_trunk_error_ignores_world_yaw(self):
        pitch, yaw = np.deg2rad([12., 77.]) / 2
        q = np.array([[0., np.sin(pitch), 0., np.cos(pitch)]])
        yaw_times_q = np.array([[-np.sin(yaw) * np.sin(pitch), np.cos(yaw) * np.sin(pitch),
                                np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch)]])
        result = trunk_metrics(yaw_times_q, q)
        self.assertAlmostEqual(result['trunk_tilt_error_mean_deg'], 0., places=10)
        self.assertAlmostEqual(result['actual_pitch_mean_deg'], 12.)


if __name__ == '__main__':
    unittest.main()
