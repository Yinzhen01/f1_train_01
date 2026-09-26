import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools.amp.audit_recorded_estimator import single_observations, replay


class RecordedEstimatorTest(unittest.TestCase):
    def inputs(self):
        n = 120
        q = np.tile((np.arange(n)*.001).astype(np.float32)[:, None], (1, 12))
        root = np.zeros((n, 13), dtype=np.float32); root[:, 6] = 1.
        action = np.zeros_like(q); action[1:] = q[:-1]
        data = dict(time=(np.arange(n)+1)*.01, root_state=root, dof_pos=q,
                    dof_vel=np.zeros_like(q), action=action,
                    base_ang_vel=np.zeros((n, 3), dtype=np.float32),
                    base_lin_vel=np.zeros((n, 3), dtype=np.float32))
        names = ['q'+str(i) for i in range(12)]
        cfg = dict(env=dict(num_single_obs=47, frame_stack=66, short_frame_stack=5, use_ref_actions=False),
            noise=dict(add_noise=False), domain_rand={},
            commands=dict(ranges=dict(lin_vel_x=[.45, .45], lin_vel_y=[0., 0.], ang_vel_yaw=[0., 0.])),
            normalization=dict(obs_scales=dict(dof_pos=1., dof_vel=.05, lin_vel=2., ang_vel=.25, quat=1.),
                               clip_observations=100., clip_actions=100.),
            init_state=dict(default_joint_angles={n: 0. for n in names}))
        return data, dict(environment=cfg, dof_names=names)

    def test_observation_layout_and_next_transition_alignment(self):
        data, manifest = self.inputs()
        obs = single_observations(data, manifest)
        self.assertEqual(tuple(obs.shape), (120, 47))
        torch.testing.assert_close(obs[5, :5], torch.tensor([0., 1., .9, 0., 0.]))
        policy = SimpleNamespace(num_short_obs=235,
            act_inference=lambda x: x[:, -42:-41].expand(-1, 12),
            state_estimator=lambda x: torch.zeros((len(x), 3)))
        row = replay(data, manifest, policy)
        self.assertEqual(row['time'][0], .66)
        self.assertEqual(row['parity']['max_action_abs_error'], 0.)
        self.assertEqual(len(row['time']), 11)
        data['action'] += .1
        with self.assertRaisesRegex(ValueError, 'action parity'):
            replay(data, manifest, policy)

    def test_lag_and_nonfixed_commands_rejected(self):
        data, manifest = self.inputs()
        manifest['environment']['domain_rand']['add_imu_lag'] = True
        with self.assertRaisesRegex(ValueError, 'lagged'):
            single_observations(data, manifest)
        manifest['environment']['domain_rand'].clear()
        manifest['environment']['commands']['ranges']['lin_vel_x'] = [0., .45]
        with self.assertRaisesRegex(ValueError, 'fixed'):
            single_observations(data, manifest)


if __name__ == '__main__':
    unittest.main()
