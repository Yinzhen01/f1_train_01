import unittest

import numpy as np

from tools.amp.analyze_progress_failures import metrics, summarize


class ProgressFailureAnalysisTest(unittest.TestCase):
    def data(self):
        n = 600
        root = np.zeros((n, 13)); root[:, 6] = 1.; root[:, 2] = .6; root[:, 7] = .45
        body = np.zeros((n, 3)); body[:, 0] = .45
        force = np.zeros((n, 2, 3)); force[:, :, 2] = 100
        return dict(time=(np.arange(n)+1)*.01, root_state=root, base_lin_vel=body, foot_force=force,
                    dof_pos=np.zeros((n, 12)), action=np.zeros((n, 12)), torque=np.zeros((n, 12)))

    def test_exact_half_open_window_and_no_shortened_failed_prefix(self):
        data = self.data()
        row = metrics(data, 2., 4.)
        self.assertEqual(row['samples'], 200)
        self.assertAlmostEqual(row['body_vx'], .45)
        self.assertAlmostEqual(row['body_progress'], .02)
        self.assertAlmostEqual(row['qdd_fd_rms'], 0.)
        self.assertEqual(row['double_foot_force_fraction'], 1.)
        short = {k: v[:350] for k, v in data.items()}
        self.assertIsNone(metrics(short, 2., 4.))
        self.assertIsNone(metrics(data, 2., 2.1))

    def test_slow_backward_motion_not_reported_as_forward(self):
        data = self.data()
        data['base_lin_vel'][:, 0] = -.2
        data['root_state'][:, 7] = -.2
        row = metrics(data, 2., 4.)
        self.assertLess(row['body_progress'], 0.)
        self.assertLess(row['world_progress'], 0.)
        self.assertAlmostEqual(row['body_vx'], -.2)

    def test_missing_control_count_explicit(self):
        self.assertEqual(summarize([None]), dict(n=0, mean={}))
        summary = summarize([dict(samples=200, vx=1.), None, dict(samples=200, vx=3.)])
        self.assertEqual(summary, dict(n=2, mean=dict(vx=2.)))


if __name__ == '__main__':
    unittest.main()
