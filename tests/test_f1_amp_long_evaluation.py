import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

from humanoid.amp.evaluation import validate_evaluation_budget, independent_mode_seeds
from tools.amp.evaluate_mounted import locate_checkpoint
from tools.amp.evaluate_long_horizon import SOURCES, audit_source


class LongEvaluationTests(unittest.TestCase):
    def test_drift_metrics_use_only_requested_valid_time_interval(self):
        from tools.amp.analyze_long_horizon import metrics
        t = (np.arange(100)+1)*.01
        root = np.zeros((100, 13)); root[:, 2] = .6; root[:, 6] = 1
        vel = np.zeros((100, 3)); vel[:, 0] = .45; vel[50:, 0] = 9
        force = np.zeros((100, 2, 3)); force[:, :, 2] = 100
        data = dict(time=t, root_state=root, base_lin_vel=vel, foot_force=force, dof_pos=np.zeros((100, 12)))
        out = metrics(data, np.zeros((100, 2)), np.ones((100, 2))*.1, 0, .5)
        self.assertAlmostEqual(out['vx_mean'], .45)
        self.assertAlmostEqual(out['vx_rmse_to_command'], 0.)
        self.assertAlmostEqual(out['slip_mean_m_s'], .1)
        self.assertIsNone(metrics(data, np.zeros((100, 2)), np.zeros((100, 2)), 0, .05))
        self.assertIsNone(metrics(data, np.zeros((100, 2)), np.zeros((100, 2)), 2, 6))

    def test_initial_pair_does_not_assume_same_seed_means_same_state(self):
        from tools.amp.verify_long_pair import initial_comparison
        a = {'standing_initial_root_state': np.zeros((16, 13))}
        b = {'standing_initial_root_state': np.zeros((16, 13))}
        self.assertTrue(initial_comparison(a, b)['all_captured_fields_exact_equal'])
        b['standing_initial_root_state'][3, 0] = .01
        result = initial_comparison(a, b)
        self.assertFalse(result['all_captured_fields_exact_equal'])
        self.assertEqual(result['fields']['standing_initial_root_state']['max_abs_difference'], .01)
        with self.assertRaises(ValueError): initial_comparison(a, {})

    def test_long_modes_do_not_depend_on_earlier_failures_rng(self):
        self.assertIsNone(independent_mode_seeds(5))
        self.assertEqual(independent_mode_seeds(5, True), {'standing': 5, 'reference': 105})

    def test_sixty_seconds_requires_explicit_flag_and_exact_budget(self):
        self.assertTrue(validate_evaluation_budget(16, 60., True))
        self.assertTrue(validate_evaluation_budget(16, 20.))
        self.assertTrue(validate_evaluation_budget(2, 1.))
        for budget in ((16, 60., False), (2, 60., True), (16, 20., True), (16, 120., True)):
            with self.assertRaises(ValueError): validate_evaluation_budget(*budget)

    def test_source_task_and_sha_cannot_be_mixed(self):
        for task, record in SOURCES.items():
            self.assertEqual(audit_source(task, record['sha256']), record)
            with self.assertRaises(ValueError): audit_source(task, '0'*64)
        with self.assertRaises(ValueError): audit_source('TASK_20260924_093', SOURCES['TASK_20260924_094']['sha256'])

    def test_exact1500_artifact_lookup_never_selects_old1000(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root/'model_1000.pt'; old.write_bytes(b'fixture old')
            new = root/'model_8801500_cloudsuffix.pt'; new.write_bytes(b'fixture exact')
            digest = hashlib.sha256(new.read_bytes()).hexdigest()
            self.assertEqual(locate_checkpoint([root], digest, 'model_8801500*.pt'), new.resolve())
            with self.assertRaises(FileNotFoundError): locate_checkpoint([root], digest)
            with self.assertRaises(ValueError): locate_checkpoint([root], digest, '*.pt')


if __name__ == '__main__':
    unittest.main()
