import hashlib
from pathlib import Path
import tempfile
import unittest

from humanoid.amp.evaluation import validate_evaluation_budget, independent_mode_seeds
from tools.amp.evaluate_mounted import locate_checkpoint
from tools.amp.evaluate_long_horizon import SOURCES, audit_source


class LongEvaluationTests(unittest.TestCase):
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
