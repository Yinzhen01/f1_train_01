import unittest

import torch

from tools.amp.diagnose_style_gap import GROUPS, score_summary


class StyleGapDiagnosticsTest(unittest.TestCase):
    def test_style_zero_region_matches_training_formula(self):
        result = score_summary(torch.tensor([-2., -1., 0., 1., 2.]))
        self.assertAlmostEqual(result['reward_mean'], .5)
        self.assertAlmostEqual(result['zero_fraction'], .4)
        self.assertAlmostEqual(result['score_mean'], 0.)

    def test_feature_groups_cover_exact_39_dimensions_without_overlap(self):
        indices = [i for start, end in GROUPS.values() for i in range(start, end)]
        self.assertEqual(indices, list(range(39)))


if __name__ == '__main__':
    unittest.main()
