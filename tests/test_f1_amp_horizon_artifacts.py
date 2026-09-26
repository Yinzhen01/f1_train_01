import copy
import unittest
from unittest.mock import Mock

from tools.amp.verify_horizon_formal import validate_updates, read_cloud_log


class HorizonArtifactTests(unittest.TestCase):
    def test_raw_full_log_and_live_api_envelope(self):
        raw = '[f1-amp-update] {"iteration":1501}\n[f1-amp-complete] {}\n'
        self.assertEqual(read_cloud_log(Mock(suffix='.log', read_text=Mock(return_value=raw))), raw)
        import json
        wrapped = json.dumps(dict(data=raw.replace('\n', '\\n')))
        self.assertEqual(read_cloud_log(Mock(suffix='.json', read_text=Mock(return_value=wrapped))), raw)
        with self.assertRaises(ValueError):
            read_cloud_log(Mock(suffix='.txt', read_text=Mock(return_value=raw)))

    def test_exact_bounded_continuation_updates(self):
        rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
                     discriminator_bridge_gradient_penalty=.1, style_negative_fraction=0)
                for i in range(1501, 1751)]
        self.assertTrue(validate_updates(rows))
        for invalid in (rows[1:], rows[:-1], rows+[rows[-1]], rows[::-1]):
            with self.assertRaises(ValueError): validate_updates(invalid)

    def test_invalid_or_changed_reward_metrics_are_rejected(self):
        rows = [dict(iteration=i, style_reward=.01, weighted_style_reward=.01,
                     discriminator_bridge_gradient_penalty=.1, style_negative_fraction=0)
                for i in range(1501, 1751)]
        for key, value in (('weighted_style_reward', 0), ('discriminator_bridge_gradient_penalty', 0),
                           ('style_negative_fraction', .1), ('style_reward', float('nan'))):
            invalid = copy.deepcopy(rows); invalid[100][key] = value
            with self.assertRaises(ValueError): validate_updates(invalid)


if __name__ == '__main__': unittest.main()
