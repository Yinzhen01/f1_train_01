import unittest
import numpy as np
from tools.amp.render_recorded_selection import select_interval


class RecordedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(time=np.array([28., 28.01, 28.02]),
            failure=np.array([False, False, True]), initial=dict(failure=False))

    def test_original_timestamps_and_failure_are_not_hidden(self):
        selected = select_interval(self.data, 28.01, 29.)
        np.testing.assert_array_equal(selected['time'], [28.01, 28.02])
        np.testing.assert_array_equal(selected['failure'], [False, True])
        self.assertEqual(len(self.data['time']), 3)

    def test_empty_interval_cannot_manufacture_video(self):
        with self.assertRaises(ValueError):
            select_interval(self.data, 29., 30.)

    def test_invalid_bounds_fail(self):
        for start, end in ((3., 2.), (-1., 2.), (np.nan, 2.), (1., np.inf)):
            with self.assertRaises(ValueError):
                select_interval(self.data, start, end)


if __name__ == '__main__':
    unittest.main()
