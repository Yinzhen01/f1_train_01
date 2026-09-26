import unittest
import numpy as np
from tools.amp.audit_gait_rhythm import rhythm_metrics, summarize


class GaitRhythmTests(unittest.TestCase):
    def test_antiphase_stride_is_not_two_times_cadence(self):
        t = np.arange(6000)*.01
        x = .2*np.sin(2*np.pi*.7*t)
        r = rhythm_metrics(np.c_[x, -x])
        self.assertTrue(r['valid'])
        for foot in r['feet']:
            self.assertAlmostEqual(foot['stride_hz'], .7, delta=.005)
        self.assertEqual(r['alternating_foremost_event_fraction'], 1.)
        self.assertEqual(r['near_simultaneous_event_fraction'], 0.)
        self.assertAlmostEqual(r['error_from_antiphase_deg'], 0., places=5)

    def test_inphase_is_not_alternation_even_if_ordering_flips(self):
        t = np.arange(6000)*.01
        x = .2*np.sin(2*np.pi*.7*t)
        r = rhythm_metrics(np.c_[x, x])
        self.assertGreater(r['near_simultaneous_event_fraction'], .49)
        self.assertAlmostEqual(r['error_from_antiphase_deg'], 180., places=5)

    def test_short_static_nonfinite_and_dt_rejected(self):
        self.assertFalse(rhythm_metrics(np.zeros((600, 2)))['valid'])
        self.assertFalse(rhythm_metrics(np.zeros((399, 2)))['valid'])
        with self.assertRaises(ValueError): rhythm_metrics(np.full((600, 2), np.nan))
        with self.assertRaises(ValueError): rhythm_metrics(np.zeros((600, 2)), .02)

    def test_high_frequency_flicker_not_extra_stride_and_failures_retained(self):
        t = np.arange(6000)*.01
        x = .2*np.sin(2*np.pi*.7*t)+.01*np.sin(2*np.pi*18*t)
        r = rhythm_metrics(np.c_[x, -x])
        self.assertAlmostEqual(r['feet'][0]['stride_hz'], .7, delta=.005)
        summary = summarize([{'failed': True, 'rhythm': r},
                             {'failed': False, 'rhythm': {'valid': False}}])
        self.assertEqual((summary['n'], summary['valid_n']), (2, 1))


if __name__ == '__main__': unittest.main()
