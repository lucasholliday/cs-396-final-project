import unittest
from scripts.audit_dataset import timing


class TimingTests(unittest.TestCase):
    def test_regular_ns(self):
        result = timing([0, 5000000, 10000000], 1e-9)
        self.assertAlmostEqual(result['median_hz'], 200)
        self.assertEqual(result['gaps_over_threshold'], 0)

    def test_clock_reset_and_gap_not_hidden(self):
        result = timing([0, .01, .02, .05, .04])
        self.assertEqual(result['nonpositive_steps'], 1)
        self.assertEqual(result['gaps_over_threshold'], 1)

    def test_single_sample(self):
        self.assertIsNone(timing([1])['median_hz'])


if __name__ == '__main__':
    unittest.main()
