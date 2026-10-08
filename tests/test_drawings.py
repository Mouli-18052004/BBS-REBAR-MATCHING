"""Hand-drawing tolerance without losing local geometric discrimination."""
import math
import unittest

import numpy as np

import matcher as vm


class HandDrawnTests(unittest.TestCase):
    def test_heading_alignment_is_independent_of_initial_tangent(self):
        profile = np.r_[np.zeros(25), np.linspace(0, 1.4, 20),
                        np.full(25, 1.4), np.linspace(1.4, .4, 24)]
        for offset in (.35, -1.1, 2 * math.pi - .2):
            self.assertLess(vm.heading_dtw(profile, profile + offset), 1e-5)

    def test_endpoint_wobble_does_not_bias_entire_profile(self):
        profile = np.r_[np.zeros(30), np.linspace(0, 1.5, 30), np.full(34, 1.5)]
        wobbled = profile.copy()
        wobbled[:7] += np.linspace(.3, 0, 7)
        wobbled -= wobbled[0]
        self.assertLess(vm.heading_dtw(profile, wobbled), 2.)
        wrong_turn = -profile
        self.assertGreater(vm.heading_dtw(profile, wrong_turn), 10.)

    def test_fit_is_symmetric_and_handles_different_sample_counts(self):
        a = np.linspace(0, 1.7, 60)
        b = np.linspace(0, 1.7, 94) + .8
        self.assertAlmostEqual(vm.heading_dtw(a, b), vm.heading_dtw(b, a), places=5)
        self.assertLess(vm.heading_dtw(a, b), 1.)

    def test_local_terminal_direction_remains_discriminative(self):
        a = np.r_[np.zeros(30), np.full(35, 1.4), np.full(29, .5)]
        b = np.r_[np.zeros(30), np.full(35, 1.4), np.full(29, 2.3)]
        self.assertGreater(vm.heading_dtw(a, b), 10.)
        self.assertEqual(vm.heading_dtw([], b), 9999.)


if __name__ == '__main__':
    unittest.main()
