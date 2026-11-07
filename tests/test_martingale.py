"""Independent checks of the martingale estimator and its information set."""

import math
import unittest

import numpy as np

from bsde_pricing.martingale import MartingaleEstimator, run_known_control_study
from bsde_pricing.problems import geometric_five_case, vanilla_case


class MartingaleEstimatorTests(unittest.TestCase):
    def test_two_step_left_endpoint_identity(self):
        case = vanilla_case()
        estimator = MartingaleEstimator(case)
        increments = np.array([[[0.25], [-0.10]], [[-0.30], [0.45]]])

        def control(t, spots):
            return np.exp(-0.03 * t) * 0.1 * spots

        samples = estimator.sample(increments, discounted_control=control)
        first = increments[:, 0, 0]
        second = increments[:, 1, 0]
        midpoint = 100 * np.exp(0.01 * 0.5 + 0.2 * first)
        terminal = 100 * np.exp(0.01 + 0.2 * (first + second))
        h0 = 10.0
        h1 = math.exp(-0.03 * 0.5) * 0.1 * midpoint
        discounted_payoff = math.exp(-0.03) * np.maximum(terminal - 100, 0)
        np.testing.assert_allclose(samples.stochastic_integral, h0 * first + h1 * second)
        np.testing.assert_allclose(samples.integrated_squared_control, (h0**2 + h1**2) / 2)
        np.testing.assert_allclose(samples.controlled_payoff, discounted_payoff - h0 * first - h1 * second)
        zero = estimator.sample(increments)
        np.testing.assert_array_equal(zero.controlled_payoff, samples.discounted_payoff)
        np.testing.assert_array_equal(zero.stochastic_integral, np.zeros(2))

    def test_future_increment_cannot_change_left_endpoint_inputs(self):
        case = geometric_five_case()
        estimator = MartingaleEstimator(case)
        increments = np.random.default_rng(25).normal(size=(3, 4, 5)) / 2
        changed = increments.copy()
        changed[0, -1] += 0.1

        def evaluate(shocks):
            seen = []

            def control(t, spots):
                seen.append((t, spots.copy()))
                return spots / 100

            return estimator.sample(shocks, discounted_control=control), seen

        before, left_before = evaluate(increments)
        after, left_after = evaluate(changed)
        self.assertEqual([t for t, _ in left_after], [0, 0.25, 0.5, 0.75])
        for (_, first), (_, second) in zip(left_before, left_after):
            np.testing.assert_array_equal(first, second)
        expected_change = (left_before[-1][1][0] / 100) @ (changed[0, -1] - increments[0, -1])
        np.testing.assert_allclose(after.stochastic_integral - before.stochastic_integral,
                                   [expected_change, 0, 0])

    def test_known_controls_reduce_variance_without_price_shift(self):
        for case in (vanilla_case(), geometric_five_case()):
            with self.subTest(case=case.name):
                result = run_known_control_study(case, n_paths=4096, seed=2026, steps=(4, 32))
                self.assertLess(abs(result["zero_control"]["error_in_standard_errors"]), 4)
                coarse, fine = result["grids"]
                self.assertGreater(coarse["variance_reduction_factor"], 1)
                self.assertGreater(fine["variance_reduction_factor"], 4 * coarse["variance_reduction_factor"])
                for row in (coarse, fine):
                    self.assertLess(abs(row["analytical_control"]["error_in_standard_errors"]), 4)
                    self.assertLess(abs(row["stochastic_integral"]["mean_in_standard_errors"]), 4)
                    self.assertLess(abs(row["isometry_gap"]["mean_in_standard_errors"]), 4)

    def test_control_shape_and_grid_validation(self):
        estimator = MartingaleEstimator(vanilla_case())
        increments = np.zeros((2, 2, 1))
        with self.assertRaisesRegex(ValueError, "discounted control must have shape"):
            estimator.sample(increments, discounted_control=lambda t, spots: np.ones(len(spots)))
        with self.assertRaisesRegex(ValueError, "divide the finest"):
            run_known_control_study(estimator.case, n_paths=2, seed=2026, steps=(3, 4))
