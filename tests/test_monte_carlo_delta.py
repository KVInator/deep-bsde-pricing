"""Pathwise risk, paired bump comparisons and derivative boundary checks."""

import unittest
from dataclasses import replace

import numpy as np

from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.monte_carlo import MonteCarloPricer
from bsde_pricing.problems import vanilla_case


class MonteCarloDeltaTests(unittest.TestCase):
    def test_pathwise_delta_against_analytical_reference(self):
        case = vanilla_case()
        for rate, dividend, time in ((0.03, 0.0, 0.0), (-0.01, 0.02, 0.25)):
            with self.subTest(rate=rate, dividend=dividend):
                model = replace(case.model, rate=rate, dividend_yields=(dividend,))
                value = MonteCarloPricer(model, case.contract).estimate_delta(
                    case.initial_spots, n_paths=100_000,
                    rng=np.random.default_rng(2026), time=time,
                )
                reference = BlackScholesReference(model, case.contract).evaluate(100.0, time)
                self.assertLess(
                    abs(value.pathwise.estimate - reference.call_delta),
                    6 * value.pathwise.standard_error,
                )

    def test_common_random_bump_and_paired_error(self):
        case = vanilla_case()
        pricer = MonteCarloPricer(case.model, case.contract)
        reference = BlackScholesReference(case.model, case.contract)
        delta = reference.evaluate(100.0).call_delta
        for fraction in (0.02, 0.01, 0.005, 0.001):
            with self.subTest(fraction=fraction):
                value = pricer.estimate_delta(
                    case.initial_spots, n_paths=100_000,
                    rng=np.random.default_rng(2026), bump_fraction=fraction,
                )
                h = value.bump_size
                secant = (
                    reference.evaluate(100.0 + h).call_price
                    - reference.evaluate(100.0 - h).call_price
                ) / (2 * h)
                self.assertLess(
                    abs(value.central_bump.estimate - secant),
                    6 * value.central_bump.standard_error,
                )
                self.assertLess(
                    abs(value.paired_difference.estimate - (delta - secant)),
                    6 * value.paired_difference.standard_error + 1e-10,
                )
        up = pricer.estimate(
            (100.0 + h,), n_paths=100_000, rng=np.random.default_rng(2026)
        )
        down = pricer.estimate(
            (100.0 - h,), n_paths=100_000, rng=np.random.default_rng(2026)
        )
        self.assertAlmostEqual(
            value.central_bump.estimate,
            (up.price_estimate - down.price_estimate) / (2 * h), delta=1e-11,
        )

    def test_deterministic_deltas_and_undefined_kinks(self):
        case = vanilla_case()
        for model, time in ((replace(case.model, volatilities=(0.0,)), 0.0), (case.model, 1.0)):
            for spot in (80.0, 120.0):
                with self.subTest(time=time, spot=spot):
                    rng = np.random.default_rng(1)
                    before = rng.bit_generator.state
                    value = MonteCarloPricer(model, case.contract).estimate_delta(
                        (spot,), n_paths=100, rng=rng, time=time,
                    )
                    exact = BlackScholesReference(model, case.contract).evaluate(spot, time)
                    self.assertAlmostEqual(value.pathwise.estimate, exact.call_delta, delta=1e-12)
                    self.assertAlmostEqual(
                        value.central_bump.estimate, exact.call_delta, delta=1e-11
                    )
                    self.assertEqual(value.pathwise.standard_error, 0.0)
                    self.assertEqual(before, rng.bit_generator.state)
        for model, time in (
            (case.model, 1.0),
            (replace(case.model, volatilities=(0.0,), rate=0.0), 0.0),
        ):
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, "undefined"):
                MonteCarloPricer(model, case.contract).estimate_delta(
                    (100.0,), n_paths=100, rng=np.random.default_rng(0), time=time,
                )
        crossing = MonteCarloPricer(case.model, case.contract).estimate_delta(
            (100.02,), n_paths=100, rng=np.random.default_rng(0), time=1.0,
        )
        self.assertEqual(crossing.pathwise.estimate, 1.0)
        self.assertEqual(crossing.bump_crossing_count, 100)
        expected_secant = (100.02 + crossing.bump_size - 100.0) / (2 * crossing.bump_size)
        self.assertAlmostEqual(crossing.central_bump.estimate, expected_secant, delta=1e-11)
        self.assertGreater(crossing.paired_difference.estimate, 0.3)

    def test_invalid_bumps_and_zero_observed_variation(self):
        case = vanilla_case()
        pricer = MonteCarloPricer(case.model, case.contract)
        for fraction in (0.0, -0.1, 1.0, float("nan"), 1e-20):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                pricer.estimate_delta(
                    (100.0,), n_paths=100, rng=np.random.default_rng(0),
                    bump_fraction=fraction,
                )
        rare = MonteCarloPricer(case.model, replace(case.contract, strike=1e6))
        value = rare.estimate_delta((100.0,), n_paths=100, rng=np.random.default_rng(0))
        self.assertEqual(value.positive_payoff_count, 0)
        self.assertIsNone(value.pathwise.confidence_low)
        self.assertIsNone(value.central_bump.confidence_low)
        self.assertIsNone(value.paired_difference.confidence_low)


if __name__ == "__main__":
    unittest.main()
