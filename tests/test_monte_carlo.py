"""Distribution, uncertainty and deterministic-limit checks for scalar simulation."""

import math
import unittest
from dataclasses import replace

import numpy as np

from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.monte_carlo import MonteCarloPricer
from bsde_pricing.problems import vanilla_case


class MonteCarloTests(unittest.TestCase):
    def test_seeded_sampler_shape_and_model_moments(self):
        case = vanilla_case()
        count = 100_000
        terminal = case.model.sample_terminal(
            case.initial_spots, 1.0, count, np.random.default_rng(2026)
        )
        repeat = case.model.sample_terminal(
            case.initial_spots, 1.0, count, np.random.default_rng(2026)
        )
        self.assertEqual(terminal.shape, (count, 1))
        self.assertTrue(np.all(terminal > 0))
        np.testing.assert_array_equal(terminal, repeat)
        log_returns = np.log(terminal[:, 0] / 100.0)
        self.assertLess(abs(log_returns.mean() - 0.01), 6 * 0.2 / math.sqrt(count))
        self.assertLess(
            abs(log_returns.var(ddof=1) - 0.04),
            6 * 0.04 * math.sqrt(2 / (count - 1)),
        )
        expected = 100.0 * math.exp(0.03)
        variance = 100.0**2 * math.exp(0.06) * math.expm1(0.04)
        self.assertLess(abs(terminal.mean() - expected), 6 * math.sqrt(variance / count))

    def test_price_uncertainty_against_analytical_reference(self):
        case = vanilla_case()
        model = replace(case.model, dividend_yields=(0.02,), rate=-0.01)
        pricer = MonteCarloPricer(model, case.contract)
        value = pricer.estimate(
            case.initial_spots, n_paths=100_000, rng=np.random.default_rng(7), time=0.25
        )
        reference = BlackScholesReference(model, case.contract).evaluate(100.0, time=0.25)
        self.assertGreater(value.standard_error, 0)
        self.assertLess(abs(value.price_estimate - reference.call_price), 6 * value.standard_error)
        self.assertAlmostEqual(
            value.standard_error * math.sqrt(value.n_paths),
            value.sample_standard_deviation,
        )
        self.assertLess(value.confidence_low, value.price_estimate)
        self.assertGreater(value.confidence_high, value.price_estimate)

    def test_standard_error_scaling_and_repeat_run_dispersion(self):
        case = vanilla_case()
        pricer = MonteCarloPricer(case.model, case.contract)
        small, large = [], []
        for seed in range(8):
            small.append(pricer.estimate(
                case.initial_spots, n_paths=2_000, rng=np.random.default_rng(seed)
            ))
            large.append(pricer.estimate(
                case.initial_spots, n_paths=20_000, rng=np.random.default_rng(seed + 100)
            ))
        ratio = np.mean([v.standard_error for v in small]) / np.mean([
            v.standard_error for v in large
        ])
        self.assertGreater(ratio, 2.8)
        self.assertLess(ratio, 3.6)
        reference = BlackScholesReference(case.model, case.contract).evaluate(100.0)
        z = np.array([
            (v.price_estimate - reference.call_price) / v.standard_error for v in large
        ])
        self.assertLess(abs(z.mean()), 2.0)
        self.assertGreater(np.mean(z**2), 0.1)
        self.assertLess(np.mean(z**2), 4.0)

    def test_deterministic_limits_and_random_state(self):
        case = vanilla_case()
        for model, time in ((replace(case.model, volatilities=(0.0,)), 0.0), (case.model, 1.0)):
            with self.subTest(time=time):
                rng = np.random.default_rng(9)
                before = rng.bit_generator.state
                value = MonteCarloPricer(model, case.contract).estimate(
                    case.initial_spots, n_paths=100, rng=rng, time=time
                )
                reference = BlackScholesReference(model, case.contract).evaluate(100.0, time=time)
                self.assertAlmostEqual(value.price_estimate, reference.call_price, delta=1e-12)
                self.assertEqual(value.standard_error, 0.0)
                self.assertEqual(value.confidence_low, value.confidence_high)
                self.assertEqual(before, rng.bit_generator.state)

    def test_zero_observed_variance_and_invalid_inputs(self):
        case = vanilla_case()
        pricer = MonteCarloPricer(case.model, replace(case.contract, strike=1e6))
        value = pricer.estimate(case.initial_spots, n_paths=100, rng=np.random.default_rng(0))
        self.assertEqual(value.positive_payoff_count, 0)
        self.assertEqual(value.standard_error, 0.0)
        self.assertIsNone(value.confidence_low)
        self.assertIsNone(value.confidence_high)
        for count in (0, 1, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                pricer.estimate(case.initial_spots, n_paths=count, rng=np.random.default_rng(0))
        with self.assertRaises(ValueError):
            case.model.sample_terminal((0.0,), 1.0, 10, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            case.model.sample_terminal((100.0,), -1.0, 10, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
