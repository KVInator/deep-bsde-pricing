"""Joint distribution, Cholesky orientation and scalar compatibility checks."""

import math
import unittest
from dataclasses import replace

import numpy as np

from bsde_pricing.monte_carlo import MonteCarloPricer
from bsde_pricing.problems import correlated_case, vanilla_case
from bsde_pricing.sampling import TerminalSampleDiagnostics


class CorrelatedSamplingTests(unittest.TestCase):
    def test_scalar_stream_matches_original_transition(self):
        case = vanilla_case()
        for horizon in (0.25, 1.0):
            normals = np.random.default_rng(2026).standard_normal((1000, 1))
            deviation = 0.2 * math.sqrt(horizon)
            variance = deviation * deviation
            drift = 0.03 * horizon - 0.5 * variance
            old_log_spots = np.full((1000, 1), math.log(100) + drift)
            old_log_spots += deviation * normals
            actual = case.model.sample_terminal(
                case.initial_spots, horizon, 1000, np.random.default_rng(2026),
            )
            np.testing.assert_array_equal(actual, np.exp(old_log_spots))

    def test_joint_moments_against_gaussian_and_lognormal_formulas(self):
        case = correlated_case()
        horizon, count = 0.7, 150_000
        terminal = case.model.sample_terminal(
            case.initial_spots, horizon, count, np.random.default_rng(2026),
        )
        self.assertEqual(terminal.shape, (count, 3))
        repeat = case.model.sample_terminal(
            case.initial_spots, horizon, count, np.random.default_rng(2026),
        )
        np.testing.assert_array_equal(terminal, repeat)
        report = TerminalSampleDiagnostics.from_sample(case.model, case.initial_spots, horizon, terminal)
        # Six-standard-error budgets cover these fixed, pointwise diagnostics;
        # a 95% interval is not a reliable all-components test threshold.
        for observed, expected, se in (
            (report.log_return_mean_observed, report.log_return_mean_expected,
             report.log_return_mean_standard_error),
            (report.log_return_covariance_observed, report.log_return_covariance_expected,
             report.log_return_covariance_standard_error),
            (report.terminal_mean_observed, report.terminal_mean_expected,
             report.terminal_mean_standard_error),
        ):
            self.assertTrue(np.all(np.abs(np.asarray(observed) - expected) < 6 * np.asarray(se)))
        # An explicit signed cross-covariance guards asset ordering and units.
        self.assertAlmostEqual(report.log_return_covariance_expected[0][2],
                               -0.2 * 0.15 * 0.35 * horizon)

    def test_cholesky_orientation_with_hand_worked_shocks(self):
        class FixedNormals:
            def standard_normal(self, shape):
                assert shape == (2, 3)
                return np.array([[1.0, 2.0, -1.0], [-0.5, 0.0, 1.5]])

        case = correlated_case()
        z = FixedNormals().standard_normal((2, 3))
        # Explicit lower-triangular factor of the fixture's correlation;
        # no call to the production factorisation in this oracle.
        y = np.column_stack((z[:, 0], 0.6 * z[:, 0] + 0.8 * z[:, 1],
             -0.2 * z[:, 0] + 0.525 * z[:, 1] + math.sqrt(0.684375) * z[:, 2]))
        terminal = case.model.sample_terminal(case.initial_spots, 1.0, 2, FixedNormals())
        actual_logs = np.log(terminal) - np.log(case.initial_spots)
        sigma = np.array(case.model.volatilities)
        expected_logs = 0.03 - np.array(case.model.dividend_yields) - 0.5 * sigma**2 + sigma * y
        np.testing.assert_allclose(actual_logs, expected_logs, rtol=0, atol=2e-15)

    def test_zero_horizon_and_mixed_zero_volatility(self):
        case = correlated_case()
        for model, horizon in (
            (case.model, 0.0), (replace(case.model, volatilities=(0, 0, 0)), 1.0),
        ):
            rng = np.random.default_rng(7)
            state = rng.bit_generator.state
            result = model.sample_terminal(case.initial_spots, horizon, 10, rng)
            self.assertEqual(state, rng.bit_generator.state)
            np.testing.assert_array_equal(result, np.broadcast_to(result[0], result.shape))
            report = TerminalSampleDiagnostics.from_sample(model, case.initial_spots, horizon, result)
            np.testing.assert_array_equal(report.log_return_covariance_expected, np.zeros((3, 3)))
            if horizon == 0:
                np.testing.assert_array_equal(result[0], case.initial_spots)
        model = replace(case.model, volatilities=(0.0, 0.25, 0.35))
        result = model.sample_terminal(case.initial_spots, 1.0, 10000, np.random.default_rng(2))
        np.testing.assert_array_equal(result[:, 0], np.full(len(result), result[0, 0]))
        self.assertGreater(result[:, 1].std(), 0.0)
        price = MonteCarloPricer(model, case.contract).estimate(
            case.initial_spots, n_paths=10000, rng=np.random.default_rng(2),
        )
        payoffs = math.exp(-model.rate) * case.contract.payoff(result)
        self.assertAlmostEqual(price.price_estimate, payoffs.mean())
        self.assertAlmostEqual(price.standard_error, payoffs.std(ddof=1) / 100)
        self.assertGreater(price.standard_error, 0)

    def test_sample_and_diagnostic_input_contracts(self):
        case = correlated_case()
        for spots, horizon, count in (
            ((80, 100), 1, 10), ((80, -100, 120), 1, 10),
            (case.initial_spots, -1, 10), (case.initial_spots, 1, True),
            (case.initial_spots, 1, 0), (case.initial_spots, 1, 1.5),
        ):
            with self.subTest(spots=spots, horizon=horizon, count=count), self.assertRaises(ValueError):
                case.model.sample_terminal(spots, horizon, count, np.random.default_rng(0))
        valid = np.ones((3, 3)) * 100
        for invalid in (np.ones(3), valid[:1], valid[:, :2], valid * 0,
                        valid * np.nan, valid.astype(complex) + 1j):
            with self.assertRaises(ValueError):
                TerminalSampleDiagnostics.from_sample(case.model, case.initial_spots, 1, invalid)
        report = TerminalSampleDiagnostics.from_sample(
            vanilla_case().model, (100,), 1, [[90], [110]],
        )
        self.assertEqual(np.asarray(report.log_return_covariance_observed).shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
