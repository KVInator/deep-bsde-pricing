"""Financial signs, Brownian coordinates and analytical residual refinement."""

import math
import unittest
from dataclasses import replace

import numpy as np

from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.bsde import AnalyticalBSDEDiagnostic, PricingBSDE
from bsde_pricing.problems import EuropeanCall, GBMModel, vanilla_case


class PricingBSDETests(unittest.TestCase):
    def setUp(self):
        self.case = vanilla_case()
        self.problem = PricingBSDE(self.case.model, self.case.contract)

    def test_brownian_coordinates_and_covariance(self):
        model = GBMModel(0.03, (0.2, 0.3), (0.01, 0.02), ((1, 0.6), (0.6, 1)))
        problem = PricingBSDE(model, EuropeanCall(100, 1, weights=(0.4, 0.6)))
        spots = np.array([[80.0, 120.0], [160.0, 240.0]])
        deltas = np.array([[0.3, 0.4], [0.3, 0.4]])
        original = spots.copy(), deltas.copy()
        diffusion = problem.diffusion(spots)
        np.testing.assert_allclose(diffusion[0], [[16, 0], [21.6, 28.8]], atol=1e-14)
        np.testing.assert_allclose(diffusion[1], 2 * diffusion[0])
        np.testing.assert_allclose(problem.control(spots, deltas), [[13.44, 11.52], [26.88, 23.04]])
        np.testing.assert_allclose(diffusion[0] @ diffusion[0].T, [[256, 345.6], [345.6, 1296]])
        np.testing.assert_allclose(problem.drift(spots), [[1.6, 1.2], [3.2, 2.4]])
        np.testing.assert_array_equal(spots, original[0])
        np.testing.assert_array_equal(deltas, original[1])
        self.assertFalse(np.shares_memory(diffusion, spots))
        self.assertEqual(problem.diffusion(spots.reshape(1, 2, 2)).shape, (1, 2, 2, 2))
        np.testing.assert_allclose(problem.terminal_value(spots), [4, 108])

    def test_driver_sign_and_pricing_pde(self):
        for rate, dividend in ((0.03, 0), (-0.02, 0.04), (0, 0.01)):
            model = replace(self.case.model, rate=rate, dividend_yields=(dividend,))
            problem = PricingBSDE(model, self.case.contract)
            reference = BlackScholesReference(model, self.case.contract)
            value = reference.evaluate(110, time=0.4)
            # Time differentiation of independently evaluated prices checks
            # the generator and driver together, including dividend carry.
            h = 1e-5
            time_derivative = (
                reference.evaluate(110, 0.4 + h).call_price
                - reference.evaluate(110, 0.4 - h).call_price
            ) / (2 * h)
            drift = problem.drift([110])[0]
            diffusion = problem.diffusion([110])[0, 0]
            pde_residual = (time_derivative + drift * value.call_delta
                            + 0.5 * diffusion**2 * value.gamma + problem.driver(value.call_price))
            self.assertAlmostEqual(float(pde_residual), 0, delta=2e-8)
            self.assertAlmostEqual(float(problem.driver(12)), -12 * rate)
        self.assertGreater(-float(self.problem.driver(12)), 0)
        self.assertAlmostEqual(self.problem.control([100], [0.5])[0], 10)
        zero_model = replace(self.case.model, volatilities=(0.0,))
        zero_problem = PricingBSDE(zero_model, self.case.contract)
        np.testing.assert_array_equal(zero_problem.control([100], [0.5]), [0])

    def test_vectorised_deltas_against_spot_bumps_and_scalar_reference(self):
        model = replace(self.case.model, rate=-0.01, dividend_yields=(0.03,))
        reference = BlackScholesReference(model, self.case.contract)
        spots = np.array([[80.0, 95.0], [100.0, 120.0]])
        for time in (0, 0.5, 0.99):
            deltas = reference.call_delta_array(spots, time)
            self.assertEqual(deltas.shape, spots.shape)
            for spot, delta in zip(spots.flat, deltas.flat):
                bump = spot * 1e-5
                numerical = (reference.evaluate(spot + bump, time).call_price
                             - reference.evaluate(spot - bump, time).call_price) / (2 * bump)
                self.assertAlmostEqual(delta, numerical, delta=2e-8)
                self.assertAlmostEqual(delta, reference.evaluate(spot, time).call_delta, delta=2e-15)
        self.assertFalse(np.shares_memory(deltas, spots))
        for values, time in (([], 0), ([0], 0), ([complex(100, 1)], 0), ([math.nan], 0),
                             ([100], -0.1), ([100], 1), ([100], math.nan)):
            with self.assertRaises(ValueError):
                reference.call_delta_array(values, time)
        with self.assertRaises(ValueError):
            BlackScholesReference(replace(model, volatilities=(0,)), self.case.contract).call_delta_array([100])

    def test_nested_coarse_path_against_manual_discounted_identity(self):
        diagnostic = AnalyticalBSDEDiagnostic(self.problem)
        result = diagnostic.run(100, n_paths=32, rng=np.random.default_rng(17), steps=(1, 2))
        brownian = np.random.default_rng(17).standard_normal((32, 2)).sum(axis=1) / math.sqrt(2)
        terminal = 100 * np.exp(0.01 + 0.2 * brownian)
        reference = BlackScholesReference(self.case.model, self.case.contract).evaluate(100)
        residual = (math.exp(-0.03) * np.maximum(terminal - 100, 0) - reference.call_price
                    - 20 * reference.call_delta * brownian)
        coarse = result.grids[0]
        self.assertAlmostEqual(coarse.residual_mean, float(residual.mean()), delta=1e-12)
        self.assertAlmostEqual(coarse.residual_standard_error, float(residual.std(ddof=1) / math.sqrt(32)), delta=1e-12)
        self.assertAlmostEqual(coarse.residual_rms, float(np.sqrt(np.mean(residual**2))), delta=1e-12)
        self.assertEqual(result, diagnostic.run(100, n_paths=32, rng=np.random.default_rng(17), steps=(1, 2)))

    def test_analytical_control_refines_across_financial_inputs(self):
        for spot, rate, dividend, volatility, maturity in (
            (100, 0.03, 0, 0.2, 1), (95, -0.01, 0.02, 0.35, 0.5),
        ):
            model = replace(self.case.model, rate=rate, dividend_yields=(dividend,), volatilities=(volatility,))
            problem = PricingBSDE(model, replace(self.case.contract, maturity=maturity))
            result = AnalyticalBSDEDiagnostic(problem).run(
                spot, n_paths=8192, rng=np.random.default_rng(123), steps=(8, 32, 128),
            )
            # A factor 16 in dt suggests about fourfold RMS improvement.
            # This broad bound tests convergence without fitting its order.
            self.assertLess(result.grids[-1].residual_rms, 0.4 * result.grids[0].residual_rms)
            for row in result.grids:
                self.assertLess(abs(row.residual_mean), 4 * row.residual_standard_error)

    def test_input_contracts_reject_before_sampling(self):
        diagnostic = AnalyticalBSDEDiagnostic(self.problem)
        rng = np.random.default_rng(2026)
        before = rng.bit_generator.state
        for count, steps in ((1, (8,)), (True, (8,)), (2, ()), (2, (8, 8)),
                             (2, (8, 12)), (2, (0, 8)), (2, (8.0, 16))):
            with self.assertRaises(ValueError):
                diagnostic.run(100, n_paths=count, rng=rng, steps=steps)
        self.assertEqual(before, rng.bit_generator.state)
        for spots in ([0], [100, 120], [[100], []], [math.inf], [complex(100, 1)]):
            with self.assertRaises(ValueError):
                self.problem.diffusion(spots)
        with self.assertRaises(ValueError):
            self.problem.control([[100]], [0.5])
        with self.assertRaises(ValueError):
            self.problem.driver(math.nan)
        with self.assertRaises(ValueError):
            PricingBSDE(self.case.model, EuropeanCall(100, 1, weights=(0.5, 0.5)))
        for model, contract in (
            (replace(self.case.model, volatilities=(0,)), self.case.contract),
            (self.case.model, replace(self.case.contract, maturity=0)),
        ):
            with self.assertRaises(ValueError):
                AnalyticalBSDEDiagnostic(PricingBSDE(model, contract))


if __name__ == "__main__":
    unittest.main()
