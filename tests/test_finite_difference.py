"""PDE accuracy, convergence and boundary checks against closed-form prices."""

import unittest
from dataclasses import replace

import numpy as np

from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.finite_difference import FiniteDifferencePricer
from bsde_pricing.problems import EuropeanCall, GBMModel, vanilla_case


class FiniteDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.case = vanilla_case()
        self.pricer = FiniteDifferencePricer(self.case.model, self.case.contract)
        self.reference = BlackScholesReference(self.case.model, self.case.contract)

    def test_price_and_greeks_across_financial_inputs(self):
        # Absolute budgets in price, delta and gamma units for a 0.5 spot mesh.
        budgets = (0.002, 0.0001, 0.00002)
        for rate, dividend, sigma, time in (
            (0.03, 0.0, 0.2, 0.0), (-0.01, 0.02, 0.35, 0.25),
            (0.06, 0.02, 0.1, 0.0), (-0.06, 0.0, 0.1, 0.0),
        ):
            model = replace(self.case.model, rate=rate, dividend_yields=(dividend,),
                            volatilities=(sigma,))
            result = FiniteDifferencePricer(model, self.case.contract).solve(
                n_space=800, n_time=800, time=time,
            )
            reference = BlackScholesReference(model, self.case.contract)
            for spot in (80.0, 100.0, 120.0):
                with self.subTest(rate=rate, dividend=dividend, sigma=sigma, spot=spot):
                    actual, expected = result.evaluate(spot), reference.evaluate(spot, time)
                    for field, budget in zip(("call_price", "call_delta", "gamma"), budgets):
                        self.assertLess(abs(getattr(actual, field) - getattr(expected, field)), budget)
            self.assertEqual(result.upwind_node_count > 0, sigma == 0.1)

    def test_aligned_spatial_refinement(self):
        expected = self.reference.evaluate(100.0)
        errors = []
        for size in (100, 200, 400):
            value = self.pricer.solve(n_space=size, n_time=2048).evaluate(100.0)
            errors.append([abs(getattr(value, key) - getattr(expected, key))
                           for key in ("call_price", "call_delta", "gamma")])
        ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
        # Strike and valuation spot are nodes on every grid. A factor-four
        # error reduction tests second-order space behaviour, not a snapshot.
        self.assertTrue(np.all((ratios > 3.7) & (ratios < 4.4)), ratios)

    def test_time_refinement_and_rannacher_gamma_damping(self):
        expected = self.reference.evaluate(100.0)
        plain = self.pricer.solve(n_space=800, n_time=8, rannacher_steps=0).evaluate(100)
        smooth = self.pricer.solve(n_space=800, n_time=8).evaluate(100)
        self.assertLess(abs(smooth.gamma - expected.gamma), 0.001)
        self.assertGreater(abs(plain.gamma - expected.gamma), 0.1)
        # A fine-time solution on the same spatial grid isolates time error.
        fine = self.pricer.solve(n_space=800, n_time=4096).evaluate(100)
        errors = [abs(self.pricer.solve(n_space=800, n_time=n).evaluate(100).call_price
                      - fine.call_price) for n in (8, 16, 32)]
        self.assertTrue(all(3.8 < a / b < 4.3 for a, b in zip(errors, errors[1:])))

    def test_boundary_enlargement_at_fixed_spot_spacing(self):
        reference = self.reference.evaluate(100).call_price
        errors = []
        for maximum in (120.0, 200.0, 400.0):
            result = self.pricer.solve(s_max=maximum, n_space=int(maximum * 2), n_time=800)
            errors.append(abs(result.evaluate(100).call_price - reference))
            self.assertEqual(result.call_values[0], 0.0)
            self.assertTrue(np.all(np.diff(result.call_values) >= -1e-10))
            self.assertTrue(np.all(result.gammas >= -1e-10))
        self.assertGreater(errors[0], 0.1)
        self.assertLess(errors[1], 0.001)
        self.assertLess(abs(errors[2] - errors[1]), 1e-8)

    def test_grid_contract_and_interpolated_evaluation(self):
        result = self.pricer.solve(n_space=800, n_time=800)
        self.assertEqual(result.spots.shape, (801,))
        self.assertEqual(result.deltas.shape, (799,))
        self.assertEqual(result.gammas.shape, (799,))
        actual, expected = result.evaluate(101.3), self.reference.evaluate(101.3)
        self.assertLess(abs(actual.call_price - expected.call_price), 0.002)
        self.assertLess(abs(actual.call_delta - expected.call_delta), 0.0001)
        self.assertLess(abs(actual.gamma - expected.gamma), 0.00002)
        with self.assertRaises(ValueError):
            result.call_values[0] = 1.0
        for spot in (0, 0.1, 400, float("nan")):
            with self.subTest(spot=spot), self.assertRaises(ValueError):
                result.evaluate(spot)

    def test_scope_and_invalid_grids(self):
        model = GBMModel(0.03, (0.2, 0.2), (0, 0), ((1, 0), (0, 1)))
        for invalid_model in (model, replace(self.case.model, volatilities=(0.0,))):
            with self.assertRaises(ValueError):
                FiniteDifferencePricer(invalid_model, self.case.contract)
        for options in (
            {"n_space": 3}, {"n_space": True}, {"n_time": 0}, {"n_time": 1.5},
            {"rannacher_steps": -1}, {"n_time": 1}, {"s_max": 100},
            {"s_max": float("inf")}, {"time": -0.1}, {"time": 1.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.pricer.solve(**options)
        model = replace(self.case.model, dividend_yields=(1.0,))
        with self.assertRaisesRegex(ValueError, "boundary"):
            FiniteDifferencePricer(model, self.case.contract).solve(s_max=200)
        model = replace(self.case.model, rate=-2.0)
        with self.assertRaisesRegex(ValueError, "smaller time step"):
            FiniteDifferencePricer(model, EuropeanCall(100, 1)).solve(n_time=2)


if __name__ == "__main__":
    unittest.main()
