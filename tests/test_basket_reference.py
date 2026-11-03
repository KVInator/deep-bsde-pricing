"""Sampling-unit uncertainty, independent references and control behaviour."""

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from bsde_pricing.basket_reference import BasketReferencePricer
from bsde_pricing.geometric_basket import GeometricBasketReference
from bsde_pricing.problems import correlated_case, geometric_case


class BasketReferenceTests(unittest.TestCase):
    def setUp(self):
        self.case = correlated_case()
        self.pricer = BasketReferencePricer(self.case.model, self.case.contract)

    def test_explicit_normal_transition_preserves_sampler_and_inputs(self):
        c = self.case
        normals = np.random.default_rng(2026).standard_normal((1000, 3))
        before = normals.copy()
        mapped = c.model.terminal_from_normals(c.initial_spots, 0.7, normals)
        sampled = c.model.sample_terminal(c.initial_spots, 0.7, 1000, np.random.default_rng(2026))
        np.testing.assert_array_equal(mapped, sampled)
        np.testing.assert_array_equal(normals, before)
        np.testing.assert_array_equal(c.model.terminal_from_normals(c.initial_spots, 0, normals)[0],
                                      c.initial_spots)
        for invalid in (np.zeros(3), np.zeros((1, 2)), np.zeros((0, 3)), normals * np.nan):
            with self.assertRaises(ValueError):
                c.model.terminal_from_normals(c.initial_spots, 1, invalid)

    def test_antithetic_standard_error_uses_pair_averages(self):
        # Reconstruct the documented evaluation stream and form pair payoffs
        # directly; counting the two dependent legs as IID would fail this.
        count, seed = 2048, 9
        stream = np.random.SeedSequence(seed, spawn_key=(0,)).spawn(2)[1]
        z = np.random.default_rng(stream).standard_normal((count, 3))
        plus = self.case.model.terminal_from_normals(self.case.initial_spots, 1, z)
        minus = self.case.model.terminal_from_normals(self.case.initial_spots, 1, -z)
        pairs = math.exp(-0.03) * (self.case.contract.payoff(plus)
                                  + self.case.contract.payoff(minus)) / 2
        result = self.pricer.antithetic(self.case.initial_spots, n_pairs=count, pilot_pairs=256, seed=seed)
        self.assertAlmostEqual(result.plain.price.estimate, pairs.mean(), delta=1e-12)
        self.assertAlmostEqual(result.plain.price.standard_error, pairs.std(ddof=1)/math.sqrt(count), delta=1e-13)
        self.assertEqual(result.n_units, count)
        self.assertEqual(result.total_terminal_states, 2*(count+256))

    def test_sobol_uncertainty_uses_independent_scramble_means(self):
        result = self.pricer.sobol(self.case.initial_spots, power=10, scrambles=8, pilot_pairs=256)
        repeated = self.pricer.sobol(self.case.initial_spots, power=10, scrambles=8, pilot_pairs=256)
        self.assertEqual(result, repeated)
        means = np.array(result.replicate_controlled)
        self.assertGreater(np.ptp(means[:, 0]), 0)
        self.assertEqual(result.n_units, 8)
        self.assertEqual(result.evaluation_terminal_states, 8*1024)
        self.assertAlmostEqual(result.controlled.price.standard_error,
                               means[:, 0].std(ddof=1)/math.sqrt(8), delta=1e-13)
        half = (result.controlled.price.confidence_high-result.controlled.price.confidence_low)/2
        self.assertAlmostEqual(half, t.ppf(0.975, 7)*result.controlled.price.standard_error, delta=1e-12)
        # Pilot coefficients do not depend on evaluation net size/count.
        larger = self.pricer.sobol(self.case.initial_spots, power=11, scrambles=8, pilot_pairs=256)
        self.assertEqual(result.control_coefficients, larger.control_coefficients)

    def test_geometric_oracle_for_both_reference_methods(self):
        c = geometric_case()
        pricer = BasketReferencePricer(c.model, c.contract)
        exact = GeometricBasketReference(c.model, c.contract).evaluate(c.initial_spots)
        for result in (pricer.antithetic(c.initial_spots, n_pairs=8192, pilot_pairs=512),
                       pricer.sobol(c.initial_spots, power=11, scrambles=8, pilot_pairs=512)):
            for stat, target in zip((result.plain.price, *result.plain.deltas),
                                    (exact.call_price, *exact.call_deltas)):
                self.assertLess(abs(stat.estimate-target), 6*stat.standard_error)
            self.assertAlmostEqual(result.controlled.price.estimate, exact.call_price, delta=1e-11)

    def test_arithmetic_agreement_bounds_and_variance_reduction(self):
        anti = self.pricer.antithetic(self.case.initial_spots, n_pairs=32768)
        sobol = self.pricer.sobol(self.case.initial_spots, power=12)
        for a, b in zip((anti.controlled.price, *anti.controlled.deltas),
                        (sobol.controlled.price, *sobol.controlled.deltas)):
            self.assertLess(abs(a.estimate-b.estimate), 6*math.hypot(a.standard_error, b.standard_error))
        for result in (anti, sobol):
            self.assertGreater(result.controlled.price.estimate, result.lower_price_bound)
            self.assertLess(result.controlled.price.estimate, result.upper_price_bound)
        self.assertLess(anti.controlled.price.standard_error, anti.plain.price.standard_error/3)
        # Independently bump the arithmetic price with matched Sobol streams.
        for j in range(3):
            up, down = list(self.case.initial_spots), list(self.case.initial_spots)
            h = up[j]*0.002
            up[j] += h
            down[j] -= h
            high = self.pricer.sobol(up, power=12, pilot_pairs=256)
            low = self.pricer.sobol(down, power=12, pilot_pairs=256)
            prices_up = np.array(high.replicate_plain)[:, 0]
            prices_down = np.array(low.replicate_plain)[:, 0]
            secants = (prices_up-prices_down)/(2*h)
            deltas = np.array(sobol.replicate_plain)[:, j+1]
            differences = secants-deltas
            self.assertLess(abs(differences.mean()), 6*differences.std(ddof=1)/4 + 2e-5)

    def test_degenerate_pilot_and_invalid_allocations(self):
        rare = BasketReferencePricer(self.case.model, replace(self.case.contract, strike=1e12))
        result = rare.antithetic(self.case.initial_spots, n_pairs=64, pilot_pairs=64)
        self.assertEqual(result.unfitted_controls, (0, 1, 2, 3))
        self.assertEqual(result.control_coefficients, (0, 0, 0, 0))
        self.assertIsNone(result.controlled.price.confidence_low)
        for options in ({"power": 0}, {"power": 31}, {"scrambles": 3},
                        {"pilot_pairs": 1}, {"seed": -1}, {"seed": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.pricer.sobol(self.case.initial_spots, **options)
        with self.assertRaises(ValueError):
            self.pricer.antithetic(self.case.initial_spots, n_pairs=1)
        with self.assertRaises(ValueError):
            self.pricer.antithetic(self.case.initial_spots, time=1)
        flat = replace(self.case.model, volatilities=(0, 0, 0))
        with self.assertRaises(ValueError):
            BasketReferencePricer(flat, self.case.contract).sobol(self.case.initial_spots)

    def test_sobol_endpoints_are_rejected_without_tail_clipping(self):
        with patch("bsde_pricing.basket_reference.qmc.Sobol") as factory:
            factory.return_value.random_base2.return_value = np.zeros((4, 3))
            with self.assertRaisesRegex(ArithmeticError, "no clipping"):
                self.pricer.sobol(self.case.initial_spots, power=2, scrambles=4, pilot_pairs=64)


if __name__ == "__main__":
    unittest.main()
