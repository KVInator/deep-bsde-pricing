"""Checks for price-only baselines and discrete matched-precision accounting."""

import unittest

import numpy as np
from scipy.stats import t

from bsde_pricing.basket_reference import BasketReferencePricer
from bsde_pricing.control_cost import CONVENTIONAL, _matched_precision, _sustained_break_even
from bsde_pricing.monte_carlo import _statistics
from bsde_pricing.problems import arithmetic_ten_case


class ControlCostTests(unittest.TestCase):
    def test_price_only_controls_reproduce_the_price_and_uncertainty(self):
        case = arithmetic_ten_case()
        pricer = BasketReferencePricer(case.model, case.contract)
        seed, pilot_pairs, scrambles = 87, 128, 8
        for branch in (0, 1):
            spots, _, tau, _, _, streams, known, beta, _ = pricer._setup(
                case.initial_spots, 0, seed, pilot_pairs, branch,
                scrambles+1 if branch else 2, price_only=True,
            )
            if branch:
                _, adjusted = pricer._sobol_means(spots, tau, 7, streams[1:], known, beta, 64, price_only=True)
                critical = float(t.ppf(0.975, scrambles-1))
                full = pricer.sobol(case.initial_spots, power=7, scrambles=scrambles, seed=seed,
                                    pilot_pairs=pilot_pairs, chunk_size=64)
            else:
                payoff, control = pricer._pairs(spots, tau, 512, np.random.default_rng(streams[1]),
                                               price_only=True)
                adjusted, critical = payoff-beta*(control-known), None
                full = pricer.antithetic(case.initial_spots, n_pairs=512, seed=seed, pilot_pairs=pilot_pairs)
            price = _statistics(adjusted[:, 0], False, critical_value=critical)
            self.assertAlmostEqual(float(beta[0]), full.control_coefficients[0], places=14)
            for field in ("estimate", "standard_error", "confidence_low", "confidence_high"):
                self.assertAlmostEqual(getattr(price, field), getattr(full.controlled.price, field), places=13)

    def test_matched_precision_uses_measured_allocations_and_charges_upfront_cost_once(self):
        anti, sobol = CONVENTIONAL
        learned = "terminal_loss_fitted"

        def measured(loose_passes, tight_passes, pilot, evaluation):
            return {"passes": {"0.025": loose_passes, "0.01": tight_passes},
                    "median_pilot_seconds": pilot, "median_evaluation_seconds": evaluation}

        results = [
            {"pairs": 4096, "methods": {
                anti: measured(16, 10, 1, 2), sobol: measured(16, 16, 1, 2),
                learned: measured(15, 0, 2, 3),
            }},
            {"pairs": 16384, "methods": {
                anti: measured(16, 16, 1, 3), sobol: measured(16, 16, 1, 3),
                learned: measured(16, 15, 2, 5),
            }},
        ]
        costs = {anti: {"training_seconds": 0}, sobol: {"training_seconds": 0},
                 learned: {"training_seconds": 10, "checkpoint_setup_seconds": 1}}
        matches = _matched_precision(results, costs, 16)
        row = matches["0.025"][learned]
        self.assertEqual(row["evaluation_states"], 32768)
        self.assertEqual(row["total_seconds"], {"1": 18, "16": 93, "100": 513})
        self.assertEqual(row["break_even_queries"], {anti: None, sobol: None})
        self.assertEqual(matches["0.01"][learned]["status"], "target_missed")
        self.assertEqual(matches["0.01"][anti]["evaluation_states"], 32768)
        self.assertEqual(matches["0.01"][sobol]["evaluation_states"], 8192)
        self.assertEqual(_sustained_break_even(10, 1, 2, 3), 4)
