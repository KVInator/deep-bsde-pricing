"""Spot-risk recovery in physical coordinates."""

import math
import unittest
from dataclasses import replace

import numpy as np

from bsde_pricing.bsde import PricingBSDE
from bsde_pricing.problems import EuropeanCall, GBMModel, vanilla_case


class ControlRecoveryTests(unittest.TestCase):
    def test_correlated_hand_worked_orientation_and_batched_recovery(self):
        model = GBMModel(0.03, (0.2, 0.3), (0, 0), ((1, 0.6), (0.6, 1)))
        problem = PricingBSDE(model, EuropeanCall(100, 1, (0.5, 0.5)))
        result = problem.recover_deltas([100, 80], [18.08, 13.44])
        np.testing.assert_allclose(result.deltas, [0.4, 0.7], atol=2e-15, rtol=0)
        self.assertLess(float(result.absolute_residual), 1e-14)
        spots = np.array([[[100, 80], [95, 90]], [[110, 85], [130, 100]]], dtype=float)
        deltas = np.array([[[0.4, 0.7], [0.2, 0.3]], [[0.8, 0.1], [0, 0]]])
        recovered = problem.recover_deltas(spots, problem.control(spots, deltas))
        np.testing.assert_allclose(recovered.deltas, deltas, atol=3e-16, rtol=0)
        self.assertEqual(recovered.relative_residual[-1, -1], 0)
        self.assertFalse(recovered.deltas.flags.writeable)

    def test_scalar_condition_number_does_not_hide_inverse_amplification(self):
        case = vanilla_case()
        problem = PricingBSDE(replace(case.model, volatilities=(1e-10,)), case.contract)
        a = problem.recover_deltas([100], [1.0])
        b = problem.recover_deltas([100], [1.0+1e-8])
        self.assertEqual(float(a.condition_number), 1)
        self.assertAlmostEqual(float(a.inverse_norm), 1e8)
        self.assertAlmostEqual(float(b.deltas[0]-a.deltas[0]), 1, places=5)

    def test_near_singular_correlation_reports_condition_without_repair(self):
        rho = 0.999999
        model = GBMModel(0.03, (0.2, 0.2), (0, 0), ((1, rho), (rho, 1)))
        problem = PricingBSDE(model, EuropeanCall(100, 1, (0.5, 0.5)))
        result = problem.recover_deltas([100, 100], problem.control([100, 100], [0.3, 0.2]))
        self.assertAlmostEqual(float(result.condition_number), math.sqrt((1+rho)/(1-rho)), delta=1e-7)
        self.assertAlmostEqual(float(result.inverse_norm), 1/(20*math.sqrt(1-rho)), delta=1e-8)
        np.testing.assert_allclose(result.deltas, [0.3, 0.2], atol=1e-12)

    def test_invalid_recovery_contract_and_singular_diffusion(self):
        case = vanilla_case();problem = PricingBSDE(case.model, case.contract)
        for spots, controls in (([0], [1]), ([100], [[1]]), ([100], [float('nan')])):
            with self.assertRaises(ValueError): problem.recover_deltas(spots, controls)
        with self.assertRaises(ValueError):
            PricingBSDE(replace(case.model, volatilities=(0,)), case.contract).recover_deltas([100], [0])
