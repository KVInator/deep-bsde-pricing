"""Local loss identities, gradient flow, analytical oracles and fault detection."""

import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from bsde_pricing.analytical_torch import AnalyticalCall
from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.bsde import PricingBSDE
from bsde_pricing.bsde_loss import LocalBSDELoss
from bsde_pricing.geometric_basket import GeometricBasketReference
from bsde_pricing.loss_diagnostics import run_loss_diagnostics
from bsde_pricing.neural import ValueNetwork, value_derivatives
from bsde_pricing.problems import GBMModel, EuropeanCall, vanilla_case, geometric_case


class LocalBSDELossTests(unittest.TestCase):
    def test_one_step_against_hand_calculated_loss(self):
        case = vanilla_case()
        loss = LocalBSDELoss(PricingBSDE(case.model, case.contract), value_scale=10, terminal_weight=2)
        times = torch.tensor([0, 1], dtype=torch.float64)
        states = torch.tensor([[[100], [110]], [[100], [90]]], dtype=torch.float64)
        increments = torch.tensor([[[0.4]], [[-0.3]]], dtype=torch.float64)
        result = loss(lambda x: (0.4*x[:, 1]+2*x[:, 0])[:, None], times, states, increments)
        discount = math.exp(-0.03)
        residuals = np.array([discount*46-40-8*0.4, discount*38-40+8*0.3])
        terminal = discount*np.array([36, 38])
        np.testing.assert_allclose(result.residuals.detach().numpy()[:, 0], residuals, atol=1e-14)
        np.testing.assert_allclose(result.terminal_residual.detach().numpy(), terminal, atol=1e-14)
        self.assertAlmostEqual(float(result.local.detach()), float(np.mean(residuals**2)/100), delta=1e-13)
        self.assertAlmostEqual(float(result.terminal.detach()), float(np.mean(terminal**2)/100), delta=1e-13)
        self.assertAlmostEqual(float(result.total.detach()), float(np.mean(residuals**2+2*terminal**2)/100), delta=1e-13)

    def test_exact_path_axes_correlation_and_input_ownership(self):
        model = GBMModel(0.03, (0.2, 0.3), (0.01, 0.02), ((1, 0.6), (0.6, 1)))
        increments = np.array([[[0.1, 0.2], [-0.3, 0.4]], [[0.2, -0.1], [0.3, 0.2]]])
        original = increments.copy()
        states = model.path_from_increments([80, 120], 1, increments)
        np.testing.assert_array_equal(states[:, 0], [[80, 120], [80, 120]])
        for index, time in ((1, 0.5), (2, 1)):
            b = increments[:, :index].sum(axis=1)
            shocks = np.column_stack((b[:, 0], 0.6*b[:, 0]+0.8*b[:, 1]))
            expected = np.array([80, 120])*np.exp(np.array([0, -0.035])*time + np.array([0.2, 0.3])*shocks)
            np.testing.assert_allclose(states[:, index], expected, rtol=2e-15)
        np.testing.assert_array_equal(increments, original)
        for horizon, bad in ((0, increments), (1, np.zeros((2, 0, 2))),
                             (1, np.zeros((2, 2))), (1, increments*math.nan)):
            with self.assertRaises(ValueError): model.path_from_increments([80, 120], horizon, bad)

    def test_tensor_financial_adapter_matches_numpy_contract(self):
        for case in (vanilla_case(), geometric_case(), replace(geometric_case(),
                     contract=replace(geometric_case().contract, basket="arithmetic"))):
            problem = PricingBSDE(case.model, case.contract)
            loss = LocalBSDELoss(problem, value_scale=100)
            spots = np.array([case.initial_spots, np.array(case.initial_spots)*1.1])
            deltas = np.full_like(spots, 0.3)
            np.testing.assert_allclose(loss.control(torch.from_numpy(spots), torch.from_numpy(deltas)),
                                       problem.control(spots, deltas), atol=2e-14)
            np.testing.assert_allclose(loss.payoff(torch.from_numpy(spots)), problem.terminal_value(spots), atol=2e-13)
            np.testing.assert_allclose(loss.driver(torch.tensor([12.0], dtype=torch.float64)), problem.driver([12]))

    def test_known_solution_values_and_controls_against_scalar_references(self):
        for case in (vanilla_case(), geometric_case()):
            oracle = AnalyticalCall(case.model, case.contract)
            for t in (0.2, 0.8):
                points = torch.tensor([[t, *case.initial_spots]], dtype=torch.float64)
                d = value_derivatives(oracle, points, compute_hessian=False)
                self.assertIsNone(d.hessian)
                if case.model.dimension == 1:
                    r = BlackScholesReference(case.model, case.contract).evaluate(case.initial_spots[0], t)
                    target_delta = [r.call_delta]
                else:
                    r = GeometricBasketReference(case.model, case.contract).evaluate(case.initial_spots, t)
                    target_delta = r.call_deltas
                self.assertAlmostEqual(float(d.values[0, 0]), r.call_price, delta=2e-12)
                np.testing.assert_allclose(d.gradient[0, 1:], target_delta, atol=2e-14)
            at_expiry = torch.tensor([[case.contract.maturity, *case.initial_spots]], dtype=torch.float64)
            self.assertAlmostEqual(float(oracle(at_expiry)[0, 0]),
                                   float(case.contract.payoff(case.initial_spots)), delta=2e-13)

    def test_loss_parameter_gradient_matches_finite_weight_bump(self):
        case = vanilla_case()
        inc = np.array([[[0.1], [0.2]], [[-0.3], [0.4]]])
        states = torch.from_numpy(case.model.path_from_increments(case.initial_spots, 1, inc))
        times, increments = torch.tensor([0, 0.5, 1], dtype=torch.float64), torch.from_numpy(inc)
        network = ValueNetwork((100,), 1, 100, width=4)
        loss = LocalBSDELoss(PricingBSDE(case.model, case.contract), value_scale=100)
        result = loss(network, times, states, increments)
        result.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()))
        weight = next(network.parameters())
        automatic, original = float(weight.grad[0, 0]), float(weight.detach()[0, 0])
        estimates = []
        for sign in (1, -1):
            with torch.no_grad(): weight[0, 0] = original+sign*1e-5
            estimates.append(float(loss(network, times, states, increments, parameter_graph=False).total))
        with torch.no_grad(): weight[0, 0] = original
        self.assertAlmostEqual(automatic, (estimates[0]-estimates[1])/2e-5, delta=1e-8)
        self.assertIsNone(states.grad)
        self.assertFalse(loss(network, times, states, increments, parameter_graph=False).total.requires_grad)
        with torch.no_grad():
            evaluation = loss(network, times, states, increments, parameter_graph=False)
        torch.testing.assert_close(evaluation.total, result.total.detach())
        with torch.inference_mode():
            with self.assertRaises(ValueError): value_derivatives(network, torch.ones((1, 2), dtype=torch.float64))

    def test_refinement_and_faults_on_coupled_known_solutions(self):
        vanilla = vanilla_case()
        short_case = replace(vanilla, model=replace(
            vanilla.model, rate=-0.01, dividend_yields=(0.02,), volatilities=(0.35,),
        ), contract=replace(vanilla.contract, maturity=0.25))
        for case in (vanilla, geometric_case(), short_case):
            r = run_loss_diagnostics(case, n_paths=1024, steps=(8, 32, 64))
            self.assertTrue(r["passed"], r)
            self.assertLess(r["grids"][-1]["local_loss"], 0.25*r["grids"][0]["local_loss"])
            self.assertLess(r["grids"][-1]["local_residual_rms"], 0.25*r["grids"][0]["local_residual_rms"])
        case = vanilla_case()
        zero_rate = replace(case, model=replace(case.model, rate=0))
        r = run_loss_diagnostics(zero_rate, n_paths=512, steps=(8, 32))
        self.assertFalse(r["driver_fault_applicable"])
        self.assertFalse(r["correlation_fault_applicable"])
        self.assertNotIn("driver_sign_detected", r["checks"])

    def test_invalid_loss_inputs_and_diagnostic_contracts(self):
        case = vanilla_case()
        problem = PricingBSDE(case.model, case.contract)
        loss = LocalBSDELoss(problem, value_scale=100)
        oracle = AnalyticalCall(case.model, case.contract)
        times = torch.tensor([0, 1], dtype=torch.float64)
        states = torch.full((2, 2, 1), 100.0, dtype=torch.float64)
        increments = torch.zeros((2, 1, 1), dtype=torch.float64)
        for t, s, b in ((times.float(), states, increments), (times, states[:, :1], increments),
                        (times, -states, increments), (times, states*math.nan, increments),
                        (times, states.clone().requires_grad_(True), increments),
                        (times*0.5, states, increments)):
            with self.assertRaises(ValueError): loss(oracle, t, s, b)
        for scale, tw in ((0, 1), (100, 0), (math.inf, 1)):
            with self.assertRaises(ValueError): LocalBSDELoss(problem, value_scale=scale, terminal_weight=tw)
        for steps in ((), (8,), (8, 8), (8, 12), (8, True)):
            with self.assertRaises(ValueError): run_loss_diagnostics(case, steps=steps)
        with self.assertRaises(ValueError):
            AnalyticalCall(replace(case.model, volatilities=(0,)), case.contract)


if __name__ == "__main__":
    unittest.main()
