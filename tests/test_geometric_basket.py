"""Geometric prices and coordinate deltas against independent numerical checks."""

import math
import unittest
from dataclasses import replace

import numpy as np
from scipy.integrate import quad

from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.geometric_basket import GeometricBasketReference
from bsde_pricing.monte_carlo import MonteCarloPricer
from bsde_pricing.problems import EuropeanCall, geometric_case, vanilla_case


class GeometricBasketTests(unittest.TestCase):
    def setUp(self):
        self.case = geometric_case()
        self.reference = GeometricBasketReference(self.case.model, self.case.contract)

    def test_price_against_direct_gaussian_payoff_integration(self):
        for rate, time in ((0.03, 0.0), (-0.01, 0.25)):
            model = replace(self.case.model, rate=rate)
            weights, sigma = self.case.contract.weights, model.volatilities
            tau = 1.0 - time
            # Direct log-distribution moments, independent of the reference's
            # effective-dividend reduction and its normal CDF calculation.
            mean = sum(w * (math.log(s) + (rate-q-0.5*v*v)*tau)
                       for w,s,q,v in zip(weights, self.case.initial_spots, model.dividend_yields, sigma))
            variance = tau * sum(weights[i]*weights[j]*sigma[i]*sigma[j]*model.correlation[i][j]
                                 for i in range(3) for j in range(3))
            deviation = math.sqrt(variance)
            threshold = (math.log(100) - mean) / deviation

            def integrand(z):
                return (math.exp(mean-rate*tau+deviation*z-0.5*z*z)
                        - 100*math.exp(-rate*tau-0.5*z*z)) / math.sqrt(2*math.pi)

            call, error = quad(integrand, threshold, math.inf, epsabs=1e-10, epsrel=1e-10)
            put, put_error = quad(lambda z: -integrand(z), -math.inf, threshold,
                                  epsabs=1e-10, epsrel=1e-10)
            actual = GeometricBasketReference(model, self.case.contract).evaluate(self.case.initial_spots, time)
            self.assertLess(max(error, put_error), 1e-8)
            self.assertAlmostEqual(actual.call_price, call, delta=1e-9)
            self.assertAlmostEqual(actual.put_price, put, delta=1e-9)
            parity = math.exp(mean + 0.5*variance - rate*tau) - 100*math.exp(-rate*tau)
            self.assertAlmostEqual(actual.call_price-actual.put_price, parity, delta=1e-11)

    def test_coordinate_bump_convergence_and_delta_parity(self):
        spots = np.array(self.case.initial_spots)
        value = self.reference.evaluate(spots)
        for j in range(3):
            errors = []
            for fraction in (0.02, 0.01, 0.005, 0.0025):
                h = spots[j] * fraction
                up, down = spots.copy(), spots.copy()
                up[j] += h
                down[j] -= h
                secant = (self.reference.evaluate(up).call_price -
                          self.reference.evaluate(down).call_price) / (2*h)
                errors.append(abs(secant-value.call_deltas[j]))
            self.assertLess(errors[-1], errors[0]/8)
            self.assertLess(errors[-1], 2e-6)
            expected = (self.case.contract.weights[j] * value.geometric_spot / spots[j]
                        * math.exp(-value.effective_dividend_yield))
            self.assertAlmostEqual(value.call_deltas[j]-value.put_deltas[j], expected, delta=1e-13)

    def test_joint_monte_carlo_price_and_coordinate_deltas(self):
        for rate, time in ((0.03, 0.0), (-0.01, 0.25)):
            model = replace(self.case.model, rate=rate)
            pricer = MonteCarloPricer(model, self.case.contract)
            reference = GeometricBasketReference(model, self.case.contract)
            exact = reference.evaluate(self.case.initial_spots, time)
            price = pricer.estimate(self.case.initial_spots, n_paths=100000,
                                   rng=np.random.default_rng(2026), time=time)
            self.assertLess(abs(price.price_estimate-exact.call_price), 6*price.standard_error)
            for j in range(3):
                risk = pricer.estimate_coordinate_delta(self.case.initial_spots, coordinate=j,
                    n_paths=100000, rng=np.random.default_rng(2026), time=time, bump_fraction=0.01)
                h = risk.bump_size
                up, down = list(self.case.initial_spots), list(self.case.initial_spots)
                up[j] += h
                down[j] -= h
                secant = (reference.evaluate(up,time).call_price-reference.evaluate(down,time).call_price)/(2*h)
                self.assertLess(abs(risk.pathwise.estimate-exact.call_deltas[j]), 6*risk.pathwise.standard_error)
                self.assertLess(abs(risk.central_bump.estimate-secant), 6*risk.central_bump.standard_error)
                self.assertLess(abs(risk.paired_difference.estimate-(exact.call_deltas[j]-secant)),
                                6*risk.paired_difference.standard_error+1e-10)
                # Reprice only the selected starting coordinate on matched streams.
                up_mc = pricer.estimate(up,n_paths=100000,rng=np.random.default_rng(2026),time=time)
                down_mc = pricer.estimate(down,n_paths=100000,rng=np.random.default_rng(2026),time=time)
                self.assertAlmostEqual(risk.central_bump.estimate,
                    (up_mc.price_estimate-down_mc.price_estimate)/(2*h), delta=2e-11)

    def test_scalar_reduction_and_equal_spot_identity(self):
        case = vanilla_case()
        contract = replace(case.contract,basket="geometric")
        for spot in (80,100,120):
            geometric = GeometricBasketReference(case.model,contract).evaluate((spot,))
            scalar = BlackScholesReference(case.model,case.contract).evaluate(spot)
            self.assertEqual(geometric.call_price, scalar.call_price)
            self.assertEqual(geometric.call_deltas, (scalar.call_delta,))
        self.assertEqual(float(self.case.contract.basket_value((100,100,100))),100.0)
        self.assertEqual(float(contract.basket_value((100,))),100.0)

    def test_deterministic_reference_limits_and_kinks(self):
        flat = replace(self.case.model,rate=0,dividend_yields=(0,0,0),volatilities=(0,0,0))
        for model,time in ((flat,0.0),(self.case.model,1.0)):
            reference = GeometricBasketReference(model,self.case.contract)
            kink = reference.evaluate((100,100,100),time)
            self.assertEqual(kink.call_price,0)
            self.assertEqual(kink.call_deltas,(None,None,None))
            value = reference.evaluate((120,120,120),time)
            self.assertAlmostEqual(value.call_price,20,delta=1e-12)
            np.testing.assert_allclose(value.call_deltas,self.case.contract.weights,rtol=0,atol=1e-14)
        model = replace(self.case.model,volatilities=(0,0.25,0.35))
        exact = GeometricBasketReference(model,self.case.contract).evaluate(self.case.initial_spots)
        delta = MonteCarloPricer(model,self.case.contract).estimate_coordinate_delta(
            self.case.initial_spots, coordinate=0,n_paths=100000,rng=np.random.default_rng(7))
        self.assertLess(abs(delta.pathwise.estimate-exact.call_deltas[0]),6*delta.pathwise.standard_error)
        deterministic_model = replace(self.case.model, volatilities=(0, 0, 0))
        value = GeometricBasketReference(deterministic_model, self.case.contract).evaluate(
            self.case.initial_spots)
        growth = sum(w * (deterministic_model.rate-q) for w,q in
                     zip(self.case.contract.weights, deterministic_model.dividend_yields))
        terminal_basket = value.geometric_spot * math.exp(growth)
        discount = math.exp(-deterministic_model.rate)
        self.assertAlmostEqual(value.call_price, discount * max(terminal_basket-100, 0), delta=1e-11)
        for j in range(3):
            expected_delta = discount * self.case.contract.weights[j] * terminal_basket / self.case.initial_spots[j]
            self.assertAlmostEqual(value.call_deltas[j], expected_delta, delta=1e-13)
        with self.assertRaises(ValueError):
            MonteCarloPricer(deterministic_model, self.case.contract).estimate_coordinate_delta(
                self.case.initial_spots, coordinate=0, n_paths=10, rng=np.random.default_rng(0))

    def test_invalid_scope_spots_and_delta_coordinates(self):
        with self.assertRaises(ValueError):
            GeometricBasketReference(self.case.model,replace(self.case.contract,basket="arithmetic"))
        with self.assertRaises(ValueError):
            GeometricBasketReference(vanilla_case().model,self.case.contract)
        for spots in ((80,100),(0,100,120),(80,float('nan'),120)):
            with self.assertRaises(ValueError): self.reference.evaluate(spots)
        for time in (-1,2,float('nan')):
            with self.assertRaises(ValueError): self.reference.evaluate(self.case.initial_spots,time)
        pricer = MonteCarloPricer(self.case.model,self.case.contract)
        for coordinate in (-1,3,True,1.5):
            with self.assertRaises(ValueError):
                pricer.estimate_coordinate_delta(self.case.initial_spots,coordinate=coordinate,
                    n_paths=10,rng=np.random.default_rng(0))
        with self.assertRaises(ValueError):
            pricer.estimate_coordinate_delta(self.case.initial_spots,coordinate=0,
                n_paths=10,rng=np.random.default_rng(0),time=1)


if __name__ == "__main__":
    unittest.main()
