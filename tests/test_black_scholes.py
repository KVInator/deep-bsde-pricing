"""Independent price integration, economic identities and Greek convergence."""

import math
import unittest
from dataclasses import replace

from scipy.integrate import quad

from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.problems import EuropeanCall, GBMModel, vanilla_case


def integrated_prices(spot, strike, rate, dividend, sigma, tau):
    """Integrate discounted lognormal payoffs without a normal CDF formula."""
    deviation = sigma * math.sqrt(tau)
    threshold = (
        math.log(strike / spot) - (rate - dividend - 0.5 * sigma**2) * tau
    ) / deviation

    def discounted_difference(z):
        stock = spot * math.exp(
            -dividend * tau - 0.5 * deviation**2 + deviation * z - 0.5 * z**2
        )
        cash = strike * math.exp(-rate * tau - 0.5 * z**2)
        return (stock - cash) / math.sqrt(2 * math.pi)

    call, call_error = quad(
        discounted_difference,
        threshold, math.inf, epsabs=1e-11, epsrel=1e-11,
    )
    put, put_error = quad(
        lambda z: -discounted_difference(z),
        -math.inf, threshold, epsabs=1e-11, epsrel=1e-11,
    )
    return call, put, max(call_error, put_error)


class BlackScholesTests(unittest.TestCase):
    def test_prices_against_independent_payoff_integration(self):
        for spot, strike, rate, dividend, sigma, tau in (
            (100.0, 100.0, 0.03, 0.0, 0.20, 1.0),
            (100.0, 90.0, -0.01, 0.02, 0.35, 0.5),
            (100.0, 110.0, 0.03, 0.01, 0.15, 1.5),
        ):
            with self.subTest(spot=spot, strike=strike, rate=rate):
                model = GBMModel(rate, (sigma,), (dividend,), ((1.0,),))
                result = BlackScholesReference(model, EuropeanCall(strike, tau)).evaluate(spot)
                call, put, error = integrated_prices(spot, strike, rate, dividend, sigma, tau)
                self.assertLess(error, 1e-9)
                self.assertAlmostEqual(result.call_price, call, delta=5e-10)
                self.assertAlmostEqual(result.put_price, put, delta=5e-10)

    def test_parity_bounds_and_delta_relation(self):
        case = vanilla_case()
        model = replace(case.model, dividend_yields=(0.02,))
        reference = BlackScholesReference(model, case.contract)
        for spot in (80.0, 100.0, 120.0):
            with self.subTest(spot=spot):
                result = reference.evaluate(spot, time=0.25)
                tau = result.time_to_maturity
                stock = spot * math.exp(-0.02 * tau)
                cash = case.contract.strike * math.exp(-model.rate * tau)
                self.assertAlmostEqual(
                    result.call_price - result.put_price, stock - cash, delta=1e-11
                )
                self.assertGreaterEqual(result.call_price + 1e-11, max(stock - cash, 0.0))
                self.assertLessEqual(result.call_price, stock + 1e-11)
                self.assertGreaterEqual(result.put_price + 1e-11, max(cash - stock, 0.0))
                self.assertLessEqual(result.put_price, cash + 1e-11)
                self.assertAlmostEqual(
                    result.call_delta - result.put_delta,
                    math.exp(-0.02 * tau), delta=1e-13,
                )
                self.assertGreaterEqual(result.gamma, 0.0)

    def test_spot_bump_convergence(self):
        case = vanilla_case()
        reference = BlackScholesReference(case.model, case.contract)
        for spot in (80.0, 100.0, 120.0):
            with self.subTest(spot=spot):
                result = reference.evaluate(spot)
                delta_errors, gamma_errors = [], []
                for fraction in (0.01, 0.005, 0.0025, 0.00125):
                    h = spot * fraction
                    up = reference.evaluate(spot + h).call_price
                    down = reference.evaluate(spot - h).call_price
                    delta_errors.append(abs((up - down) / (2 * h) - result.call_delta))
                    gamma_errors.append(
                        abs((up - 2 * result.call_price + down) / h**2 - result.gamma)
                    )
                self.assertLess(delta_errors[-1], delta_errors[0] / 8)
                self.assertLess(gamma_errors[-1], gamma_errors[0] / 8)
                self.assertLess(delta_errors[-1], 3e-6)
                self.assertLess(gamma_errors[-1], 2e-7)

    def test_expiry_and_kink_conventions(self):
        case = vanilla_case()
        reference = BlackScholesReference(case.model, case.contract)
        for spot, call, put, delta in ((80.0, 0.0, 20.0, 0.0), (120.0, 20.0, 0.0, 1.0)):
            result = reference.evaluate(spot, time=1.0)
            self.assertEqual((result.call_price, result.put_price), (call, put))
            self.assertEqual(result.call_delta, delta)
            self.assertEqual(result.gamma, 0.0)
        kink = reference.evaluate(100.0, time=1.0)
        self.assertEqual(kink.call_price, 0.0)
        self.assertIsNone(kink.call_delta)
        self.assertIsNone(kink.put_delta)
        self.assertIsNone(kink.gamma)

    def test_zero_volatility_limit(self):
        case = vanilla_case()
        model = replace(case.model, volatilities=(0.0,), dividend_yields=(0.02,))
        reference = BlackScholesReference(model, case.contract)
        for spot in (80.0, 120.0):
            result = reference.evaluate(spot)
            gap = spot * math.exp(-0.02) - 100.0 * math.exp(-0.03)
            self.assertAlmostEqual(result.call_price, max(gap, 0.0))
            self.assertEqual(result.gamma, 0.0)
            limit = BlackScholesReference(
                replace(model, volatilities=(1e-8,)), case.contract
            ).evaluate(spot)
            self.assertAlmostEqual(result.call_price, limit.call_price, delta=1e-11)
            self.assertAlmostEqual(result.call_delta, limit.call_delta, delta=1e-12)
        flat = replace(model, rate=0.0, dividend_yields=(0.0,))
        self.assertIsNone(BlackScholesReference(flat, case.contract).evaluate(100.0).gamma)

    def test_reference_scope_and_evaluation_inputs(self):
        case = vanilla_case()
        reference = BlackScholesReference(case.model, case.contract)
        for spot in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(spot=spot), self.assertRaises(ValueError):
                reference.evaluate(spot)
        for time in (-0.1, 1.1, float("nan")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                reference.evaluate(100.0, time=time)
        for rate in (-1000.0, 1000.0):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                BlackScholesReference(
                    replace(case.model, rate=rate), case.contract
                ).evaluate(100.0)
        with self.assertRaisesRegex(ValueError, "one asset"):
            BlackScholesReference(
                GBMModel(0.03, (0.2, 0.2), (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),
                EuropeanCall(100.0, 1.0, weights=(0.5, 0.5)),
            )


if __name__ == "__main__":
    unittest.main()
