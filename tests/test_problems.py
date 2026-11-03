"""Economic examples and invalid-input checks for pricing definitions."""

import json
import unittest
from dataclasses import replace

import numpy as np

from bsde_pricing.problems import EuropeanCall, GBMModel, PricingCase, vanilla_case


class PricingProblemTests(unittest.TestCase):
    def test_vanilla_case_and_terminal_payoffs(self):
        case = vanilla_case()
        self.assertEqual(case.initial_spots, (100.0,))
        self.assertEqual(case.model.rate, 0.03)
        self.assertEqual(case.model.volatilities, (0.20,))
        np.testing.assert_array_equal(
            case.contract.payoff([[90.0], [100.0], [110.0]]), [0.0, 0.0, 10.0]
        )
        self.assertEqual(case.contract.payoff([110.0]).shape, ())
        conventions = json.loads(case.to_json())["conventions"]
        self.assertEqual(conventions["measure"], "risk_neutral")

    def test_arithmetic_and_geometric_basket_economics(self):
        call = EuropeanCall(110.0, 1.0, weights=(0.5, 0.5))
        np.testing.assert_allclose(call.payoff([[100.0, 144.0]]), [12.0])
        geometric = replace(call, basket="geometric")
        np.testing.assert_allclose(geometric.payoff([[100.0, 144.0]]), [10.0])
        with self.assertRaisesRegex(ValueError, "shape"):
            call.payoff([[100.0]])
        np.testing.assert_array_equal(geometric.payoff([[0.0, 144.0]]), [0.0])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            geometric.payoff([[-1.0, 144.0]])

    def test_model_rejects_invalid_parameters_and_correlations(self):
        model = GBMModel(
            0.03, (0.2, 0.3), (0.0, 0.0), ((1.0, 0.3), (0.3, 1.0))
        )
        invalid = (
            {"rate": float("nan")},
            {"volatilities": (0.2, -0.3)},
            {"volatilities": (0.2, float("inf"))},
            {"volatilities": np.array([0.2 + 0.1j, 0.3])},
            {"dividend_yields": (0.0,)},
            {"correlation": ((1.0,),)},
            {"correlation": ((1.0, 0.3), (0.2, 1.0))},
            {"correlation": ((2.0, 0.3), (0.3, 1.0))},
            {"correlation": ((1.0, 1.0), (1.0, 1.0))},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(model, **changes)

    def test_contract_rejects_invalid_inputs_and_preserves_limits(self):
        call = vanilla_case().contract
        invalid = (
            {"strike": 0.0},
            {"maturity": -1.0},
            {"weights": (0.6, 0.6)},
            {"weights": (0.0, 1.0)},
            {"weights": (float("nan"),)},
            {"basket": "unknown"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(call, **changes)
        self.assertEqual(replace(call, maturity=0.0).maturity, 0.0)
        model = replace(vanilla_case().model, rate=-0.01, volatilities=(0.0,))
        self.assertEqual(model.rate, -0.01)

    def test_case_rejects_invalid_spots_and_dimension_mismatch(self):
        case = vanilla_case()
        for spots in ((0.0,), (float("inf"),), (100.0, 100.0)):
            with self.subTest(spots=spots), self.assertRaises(ValueError):
                replace(case, initial_spots=spots)
        with self.assertRaisesRegex(ValueError, "weights"):
            PricingCase(
                "invalid",
                case.model,
                EuropeanCall(100.0, 1.0, weights=(0.5, 0.5)),
                (100.0,),
            )


if __name__ == "__main__":
    unittest.main()
