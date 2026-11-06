"""Exact European prices and coordinate deltas for a weighted geometric basket."""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .black_scholes import BlackScholesReference
from .problems import EuropeanCall, GBMModel, PricingCase, _finite_array, _finite_vector


@dataclass(frozen=True)
class GeometricBasketResult:
    spots: tuple[float, ...]
    time: float
    time_to_maturity: float
    geometric_spot: float
    effective_volatility: float
    effective_dividend_yield: float
    basket_drift: float
    call_price: float
    put_price: float
    call_deltas: tuple[float | None, ...]
    put_deltas: tuple[float | None, ...]


@dataclass(frozen=True)
class GeometricBasketReference:
    model: GBMModel
    contract: EuropeanCall
    _scalar_reference: BlackScholesReference = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.contract.basket != "geometric":
            raise ValueError("Geometric reference requires a geometric-basket contract.")
        if self.model.dimension != self.contract.dimension:
            raise ValueError("Model and contract dimensions must agree.")
        weights = np.array(self.contract.weights)
        sigma = np.array(self.model.volatilities)
        with np.errstate(over="raise", under="raise", invalid="raise"):
            loading = np.linalg.cholesky(self.model.correlation).T @ (weights * sigma)
            variance = float(loading @ loading)
            if np.any(sigma > 0) and variance == 0:
                raise ValueError("Effective variance is below floating-point resolution.")
            weighted_variance = math.fsum(w * s * s for w, s in zip(weights, sigma))
        effective_dividend = math.fsum((
            self.model.rate * (1.0 - math.fsum(self.contract.weights)),
            math.fsum(w * q for w, q in zip(weights, self.model.dividend_yields)),
            0.5 * weighted_variance, -0.5 * variance,
        ))
        scalar_model = GBMModel(self.model.rate, (math.sqrt(variance),),
                               (effective_dividend,), ((1.0,),))
        scalar_contract = EuropeanCall(self.contract.strike, self.contract.maturity)
        object.__setattr__(self, "_scalar_reference",
                           BlackScholesReference(scalar_model, scalar_contract))

    def call_delta_array(self, spots: ArrayLike, time: float = 0.0) -> NDArray[np.float64]:
        """Return pre-expiry coordinate deltas for spots of shape (..., d)."""
        spots = _finite_array(spots, "spots")
        if (spots.ndim < 1 or spots.shape[-1] != self.model.dimension
                or not spots.size or np.any(spots <= 0)):
            raise ValueError("Spots must be positive with shape (..., dimension).")
        basket = self.contract.basket_value(spots)
        basket_delta = self._scalar_reference.call_delta_array(basket, time)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return (basket_delta*basket)[..., None]*np.asarray(self.contract.weights)/spots

    def evaluate(self, spots: ArrayLike, time: float = 0.0) -> GeometricBasketResult:
        spots = _finite_vector(spots, "spots")
        if len(spots) != self.model.dimension or any(s <= 0 for s in spots):
            raise ValueError("spots must be positive and match the model dimension.")
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            geometric_spot = float(self.contract.basket_value(spots))
            scalar = self._scalar_reference.evaluate(geometric_spot, time)
            chain = np.array(self.contract.weights) * geometric_spot / np.array(spots)
            call_deltas = (tuple(None for _ in spots) if scalar.call_delta is None else
                           tuple(float(scalar.call_delta * x) for x in chain))
            put_deltas = (tuple(None for _ in spots) if scalar.put_delta is None else
                          tuple(float(scalar.put_delta * x) for x in chain))
        if not all(x is None or math.isfinite(x) for x in (*call_deltas, *put_deltas)):
            raise ArithmeticError("Geometric-basket deltas exceeded floating-point range.")
        effective_model = self._scalar_reference.model
        return GeometricBasketResult(
            spots, scalar.time, scalar.time_to_maturity, geometric_spot,
            effective_model.volatilities[0], effective_model.dividend_yields[0],
            self.model.rate - effective_model.dividend_yields[0],
            scalar.call_price, scalar.put_price, call_deltas, put_deltas,
        )


def call_reference(case: PricingCase) -> BlackScholesReference | GeometricBasketReference | None:
    """Return an exact call reference when one is available for this basket."""
    if case.model.dimension == 1:
        return BlackScholesReference(case.model, case.contract)
    if case.contract.basket == "arithmetic":
        return None
    return GeometricBasketReference(case.model, case.contract)


def reference_deltas_on_grid(
    reference: BlackScholesReference | GeometricBasketReference,
    states: NDArray[np.float64],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return (paths, intervals, assets) for a grid including expiry."""
    return np.stack([reference.call_delta_array(states[:, m], t)
                     for m, t in enumerate(times[:-1])], axis=1)
