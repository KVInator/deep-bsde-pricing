"""Single-asset European prices and spot Greeks under constant GBM parameters."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from .problems import EuropeanCall, GBMModel, _finite_array


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _normal_arguments(
    log_spot: float | NDArray[np.float64], strike: float, rate: float,
    dividend: float, tau: float, deviation: float,
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    log_forward_moneyness = log_spot - math.log(strike) + (rate - dividend) * tau
    d1 = log_forward_moneyness / deviation + 0.5 * deviation
    return d1, d1 - deviation


@dataclass(frozen=True)
class BlackScholesResult:
    spot: float
    time: float
    time_to_maturity: float
    call_price: float
    put_price: float
    call_delta: float | None
    put_delta: float | None
    gamma: float | None


@dataclass(frozen=True)
class BlackScholesReference:
    model: GBMModel
    contract: EuropeanCall

    def __post_init__(self) -> None:
        if self.model.dimension != 1 or self.contract.weights != (1.0,):
            raise ValueError("Black-Scholes reference requires one asset and unit weight.")

    def call_delta_array(self, spots: ArrayLike, time: float = 0.0) -> NDArray[np.float64]:
        """Evaluate delta at one time, retaining the input array's shape.

        Path diagnostics use this vectorised form before expiry with positive
        volatility. Deterministic limits and kinks remain in evaluate().
        """
        spots = _finite_array(spots, "spots")
        time = float(time)
        if spots.size == 0 or np.any(spots <= 0):
            raise ValueError("spots must be nonempty and strictly positive.")
        if not math.isfinite(time) or not 0 <= time < self.contract.maturity:
            raise ValueError("Array deltas require a time strictly before maturity.")
        tau = self.contract.maturity - time
        sigma = self.model.volatilities[0]
        deviation = sigma * math.sqrt(tau)
        if not math.isfinite(deviation) or deviation == 0:
            raise ValueError("Array deltas require positive finite integrated volatility.")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            d1, _ = _normal_arguments(
                np.log(spots), self.contract.strike, self.model.rate,
                self.model.dividend_yields[0], tau, deviation,
            )
            discount_q = math.exp(-self.model.dividend_yields[0] * tau)
            deltas = np.asarray(discount_q * ndtr(d1), dtype=np.float64)
        if discount_q == 0 or not np.isfinite(deltas).all():
            raise ArithmeticError("Array deltas exceeded the supported numerical range.")
        return deltas

    def evaluate(self, spot: float, time: float = 0.0) -> BlackScholesResult:
        spot, time = float(spot), float(time)
        if not math.isfinite(spot) or spot <= 0:
            raise ValueError("spot must be finite and strictly positive.")
        if not math.isfinite(time) or not 0 <= time <= self.contract.maturity:
            raise ValueError("time must be finite and between zero and maturity.")

        tau = self.contract.maturity - time
        rate = self.model.rate
        dividend = self.model.dividend_yields[0]
        sigma = self.model.volatilities[0]
        strike = self.contract.strike
        try:
            discount_r = math.exp(-rate * tau)
            discount_q = math.exp(-dividend * tau)
        except OverflowError as exc:
            raise ValueError("Discount factors exceed floating-point range.") from exc
        discounted_spot = spot * discount_q
        discounted_strike = strike * discount_r
        if (
            discount_r == 0
            or discount_q == 0
            or not math.isfinite(discounted_spot)
            or not math.isfinite(discounted_strike)
        ):
            raise ValueError("Discounted inputs are outside the supported numerical range.")

        if tau == 0 or sigma == 0:
            gap = discounted_spot - discounted_strike
            if gap == 0:
                call_delta = put_delta = gamma = None
            else:
                call_delta = discount_q if gap > 0 else 0.0
                put_delta = -discount_q if gap < 0 else 0.0
                gamma = 0.0
            return BlackScholesResult(
                spot=spot,
                time=time,
                time_to_maturity=tau,
                call_price=max(gap, 0.0),
                put_price=max(-gap, 0.0),
                call_delta=call_delta,
                put_delta=put_delta,
                gamma=gamma,
            )

        deviation = sigma * math.sqrt(tau)
        if not math.isfinite(deviation) or deviation == 0:
            raise ValueError("Integrated volatility is outside the supported numerical range.")
        d1, d2 = _normal_arguments(
            math.log(spot), strike, rate, dividend, tau, deviation,
        )
        call_price = math.fsum(
            (discounted_spot * _normal_cdf(d1),
             -discounted_strike * _normal_cdf(d2))
        )
        put_price = math.fsum(
            (discounted_strike * _normal_cdf(-d2),
             -discounted_spot * _normal_cdf(-d1))
        )
        density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        gamma = (discount_q * density / deviation) / spot
        values = (call_price, put_price, gamma)
        if not all(math.isfinite(value) for value in values):
            raise ArithmeticError("Black-Scholes calculation exceeded floating-point range.")
        if call_price < 0 or put_price < 0:
            raise ArithmeticError("Precision loss produced a negative option price.")
        return BlackScholesResult(
            spot=spot,
            time=time,
            time_to_maturity=tau,
            call_price=call_price,
            put_price=put_price,
            call_delta=discount_q * _normal_cdf(d1),
            put_delta=-discount_q * _normal_cdf(-d1),
            gamma=gamma,
        )
