"""IID Monte Carlo prices from exact terminal model states."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

from .problems import EuropeanCall, GBMModel


@dataclass(frozen=True)
class MonteCarloResult:
    price_estimate: float
    sample_standard_deviation: float
    standard_error: float
    confidence_level: float
    confidence_low: float | None
    confidence_high: float | None
    n_paths: int
    positive_payoff_count: int
    time: float
    time_to_maturity: float


@dataclass(frozen=True)
class MonteCarloPricer:
    model: GBMModel
    contract: EuropeanCall

    def __post_init__(self) -> None:
        if self.model.dimension != self.contract.dimension:
            raise ValueError("Model and contract dimensions must agree.")

    def estimate(
        self, initial_spots: ArrayLike, *, n_paths: int,
        rng: np.random.Generator, time: float = 0.0,
    ) -> MonteCarloResult:
        if (
            isinstance(n_paths, (bool, np.bool_))
            or not isinstance(n_paths, (int, np.integer)) or n_paths < 2
        ):
            raise ValueError("n_paths must be an integer of at least two.")
        n_paths = int(n_paths)
        time = float(time)
        if not math.isfinite(time) or not 0 <= time <= self.contract.maturity:
            raise ValueError("time must be finite and between zero and maturity.")
        tau = self.contract.maturity - time
        discount = math.exp(-self.model.rate * tau)
        if discount == 0 or not math.isfinite(discount):
            raise ValueError("Discount factor is outside the supported numerical range.")
        terminal = self.model.sample_terminal(initial_spots, tau, n_paths, rng)
        with np.errstate(over="raise", under="raise", invalid="raise"):
            discounted_payoffs = discount * self.contract.payoff(terminal)
            deterministic = tau == 0 or self.model.volatilities[0] == 0
            if deterministic:
                price = float(discounted_payoffs[0])
                sample_std = 0.0
            else:
                price = float(discounted_payoffs.mean())
                sample_std = float(discounted_payoffs.std(ddof=1))
        standard_error = sample_std / math.sqrt(n_paths)
        positive_count = int(np.count_nonzero(discounted_payoffs > 0))
        if deterministic:
            lower = upper = price
        elif standard_error == 0:
            lower = upper = None
        else:
            half_width = NormalDist().inv_cdf(0.975) * standard_error
            lower, upper = price - half_width, price + half_width
        return MonteCarloResult(
            price_estimate=price,
            sample_standard_deviation=sample_std,
            standard_error=standard_error,
            confidence_level=0.95,
            confidence_low=lower,
            confidence_high=upper,
            n_paths=n_paths,
            positive_payoff_count=positive_count,
            time=time,
            time_to_maturity=tau,
        )
