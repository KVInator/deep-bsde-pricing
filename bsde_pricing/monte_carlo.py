"""IID prices and pathwise spot sensitivities from exact terminal states."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .problems import EuropeanCall, GBMModel, _finite_vector


@dataclass(frozen=True)
class EstimateStatistics:
    estimate: float
    sample_standard_deviation: float
    standard_error: float
    confidence_level: float
    confidence_low: float | None
    confidence_high: float | None


def _statistics(samples: NDArray[np.float64], deterministic: bool) -> EstimateStatistics:
    with np.errstate(over="raise", under="raise", invalid="raise"):
        mean = float(samples[0]) if deterministic else float(samples.mean())
        sample_std = 0.0 if deterministic else float(samples.std(ddof=1))
    standard_error = sample_std / math.sqrt(len(samples))
    if deterministic:
        lower = upper = mean
    elif standard_error == 0:
        lower = upper = None
    else:
        half_width = NormalDist().inv_cdf(0.975) * standard_error
        lower, upper = mean - half_width, mean + half_width
    return EstimateStatistics(
        estimate=mean,
        sample_standard_deviation=sample_std,
        standard_error=standard_error,
        confidence_level=0.95,
        confidence_low=lower,
        confidence_high=upper,
    )


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
class MonteCarloDeltaResult:
    pathwise: EstimateStatistics
    central_bump: EstimateStatistics
    paired_difference: EstimateStatistics
    bump_fraction: float
    bump_size: float
    bump_crossing_count: int
    positive_payoff_count: int
    n_paths: int
    time: float
    time_to_maturity: float


@dataclass(frozen=True)
class MonteCarloPricer:
    model: GBMModel
    contract: EuropeanCall

    def __post_init__(self) -> None:
        if self.model.dimension != self.contract.dimension:
            raise ValueError("Model and contract dimensions must agree.")

    def _simulate(
        self, initial_spots: ArrayLike, *, n_paths: int,
        rng: np.random.Generator, time: float = 0.0,
    ) -> tuple[NDArray[np.float64], float, float]:
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
        return terminal, tau, discount

    def estimate(
        self, initial_spots: ArrayLike, *, n_paths: int,
        rng: np.random.Generator, time: float = 0.0,
    ) -> MonteCarloResult:
        terminal, tau, discount = self._simulate(
            initial_spots, n_paths=n_paths, rng=rng, time=time
        )
        deterministic = tau == 0 or self.model.volatilities[0] == 0
        with np.errstate(over="raise", under="raise", invalid="raise"):
            discounted_payoffs = discount * self.contract.payoff(terminal)
        stats = _statistics(discounted_payoffs, deterministic)
        positive_count = int(np.count_nonzero(discounted_payoffs > 0))
        return MonteCarloResult(
            price_estimate=stats.estimate,
            sample_standard_deviation=stats.sample_standard_deviation,
            standard_error=stats.standard_error,
            confidence_level=stats.confidence_level,
            confidence_low=stats.confidence_low,
            confidence_high=stats.confidence_high,
            n_paths=len(terminal),
            positive_payoff_count=positive_count,
            time=float(time),
            time_to_maturity=tau,
        )

    def estimate_delta(
        self, initial_spots: ArrayLike, *, n_paths: int,
        rng: np.random.Generator, time: float = 0.0, bump_fraction: float = 0.001,
    ) -> MonteCarloDeltaResult:
        """Compare pathwise delta and a central bump on paired terminal states."""
        if self.model.dimension != 1 or self.contract.weights != (1.0,):
            raise ValueError("Delta estimation requires one asset and unit weight.")
        bump_fraction = float(bump_fraction)
        if not math.isfinite(bump_fraction) or not 0 < bump_fraction < 1:
            raise ValueError("bump_fraction must be finite and strictly between zero and one.")
        spots = _finite_vector(initial_spots, "initial_spots")
        if len(spots) != 1 or spots[0] <= 0:
            raise ValueError("initial_spots must contain one strictly positive spot.")
        spot = spots[0]
        bump = spot * bump_fraction
        upper_spot, lower_spot = spot + bump, spot - bump
        if not math.isfinite(upper_spot) or not 0 < lower_spot < spot < upper_spot:
            raise ValueError("Spot bump is outside the representable positive range.")
        terminal, tau, discount = self._simulate(
            spots, n_paths=n_paths, rng=rng, time=time
        )
        deterministic = tau == 0 or self.model.volatilities[0] == 0
        exercise = terminal[:, 0] > self.contract.strike
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            if deterministic:
                dividend_discount = math.exp(-self.model.dividend_yields[0] * tau)
                gap = spot * dividend_discount - self.contract.strike * discount
                if gap == 0:
                    raise ValueError("Delta is undefined at a deterministic exercise boundary.")
                pathwise = np.full(len(terminal), dividend_discount if gap > 0 else 0.0)
            else:
                pathwise = discount * exercise * (terminal[:, 0] / spot)
            upper = terminal * (upper_spot / spot)
            lower = terminal * (lower_spot / spot)
            bumped = discount * (
                self.contract.payoff(upper) - self.contract.payoff(lower)
            ) / (2 * bump)
            difference = pathwise - bumped
        crossing_count = int(np.count_nonzero(
            (upper[:, 0] > self.contract.strike) != (lower[:, 0] > self.contract.strike)
        ))
        return MonteCarloDeltaResult(
            pathwise=_statistics(pathwise, deterministic),
            central_bump=_statistics(bumped, deterministic),
            paired_difference=_statistics(difference, deterministic),
            bump_fraction=bump_fraction,
            bump_size=bump,
            bump_crossing_count=crossing_count,
            positive_payoff_count=int(np.count_nonzero(exercise)),
            n_paths=len(terminal),
            time=float(time),
            time_to_maturity=tau,
        )
