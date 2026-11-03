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


def _statistics(
    samples: NDArray[np.float64], deterministic: bool, *, critical_value: float | None = None,
) -> EstimateStatistics:
    with np.errstate(over="raise", under="raise", invalid="raise"):
        mean = float(samples[0]) if deterministic else float(samples.mean())
        sample_std = 0.0 if deterministic else float(samples.std(ddof=1))
    standard_error = sample_std / math.sqrt(len(samples))
    if deterministic:
        lower = upper = mean
    elif standard_error == 0:
        lower = upper = None
    else:
        critical = NormalDist().inv_cdf(0.975) if critical_value is None else critical_value
        half_width = critical * standard_error
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
        deterministic = tau == 0 or all(sigma == 0 for sigma in self.model.volatilities)
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
        return self._estimate_delta(initial_spots, coordinate=0, n_paths=n_paths,
            rng=rng, time=time, bump_fraction=bump_fraction)

    def estimate_coordinate_delta(
        self, initial_spots: ArrayLike, *, coordinate: int, n_paths: int,
        rng: np.random.Generator, time: float = 0.0, bump_fraction: float = 0.001,
    ) -> MonteCarloDeltaResult:
        """Geometric-basket coordinate delta at positive remaining variance.

        coordinate is zero based. Only that starting spot is bumped; all
        terminal coordinates retain the same underlying normal draws.
        """
        if self.contract.basket != "geometric":
            raise ValueError("Coordinate delta currently requires a geometric basket.")
        if not 0 <= time < self.contract.maturity or all(s == 0 for s in self.model.volatilities):
            raise ValueError("Geometric Monte Carlo delta requires positive remaining variance.")
        return self._estimate_delta(initial_spots, coordinate=coordinate, n_paths=n_paths,
            rng=rng, time=time, bump_fraction=bump_fraction)

    def _estimate_delta(
        self, initial_spots: ArrayLike, *, coordinate: int, n_paths: int,
        rng: np.random.Generator, time: float, bump_fraction: float,
    ) -> MonteCarloDeltaResult:
        if (isinstance(coordinate, (bool, np.bool_))
            or not isinstance(coordinate, (int, np.integer))
            or not 0 <= coordinate < self.model.dimension):
            raise ValueError("coordinate must be an integer in the model's asset range.")
        bump_fraction = float(bump_fraction)
        if not math.isfinite(bump_fraction) or not 0 < bump_fraction < 1:
            raise ValueError("bump_fraction must be finite and strictly between zero and one.")
        spots = _finite_vector(initial_spots, "initial_spots")
        if len(spots) != self.model.dimension or any(s <= 0 for s in spots):
            raise ValueError("initial_spots must be positive and match the model dimension.")
        spot = spots[coordinate]
        bump = spot * bump_fraction
        upper_spot, lower_spot = spot + bump, spot - bump
        if not math.isfinite(upper_spot) or not 0 < lower_spot < spot < upper_spot:
            raise ValueError("Spot bump is outside the representable positive range.")
        terminal, tau, discount = self._simulate(
            spots, n_paths=n_paths, rng=rng, time=time
        )
        deterministic = tau == 0 or all(s == 0 for s in self.model.volatilities)
        basket = self.contract.basket_value(terminal)
        exercise = basket > self.contract.strike
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            if deterministic:
                dividend_discount = math.exp(-self.model.dividend_yields[0] * tau)
                gap = spot * dividend_discount - self.contract.strike * discount
                if gap == 0:
                    raise ValueError("Delta is undefined at a deterministic exercise boundary.")
                pathwise = np.full(len(terminal), dividend_discount if gap > 0 else 0.0)
            elif self.contract.basket == "geometric":
                pathwise = discount * exercise * (self.contract.weights[coordinate] * basket / spot)
            else:
                pathwise = discount * exercise * (terminal[:, 0] / spot)
            upper, lower = terminal.copy(), terminal.copy()
            upper[:, coordinate] *= upper_spot / spot
            lower[:, coordinate] *= lower_spot / spot
            upper_payoff = self.contract.payoff(upper)
            lower_payoff = self.contract.payoff(lower)
            bumped = discount * (upper_payoff - lower_payoff) / (2 * bump)
            difference = pathwise - bumped
        crossing_count = int(np.count_nonzero(
            (upper_payoff > 0) != (lower_payoff > 0)
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
