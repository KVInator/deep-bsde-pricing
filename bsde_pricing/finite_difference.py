"""Scalar European-call PDE on a uniform spot grid, using CPU float64."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from .problems import EuropeanCall, GBMModel


def _integer(value: int, minimum: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) or value < minimum
    ):
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return int(value)


@dataclass(frozen=True)
class FiniteDifferenceValue:
    spot: float
    time: float
    time_to_maturity: float
    call_price: float
    call_delta: float
    gamma: float


@dataclass(frozen=True)
class FiniteDifferenceResult:
    """One valuation-time slice; Greeks live only on spots[1:-1].

    Arrays are read-only. Interpolation acts separately on nodal prices and
    finite-difference Greeks; it does not differentiate a linear interpolant.
    """

    spots: NDArray[np.float64]
    call_values: NDArray[np.float64]
    deltas: NDArray[np.float64]
    gammas: NDArray[np.float64]
    time: float
    time_to_maturity: float
    n_time: int
    rannacher_steps: int
    upwind_node_count: int

    def evaluate(self, spot: float) -> FiniteDifferenceValue:
        spot = float(spot)
        if not math.isfinite(spot) or not self.spots[1] <= spot <= self.spots[-2]:
            raise ValueError("spot must lie between the first and last interior grid nodes.")
        return FiniteDifferenceValue(
            spot, self.time, self.time_to_maturity,
            float(np.interp(spot, self.spots, self.call_values)),
            float(np.interp(spot, self.spots[1:-1], self.deltas)),
            float(np.interp(spot, self.spots[1:-1], self.gammas)),
        )


@dataclass(frozen=True)
class FiniteDifferencePricer:
    model: GBMModel
    contract: EuropeanCall

    def __post_init__(self) -> None:
        if self.model.dimension != 1 or self.contract.weights != (1.0,):
            raise ValueError("Finite differences require one asset and unit weight.")
        if self.model.volatilities[0] <= 0:
            raise ValueError("The parabolic PDE solver requires positive volatility.")

    def solve(
        self, *, s_max: float = 400.0, n_space: int = 400,
        n_time: int = 400, rannacher_steps: int = 2, time: float = 0.0,
    ) -> FiniteDifferenceResult:
        """March from payoff to time, retaining O(n_space) working storage.

        n_space counts intervals on [0, s_max]. Each initial Rannacher interval
        is replaced by two backward-Euler half steps, then CN resumes.
        """
        n_space = _integer(n_space, 4, "n_space")
        n_time = _integer(n_time, 1, "n_time")
        rannacher_steps = _integer(rannacher_steps, 0, "rannacher_steps")
        if rannacher_steps > n_time:
            raise ValueError("rannacher_steps cannot exceed n_time.")
        s_max, time = float(s_max), float(time)
        if not math.isfinite(s_max) or s_max <= self.contract.strike:
            raise ValueError("s_max must be finite and greater than strike.")
        if not math.isfinite(time) or not 0 <= time < self.contract.maturity:
            raise ValueError("time must be finite, nonnegative and strictly before maturity.")

        tau = self.contract.maturity - time
        dt = tau / n_time
        ds = s_max / n_space
        rate = self.model.rate
        dividend = self.model.dividend_yields[0]
        sigma = self.model.volatilities[0]
        if dt == 0 or ds == 0:
            raise ValueError("Grid spacing is below floating-point range.")
        if 1.0 + dt * rate <= 0:
            raise ValueError("Negative-rate discounting requires a smaller time step.")

        def upper_boundary(elapsed: float) -> float:
            try:
                dr, dq = math.exp(-rate * elapsed), math.exp(-dividend * elapsed)
                value = s_max * dq - self.contract.strike * dr
            except OverflowError as exc:
                raise ValueError("Boundary discount factors exceed floating-point range.") from exc
            if dr == 0 or dq == 0 or not math.isfinite(value) or value <= 0:
                raise ValueError("Upper call boundary is invalid; enlarge s_max or revise inputs.")
            return value

        # The sign of the discounted boundary gap is controlled by an
        # exponential ratio, so checking both endpoints covers the horizon.
        upper_boundary(tau)
        spots = np.linspace(0.0, s_max, n_space + 1)
        i = np.arange(1, n_space, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            diffusion = 0.5 * sigma * sigma * i * i
            drift = (rate - dividend) * i
            lower = diffusion - 0.5 * drift
            upper = diffusion + 0.5 * drift
            # Preserve nonnegative spatial off-diagonals near S=0 when
            # convection dominates. Ordinary rows use central differences.
            upwind = (lower < 0) | (upper < 0)
            lower[upwind] = diffusion[upwind] + np.maximum(-drift[upwind], 0)
            upper[upwind] = diffusion[upwind] + np.maximum(drift[upwind], 0)
            diagonal = -lower - upper - rate

        values = self.contract.payoff(spots[:, None])

        def advance(elapsed: float, step: float, theta: float) -> None:
            # Old boundary values enter through the explicit operator; the
            # new Dirichlet boundary enters on the implicit right-hand side.
            interior = values[1:-1]
            weight = (1.0 - theta) * step
            rhs = interior + weight * (
                lower * values[:-2] + diagonal * interior + upper * values[2:]
            )
            boundary = upper_boundary(elapsed)
            rhs[-1] += theta * step * upper[-1] * boundary
            banded = np.zeros((3, n_space - 1), dtype=np.float64)
            banded[0, 1:] = -theta * step * upper[:-1]
            banded[1] = 1.0 - theta * step * diagonal
            banded[2, :-1] = -theta * step * lower[1:]
            values[1:-1] = solve_banded((1, 1), banded, rhs)
            values[0], values[-1] = 0.0, boundary

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index in range(n_time):
                if index < rannacher_steps:
                    advance((index + 0.5) * dt, dt / 2.0, 1.0)
                    advance((index + 1) * dt, dt / 2.0, 1.0)
                else:
                    advance((index + 1) * dt, dt, 0.5)
            deltas = (values[2:] - values[:-2]) / (2.0 * ds)
            gammas = ((values[2:] - values[1:-1]) -
                      (values[1:-1] - values[:-2])) / ds / ds
        for array in (spots, values, deltas, gammas):
            if not np.isfinite(array).all():
                raise ArithmeticError("Finite-difference solution exceeded floating-point range.")
            array.setflags(write=False)
        return FiniteDifferenceResult(
            spots, values, deltas, gammas, time, tau, n_time,
            rannacher_steps, int(np.count_nonzero(upwind)),
        )
