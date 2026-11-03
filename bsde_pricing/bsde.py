"""Risk-neutral pricing BSDE and a vanilla analytical-control diagnostic."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .black_scholes import BlackScholesReference
from .problems import EuropeanCall, GBMModel, _finite_array


@dataclass(frozen=True)
class PricingBSDE:
    """dY = -f(Y) dt + Z.dB, with independent Brownian coordinates B.

    The pricing driver is f(Y) = -rY. State and delta arrays have shape
    (..., d); diffusion arrays have shape (..., d, d), with asset rows and
    independent Brownian columns. Methods return new arrays.
    """

    model: GBMModel
    contract: EuropeanCall

    def __post_init__(self) -> None:
        if self.model.dimension != len(self.contract.weights):
            raise ValueError("Model and contract dimensions must match.")

    def _states(self, spots: ArrayLike) -> NDArray[np.float64]:
        values = _finite_array(spots, "spots")
        if values.ndim < 1 or values.shape[-1] != self.model.dimension or values.size == 0:
            raise ValueError("spots must have nonempty shape (..., dimension).")
        if np.any(values <= 0):
            raise ValueError("spots must be strictly positive.")
        return values

    def drift(self, spots: ArrayLike) -> NDArray[np.float64]:
        values = self._states(spots)
        with np.errstate(over="raise", invalid="raise"):
            return values * (self.model.rate - np.asarray(self.model.dividend_yields))

    def diffusion(self, spots: ArrayLike) -> NDArray[np.float64]:
        values = self._states(spots)
        with np.errstate(over="raise", invalid="raise"):
            return (values * self.model.volatilities)[..., :, None] * np.linalg.cholesky(
                self.model.correlation,
            )

    def driver(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return f = -rY; the forward increment uses minus this driver."""
        values = _finite_array(values, "values")
        with np.errstate(over="raise", invalid="raise"):
            return -self.model.rate * values

    def control(self, spots: ArrayLike, deltas: ArrayLike) -> NDArray[np.float64]:
        """Map spot deltas to independent-Brownian exposures Z = A.T delta."""
        values = self._states(spots)
        deltas = _finite_array(deltas, "deltas")
        if deltas.shape != values.shape:
            raise ValueError("deltas must have the same shape as spots.")
        with np.errstate(over="raise", invalid="raise"):
            # Row batches multiply L, whereas Brownian shocks multiply L.T.
            return (deltas * values * self.model.volatilities) @ np.linalg.cholesky(
                self.model.correlation,
            )

    def terminal_value(self, spots: ArrayLike) -> NDArray[np.float64]:
        return self.contract.payoff(self._states(spots))


@dataclass(frozen=True)
class BSDEGridDiagnostic:
    n_steps: int
    time_step: float
    residual_mean: float
    residual_standard_error: float
    residual_rms: float


@dataclass(frozen=True)
class BSDERefinementResult:
    reference_call_price: float
    n_paths: int
    bit_generator: str
    finest_steps: int
    grids: tuple[BSDEGridDiagnostic, ...]


@dataclass(frozen=True)
class AnalyticalBSDEDiagnostic:
    """Test the discounted identity on nested uniform vanilla time grids.

    Each grid uses left-endpoint analytical deltas and the same Brownian
    paths. GBM states are exact at grid points. Only the stochastic integral
    is discretised. Returned statistics are across independent paths.
    """

    problem: PricingBSDE

    def __post_init__(self) -> None:
        BlackScholesReference(self.problem.model, self.problem.contract)
        if self.problem.contract.maturity <= 0 or self.problem.model.volatilities[0] <= 0:
            raise ValueError("The diagnostic requires positive maturity and volatility.")

    def run(
        self, initial_spot: float, *, n_paths: int, rng: np.random.Generator,
        steps: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    ) -> BSDERefinementResult:
        if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
            raise ValueError("n_paths must be an integer of at least two.")
        steps = tuple(steps)
        if not steps or any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in steps
        ):
            raise ValueError("steps must contain positive integers.")
        if any(a >= b for a, b in zip(steps, steps[1:])) or any(steps[-1] % n for n in steps):
            raise ValueError("steps must increase strictly and divide the finest step count.")
        reference = BlackScholesReference(self.problem.model, self.problem.contract)
        price = reference.evaluate(initial_spot).call_price
        model, contract = self.problem.model, self.problem.contract
        maturity = contract.maturity
        fine_dt = maturity / steps[-1]
        if fine_dt == 0:
            raise ValueError("The finest time step is below floating-point resolution.")
        # One finest-grid stream provides common paths for every coarser grid.
        increments = rng.standard_normal((n_paths, steps[-1])) * math.sqrt(fine_dt)
        terminal_brownian = increments.sum(axis=1)
        terminal = model.terminal_from_normals(
            (initial_spot,), maturity, (terminal_brownian / math.sqrt(maturity))[:, None],
        )
        with np.errstate(over="raise", invalid="raise"):
            discounted_payoff = math.exp(-model.rate * maturity) * self.problem.terminal_value(terminal)
        rows = []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for n_steps in steps:
                dt = maturity / n_steps
                coarse = increments.reshape(n_paths, n_steps, steps[-1] // n_steps).sum(axis=2)
                brownian = np.zeros(n_paths)
                integral = np.zeros(n_paths)
                for index in range(n_steps):
                    time = index * dt
                    states = (
                        np.full((n_paths, 1), initial_spot) if index == 0 else
                        model.terminal_from_normals(
                            (initial_spot,), time, (brownian / math.sqrt(time))[:, None],
                        )
                    )
                    deltas = reference.call_delta_array(states, time)
                    control = self.problem.control(states, deltas)[:, 0]
                    integral += math.exp(-model.rate * time) * control * coarse[:, index]
                    brownian += coarse[:, index]
                residual = discounted_payoff - price - integral
                rows.append(BSDEGridDiagnostic(
                    n_steps=int(n_steps), time_step=dt,
                    residual_mean=float(residual.mean()),
                    residual_standard_error=float(residual.std(ddof=1) / math.sqrt(n_paths)),
                    residual_rms=float(np.sqrt(np.mean(residual * residual))),
                ))
        return BSDERefinementResult(
            reference_call_price=price, n_paths=int(n_paths),
            bit_generator=type(rng.bit_generator).__name__, finest_steps=int(steps[-1]),
            grids=tuple(rows),
        )
