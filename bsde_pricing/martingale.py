"""IID European payoff estimates with a predictable Brownian control."""

import math
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

from .bsde import PricingBSDE
from .geometric_basket import call_reference
from .monte_carlo import EstimateStatistics, _statistics
from .problems import PricingCase, _finite_array


DiscountedControl = Callable[[float, NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True)
class MartingaleSamples:
    """One observation per IID path, in discounted currency units.

    The integrated squared control has units of currency squared.
    """

    discounted_payoff: NDArray[np.float64]
    stochastic_integral: NDArray[np.float64]
    integrated_squared_control: NDArray[np.float64]

    @property
    def controlled_payoff(self) -> NDArray[np.float64]:
        return self.discounted_payoff - self.stochastic_integral


@dataclass(frozen=True)
class MartingaleEstimator:
    """Subtract a discrete martingale integral from an exact terminal payoff."""

    case: PricingCase

    def __post_init__(self) -> None:
        if self.case.contract.maturity <= 0:
            raise ValueError("The martingale estimator requires positive maturity.")

    def sample(
        self, increments: ArrayLike, *, discounted_control: DiscountedControl | None = None,
    ) -> MartingaleSamples:
        """Evaluate uniform-grid increments of shape (paths, intervals, dimension).

        Increments are in independent Brownian coordinates, in sqrt(year)
        units. A frozen control receives time and left-endpoint spots only
        and returns discounted Brownian exposures of shape (paths, dimension).
        Supplied increments and control state must preserve path independence
        and predictability for IID inference.
        """
        increments = _finite_array(increments, "increments")
        if (increments.ndim != 3 or increments.shape[0] < 2 or increments.shape[1] < 1
                or increments.shape[2] != self.case.model.dimension):
            raise ValueError("increments must have shape (paths >= 2, intervals >= 1, dimension).")

        case = self.case
        maturity = case.contract.maturity
        with np.errstate(over="raise", invalid="raise"):
            terminal_spots = case.model.terminal_from_normals(
                case.initial_spots, maturity, increments.sum(axis=1) / math.sqrt(maturity),
            )
            payoff = math.exp(-case.model.rate * maturity) * case.contract.payoff(terminal_spots)
        if discounted_control is None:
            return MartingaleSamples(payoff, np.zeros_like(payoff), np.zeros_like(payoff))
        integral, energy = self._integrate(increments, discounted_control)
        return MartingaleSamples(payoff, integral, energy)

    def _integrate(
        self, increments: NDArray[np.float64], control: DiscountedControl,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_paths, n_steps, dimension = increments.shape
        case = self.case
        dt = case.contract.maturity / n_steps
        if dt == 0:
            raise ValueError("Time step is below floating-point resolution.")
        brownian = np.zeros((n_paths, dimension))
        integral = np.zeros(n_paths)
        energy = np.zeros(n_paths)
        with np.errstate(over="raise", invalid="raise"):
            for index in range(n_steps):
                time_point = index * dt
                spots = (
                    np.broadcast_to(case.initial_spots, brownian.shape) if index == 0 else
                    case.model.terminal_from_normals(
                        case.initial_spots, time_point, brownian / math.sqrt(time_point),
                    )
                )
                exposure = _finite_array(control(time_point, spots), "discounted control")
                if exposure.shape != brownian.shape:
                    raise ValueError("The discounted control must have shape (paths, dimension).")
                shock = increments[:, index]
                integral += np.einsum("ij,ij->i", exposure, shock)
                energy += dt * np.einsum("ij,ij->i", exposure, exposure)
                brownian += shock
        return integral, energy


def _standardised_gap(statistics: EstimateStatistics, target: float) -> float | None:
    if statistics.standard_error == 0:
        return None
    return (statistics.estimate - target) / statistics.standard_error


def _price_summary(samples: NDArray[np.float64], reference_price: float) -> dict:
    statistics = _statistics(samples, False)
    return {
        **asdict(statistics),
        "sample_variance": statistics.sample_standard_deviation**2,
        "error": statistics.estimate - reference_price,
        "error_in_standard_errors": _standardised_gap(statistics, reference_price),
    }


def run_known_control_study(
    case: PricingCase, *, n_paths: int, seed: int, steps: tuple[int, ...],
) -> dict:
    """Compare zero and analytical controls on shared IID GBM paths."""
    if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least two.")
    steps = tuple(steps)
    if not steps or any(
        isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1
        for n in steps
    ):
        raise ValueError("steps must contain positive integers.")
    if any(a >= b for a, b in zip(steps, steps[1:])) or any(steps[-1] % n for n in steps):
        raise ValueError("steps must increase strictly and divide the finest step count.")

    estimator = MartingaleEstimator(case)
    reference = call_reference(case)
    if reference is None or not any(case.model.volatilities):
        raise ValueError("The study requires a stochastic call with an analytical reference.")
    spots = case.initial_spots[0] if case.model.dimension == 1 else case.initial_spots
    reference_price = reference.evaluate(spots).call_price
    problem = PricingBSDE(case.model, case.contract)

    def analytical_control(t: float, spots: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = reference.call_delta_array(spots, t)
        return math.exp(-case.model.rate * t) * problem.control(spots, delta)

    spawn_key = (25, case.model.dimension)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=spawn_key))
    started = time.perf_counter()
    finest = int(steps[-1])
    maturity = case.contract.maturity
    fine_dt = maturity / finest
    if fine_dt == 0:
        raise ValueError("The finest time step is below floating-point resolution.")
    fine = rng.standard_normal((n_paths, finest, case.model.dimension))
    fine *= math.sqrt(fine_dt)
    payoff = estimator.sample(fine).discounted_payoff
    baseline = _price_summary(payoff, reference_price)
    rows = []
    for count in steps:
        count = int(count)
        coarse = fine if count == finest else fine.reshape(
            n_paths, count, finest // count, case.model.dimension,
        ).sum(axis=2)
        integral, energy = estimator._integrate(coarse, analytical_control)
        samples = MartingaleSamples(payoff, integral, energy)
        estimate = _price_summary(samples.controlled_payoff, reference_price)
        integral_statistics = _statistics(integral, False)
        with np.errstate(over="raise", invalid="raise"):
            squared_integral = integral**2
            isometry = _statistics(squared_integral - energy, False)
        rows.append({
            "steps": count,
            "time_step": maturity / count,
            "analytical_control": estimate,
            "variance_reduction_factor": (
                baseline["sample_variance"] / estimate["sample_variance"]
                if estimate["sample_variance"] > 0 else None
            ),
            "stochastic_integral": {
                **asdict(integral_statistics),
                "mean_in_standard_errors": _standardised_gap(integral_statistics, 0.0),
            },
            "mean_squared_integral": float(squared_integral.mean()),
            "mean_integrated_squared_control": float(energy.mean()),
            "isometry_gap": {
                **asdict(isometry),
                "mean_in_standard_errors": _standardised_gap(isometry, 0.0),
            },
        })
    return {
        "method": "iid_known_martingale_controls",
        "case": case.name,
        "reference_call_price": reference_price,
        "n_paths": int(n_paths),
        "seed": int(seed),
        "spawn_key": list(spawn_key),
        "bit_generator": type(rng.bit_generator).__name__,
        "zero_control": baseline,
        "grids": rows,
        "wall_seconds": time.perf_counter() - started,
        "runtime": {"python": platform.python_version(), "system": platform.platform(),
                    "numpy": np.__version__, "scipy": scipy.__version__},
    }
