"""Distribution diagnostics for exact GBM terminal samples."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .problems import GBMModel, _finite_array, _finite_vector


@dataclass(frozen=True)
class TerminalSampleDiagnostics:
    """Expected and observed moments, with theoretical sampling standard errors.

    Log-return covariance uses the unbiased N-1 denominator. Its standard
    errors follow the Gaussian covariance formula; terminal-mean standard
    errors use the marginal lognormal variance. These are pointwise checks.
    """

    n_paths: int
    horizon: float
    log_return_mean_expected: tuple[float, ...]
    log_return_mean_observed: tuple[float, ...]
    log_return_mean_standard_error: tuple[float, ...]
    log_return_covariance_expected: tuple[tuple[float, ...], ...]
    log_return_covariance_observed: tuple[tuple[float, ...], ...]
    log_return_covariance_standard_error: tuple[tuple[float, ...], ...]
    terminal_mean_expected: tuple[float, ...]
    terminal_mean_observed: tuple[float, ...]
    terminal_mean_standard_error: tuple[float, ...]

    @classmethod
    def from_sample(
        cls, model: GBMModel, initial_spots: ArrayLike, horizon: float,
        terminal_spots: ArrayLike,
    ) -> "TerminalSampleDiagnostics":
        spots = np.asarray(_finite_vector(initial_spots, "initial_spots"))
        if len(spots) != model.dimension or np.any(spots <= 0):
            raise ValueError("initial_spots must be positive and match the model dimension.")
        horizon = float(horizon)
        if not math.isfinite(horizon) or horizon < 0:
            raise ValueError("horizon must be finite and nonnegative.")
        terminal = _finite_array(terminal_spots, "terminal_spots")
        if terminal.ndim != 2 or terminal.shape[1] != model.dimension or len(terminal) < 2:
            raise ValueError("terminal_spots must have shape (n_paths >= 2, dimension).")
        if np.any(terminal <= 0):
            raise ValueError("terminal_spots must be strictly positive.")
        count = len(terminal)
        sigma = np.asarray(model.volatilities)
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            deviation = sigma * math.sqrt(horizon)
            variance = deviation**2
            mean = (model.rate - np.asarray(model.dividend_yields)) * horizon - 0.5 * variance
            covariance = np.outer(deviation, deviation) * np.asarray(model.correlation)
            logs = np.log(terminal) - np.log(spots)
            observed_covariance = np.atleast_2d(np.cov(logs, rowvar=False, ddof=1))
            covariance_se = np.sqrt(
                (np.outer(variance, variance) + covariance**2) / (count - 1)
            )
            terminal_mean = spots * np.exp(
                (model.rate - np.asarray(model.dividend_yields)) * horizon
            )
            terminal_mean_se = terminal_mean * np.sqrt(np.expm1(variance) / count)
            observed_mean = logs.mean(axis=0)
            observed_terminal_mean = terminal.mean(axis=0)
        vectors = (mean, observed_mean, np.sqrt(variance / count),
                   terminal_mean, observed_terminal_mean, terminal_mean_se)
        matrices = (covariance, observed_covariance, covariance_se)
        if not all(np.isfinite(x).all() for x in (*vectors, *matrices)):
            raise ArithmeticError("Terminal diagnostics exceeded floating-point range.")
        v = [tuple(float(x) for x in vector) for vector in vectors]
        m = [tuple(tuple(float(x) for x in row) for row in matrix) for matrix in matrices]
        return cls(count, horizon, v[0], v[1], v[2], m[0], m[1], m[2], v[3], v[4], v[5])
