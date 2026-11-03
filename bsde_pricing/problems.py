"""Financial model, payoff and initial-state contracts for European pricing."""

import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values.")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values.")
    return array


def _finite_vector(values: ArrayLike, name: str) -> tuple[float, ...]:
    array = _finite_array(values, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector.")
    return tuple(float(value) for value in array)


@dataclass(frozen=True)
class GBMModel:
    """Constant risk-neutral parameters; rates are continuous and time is in years."""

    rate: float
    volatilities: tuple[float, ...]
    dividend_yields: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        rate = float(self.rate)
        if not math.isfinite(rate):
            raise ValueError("rate must be finite.")
        volatilities = _finite_vector(self.volatilities, "volatilities")
        dividends = _finite_vector(self.dividend_yields, "dividend_yields")
        if any(value < 0 for value in volatilities):
            raise ValueError("volatilities must be nonnegative.")
        if len(dividends) != len(volatilities):
            raise ValueError("dividend_yields must match the model dimension.")

        correlation = _finite_array(self.correlation, "correlation")
        dimension = len(volatilities)
        if correlation.shape != (dimension, dimension):
            raise ValueError("correlation must have shape (dimension, dimension).")
        if not np.allclose(correlation, correlation.T, rtol=0, atol=1e-12):
            raise ValueError("correlation must be symmetric.")
        if not np.allclose(np.diag(correlation), 1.0, rtol=0, atol=1e-12):
            raise ValueError("correlation must have unit diagonal.")
        try:
            np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("correlation must be positive definite.") from exc

        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "volatilities", volatilities)
        object.__setattr__(self, "dividend_yields", dividends)
        rows = tuple(tuple(float(value) for value in row) for row in correlation)
        object.__setattr__(self, "correlation", rows)

    @property
    def dimension(self) -> int:
        return len(self.volatilities)


@dataclass(frozen=True)
class EuropeanCall:
    """Call on a positive-weight arithmetic or geometric basket."""

    strike: float
    maturity: float
    weights: tuple[float, ...] = (1.0,)
    basket: Literal["arithmetic", "geometric"] = "arithmetic"

    def __post_init__(self) -> None:
        strike, maturity = float(self.strike), float(self.maturity)
        if not math.isfinite(strike) or strike <= 0:
            raise ValueError("strike must be finite and strictly positive.")
        if not math.isfinite(maturity) or maturity < 0:
            raise ValueError("maturity must be finite and nonnegative.")
        weights = _finite_vector(self.weights, "weights")
        if any(weight <= 0 for weight in weights):
            raise ValueError("weights must be strictly positive.")
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("weights must sum to one.")
        if self.basket not in ("arithmetic", "geometric"):
            raise ValueError("basket must be arithmetic or geometric.")
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "weights", weights)

    @property
    def dimension(self) -> int:
        return len(self.weights)

    def payoff(self, terminal_spots: ArrayLike) -> NDArray[np.float64]:
        """Map shape (..., dimension) to the leading shape (...), including ()."""
        spots = _finite_array(terminal_spots, "terminal_spots")
        if spots.ndim < 1 or spots.shape[-1] != self.dimension:
            raise ValueError("terminal_spots must have shape (..., dimension).")
        if np.any(spots < 0):
            raise ValueError("terminal_spots must be nonnegative.")
        weights = np.asarray(self.weights)
        if self.basket == "arithmetic":
            basket_value = spots @ weights
        else:
            log_spots = np.zeros_like(spots)
            np.log(spots, out=log_spots, where=spots > 0)
            basket_value = np.exp(log_spots @ weights)
            basket_value = np.where(np.any(spots == 0, axis=-1), 0.0, basket_value)
        return np.asarray(np.maximum(basket_value - self.strike, 0.0))


@dataclass(frozen=True)
class PricingCase:
    name: str
    model: GBMModel
    contract: EuropeanCall
    initial_spots: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty.")
        spots = _finite_vector(self.initial_spots, "initial_spots")
        if any(spot <= 0 for spot in spots):
            raise ValueError("initial_spots must be strictly positive.")
        if len(spots) != self.model.dimension:
            raise ValueError("initial_spots must match the model dimension.")
        if self.contract.dimension != self.model.dimension:
            raise ValueError("contract weights must match the model dimension.")
        object.__setattr__(self, "initial_spots", spots)

    def to_json(self) -> str:
        description = asdict(self)
        description["conventions"] = {
            "measure": "risk_neutral",
            "time_unit": "year",
            "rate_compounding": "continuous",
            "volatility_unit": "per_sqrt_year",
            "payoff_units": "same_currency_unit_as_spots_and_strike",
        }
        return json.dumps(description, indent=2, sort_keys=True)


def vanilla_case() -> PricingCase:
    return PricingCase(
        name="vanilla",
        model=GBMModel(
            rate=0.03,
            volatilities=(0.20,),
            dividend_yields=(0.0,),
            correlation=((1.0,),),
        ),
        contract=EuropeanCall(strike=100.0, maturity=1.0),
        initial_spots=(100.0,),
    )
