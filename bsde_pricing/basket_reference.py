"""Basket prices and coordinate deltas with pilot-fitted geometric controls."""

import math
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri
from scipy.stats import qmc, t

from .black_scholes import BlackScholesReference
from .geometric_basket import GeometricBasketReference
from .monte_carlo import EstimateStatistics, _statistics
from .problems import EuropeanCall, GBMModel, _finite_vector


def _integer(value: int, minimum: int, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        or value < minimum):
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return int(value)


@dataclass(frozen=True)
class BasketEstimate:
    price: EstimateStatistics
    deltas: tuple[EstimateStatistics, ...]


class _RunningMoments:
    """Merge independent contribution blocks without retaining terminal samples."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def add(self, values):
        if not np.isfinite(values).all():
            raise ArithmeticError("Nonfinite reference contributions.")
        count = len(values)
        mean = values.mean(axis=0)
        m2 = np.sum((values-mean)**2, axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        difference = mean-self.mean
        total = self.count+count
        self.m2 += m2+difference**2*(self.count*count/total)
        self.mean += difference*(count/total)
        self.count = total

    def statistics(self):
        if not np.isfinite(self.mean).all() or not np.isfinite(self.m2).all():
            raise ArithmeticError("Reference moments exceeded floating-point range.")
        standard_deviation = np.sqrt(self.m2/(self.count-1))
        se = standard_deviation/math.sqrt(self.count)
        critical = NormalDist().inv_cdf(0.975)
        stats = [EstimateStatistics(float(m), float(s), float(e), 0.95,
                 float(m-critical*e) if e > 0 else None,
                 float(m+critical*e) if e > 0 else None)
                 for m, s, e in zip(self.mean, standard_deviation, se)]
        return BasketEstimate(stats[0], tuple(stats[1:]))


@dataclass(frozen=True)
class BasketReferenceResult:
    method: str
    seed: int
    stream_branch: int
    time: float
    time_to_maturity: float
    sampling_unit: str
    n_units: int
    pilot_pairs: int
    evaluation_terminal_states: int
    total_terminal_states: int
    points_per_scramble: int | None
    sobol_bits: int | None
    plain: BasketEstimate
    controlled: BasketEstimate
    control_coefficients: tuple[float, ...]
    control_expectations: tuple[float, ...]
    unfitted_controls: tuple[int, ...]
    lower_price_bound: float | None
    upper_price_bound: float | None
    replicate_plain: tuple[tuple[float, ...], ...] | None
    replicate_controlled: tuple[tuple[float, ...], ...] | None


@dataclass(frozen=True)
class BasketReferencePricer:
    """Compare antithetic and scrambled-net estimates with independent pilots."""
    model: GBMModel
    contract: EuropeanCall

    def __post_init__(self) -> None:
        if self.model.dimension != self.contract.dimension:
            raise ValueError("Model and contract dimensions must agree.")

    def _setup(self, spots, time, seed, pilot_pairs, branch, children):
        spots = np.array(_finite_vector(spots, "initial_spots"))
        if len(spots) != self.model.dimension or np.any(spots <= 0):
            raise ValueError("initial_spots must be positive and match the model dimension.")
        time = float(time)
        if not math.isfinite(time) or not 0 <= time < self.contract.maturity:
            raise ValueError("Reference estimates require time strictly before maturity.")
        if all(s == 0 for s in self.model.volatilities):
            raise ValueError("Reference estimates require positive remaining variance.")
        seed = _integer(seed, 0, "seed")
        pilot_pairs = _integer(pilot_pairs, 2, "pilot_pairs")
        tau = self.contract.maturity - time
        geometric = replace(self.contract, basket="geometric")
        reference = GeometricBasketReference(self.model, geometric).evaluate(spots, time)
        known = np.array((reference.call_price, *reference.call_deltas), dtype=np.float64)
        streams = np.random.SeedSequence(seed, spawn_key=(branch,)).spawn(children)
        pilot_x, pilot_y = self._pairs(spots, tau, pilot_pairs, np.random.default_rng(streams[0]))
        x = pilot_x - pilot_x.mean(axis=0)
        y = pilot_y - pilot_y.mean(axis=0)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            denominator = np.einsum("ij,ij->j", y, y)
            numerator = np.einsum("ij,ij->j", x, y)
            beta = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                             where=denominator > 0)
        if not np.isfinite(beta).all():
            raise ArithmeticError("Pilot coefficients exceeded floating-point range.")
        missing = tuple(int(j) for j in np.flatnonzero(denominator == 0))
        return spots, time, tau, seed, pilot_pairs, streams, known, beta, missing

    def _contributions(
        self, spots: NDArray[np.float64], tau: float, normals: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        terminal = self.model.terminal_from_normals(spots, tau, normals)
        discount = math.exp(-self.model.rate * tau)
        if not math.isfinite(discount) or discount == 0:
            raise ValueError("Discount factor is outside the supported numerical range.")
        geometric = replace(self.contract, basket="geometric")
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            basket = self.contract.basket_value(terminal)
            geo = geometric.basket_value(terminal)
            weights = np.array(self.contract.weights)
            scale = (terminal if self.contract.basket == "arithmetic" else basket[:, None])
            target_delta = discount * (basket > self.contract.strike)[:, None] * weights * scale / spots
            control_delta = discount * (geo > self.contract.strike)[:, None] * weights * geo[:, None] / spots
            target = np.column_stack((discount * self.contract.payoff(terminal), target_delta))
            control = np.column_stack((discount * geometric.payoff(terminal), control_delta))
        return target, control

    def _pairs(
        self, spots: NDArray[np.float64], tau: float, count: int, rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        normals = rng.standard_normal((count, self.model.dimension))
        plus, plus_control = self._contributions(spots, tau, normals)
        minus, minus_control = self._contributions(spots, tau, -normals)
        return (plus + minus) / 2, (plus_control + minus_control) / 2

    def antithetic(
        self, initial_spots: ArrayLike, *, n_pairs: int = 65_536,
        pilot_pairs: int = 4096, seed: int = 2026, time: float = 0.0,
        chunk_size: int | None = None,
    ) -> BasketReferenceResult:
        """Fit on independent pilot pairs, then estimate using IID pair averages."""
        n_pairs = _integer(n_pairs, 2, "n_pairs")
        if chunk_size is not None:
            chunk_size = _integer(chunk_size, 2, "chunk_size")
        setup = self._setup(initial_spots, time, seed, pilot_pairs, 0, 2)
        spots, time, tau, seed, pilot_pairs, streams, known, beta, missing = setup
        if chunk_size is not None:
            plain, adjusted = _RunningMoments(), _RunningMoments()
            rng = np.random.default_rng(streams[1])
            for start in range(0, n_pairs, chunk_size):
                target, control = self._pairs(spots, tau, min(chunk_size, n_pairs-start), rng)
                plain.add(target)
                adjusted.add(target-beta*(control-known))
            return self._result("antithetic_geometric_control", spots, time, tau, seed, 0,
                pilot_pairs, plain, adjusted, known, beta, missing, 2*n_pairs, None)
        plain, control = self._pairs(spots, tau, n_pairs, np.random.default_rng(streams[1]))
        with np.errstate(over="raise", invalid="raise"):
            adjusted = plain - beta * (control - known)
        return self._result("antithetic_geometric_control", spots, time, tau, seed, 0,
            pilot_pairs, plain, adjusted, known, beta, missing, 2*n_pairs, None)

    def sobol(
        self, initial_spots: ArrayLike, *, power: int = 13, scrambles: int = 16,
        pilot_pairs: int = 4096, seed: int = 2026, time: float = 0.0,
        chunk_size: int | None = None,
    ) -> BasketReferenceResult:
        """Estimate error across independent scrambled-net means, not points."""
        power = _integer(power, 1, "power")
        if power > 30:
            raise ValueError("power cannot exceed 30; select a feasible allocation before running.")
        scrambles = _integer(scrambles, 4, "scrambles")
        if chunk_size is not None:
            chunk_size = _integer(chunk_size, 2, "chunk_size")
            if chunk_size & (chunk_size-1):
                raise ValueError("Sobol chunk_size must be a power of two.")
        setup = self._setup(initial_spots, time, seed, pilot_pairs, 1, scrambles + 1)
        spots, time, tau, seed, pilot_pairs, streams, known, beta, missing = setup
        plain, adjusted = [], []
        for stream in streams[1:]:
            engine = qmc.Sobol(self.model.dimension, scramble=True, bits=52,
                               rng=np.random.default_rng(stream))
            size = 2**power if chunk_size is None else min(chunk_size, 2**power)
            sums = np.zeros(self.model.dimension+1)
            control_sums = np.zeros_like(sums)
            for start in range(0, 2**power, size):
                uniforms = engine.random_base2(int(math.log2(size))) if start == 0 else engine.random(size)
                if np.any((uniforms <= 0) | (uniforms >= 1)):
                    raise ArithmeticError("A Sobol endpoint cannot be mapped to a finite normal; no clipping applied.")
                target, control = self._contributions(spots, tau, ndtri(uniforms))
                sums += target.sum(axis=0)
                control_sums += control.sum(axis=0)
            means = sums/2**power
            plain.append(means)
            with np.errstate(over="raise", invalid="raise"):
                adjusted.append(means - beta * (control_sums/2**power - known))
        return self._result("scrambled_sobol_geometric_control", spots, time, tau, seed, 1,
            pilot_pairs, np.array(plain), np.array(adjusted), known, beta, missing,
            scrambles * 2**power, 2**power)

    def _result(self, method, spots, time, tau, seed, branch, pilot_pairs, plain,
                adjusted, known, beta, missing, evaluation_states, points):
        streaming = isinstance(plain, _RunningMoments)
        if not streaming and (not np.isfinite(plain).all() or not np.isfinite(adjusted).all()):
            raise ArithmeticError("Reference contributions exceeded floating-point range.")
        critical = float(t.ppf(0.975, len(plain)-1)) if points is not None else None

        def statistics(values):
            stats = tuple(_statistics(values[:, j], False, critical_value=critical)
                          for j in range(values.shape[1]))
            return BasketEstimate(stats[0], stats[1:])

        lower = upper = None
        if math.fsum(self.contract.weights) == 1.0:
            lower = float(known[0])
            upper = math.fsum(w * BlackScholesReference(
                GBMModel(self.model.rate, (vol,), (dividend,), ((1.0,),)),
                EuropeanCall(self.contract.strike, self.contract.maturity),
            ).evaluate(spot, time).call_price for w, vol, dividend, spot in zip(
                self.contract.weights, self.model.volatilities, self.model.dividend_yields, spots))
        def rows(a):
            return tuple(tuple(float(x) for x in row) for row in a)
        return BasketReferenceResult(method, seed, branch, time, tau,
            "scramble_mean" if points is not None else "antithetic_pair_mean",
            plain.count if streaming else len(plain), pilot_pairs, evaluation_states, evaluation_states + 2*pilot_pairs,
            points, 52 if points is not None else None,
            plain.statistics() if streaming else statistics(plain),
            adjusted.statistics() if streaming else statistics(adjusted),
            tuple(float(x) for x in beta), tuple(float(x) for x in known), missing,
            lower, upper, rows(plain) if points is not None else None,
            rows(adjusted) if points is not None else None)
