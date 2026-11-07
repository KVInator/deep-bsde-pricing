"""Matched-precision costs for repeated estimates of one arithmetic call."""

import math
import time
from dataclasses import asdict
from statistics import median

import numpy as np
from scipy.stats import t

from .basket_reference import BasketReferencePricer
from .experiment_design import ExperimentDesign
from .learned_control import FrozenControls, _pair_samples, _pilot_coefficient
from .martingale import MartingaleEstimator
from .monte_carlo import _statistics


HALF_WIDTH_TARGETS = (0.025, 0.01)
QUERY_COUNTS = (1, 16, 100)
CONVENTIONAL = ("antithetic_geometric_control", "scrambled_sobol_geometric_control")


def _sustained_break_even(upfront: float, evaluation: float, other_upfront: float, other_evaluation: float):
    """First positive query count after which the cost advantage persists."""
    saving = other_evaluation-evaluation
    if saving > 0:
        return max(1, math.ceil((upfront-other_upfront)/saving))
    return 1 if saving == 0 and upfront <= other_upfront else None


def _row(samples, coefficient, pilot_seconds, evaluation_seconds, *, critical=None):
    stats = _statistics(samples, False, critical_value=critical)
    half_width = None if stats.confidence_high is None else stats.confidence_high-stats.estimate
    return {"price": asdict(stats), "half_width": half_width, "coefficient": coefficient,
            "pilot_seconds": pilot_seconds, "evaluation_seconds": evaluation_seconds}


def _summarise(rows: list[dict], reference: float) -> dict:
    estimates = np.array([r["price"]["estimate"] for r in rows])
    widths = [r["half_width"] for r in rows]
    return {
        "mean_price": float(estimates.mean()),
        "empirical_price_sd": float(estimates.std(ddof=1)),
        "rmse_vs_reference": float(np.sqrt(np.mean((estimates-reference)**2))),
        "median_half_width": None if None in widths else median(widths),
        "max_half_width": None if None in widths else max(widths),
        "median_pilot_seconds": median(r["pilot_seconds"] for r in rows),
        "median_evaluation_seconds": median(r["evaluation_seconds"] for r in rows),
        "passes": {str(h): sum(w is not None and w <= h for w in widths) for h in HALF_WIDTH_TARGETS},
    }


def _matched_precision(allocations: list[dict], fixed_costs: dict, repetitions: int) -> dict:
    matches = {}
    for target in HALF_WIDTH_TARGETS:
        matched = {}
        for name, fixed in fixed_costs.items():
            qualified = next((a for a in allocations
                              if a["methods"][name]["passes"][str(target)] == repetitions), None)
            if qualified is None:
                matched[name] = {"status": "target_missed"}
                continue
            cost = qualified["methods"][name]
            upfront = sum(fixed.values())+cost["median_pilot_seconds"]
            evaluation = cost["median_evaluation_seconds"]
            matched[name] = {"status": "target_met", "evaluation_states": 2*qualified["pairs"],
                             "upfront_seconds": upfront, "evaluation_seconds": evaluation,
                             "total_seconds": {str(q): upfront+q*evaluation for q in QUERY_COUNTS}}
        for name, row in matched.items():
            if row["status"] == "target_met" and name not in CONVENTIONAL:
                row["break_even_queries"] = {
                    other: _sustained_break_even(
                        row["upfront_seconds"], row["evaluation_seconds"],
                        matched[other]["upfront_seconds"], matched[other]["evaluation_seconds"],
                    ) for other in CONVENTIONAL if matched[other]["status"] == "target_met"
                }
        matches[str(target)] = matched
    return matches


class ControlCostStudy:
    """Evaluate frozen controls and conventional estimators on nested allocations."""

    def __init__(self, design: ExperimentDesign, frozen: FrozenControls):
        spec = design.data
        protocol = spec["estimators"]
        self.case = design.case(protocol["reference_case"])
        self.counts = tuple(protocol["antithetic_pairs"])
        self.pilot_pairs = protocol["coefficient_pilot_pairs"]
        self.scrambles = spec["reference"]["scrambles"]
        self.chunk_paths = spec["resources"]["evaluation_chunk_paths"]
        self.reference_chunk = spec["reference"]["chunk_terminal_states"]
        if protocol["repetitions"] < 2 or self.chunk_paths < 2:
            raise ValueError("Cost studies require independent repeats and a complete pair per chunk.")
        self.powers = {}
        for count in self.counts:
            points, remainder = divmod(2*count, self.scrambles)
            if remainder or points < 2 or points & (points-1):
                raise ValueError("Each allocation must match a power of two per Sobol scramble.")
            self.powers[count] = points.bit_length()-1
        self.frozen = frozen
        self.estimator = MartingaleEstimator(self.case)
        self.pricer = BasketReferencePricer(self.case.model, self.case.contract)
        self.controls = {"analytical_geometric_proxy": self.frozen.geometric_proxy(),
                         "value_gradient": self.frozen.value_gradient,
                         "terminal_loss": self.frozen.terminal_loss}

    def _path_pairs(self, rng, count, control):
        shocks = rng.standard_normal((count, self.frozen.steps, self.case.model.dimension))
        shocks *= math.sqrt(self.frozen.dt)
        return _pair_samples(self.estimator, shocks, control, self.chunk_paths)

    def _martingale_rows(self, rows, pilot_stream, evaluation_stream):
        for kind, control in self.controls.items():
            fit_coefficient = kind != "analytical_geometric_proxy"
            pilot_seconds, coefficient = 0.0, None
            if fit_coefficient:
                start = time.perf_counter()
                pilot = self._path_pairs(np.random.default_rng(pilot_stream), self.pilot_pairs, control)
                coefficient = _pilot_coefficient(*pilot)
                pilot_seconds = time.perf_counter()-start
            rng = np.random.default_rng(evaluation_stream)
            payoff = np.empty(self.counts[-1])
            integral = np.empty_like(payoff)
            elapsed, previous = 0.0, 0
            for count in self.counts:
                start = time.perf_counter()
                payoff[previous:count], integral[previous:count] = self._path_pairs(rng, count-previous, control)
                elapsed += time.perf_counter()-start
                unit = _row(payoff[:count]-integral[:count], 1.0, 0.0, elapsed)
                rows[count][kind+"_unit"] = unit
                if fit_coefficient:
                    beta = 0.0 if coefficient is None else coefficient
                    fitted = _row(payoff[:count]-beta*integral[:count], beta, pilot_seconds, elapsed)
                    fitted["degenerate_pilot"] = coefficient is None
                    rows[count][kind+"_fitted"] = fitted
                previous = count

    def _conventional_rows(self, rows, anti_stream, sobol_stream):
        for branch, stream in enumerate((anti_stream, sobol_stream)):
            seed = int(stream.generate_state(1, dtype=np.uint64)[0])
            start = time.perf_counter()
            spots, _, tau, _, _, streams, known, beta, missing = self.pricer._setup(
                self.case.initial_spots, 0.0, seed, self.pilot_pairs, branch,
                self.scrambles+1 if branch else 2, price_only=True,
            )
            pilot_seconds = time.perf_counter()-start
            for count in self.counts:
                start = time.perf_counter()
                if branch:
                    _, adjusted = self.pricer._sobol_means(
                        spots, tau, self.powers[count], streams[1:], known, beta,
                        self.reference_chunk, price_only=True,
                    )
                    critical = float(t.ppf(0.975, self.scrambles-1))
                else:
                    payoff, control = self.pricer._pairs(
                        spots, tau, count, np.random.default_rng(streams[1]), price_only=True,
                    )
                    adjusted = payoff-beta*(control-known)
                    critical = None
                row = _row(adjusted[:, 0], float(beta[0]), pilot_seconds,
                           time.perf_counter()-start, critical=critical)
                row["degenerate_pilot"] = bool(missing)
                rows[count][CONVENTIONAL[branch]] = row

    def repetition(self, pilot, evaluation, antithetic, sobol) -> dict:
        rows = {n: {} for n in self.counts}
        self._martingale_rows(rows, pilot, evaluation)
        self._conventional_rows(rows, antithetic, sobol)
        return rows
