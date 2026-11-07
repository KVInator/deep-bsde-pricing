"""Development comparison of frozen martingale and geometric payoff controls."""

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy
import torch

from .basket_reference import BasketReferencePricer
from .bsde import PricingBSDE
from .experiment_design import ExperimentDesign
from .geometric_basket import GeometricBasketReference
from .martingale import DiscountedControl, MartingaleEstimator
from .monte_carlo import EstimateStatistics, _statistics
from .neural import value_derivatives
from .terminal_bsde import TerminalBSDESolver
from .training import FBSNNSolver


class FrozenControls:
    """Expose both trained formulations as discounted left-endpoint controls."""

    def __init__(self, value: FBSNNSolver, terminal: TerminalBSDESolver) -> None:
        if value.case != terminal.case or value.config.n_steps != terminal.config.n_steps:
            raise ValueError("Checkpoints must share a financial case and control grid.")
        self.value, self.terminal = value, terminal
        self.problem = PricingBSDE(value.case.model, value.case.contract)
        self.steps = terminal.config.n_steps
        self.dt = terminal.case.contract.maturity / self.steps
        for solver in (value, terminal):
            solver.network.eval().requires_grad_(False)

    def value_gradient(self, t: float, spots: np.ndarray) -> np.ndarray:
        points = torch.from_numpy(np.column_stack((np.full(len(spots), t), spots)))
        delta = value_derivatives(self.value.network, points, compute_hessian=False).gradient[:, 1:]
        return math.exp(-self.problem.model.rate*t) * self.problem.control(spots, delta.numpy())

    @torch.no_grad()
    def terminal_loss(self, t: float, spots: np.ndarray) -> np.ndarray:
        step = round(t/self.dt)
        if not 0 <= step < self.steps or not math.isclose(t, step*self.dt, abs_tol=1e-14, rel_tol=0):
            raise ValueError("Terminal control time must be a left endpoint on the checkpoint grid.")
        # The checkpoint output already incorporates the discount convention.
        return self.terminal.network.discounted_control_at(step, torch.tensor(spots)).numpy()


def _pair_samples(
    estimator: MartingaleEstimator, positive: np.ndarray,
    control: DiscountedControl | None, chunk_paths: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average the two payoffs and integrals in each IID antithetic pair."""
    payoff = np.empty(len(positive))
    integral = np.empty_like(payoff)
    pair_chunk = chunk_paths//2
    for start in range(0, len(positive), pair_chunk):
        block = positive[start:start+pair_chunk]
        n = len(block)
        samples = estimator.sample(np.concatenate((block, -block)), discounted_control=control)
        payoff[start:start+n] = (samples.discounted_payoff[:n] + samples.discounted_payoff[n:])/2
        integral[start:start+n] = (samples.stochastic_integral[:n] + samples.stochastic_integral[n:])/2
    return payoff, integral


def _pilot_coefficient(payoff: np.ndarray, integral: np.ndarray) -> float | None:
    x, z = payoff-payoff.mean(), integral-integral.mean()
    variance = z @ z
    return None if variance == 0 else float((x @ z)/variance)


def _method_row(
    price: EstimateStatistics, baseline: EstimateStatistics, reference: dict, *,
    unit: str, units: int, coefficient: float, seconds: float, timing_scope: str,
) -> dict:
    variance = price.standard_error**2
    difference = price.estimate-reference["estimate"]
    combined_se = math.hypot(price.standard_error, reference["standard_error"])
    return {
        "price": asdict(price), "sampling_unit": unit, "sampling_units": units,
        "coefficient": coefficient, "estimator_variance": variance,
        "variance_reduction_factor": baseline.standard_error**2/variance if variance > 0 else None,
        "variance_increased": variance > baseline.standard_error**2,
        "signed_reference_difference": difference, "combined_standard_error": combined_se,
        "reference_difference_in_standard_errors": difference/combined_se if combined_se else None,
        "measured_seconds": seconds, "timing_scope": timing_scope,
    }


def run_learned_control_study(pilot_directory: Path) -> dict:
    """Run the first declared allocation with one validation-selected development pair."""
    started = time.perf_counter()
    design = ExperimentDesign.load()
    spec = design.data
    protocol = spec["estimators"]
    case = design.case(protocol["reference_case"])
    path = Path(pilot_directory)
    report_bytes = (path/"report.json").read_bytes()
    pilot = json.loads(report_bytes)
    if pilot["status"] != "completed":
        raise ValueError("A completed arithmetic pilot is required.")
    source_reference = pilot["reference"]
    reference = source_reference["primary"]["price"]
    if (source_reference["reference_price_half_width"] > spec["targets"]["reference_fraction"]*spec["targets"]["price_absolute"]
            or not source_reference["independent_method_agreement"][0]["within_four_combined_standard_errors"]):
        raise ValueError("The pilot price reference must pass its precision and agreement checks.")

    seed = spec["development_seeds"][0]
    training = dict(spec["training"])
    training["n_steps"] = training.pop("base_steps")
    training["seed"] = seed
    checkpoints, solvers = {}, []
    for name, solver_type in (("value_gradient", FBSNNSolver), ("terminal_loss", TerminalBSDESolver)):
        checkpoint = path/name/"model.pt"
        source = pilot["models"][name]["source"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source["model_sha256"]:
            raise ValueError(f"{name} checkpoint differs from the pilot report.")
        solver = solver_type.load(checkpoint)
        expected = training if name == "value_gradient" else {
            k: v for k, v in training.items() if k not in {"terminal_weight", "patience"}
        }
        if (replace(solver.case, name=case.name) != case or asdict(solver.config) != expected
                or pilot["problem"] != json.loads(solver.case.to_json())
                or source["completed_updates"] != expected["iterations"]):
            raise ValueError(f"{name} checkpoint does not match the declared development fit.")
        checkpoints[name] = source
        solvers.append(solver)
    frozen = FrozenControls(*solvers)
    setup_seconds = time.perf_counter()-started

    n_pairs = protocol["antithetic_pairs"][0]
    pilot_pairs = protocol["coefficient_pilot_pairs"]
    scrambles = spec["reference"]["scrambles"]
    evaluation_states = 2*n_pairs
    net_size, remainder = divmod(evaluation_states, scrambles)
    if remainder or net_size < 2 or net_size & (net_size-1):
        raise ValueError("The matched allocation must supply a power of two per scramble.")
    chunk_paths = spec["resources"]["evaluation_chunk_paths"]
    if chunk_paths < 2:
        raise ValueError("Evaluation chunks must accommodate one complete antithetic pair.")
    streams = np.random.SeedSequence(seed, spawn_key=(26,)).spawn(4)
    increments_scale = math.sqrt(frozen.dt)
    estimator = MartingaleEstimator(case)
    pricer = BasketReferencePricer(case.model, case.contract)
    proxy = GeometricBasketReference(case.model, replace(case.contract, basket="geometric"))

    def analytical_proxy(t: float, spots: np.ndarray) -> np.ndarray:
        delta = proxy.call_delta_array(spots, t)
        return math.exp(-case.model.rate*t)*frozen.problem.control(spots, delta)

    def draw(count: int, stream: np.random.SeedSequence) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        increments = np.random.default_rng(stream).standard_normal((count, frozen.steps, case.model.dimension))
        increments *= increments_scale
        return increments, time.perf_counter()-start

    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        pilot_paths, pilot_draw_seconds = draw(pilot_pairs, streams[0])
        evaluation_paths, draw_seconds = draw(n_pairs, streams[1])
        begin = time.perf_counter()
        baseline_payoff, _ = _pair_samples(estimator, evaluation_paths, None, chunk_paths)
        baseline = _statistics(baseline_payoff, False)
        pair_options = {"unit": "antithetic_pair_mean", "units": n_pairs}
        methods = {"plain_antithetic": _method_row(
            baseline, baseline, reference, **pair_options, coefficient=0.0,
            seconds=draw_seconds+time.perf_counter()-begin, timing_scope="path draw and evaluation",
        )}
        for kind, control in (("analytical_geometric_proxy", analytical_proxy),
                              ("value_gradient", frozen.value_gradient),
                              ("terminal_loss", frozen.terminal_loss)):
            begin = time.perf_counter()
            payoff, integral = _pair_samples(estimator, evaluation_paths, control, chunk_paths)
            evaluation_seconds = draw_seconds+time.perf_counter()-begin
            integral_stats = asdict(_statistics(integral, False))
            methods[kind+"_unit"] = _method_row(
                _statistics(payoff-integral, False), baseline, reference, **pair_options, coefficient=1.0,
                seconds=evaluation_seconds, timing_scope="path draw and evaluation with frozen control",
            )
            methods[kind+"_unit"]["integral"] = integral_stats
            if kind != "analytical_geometric_proxy":
                begin = time.perf_counter()
                pilot_payoff, pilot_integral = _pair_samples(estimator, pilot_paths, control, chunk_paths)
                coefficient = _pilot_coefficient(pilot_payoff, pilot_integral)
                pilot_seconds = pilot_draw_seconds+time.perf_counter()-begin
                row = _method_row(
                    _statistics(payoff-(coefficient if coefficient is not None else 0.0)*integral, False),
                    baseline, reference, **pair_options, coefficient=coefficient if coefficient is not None else 0.0,
                    seconds=pilot_seconds+evaluation_seconds, timing_scope="coefficient pilot and evaluation",
                )
                row.update(pilot_seconds=pilot_seconds, evaluation_seconds=evaluation_seconds,
                           degenerate_pilot=coefficient is None, integral=integral_stats)
                methods[kind+"_fitted"] = row

        conventional_seeds = [int(stream.generate_state(1, dtype=np.uint64)[0]) for stream in streams[2:]]
        for fn, allocation, method_seed in (
            (pricer.antithetic, {"n_pairs": n_pairs}, conventional_seeds[0]),
            (pricer.sobol, {"power": net_size.bit_length()-1, "scrambles": scrambles}, conventional_seeds[1]),
        ):
            begin = time.perf_counter()
            result = fn(case.initial_spots, seed=method_seed, pilot_pairs=pilot_pairs, **allocation)
            methods[result.method] = _method_row(
                result.controlled.price, baseline, reference, unit=result.sampling_unit,
                units=result.n_units, coefficient=result.control_coefficients[0],
                seconds=time.perf_counter()-begin, timing_scope="coefficient pilot and price/delta evaluation",
            )
    finally:
        torch.set_num_threads(old_threads)

    return {
        "method": "development_learned_martingale_controls", "case": json.loads(case.to_json()),
        "specification_sha256": design.sha256, "pilot_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "seed": seed, "spawn_keys": [list(stream.spawn_key) for stream in streams],
        "conventional_seeds": conventional_seeds, "steps": frozen.steps, "evaluation_pairs": n_pairs,
        "evaluation_states_per_method": evaluation_states, "coefficient_pilot_pairs": pilot_pairs,
        "sobol_points_per_scramble": net_size, "reference_price": reference,
        "checkpoints": checkpoints, "methods": methods, "setup_seconds": setup_seconds,
        "wall_seconds": time.perf_counter()-started,
        "runtime": {"python": platform.python_version(), "system": platform.platform(),
                    "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
    }
