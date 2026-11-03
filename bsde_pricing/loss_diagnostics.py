"""Known-solution refinement and deliberate faults for the local BSDE loss."""

import math
import json
import platform

import numpy as np
import torch

from .analytical_torch import AnalyticalCall
from .bsde import PricingBSDE
from .bsde_loss import LocalBSDELoss
from .problems import PricingCase


class _ReversedDriver(LocalBSDELoss):
    def driver(self, values):
        return -super().driver(values)


class _TransposedControl(LocalBSDELoss):
    def control(self, spots, deltas):
        return (spots*self.sigma*deltas) @ self.cholesky.T


def _moments(samples: torch.Tensor) -> dict[str, float]:
    return {"mean": float(samples.mean()),
            "standard_error": float(samples.std(unbiased=True)/math.sqrt(len(samples)))}


def run_loss_diagnostics(
    case: PricingCase, *, n_paths: int = 2048, seed: int = 2026,
    steps: tuple[int, ...] = (8, 16, 32, 64),
) -> dict[str, object]:
    """CPU float64, exact states and coupled Brownian paths across grids.

    Faults share each correct run's paths. Reported uncertainty uses
    independent path sums, not a fictitious sample of independent time steps.
    """
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least two.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    steps = tuple(steps)
    if (len(steps) < 2 or any(isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in steps)
            or any(a >= b for a, b in zip(steps, steps[1:]))
            or any(steps[-1] % m for m in steps)):
        raise ValueError("At least two increasing nested positive step counts are required.")
    oracle = AnalyticalCall(case.model, case.contract)
    problem = PricingBSDE(case.model, case.contract)
    correct = LocalBSDELoss(problem, value_scale=case.contract.strike)
    reversed_driver = _ReversedDriver(problem, value_scale=case.contract.strike)
    transposed_control = _TransposedControl(problem, value_scale=case.contract.strike)
    horizon = case.contract.maturity
    rng = np.random.default_rng(seed)
    fine = rng.standard_normal((n_paths, steps[-1], case.model.dimension))*math.sqrt(horizon/steps[-1])
    cholesky = np.linalg.cholesky(case.model.correlation)
    correlation_applicable = not np.array_equal(cholesky, cholesky.T)
    driver_applicable = case.model.rate != 0
    rows = []
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for m in steps:
            increments = fine.reshape(n_paths, m, steps[-1]//m, case.model.dimension).sum(axis=2)
            states = torch.from_numpy(case.model.path_from_increments(case.initial_spots, horizon, increments))
            times = torch.from_numpy(np.linspace(0, horizon, m+1))
            increments = torch.from_numpy(increments)
            result = correct(oracle, times, states, increments, parameter_graph=False)
            faulty_driver = reversed_driver(oracle, times, states, increments, parameter_graph=False)
            path_rate = result.residuals.sum(dim=1)/horizon
            paired_driver = (faulty_driver.residuals-result.residuals).sum(dim=1)/horizon
            row = {"n_steps": m, "time_step": horizon/m, "local_loss": float(result.local),
                   "terminal_loss": float(result.terminal), "total_loss": float(result.total),
                   "local_residual_rms": float(torch.sqrt(result.residuals.square().mean())),
                   "summed_residual_rate": _moments(path_rate),
                   "reversed_driver_local_loss": float(faulty_driver.local),
                   "reversed_driver_residual_rate": _moments(faulty_driver.residuals.sum(dim=1)/horizon),
                   "paired_driver_rate_difference": _moments(paired_driver),
                   "transposed_control_local_loss": None}
            if correlation_applicable:
                faulty_control = transposed_control(oracle, times, states, increments, parameter_graph=False)
                row["transposed_control_local_loss"] = float(faulty_control.local)
            rows.append(row)
    finally:
        torch.set_num_threads(old_threads)
    last = rows[-1]
    gap = last["paired_driver_rate_difference"]
    checks = {
        "local_loss_decreases": rows[-1]["local_loss"] < rows[0]["local_loss"],
        "terminal_match": all(r["terminal_loss"] <= 1e-24 for r in rows),
        "residual_mean_consistent": all(
            abs(r["summed_residual_rate"]["mean"]) <= 4*r["summed_residual_rate"]["standard_error"]
            for r in rows),
    }
    if driver_applicable:
        checks["driver_sign_detected"] = abs(gap["mean"]) > 5*gap["standard_error"]
    if correlation_applicable:
        checks["correlation_transpose_detected"] = last["transposed_control_local_loss"] > 3*last["local_loss"]
    return {
        "method": "known_solution_local_bsde_loss", "case": case.name,
        "problem": json.loads(case.to_json()),
        "seed": seed, "bit_generator": type(rng.bit_generator).__name__, "n_paths": n_paths,
        "device": "cpu", "dtype": "torch.float64", "cpu_threads": 1,
        "torch": torch.__version__, "python": platform.python_version(), "numpy": np.__version__,
        "system": platform.platform(), "value_scale": case.contract.strike, "terminal_weight": 1,
        "correlation_fault_applicable": correlation_applicable,
        "driver_fault_applicable": driver_applicable,
        "checks": checks, "passed": all(checks.values()), "grids": rows,
    }
