"""A bounded arithmetic-basket pilot with independently measured references."""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .training import PathTrainingConfig


def _default_training():
    from .training import PathTrainingConfig

    return PathTrainingConfig()


@dataclass(frozen=True)
class ArithmeticPilotConfig:
    training: "PathTrainingConfig" = field(default_factory=_default_training)
    coarse_power: int = 12
    reference_power: int = 16
    scrambles: int = 16
    pilot_pairs: int = 4096
    check_pairs: int = 131_072
    reference_seed: int = 2032
    check_seed: int = 2033
    price_target: float = 0.01
    scaled_delta_target: float = 0.01
    reference_fraction: float = 0.20

    def __post_init__(self):
        from .training import PathTrainingConfig

        if type(self.training) is not PathTrainingConfig:
            raise ValueError("The arithmetic pilot uses one common PathTrainingConfig.")
        for name, minimum in (("coarse_power", 1), ("reference_power", 1), ("scrambles", 4),
                              ("pilot_pairs", 2), ("check_pairs", 2),
                              ("reference_seed", 0), ("check_seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}.")
        if not self.coarse_power < self.reference_power <= 20:
            raise ValueError("Require coarse_power < reference_power <= 20 for a bounded pilot.")
        if self.reference_seed == self.check_seed or self.training.seed in (self.reference_seed, self.check_seed):
            raise ValueError("Training, Sobol reference and IID check seeds must differ.")
        for value in (self.price_target, self.scaled_delta_target, self.reference_fraction):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Targets and reference_fraction must be positive and finite.")
        if self.reference_fraction > 1:
            raise ValueError("reference_fraction cannot exceed one.")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")


def _peak_rss_mib() -> float:
    """Use the executed image's high-water mark on Linux and rusage on macOS."""
    if platform.system() == "Linux":
        # Linux rusage can retain a fork/exec startup floor from the controller.
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1])/1024
        raise RuntimeError("Linux process status does not expose VmHWM.")
    if platform.system() == "Darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    raise RuntimeError("Peak resident memory measurement requires macOS or Linux.")


class WorkerLimitError(RuntimeError):
    """An isolated task exceeded its declared wall or resident-memory budget."""


def _child_rss_mib(pid: int) -> float | None:
    if platform.system() == "Linux":
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        except FileNotFoundError:
            return None
        for line in lines:
            if line.startswith("VmRSS:"):
                return float(line.split()[1])/1024
        return None  # A process that has exited can remain briefly as a zombie.
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    if not result.stdout.strip():
        return None
    return float(result.stdout.strip())/1024


def _guarded_worker(command, environment, log, directory, kind, limits):
    started = time.perf_counter()
    peak, status = 0.0, "completed"
    with subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT) as process:
        try:
            while process.poll() is None:
                rss = _child_rss_mib(process.pid)
                peak = max(peak, rss if rss is not None else 0)
                if peak > limits["worker_peak_resident_mib"]:
                    status = "memory_limit"
                    break
                if time.perf_counter()-started >= limits["worker_wall_seconds"]:
                    status = "time_limit"
                    break
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        resource_path = directory/f"{kind}_resources.json"
        if resource_path.exists():
            peak = max(peak, json.loads(resource_path.read_text())["peak_resident_mib"])
        if peak > limits["worker_peak_resident_mib"]:
            status = "memory_limit"
        _write_json(directory/f"{kind}_guard.json", {
            "status": status if status != "completed" or process.returncode == 0 else "numerical_failure",
            "peak_resident_mib": peak, "process_wall_seconds": time.perf_counter()-started,
            "returncode": process.returncode, "poll_seconds": 0.2, "limits": limits,
        })
        if status != "completed":
            raise WorkerLimitError(f"{kind}: {status}")
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def _reference_report(case, config: dict) -> dict:
    import numpy as np

    from .basket_reference import BasketReferencePricer

    pricer = BasketReferencePricer(case.model, case.contract)
    levels = []
    chunk = {"chunk_size": config["chunk_size"]} if "chunk_size" in config else {}
    for power in config.get("powers", (config["coarse_power"], config["reference_power"])):
        started = time.perf_counter()
        result = pricer.sobol(case.initial_spots, power=power, scrambles=config["scrambles"],
                              pilot_pairs=config["pilot_pairs"], seed=config["reference_seed"], **chunk)
        levels.append({"power": power, "seconds": time.perf_counter()-started, **asdict(result)})
    started = time.perf_counter()
    check = pricer.antithetic(case.initial_spots, n_pairs=config["check_pairs"],
                              pilot_pairs=config["pilot_pairs"], seed=config["check_seed"], **chunk)
    check_seconds = time.perf_counter()-started
    primary = levels[-1]["controlled"]
    half = [_half_width(s) for s in (primary["price"], *primary["deltas"])]
    weights = np.asarray(case.contract.weights)
    scaled_half = [None if h is None else h/w for h, w in zip(half[1:], weights)]
    price_ok = half[0] is not None and half[0] <= config["reference_fraction"]*config["price_target"]
    delta_ok = all(h is not None and h <= config["reference_fraction"]*config["scaled_delta_target"]
                   for h in scaled_half)
    agreement = []
    for a, b in zip((primary["price"], *primary["deltas"]),
                    (asdict(check.controlled.price), *(asdict(s) for s in check.controlled.deltas))):
        se = math.hypot(a["standard_error"], b["standard_error"])
        difference = a["estimate"]-b["estimate"]
        agreement.append({"signed_difference": difference, "combined_standard_error": se,
                          "difference_in_standard_errors": difference/se if se > 0 else None,
                          "within_four_combined_standard_errors": abs(difference) <= 4*se})
    return {"sobol_levels": levels, "antithetic_check": {"seconds": check_seconds, **asdict(check)},
            "primary": primary, "reference_price_half_width": half[0],
            "reference_scaled_delta_half_widths": scaled_half,
            "error_budget_met": {"price": price_ok, "all_scaled_deltas": delta_ok},
            "independent_method_agreement": agreement,
            "interval_scope": "pointwise approximate 95% intervals; no simultaneous coverage claim",
            "refinement_scope": ("one allocation; refinement records retained by the development study" if "powers" in config else
                                 "same scramble seeds and pilot at both Sobol sizes; levels are dependent")}


def _half_width(stat: dict) -> float | None:
    low, high = stat["confidence_low"], stat["confidence_high"]
    return None if low is None or high is None else (high-low)/2


def _error_assessment(learned: float, stat: dict, target: float, scale: float = 1.0) -> dict:
    """Propagate reference uncertainty into a signed, optionally scaled error."""
    error = (learned-stat["estimate"])/scale
    half = _half_width(stat)
    if half is None:
        interval, classification = None, "reference_uncertainty_unavailable"
    else:
        half /= scale
        interval = [error-half, error+half]
        classification = ("within_target" if abs(error)+half <= target else
                          "outside_target" if abs(error)-half > target else "unresolved_at_reference_precision")
    return {"estimate": learned, "reference_estimate": stat["estimate"],
            "signed_error": error, "signed_error_interval": interval, "scale": scale,
            "absolute_error_target": target, "classification": classification}


def _worker(kind: str, directory: Path) -> None:
    """Run exactly one task in a fresh process, retaining its own RSS peak."""
    design = json.loads((directory/"design.json").read_text())
    config = design["config"]
    from .problems import EuropeanCall, GBMModel, PricingCase

    problem = design["problem"]
    case = PricingCase(problem["name"], GBMModel(**problem["model"]),
                       EuropeanCall(**problem["contract"]), problem["initial_spots"])
    started = time.perf_counter()
    baseline_peak = _peak_rss_mib()
    if kind == "reference":
        result = _reference_report(case, config)
        _write_json(directory/"reference.json", result)
    elif kind == "diagnostic":
        from .development_study import profile_update_components
        from .training import PathTrainingConfig

        _write_json(directory/"components.json", profile_update_components(case, PathTrainingConfig(**config["training"])))
    elif kind == "greeks":
        from .greek_study import GreekDiagnostics

        _write_json(directory/"greeks.json", GreekDiagnostics(case).run(directory, design["greek_options"]))
    elif kind == "comparison":
        from .training import PathTrainingConfig

        options = {k: v for k, v in config.items() if k not in ("powers", "chunk_size")}
        options["training"] = PathTrainingConfig(**options["training"])
        reference = json.loads((directory/"reference.json").read_text())
        _write_json(directory/"comparison.json", ArithmeticPilot(ArithmeticPilotConfig(**options))._compare(directory, reference))
    else:
        from .training import PathTrainingConfig, TrainingConfig, train_and_save
        from .terminal_bsde import train_terminal_and_save

        if kind == "value_gradient":
            train_and_save(case, TrainingConfig(**config["training"]), directory/kind)
        elif kind == "terminal_loss":
            train_terminal_and_save(case, PathTrainingConfig(**config["training"]), directory/kind)
        else:
            raise ValueError("Unknown arithmetic-pilot worker.")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    _write_json(directory/f"{kind}_resources.json", {
        "peak_resident_mib": _peak_rss_mib(), "baseline_peak_resident_mib": baseline_peak,
        "worker_seconds": time.perf_counter()-started,
        "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
        "memory_scope": "fresh process lifetime peak RSS including imports, task, evaluation and saving",
        "memory_source": "proc VmHWM" if platform.system() == "Linux" else "getrusage ru_maxrss bytes",
        "system": platform.platform(), "python": platform.python_version(),
        "executable": sys.executable,
    })


class ArithmeticPilot:
    """One fixed-seed pair of fits, with frozen references and resource costs.

    Three tasks run sequentially in fresh Python processes. Reference values
    never enter either training process. Existing directories are protected.
    """

    def __init__(self, config: ArithmeticPilotConfig | None = None):
        self.config = ArithmeticPilotConfig() if config is None else config

    def _run_worker(self, kind: str, directory: Path, *, limits: dict | None = None) -> None:
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            environment[name] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with (directory/f"{kind}.log").open("w") as log:
            command = [sys.executable, "-B", "-m", "bsde_pricing.arithmetic_pilot",
                       "--worker", kind, "--run-dir", str(directory.resolve())]
            if limits is None:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=self.config.training.max_seconds+120 if kind != "reference" else 180)
            else:
                _guarded_worker(command, environment, log, directory, kind, limits)

    def _compare(self, directory: Path, reference: dict) -> dict:
        import numpy as np
        import torch

        from .risk_comparison import SolverRiskComparison
        from .terminal_bsde import TerminalBSDESolver
        from .training import FBSNNSolver

        value = FBSNNSolver.load(directory/"value_gradient/model.pt")
        terminal = TerminalBSDESolver.load(directory/"terminal_loss/model.pt")
        comparison = SolverRiskComparison(value, terminal)
        primary = reference["primary"]
        models = {}
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for kind, solver in (("value_gradient", value), ("terminal_loss", terminal)):
                source_path = directory/kind/"report.json"
                source = json.loads(source_path.read_text())
                initial = comparison.initial_risk(kind)
                recorded = source["evaluation"]["initial"]
                pairs = (("price", "price"), ("delta", "delta")) if kind == "value_gradient" else (
                    ("price", "price"), ("brownian_control", "discounted_initial_control"))
                if source["problem"] != json.loads(solver.case.to_json()) or source["config"] != asdict(solver.config):
                    raise ValueError("Saved arithmetic model and report metadata disagree.")
                for actual, expected in pairs:
                    if (np.shape(initial[actual]) != np.shape(recorded[expected]) or
                            not np.allclose(initial[actual], recorded[expected], rtol=0, atol=1e-10)):
                        raise ValueError("Saved arithmetic model and report values disagree.")
                for _ in range(2):
                    comparison.initial_risk(kind)
                seconds = []
                for _ in range(5):
                    started = time.perf_counter()
                    comparison.initial_risk(kind)
                    seconds.append(time.perf_counter()-started)
                weights = solver.case.contract.weights
                scaled = [_error_assessment(d, s, self.config.scaled_delta_target, w)
                          for d, s, w in zip(initial["delta"], primary["deltas"], weights)]
                upper = np.asarray(weights)*np.exp(-np.asarray(solver.case.model.dividend_yields)*solver.case.contract.maturity)
                deltas = np.asarray(initial["delta"])
                models[kind] = {
                    "initial": initial,
                    "price_check": _error_assessment(initial["price"], primary["price"], self.config.price_target),
                    "scaled_delta_checks": scaled,
                    "initial_delta_bound_violations": int(np.count_nonzero((deltas < -1e-10) | (deltas > upper+1e-10))),
                    "parameter_count": sum(p.numel() for p in solver.network.parameters()),
                    "source": {"directory": kind, "status": source["status"],
                               "completed_updates": source["completed_updates"], "selected_step": source["selected_step"],
                               "checkpoint_rule": source["checkpoint_rule"],
                               "training_seconds": source["training_seconds"], "evaluation_seconds": source["evaluation_seconds"],
                               "training_runtime": {k: source[k] for k in ("python", "numpy", "torch", "system", "device", "dtype")},
                               "model_sha256": hashlib.sha256((directory/kind/"model.pt").read_bytes()).hexdigest(),
                               "report_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
                    "initial_inference_seconds": seconds, "initial_inference_median_seconds": float(np.median(seconds)),
                    "timing_scope": "two warmups and five repeats; initial price, delta, control and diffusion diagnostics",
                    "path_delta_rmse": None,
                    "path_reference_scope": "no conditional arithmetic reference evaluated on the path grid",
                }
        finally:
            torch.set_num_threads(old_threads)
        return models

    def run(self, directory: Path) -> dict:
        from .environment import _physical_memory_gib
        from .problems import arithmetic_ten_case

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        design = {"method": "ten_asset_arithmetic_pilot", "problem": json.loads(arithmetic_ten_case().to_json()),
                  "config": asdict(self.config), "tasks": ["reference", "value_gradient", "terminal_loss"],
                  "training_reference_labels": False, "selection": "each solver's fixed validation objective",
                  "scope": "one development seed; experiment budgets remain to be locked"}
        _write_json(directory/"design.json", design)
        resources = {}
        for kind in design["tasks"]:
            worker_started = time.perf_counter()
            try:
                self._run_worker(kind, directory)
            except (OSError, subprocess.SubprocessError) as exc:
                _write_json(directory/"failure.json", {"status": "failed", "task": kind,
                            "error_type": type(exc).__name__, "error": str(exc), "design": design})
                raise RuntimeError(f"Arithmetic pilot task {kind} failed; inspect {directory/kind}.log.") from exc
            resources[kind] = json.loads((directory/f"{kind}_resources.json").read_text())
            resources[kind]["process_wall_seconds"] = time.perf_counter()-worker_started
            _write_json(directory/f"{kind}_resources.json", resources[kind])
        reference = json.loads((directory/"reference.json").read_text())
        try:
            models = self._compare(directory, reference)
        except (OSError, ValueError, RuntimeError, ArithmeticError) as exc:
            _write_json(directory/"failure.json", {"status": "failed", "task": "comparison",
                        "error_type": type(exc).__name__, "error": str(exc), "design": design})
            raise
        report = {**design, "status": "completed" if all(m["source"]["status"] == "completed" for m in models.values())
                  else "time_budget_reached", "reference": reference, "models": models,
                  "resources": resources, "physical_memory_gib": _physical_memory_gib(),
                  "controller_peak_resident_mib": _peak_rss_mib(),
                  "wall_seconds": time.perf_counter()-started,
                  "budget_projection": {
                      "basis": "three repeats of this exact case and configuration; excludes later dimensions and tuning",
                      "three_seed_fit_seconds": 3*sum(m["source"]["training_seconds"] for m in models.values()),
                      "three_seed_fit_and_evaluation_seconds": 3*sum(m["source"]["training_seconds"]+m["source"]["evaluation_seconds"] for m in models.values()),
                      "worker_memory_scope": "largest individual worker RSS; controller and other applications are additional",
                      "largest_worker_peak_resident_mib": max(r["peak_resident_mib"] for r in resources.values()),
                  }}
        _write_json(directory/"report.json", report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal isolated arithmetic-pilot worker")
    parser.add_argument("--worker", required=True, choices=("reference", "value_gradient", "terminal_loss", "comparison", "diagnostic", "greeks"))
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    _worker(arguments.worker, arguments.run_dir)
