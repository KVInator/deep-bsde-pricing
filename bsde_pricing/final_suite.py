"""Run the locked confirmatory design in resumable, guarded tasks."""

import argparse
import fcntl
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from ._workers import WorkerLimitError, _guarded_worker, _peak_rss_mib, _worker_environment, _write_json
from .basket_reference import BasketReferencePricer
from .bsde import ControlRecovery, PricingBSDE
from .experiment_design import ExperimentDesign
from .geometric_basket import call_reference


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_reference(case) -> dict | None:
    reference = call_reference(case)
    if reference is None:
        return None
    scalar = case.model.dimension == 1
    result = reference.evaluate(case.initial_spots[0] if scalar else case.initial_spots)

    def stat(value):
        return {"estimate": value, "confidence_low": value, "confidence_high": value,
                "standard_error": 0.0}

    deltas = (result.call_delta,) if scalar else result.call_deltas
    return {"price": stat(result.call_price), "deltas": [stat(value) for value in deltas]}


def _reference_report(case, config: dict) -> dict:
    pricer = BasketReferencePricer(case.model, case.contract)
    started = time.perf_counter()
    sobol = pricer.sobol(case.initial_spots, power=config["power"], scrambles=config["scrambles"],
                         pilot_pairs=config["pilot_pairs"], seed=config["reference_seed"],
                         chunk_size=config["chunk_size"])
    level = {"power": config["power"], "seconds": time.perf_counter()-started, **asdict(sobol)}
    started = time.perf_counter()
    check = pricer.antithetic(case.initial_spots, n_pairs=config["check_pairs"],
                              pilot_pairs=config["pilot_pairs"], seed=config["check_seed"], chunk_size=config["chunk_size"])
    check_seconds = time.perf_counter()-started
    primary = level["controlled"]
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
    return {"sobol_levels": [level], "antithetic_check": {"seconds": check_seconds, **asdict(check)},
            "primary": primary, "reference_price_half_width": half[0],
            "reference_scaled_delta_half_widths": scaled_half,
            "error_budget_met": {"price": price_ok, "all_scaled_deltas": delta_ok},
            "independent_method_agreement": agreement,
            "interval_scope": "approximate pointwise 95% intervals",
            "refinement_scope": "one Sobol allocation and its independent antithetic check"}


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


def _recovery_summary(recovery: ControlRecovery) -> dict[str, float]:
    return {"max_absolute_reconstruction_residual": float(np.max(recovery.absolute_residual)),
            "max_relative_reconstruction_residual": float(np.max(recovery.relative_residual)),
            "condition_number_min": float(np.min(recovery.condition_number)),
            "condition_number_max": float(np.max(recovery.condition_number)),
            "smallest_diffusion_singular_value": float(np.min(recovery.smallest_singular_value)),
            "largest_inverse_diffusion_norm": float(np.max(recovery.inverse_norm))}


def _fit(case, directory: Path, task: dict, spec: dict) -> dict:
    import torch

    from .neural import value_derivatives
    from .terminal_bsde import TerminalBSDESolver, train_terminal_and_save
    from .training import FBSNNSolver, PathTrainingConfig, TrainingConfig, train_and_save

    torch.set_num_threads(1)
    options = dict(spec["training"])
    options.pop("base_steps")
    options.update(seed=task["seed"], n_steps=task["n_steps"])
    value = task["solver"] == "value_gradient"
    if not value:
        options.pop("terminal_weight")
        options.pop("patience")
    config = (TrainingConfig if value else PathTrainingConfig)(**options)
    target = directory/"checkpoint"
    report = (train_and_save if value else train_terminal_and_save)(case, config, target)
    solver = (FBSNNSolver if value else TerminalBSDESolver).load(target/"model.pt")
    started = time.perf_counter()
    initial = solver.initial_value()
    problem = PricingBSDE(case.model, case.contract)
    spots = np.array(case.initial_spots)
    weights = np.array(case.contract.weights)
    if value:
        delta = np.atleast_1d(initial["delta"])
        control = problem.control(spots, delta)
    else:
        # The terminal solver's time-zero discount is one.
        control = np.atleast_1d(initial["discounted_initial_control"])
    recovery = problem.recover_deltas(spots, control)
    if not value:
        delta = recovery.deltas
    if case.contract.basket == "geometric":
        exact = solver.reference.evaluate(spots)
        upper = weights*exact.geometric_spot/spots * np.exp(
            -exact.effective_dividend_yield*case.contract.maturity,
        )
    else:
        upper = weights*np.exp(-np.array(case.model.dividend_yields)*case.contract.maturity)

    price_check, delta_checks = None, []
    reference = task["reference"]
    if reference is not None:
        price_check = _error_assessment(initial["price"], reference["price"], spec["targets"]["price_absolute"])
        delta_checks = [_error_assessment(float(x), ref, spec["targets"]["weight_scaled_delta_absolute"], float(w))
                        for x, ref, w in zip(delta, reference["deltas"], weights)]
    if not task["reference_usable"]:
        accuracy = "reference_inconclusive"
    else:
        classes = {check["classification"] for check in [price_check, *delta_checks]}
        accuracy = ("accuracy_miss" if "outside_target" in classes else
                    "within_target" if classes == {"within_target"} else "reference_inconclusive")

    rng = np.random.default_rng(np.random.SeedSequence(task["path_seed"], spawn_key=(3,)))
    rmse_keys = ("price_rmse_before_expiry", "delta_rmse_before_expiry", "terminal_payoff_rmse",
                 "discounted_terminal_rmse", "analytical_discounted_terminal_rmse")
    squared = Counter()
    chunk = spec["resources"]["evaluation_chunk_paths"]
    for start in range(0, config.evaluation_paths, chunk):
        count = min(chunk, config.evaluation_paths-start)
        sample = solver._evaluate(rng, count).summary
        for key in rmse_keys:
            if sample.get(key) is not None:
                squared[key] += count*sample[key]**2
    gamma = None
    if value and case.model.dimension == 1:
        point = torch.tensor([[0.0, spots[0]]], dtype=torch.float64)
        gamma = {"estimate": float(value_derivatives(solver.network, point).hessian[0, 1, 1]),
                 "reference": solver.reference.evaluate(float(spots[0])).gamma}
    return {
        "training_status": report["status"], "completed_updates": report["completed_updates"],
        "selected_step": report["selected_step"], "training_seconds": report["training_seconds"],
        "postfit_seconds": report["evaluation_seconds"], "checkpoint_sha256": _digest(target/"model.pt"),
        "price": initial["price"], "delta": delta.tolist(), "directional_delta": float(delta.sum()),
        "initial_bound_violations": int(np.count_nonzero((delta < -1e-10) | (delta > upper+1e-10))),
        "parameters": sum(p.numel() for p in solver.network.parameters()),
        "diffusion": _recovery_summary(recovery), "gamma": gamma,
        "price_check": price_check, "delta_checks": delta_checks, "accuracy": accuracy,
        "path_seed": task["path_seed"], "path_count": config.evaluation_paths,
        "path_rmse": {k: (v/config.evaluation_paths)**0.5 for k, v in squared.items()},
        "final_evaluation_seconds": time.perf_counter()-started,
    }


def _estimator(task: dict, design: ExperimentDesign) -> dict:
    import torch

    from .control_cost import ControlCostStudy
    from .learned_control import FrozenControls
    from .terminal_bsde import TerminalBSDESolver
    from .training import FBSNNSolver

    torch.set_num_threads(1)
    solvers, seconds = [], {}
    for name, loader in (("value_gradient", FBSNNSolver), ("terminal_loss", TerminalBSDESolver)):
        started = time.perf_counter()
        path = Path(task["checkpoints"][name])
        source = task["sources"][name]
        if _digest(path) != source["checkpoint_sha256"]:
            raise ValueError("Estimator checkpoint digest differs from the final fit.")
        solvers.append(loader.load(path))
        seconds[name] = time.perf_counter()-started
    frozen = FrozenControls(*solvers)
    study = ControlCostStudy(design, frozen)
    key = (task["seed"], task["repetition"])
    pilot = np.random.SeedSequence(task["pilot_seed"], spawn_key=key)
    evaluation, antithetic, sobol = np.random.SeedSequence(task["evaluation_seed"], spawn_key=key).spawn(3)
    return {"methods_by_pairs": study.repetition(pilot, evaluation, antithetic, sobol),
            "checkpoint_setup_seconds": seconds}


def _worker(directory: Path) -> None:
    task = _read(directory/"task.json")
    design = ExperimentDesign(task["specification"])
    case = design.case(task["case"])
    if task["kind"] == "reference":
        result = _reference_report(case, task["config"])
    elif task["kind"] == "fit":
        result = _fit(case, directory, task, design.data)
    else:
        result = _estimator(task, design)
    _write_json(directory/"result.json", result)
    _write_json(directory/"task_resources.json", {"peak_resident_mib": _peak_rss_mib()})


class FinalSuite:
    """A serial ledger for one immutable source, runtime and specification."""

    def __init__(self, directory: Path, design: ExperimentDesign | None = None):
        self.directory = Path(directory).resolve()
        self.design = ExperimentDesign.load() if design is None else design
        self.spec = self.design.data
        self.resources = self.spec["resources"]
        self.limit = 3600*self.resources["final_total_wall_hours"]
        self.manifest = {
            "specification_sha256": self.design.sha256, "specification": self.spec,
            "source_sha256": {p.name: _digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "runtime": {"system": platform.platform(), "python": platform.python_version(),
                        **{name: version(name) for name in ("numpy", "scipy", "torch")}},
        }

    def _save(self):
        temp = self.directory/"report.tmp"
        _write_json(temp, self.report)
        temp.replace(self.directory/"report.json")

    def _execute(self, row: dict, relative: Path, task: dict, cap: float) -> None:
        if row["status"] not in {"pending", "running", "interrupted"}:
            return
        remaining = self.limit-self.report["elapsed_seconds"]
        if remaining <= 0:
            row["status"] = "not_attempted_budget"
            self._save()
            return
        attempts = row.setdefault("attempts", [])
        if row["status"] == "running":
            attempts[-1]["status"] = "interrupted"
            self.report["elapsed_seconds"] += max(0, time.time()-attempts[-1]["started_at"])
            remaining = self.limit-self.report["elapsed_seconds"]
            if remaining <= 0:
                row["status"] = "not_attempted_budget"
                self._save()
                return
        path = relative/f"attempt-{len(attempts)+1}"
        directory = self.directory/path
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(directory/"task.json", {**task, "specification": self.spec})
        attempt = {"directory": str(path), "started_at": time.time(), "status": "running"}
        attempts.append(attempt)
        row["status"] = "running"
        self._save()
        print(f'{task["kind"]} {relative}', file=sys.stderr, flush=True)
        started = time.perf_counter()
        limits = {**self.resources, "worker_wall_seconds": min(cap, remaining)}
        command = [sys.executable, "-B", "-m", "bsde_pricing.final_suite", str(directory)]
        try:
            with (directory/"worker.log").open("w", encoding="utf-8") as log:
                _guarded_worker(command, _worker_environment(), log, directory, "task", limits)
        except KeyboardInterrupt:
            row["status"] = attempt["status"] = "interrupted"
            raise
        except (OSError, subprocess.SubprocessError, WorkerLimitError) as exc:
            guard = directory/"task_guard.json"
            row["status"] = _read(guard)["status"] if guard.exists() else "worker_failure"
            row["error"] = str(exc)
        else:
            result = _read(directory/"result.json")
            row.update(status="completed", result=result)
            if task["kind"] == "fit" and result["training_status"] == "time_budget_reached":
                row["status"] = "time_limit"
        finally:
            attempt["status"] = row["status"]
            attempt["seconds"] = time.perf_counter()-started
            self.report["elapsed_seconds"] += attempt["seconds"]
            if (directory/"task_guard.json").exists():
                attempt["guard"] = _read(directory/"task_guard.json")
            self._save()

    def _references(self):
        settings = self.spec["reference"]
        targets = self.spec["targets"]
        for index, case_row in enumerate(self.spec["cases"]):
            name = case_row["name"]
            row = self.report["references"][name]
            if row["status"] != "pending":
                continue
            exact = _exact_reference(self.design.case(name))
            if exact is not None:
                row.update(status="exact", primary=exact)
                self._save()
                continue
            streams = self.spec["final_streams"]
            for level, (power, pairs) in enumerate(zip(settings["sobol_powers"], settings["check_pairs"])):
                entry = row["levels"][level]
                config = {"power": power,
                          "scrambles": settings["scrambles"], "pilot_pairs": settings["pilot_pairs"],
                          "check_pairs": pairs, "reference_seed": streams["reference_seed_base"]+index,
                          "check_seed": streams["independent_check_seed_base"]+index,
                          "chunk_size": settings["chunk_terminal_states"], "reference_fraction": targets["reference_fraction"],
                          "price_target": targets["price_absolute"], "scaled_delta_target": targets["weight_scaled_delta_absolute"]}
                self._execute(entry, Path("references")/name/f"level-{level+1}",
                              {"kind": "reference", "case": name, "config": config},
                              settings["max_worker_seconds_per_level"])
                if entry["status"] != "completed":
                    continue
                result = entry["result"]
                row["primary"] = result["primary"]
                if all(result["error_budget_met"].values()) and all(
                    check["within_four_combined_standard_errors"] for check in result["independent_method_agreement"]
                ):
                    row["status"] = "usable"
                    for unused in row["levels"][level+1:]:
                        unused["status"] = "not_required"
                    break
            if row["status"] == "pending":
                row["status"] = "inconclusive"
            self._save()

    def _fits(self):
        indices = {row["name"]: i for i, row in enumerate(self.spec["cases"])}
        for row in self.report["fits"]:
            job = row["job"]
            name, steps, solver, seed = (job[k] for k in ("case", "n_steps", "solver", "seed"))
            reference = self.report["references"][name]
            self._execute(row, Path("fits")/name/str(steps)/solver/str(seed),
                          {**job, "kind": "fit", "reference": reference.get("primary"),
                           "reference_usable": reference["status"] in {"exact", "usable"},
                           "path_seed": self.spec["final_streams"]["path_evaluation_seed_base"]+indices[name]},
                          self.resources["worker_wall_seconds"])

    def _estimators(self):
        name = self.spec["estimators"]["reference_case"]
        steps = self.spec["training"]["base_steps"]
        index = next(i for i, row in enumerate(self.spec["cases"]) if row["name"] == name)
        streams = self.spec["final_streams"]
        reference = self.report["references"][name]
        for row in self.report["estimators"]:
            seed, repeat = row["seed"], row["repetition"]
            fits = {r["job"]["solver"]: r for r in self.report["fits"]
                    if (r["job"]["case"], r["job"]["n_steps"], r["job"]["seed"]) == (name, steps, seed)}
            if reference["status"] != "usable" or not all(r["status"] == "completed" for r in fits.values()):
                row["status"] = "dependency_unavailable"
                self._save()
                continue
            paths = {kind: str(self.directory/r["attempts"][-1]["directory"]/"checkpoint/model.pt")
                     for kind, r in fits.items()}
            task = {"kind": "estimator", "case": name, "seed": seed, "repetition": repeat,
                    "checkpoints": paths, "sources": {k: r["result"] for k, r in fits.items()},
                    "pilot_seed": streams["coefficient_pilot_seed_base"]+index,
                    "evaluation_seed": streams["estimator_evaluation_seed_base"]+index}
            self._execute(row, Path("estimators")/str(seed)/str(repeat+1), task,
                          self.resources["worker_wall_seconds"])

    def run(self) -> dict:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("The final suite already has a running controller.") from exc
            path = self.directory/"report.json"
            if path.exists():
                self.report = _read(path)
                if self.report["manifest"] != self.manifest:
                    raise ValueError("Resume requires the original source, runtime and locked specification.")
            else:
                if any(p.name != ".lock" for p in self.directory.iterdir()):
                    raise FileExistsError("The final-suite directory contains an unrecognised run.")
                self.report = {
                    "manifest": self.manifest, "elapsed_seconds": 0.0,
                    "references": {row["name"]: {"status": "pending", "levels": [
                        {"status": "pending"} for _ in self.spec["reference"]["sobol_powers"]
                    ] if row["family"] == "arithmetic" else []} for row in self.spec["cases"]},
                    "fits": [{"job": job, "status": "pending"} for job in self.design.jobs()],
                    "estimators": [{"seed": seed, "repetition": repeat, "status": "pending"}
                                   for seed in self.spec["final_training_seeds"]
                                   for repeat in range(self.spec["estimators"]["repetitions"])],
                }
                self._save()
            self._references()
            self._fits()
            self._estimators()
            summary = {
                "report": str(path), "elapsed_seconds": self.report["elapsed_seconds"],
                "fit_status": dict(Counter(r["status"] for r in self.report["fits"])),
                "fit_accuracy": dict(Counter(r.get("result", {}).get("accuracy", "unavailable")
                                             for r in self.report["fits"])),
                "reference_status": dict(Counter(r["status"] for r in self.report["references"].values())),
                "estimator_status": dict(Counter(r["status"] for r in self.report["estimators"])),
            }
            self.report["summary"] = summary
            self._save()
            return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute one final-suite task.")
    parser.add_argument("directory", type=Path)
    _worker(parser.parse_args().directory)
