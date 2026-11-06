"""Bounded arithmetic development using the locked settings and development streams."""

import json
import hashlib
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .arithmetic_pilot import ArithmeticPilot, ArithmeticPilotConfig, WorkerLimitError, _write_json, _peak_rss_mib
from .environment import _physical_memory_gib
from .experiment_design import ExperimentDesign
from .training import PathTrainingConfig


def profile_update_components(case, config: PathTrainingConfig, *, repeats: int = 7) -> dict:
    """Time fresh diagnostic models; these updates never supply accuracy results."""
    import torch
    from .training import FBSNNSolver, TrainingConfig
    from .terminal_bsde import TerminalBSDESolver

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("repeats must be an integer at least two.")
    old_threads = torch.get_num_threads()
    result = {"seed": 2034, "warmups": 2, "repeats": repeats,
              "scope": "fresh diagnostic models and Adam state; excludes imports and saved-fit accuracy",
              "models": {}}
    try:
        torch.set_num_threads(1)
        for kind in ("value_gradient", "terminal_loss"):
            solver = (FBSNNSolver(case, TrainingConfig(**asdict(config))) if kind == "value_gradient"
                      else TerminalBSDESolver(case, config))
            optimizer = torch.optim.Adam(solver.network.parameters(), lr=config.learning_rate)
            rng = np.random.default_rng(2034)
            samples = {k: [] for k in ("sample_paths", "objective_forward", "backward", "optimizer_step")}
            for iteration in range(2+repeats):
                started = time.perf_counter()
                states, increments = solver._batch(rng, config.batch_size)
                sampled = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = (solver.loss(solver.network, solver.times, states, increments).total
                        if kind == "value_gradient" else solver._loss(states, increments))
                forwarded = time.perf_counter()
                loss.backward()
                backwarded = time.perf_counter()
                optimizer.step()
                updated = time.perf_counter()
                if not bool(torch.isfinite(loss)):
                    raise ArithmeticError("Nonfinite diagnostic loss.")
                if iteration >= 2:
                    for key, value in zip(samples, (sampled-started, forwarded-sampled,
                                                   backwarded-forwarded, updated-backwarded)):
                        samples[key].append(value)
            medians = {k: float(np.median(v)) for k, v in samples.items()}
            result["models"][kind] = {"seconds": samples, "median_seconds": medians,
                                       "largest_median_component": max(medians, key=medians.get)}
    finally:
        torch.set_num_threads(old_threads)
    return result


class ArithmeticDevelopmentStudy:
    """Three paired development seeds with one shared independent reference.

    Final seed namespaces are never used. A failed task retains its row and
    guard record. Reference refinement depends only on declared precision checks.
    """

    def __init__(self, case_name: str = "arithmetic-25", design: ExperimentDesign | None = None):
        self.design = ExperimentDesign.load() if design is None else design
        self.spec = self.design.data
        self.case = self.design.case(case_name)
        if self.case.contract.basket != "arithmetic" or self.case.model.dimension < 10:
            raise ValueError("Development study requires a declared multi-asset arithmetic case.")

    def _config(self, seed, power, pairs, *, n_steps=None):
        common = dict(self.spec["training"])
        common["n_steps"] = common.pop("base_steps")
        if n_steps is not None:
            common["n_steps"] = n_steps
        common.pop("terminal_weight")
        common.pop("patience")
        common["seed"] = seed
        return ArithmeticPilotConfig(
            training=PathTrainingConfig(**common), reference_power=power, coarse_power=max(1, power-1),
            scrambles=self.spec["reference"]["scrambles"], pilot_pairs=self.spec["reference"]["pilot_pairs"],
            check_pairs=pairs, reference_seed=2032, check_seed=2033,
            price_target=self.spec["targets"]["price_absolute"],
            scaled_delta_target=self.spec["targets"]["weight_scaled_delta_absolute"],
            reference_fraction=self.spec["targets"]["reference_fraction"],
        )

    def _task(self, runner, kind, directory, limits):
        try:
            runner._run_worker(kind, directory, limits=limits)
        except (OSError, subprocess.SubprocessError, WorkerLimitError) as exc:
            failure = {"status": "failed", "task": kind, "error_type": type(exc).__name__, "error": str(exc)}
            guard = directory/f"{kind}_guard.json"
            if guard.exists():
                failure["status"] = json.loads(guard.read_text())["status"]
            _write_json(directory/f"{kind}_failure.json", failure)
            outcome = failure
        else:
            outcome = {"status": "completed"}
        guard = directory/f"{kind}_guard.json"
        if guard.exists():
            outcome["guard"] = json.loads(guard.read_text())
        return outcome

    def _design(self):
        design = {"method": "arithmetic_development_study", "problem": json.loads(self.case.to_json()),
                  "specification_sha256": self.design.sha256, "specification": self.spec,
                  "seeds": self.spec["development_seeds"], "reference_seeds": [2032, 2033],
                  "scope": "development only; final streams remain reserved"}
        design["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path(__file__).parent.glob("*.py"))}
        weights = np.asarray(self.case.contract.weights)
        spots = np.asarray(self.case.initial_spots)
        sigma = np.asarray(self.case.model.volatilities)
        covariance = np.outer(sigma, sigma)*np.asarray(self.case.model.correlation)
        initial_weights = weights*spots/float(weights@spots)
        forwards = weights*spots*np.exp((self.case.model.rate-np.asarray(self.case.model.dividend_yields))*self.case.contract.maturity)
        forward_weights = forwards/forwards.sum()
        eigenvalues = np.linalg.eigvalsh(self.case.model.correlation)
        design["basket_diagnostics"] = {
            "dimension": self.case.model.dimension, "weight_sum": float(weights.sum()),
            "initial_basket": float(weights@spots),
            "initial_instantaneous_relative_variance": float(initial_weights@covariance@initial_weights),
            "terminal_relative_variance": float(forward_weights@np.expm1(covariance*self.case.contract.maturity)@forward_weights),
            "correlation_min_eigenvalue": float(eigenvalues[0]), "correlation_max_eigenvalue": float(eigenvalues[-1]),
        }
        return design

    def run(self, directory: Path) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        design = self._design()
        _write_json(directory/"design.json", design)
        component_profile = None
        if self.case.model.dimension >= 50:
            diagnostic_dir = directory/"diagnostic"
            diagnostic_dir.mkdir()
            diagnostic_config = self._config(self.spec["development_seeds"][0],
                self.spec["reference"]["sobol_powers"][0], self.spec["reference"]["check_pairs"][0])
            _write_json(diagnostic_dir/"design.json", {"problem": design["problem"], "config": asdict(diagnostic_config)})
            outcome = self._task(ArithmeticPilot(diagnostic_config), "diagnostic", diagnostic_dir, self.spec["resources"])
            component_profile = {"task": outcome}
            if outcome["status"] == "completed":
                component_profile["timings"] = json.loads((diagnostic_dir/"components.json").read_text())
        levels, reference, usable = self._reference(directory, design)
        runs = []
        for seed in self.spec["development_seeds"]:
            row = self._run_seed(directory, design, seed, reference, usable)
            runs.append(row)
            report = {**design, "reference_levels": levels, "reference": reference,
                      "component_profile": component_profile,
                      "reference_usable": usable, "runs": runs, "wall_seconds": time.perf_counter()-started,
                      "status": "completed" if len(runs) == len(self.spec["development_seeds"]) and usable and all(
                          r.get("comparison_status") == "completed" and all(t["status"] == "completed" for t in r["tasks"].values())
                          for r in runs) else "incomplete"}
            report.update(controller_peak_resident_mib=_peak_rss_mib(), physical_memory_gib=_physical_memory_gib())
            report["summary"] = self._summary(runs)
            _write_json(directory/"report.json", report)
        return report

    def _reference(self, directory, design):
        levels = []
        reference, usable = None, False
        base_seed = self.spec["development_seeds"][0]
        for index, (power, pairs) in enumerate(zip(self.spec["reference"]["sobol_powers"], self.spec["reference"]["check_pairs"])):
            level_dir = directory/f"reference_level_{index}"
            level_dir.mkdir()
            config = self._config(base_seed, power, pairs)
            options = {**asdict(config), "powers": [power], "chunk_size": self.spec["reference"]["chunk_terminal_states"]}
            _write_json(level_dir/"design.json", {"problem": design["problem"], "config": options})
            limits = {**self.spec["resources"], "worker_wall_seconds": self.spec["reference"]["max_worker_seconds_per_level"]}
            outcome = self._task(ArithmeticPilot(config), "reference", level_dir, limits)
            row = {"directory": level_dir.name, "power": power, **outcome}
            if outcome["status"] == "completed":
                reference = json.loads((level_dir/"reference.json").read_text())
                usable = (all(reference["error_budget_met"].values()) and all(
                    r["within_four_combined_standard_errors"] for r in reference["independent_method_agreement"]))
                row.update(error_budget_met=reference["error_budget_met"], agreement_met=all(
                    r["within_four_combined_standard_errors"] for r in reference["independent_method_agreement"]))
            levels.append(row)
            if usable or outcome["status"] != "completed":
                break
        return levels, reference, usable

    def _run_seed(self, directory, design, seed, reference, usable, *, n_steps=None):
        run_dir = directory/f"seed_{seed}"
        run_dir.mkdir()
        config = self._config(seed, self.spec["reference"]["sobol_powers"][0], self.spec["reference"]["check_pairs"][0], n_steps=n_steps)
        _write_json(run_dir/"design.json", {"problem": design["problem"], "config": asdict(config)})
        runner = ArithmeticPilot(config)
        row = {"seed": seed, "directory": run_dir.name, "tasks": {}}
        for kind in ("value_gradient", "terminal_loss"):
            outcome = self._task(runner, kind, run_dir, self.spec["resources"])
            if outcome["status"] == "completed":
                source = json.loads((run_dir/kind/"report.json").read_text())
                outcome.update(status=source["status"], completed_updates=source["completed_updates"],
                               selected_step=source["selected_step"], training_seconds=source["training_seconds"],
                               evaluation_seconds=source["evaluation_seconds"],
                               selected_validation_objective=source[
                                   "selected_validation_total" if kind == "value_gradient"
                                   else "selected_validation_terminal"])
            row["tasks"][kind] = outcome
        if reference is not None and all((run_dir/k/"model.pt").exists() and
                row["tasks"][k]["status"] in ("completed", "time_budget_reached") for k in row["tasks"]):
            _write_json(run_dir/"reference.json", reference)
            comparison = self._task(runner, "comparison", run_dir, self.spec["resources"])
            row["comparison"] = comparison
            row["comparison_status"] = comparison["status"]
            if comparison["status"] == "completed":
                row["models"] = json.loads((run_dir/"comparison.json").read_text())
                for model in row["models"].values():
                    if not usable:
                        model["price_check"]["classification"] = "reference_inconclusive"
                        for check in model["scaled_delta_checks"]:
                            check["classification"] = "reference_inconclusive"
                    model["directional_delta"] = float(sum(model["initial"]["delta"]))
                    model["maximum_absolute_scaled_delta_error"] = float(max(abs(c["signed_error"]) for c in model["scaled_delta_checks"]))
        return row

    @staticmethod
    def _summary(runs):
        result = {}
        for kind in ("value_gradient", "terminal_loss"):
            rows = [r["models"][kind] for r in runs if "models" in r]
            result[kind] = {"scheduled_runs": len(runs), "compared_runs": len(rows)}
            for key, values in {
                "price_signed_error": [r["price_check"]["signed_error"] for r in rows],
                "maximum_absolute_scaled_delta_error": [r["maximum_absolute_scaled_delta_error"] for r in rows],
                "training_seconds": [r["tasks"][kind]["training_seconds"] for r in runs
                                     if "training_seconds" in r["tasks"][kind]],
                "worker_wall_seconds_all_attempts": [r["tasks"][kind]["guard"]["process_wall_seconds"] for r in runs
                                                    if "guard" in r["tasks"][kind]],
            }.items():
                result[kind][key] = {"values": values, "mean": float(np.mean(values)) if values else None,
                                    "sample_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                    "min": min(values) if values else None, "max": max(values) if values else None}
        return result


class TimeGridStudy(ArithmeticDevelopmentStudy):
    """Locked grid levels and development seeds, sharing one terminal reference.

    Fixed optimiser updates do not isolate discretisation bias. Seed labels
    repeat across grids, but Brownian paths are not coupled between resolutions.
    """

    def __init__(self, design: ExperimentDesign | None = None):
        design = ExperimentDesign.load() if design is None else design
        super().__init__(design.data["grid_case"], design)

    def run(self, directory: Path) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        design = {**self._design(), "method": "time_grid_development_study",
                  "grid_steps": sorted(self.spec["grid_steps"]),
                  "scope": "fixed-budget development grid study; no coupled cross-grid paths; final streams reserved"}
        _write_json(directory/"design.json", design)
        levels, reference, usable = self._reference(directory, design)
        grids = []
        for n_steps in design["grid_steps"]:
            grid_dir = directory/f"steps_{n_steps}"
            grid_dir.mkdir()
            grid_started = time.perf_counter()
            grid = {"n_steps": n_steps, "time_step": self.case.contract.maturity/n_steps,
                    "directory": grid_dir.name, "runs": []}
            grids.append(grid)
            for seed in self.spec["development_seeds"]:
                grid["runs"].append(self._run_seed(grid_dir, design, seed, reference, usable, n_steps=n_steps))
                grid["summary"] = self._summary(grid["runs"])
                grid["wall_seconds"] = time.perf_counter()-grid_started
                grid["status"] = "completed" if usable and len(grid["runs"]) == len(self.spec["development_seeds"]) and all(
                    r.get("comparison_status") == "completed" and all(t["status"] == "completed" for t in r["tasks"].values())
                    for r in grid["runs"]) else "incomplete"
                report = {**design, "reference_levels": levels, "reference": reference,
                          "reference_usable": usable, "grids": grids,
                          "scheduled_fits": len(design["grid_steps"])*len(self.spec["development_seeds"])*2,
                          "summary": {str(g["n_steps"]): g["summary"] for g in grids},
                          "wall_seconds": time.perf_counter()-started,
                          "controller_peak_resident_mib": _peak_rss_mib(),
                          "physical_memory_gib": _physical_memory_gib(),
                          "status": "completed" if len(grids) == len(design["grid_steps"]) and all(
                              g["status"] == "completed" for g in grids) else "incomplete"}
                _write_json(directory/"report.json", report)
        return report
