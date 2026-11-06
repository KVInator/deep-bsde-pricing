"""Locked financial stresses and independent initial Greek diagnostics."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from .arithmetic_pilot import ArithmeticPilot, _error_assessment, _peak_rss_mib, _write_json
from .basket_reference import BasketReferencePricer
from .development_study import ArithmeticDevelopmentStudy
from .experiment_design import ExperimentDesign
from .geometric_basket import GeometricBasketReference
from .monte_carlo import _statistics
from .neural import value_derivatives
from .risk_comparison import SolverRiskComparison
from .terminal_bsde import TerminalBSDESolver
from .training import FBSNNSolver


def scramble_statistics(values: np.ndarray) -> dict:
    """Pointwise t interval across independent scramble means, not net points."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("At least four finite scalar scramble estimates are required.")
    return asdict(_statistics(values, False, critical_value=float(t.ppf(0.975, len(values)-1))))


class GreekDiagnostics:
    """Initial directional deltas and selected directional curvatures.

    Curvature references are central secants of pathwise deltas using common
    random numbers. Their intervals describe sampling error at a finite bump.
    """

    def __init__(self, case):
        self.case = case
        d = case.model.dimension
        coordinate = int(np.argmin(case.contract.weights))
        self.directions = {"parallel": np.ones(d), f"coordinate_{coordinate+1}": np.eye(d)[coordinate]}

    def curvature_references(self, options: dict) -> dict:
        spots = np.asarray(self.case.initial_spots)
        geometric = self.case.contract.basket == "geometric"
        oracle = GeometricBasketReference(self.case.model, self.case.contract) if geometric else None
        pricer = BasketReferencePricer(self.case.model, self.case.contract)
        result = {}
        for name, direction in self.directions.items():
            rows = []
            for bump in options["bumps"]:
                if not np.isfinite(bump) or bump <= 0 or np.any(spots-bump*np.abs(direction) <= 0):
                    raise ValueError("Curvature bumps must be positive and preserve positive spots.")
                if geometric:
                    plus = oracle.call_delta_array(spots+bump*direction)
                    minus = oracle.call_delta_array(spots-bump*direction)
                    statistic = {"estimate": float((plus-minus)@direction/(2*bump)),
                                 "standard_error": 0.0, "confidence_low": None, "confidence_high": None}
                    replicates = None
                else:
                    paired = []
                    for sign in (1, -1):
                        estimate = pricer.sobol(spots+sign*bump*direction, power=options["power"],
                            scrambles=options["scrambles"], pilot_pairs=options["pilot_pairs"],
                            seed=options["seed"], chunk_size=options["chunk_size"])
                        paired.append(np.asarray(estimate.replicate_controlled)[:, 1:]@direction)
                    replicates = (paired[0]-paired[1])/(2*bump)
                    statistic = scramble_statistics(replicates)
                rows.append({"bump": bump, "statistics": statistic,
                             "scramble_estimates": None if replicates is None else replicates.tolist()})
            result[name] = {"direction": direction.tolist(), "bumps": rows,
                            "exact_gamma": float(direction@oracle.call_hessian(spots)@direction) if geometric else None}
        return result

    def model_curvature(self, solver: FBSNNSolver, reference: dict) -> dict:
        point = torch.tensor([[0, *self.case.initial_spots]], dtype=torch.float64)
        hessian = value_derivatives(solver.network, point).hessian[0, 1:, 1:].numpy()
        result = {}
        for name, direction in self.directions.items():
            gamma = float(direction@hessian@direction)
            rows = []
            for row in reference[name]["bumps"]:
                bump = row["bump"]
                states = np.asarray(self.case.initial_spots)[None, :]+np.array([-bump, bump])[:, None]*direction
                inputs = torch.tensor(np.column_stack((np.zeros(2), states)), dtype=torch.float64)
                deltas = value_derivatives(solver.network, inputs, compute_hessian=False).gradient[:, 1:].numpy()
                network_secant = float((deltas[1]-deltas[0])@direction/(2*bump))
                stat = row["statistics"]
                error = gamma-stat["estimate"]
                interval = None if stat["confidence_low"] is None else [gamma-stat["confidence_high"], gamma-stat["confidence_low"]]
                rows.append({"bump": bump, "network_delta_secant": network_secant,
                             "secant_minus_network_hessian": network_secant-gamma,
                             "gamma_minus_reference_secant": error, "signed_error_interval": interval})
            exact = reference[name]["exact_gamma"]
            result[name] = {"gamma": gamma, "exact_gamma_error": None if exact is None else gamma-exact,
                            "bump_checks": rows}
        return result

    def run(self, directory: Path, options: dict) -> dict:
        started = time.perf_counter()
        reference = self.curvature_references(options)
        numerical = (json.loads((directory/"reference.json").read_text())
                     if self.case.contract.basket == "arithmetic" else None)
        rows = []
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for run in options["runs"]:
                paths = {kind: directory/run[kind] for kind in ("value_gradient", "terminal_loss")}
                row = {"seed": run["seed"], "status": "models_unavailable"}
                rows.append(row)
                if not all((p/"model.pt").is_file() for p in paths.values()):
                    continue
                value = FBSNNSolver.load(paths["value_gradient"]/"model.pt")
                terminal = TerminalBSDESolver.load(paths["terminal_loss"]/"model.pt")
                if value.case != self.case or terminal.case != self.case:
                    raise ValueError("Greek diagnostics and saved models require identical cases.")
                if value.config.seed != run["seed"] or terminal.config.seed != run["seed"]:
                    raise ValueError("Saved model seeds disagree with the diagnostic row.")
                comparison = SolverRiskComparison(value, terminal)
                row.update(status="completed", models={}, source_model_sha256={
                    k: hashlib.sha256((p/"model.pt").read_bytes()).hexdigest() for k, p in paths.items()})
                for kind in paths:
                    initial = comparison.initial_risk(kind)
                    directions = {}
                    for name, direction in self.directions.items():
                        if numerical is None:
                            exact = GeometricBasketReference(self.case.model, self.case.contract).evaluate(self.case.initial_spots)
                            stat = {"estimate": float(np.asarray(exact.call_deltas)@direction),
                                    "confidence_low": None, "confidence_high": None, "standard_error": 0.0}
                        else:
                            replicates = np.asarray(numerical["sobol_levels"][-1]["replicate_controlled"])[:, 1:]
                            stat = scramble_statistics(replicates@direction)
                        estimate = float(np.asarray(initial["delta"])@direction)
                        target = options["scaled_delta_target"]*float(np.asarray(self.case.contract.weights)@np.abs(direction))
                        check = _error_assessment(estimate, stat, target)
                        if numerical is None:
                            check["classification"] = "within_target" if abs(check["signed_error"]) <= target else "outside_target"
                        elif not options["reference_usable"]:
                            check["classification"] = "reference_inconclusive"
                        directions[name] = {"reference": stat, **check}
                    row["models"][kind] = {"initial": initial, "directional_deltas": directions}
                    if numerical is None:
                        exact = GeometricBasketReference(self.case.model, self.case.contract).evaluate(self.case.initial_spots)
                        checks = []
                        for delta, target_delta, weight in zip(initial["delta"], exact.call_deltas, self.case.contract.weights):
                            error = (delta-target_delta)/weight
                            checks.append({"signed_error": error, "scale": weight,
                                           "absolute_error_target": options["scaled_delta_target"],
                                           "classification": "within_target" if abs(error) <= options["scaled_delta_target"] else "outside_target"})
                        row["models"][kind].update(scaled_delta_checks=checks,
                                                   price_error=initial["price"]-exact.call_price)
                row["value_gradient_curvature"] = self.model_curvature(value, reference)
                row["terminal_curvature"] = None
                row["terminal_curvature_reason"] = "fixed-start initial control; no global price Hessian"
                if numerical is None:
                    risk = comparison.run(n_paths=options["path_evaluation_paths"], seed=2031)
                    row["analytical_path_risk"] = risk.report
                    risk.save(directory/"asymmetric_paths")
        finally:
            torch.set_num_threads(previous_threads)
        return {"status": "completed" if all(r["status"] == "completed" for r in rows) else "incomplete",
                "problem": json.loads(self.case.to_json()), "options": options,
                "curvature_reference": reference, "runs": rows, "seconds": time.perf_counter()-started,
                "scope": "initial directions only; numerical curvature intervals exclude finite-bump bias; no curvature accuracy target",
                "sampling": "paired plus/minus scramble means with independent geometric-control pilots; seed shared across bumps"}


class GreekStressStudy:
    """Four declared arithmetic stresses and existing asymmetric geometric fits."""

    def __init__(self, design: ExperimentDesign | None = None):
        self.design = ExperimentDesign.load() if design is None else design
        self.spec = self.design.data
        self.cases = [c["name"] for c in self.spec["cases"]
                      if c["family"] == "arithmetic" and set(c)-{"name", "family", "dimension"}]

    def _diagnostics(self, directory, study, runs, reference, usable, *, case=None):
        directory.mkdir()
        case = study.case if case is None else case
        options = {"bumps": [1.0, 0.5, 0.1], "power": self.spec["reference"]["sobol_powers"][0],
                   "scrambles": self.spec["reference"]["scrambles"],
                   "pilot_pairs": self.spec["reference"]["pilot_pairs"], "seed": 2035,
                   "chunk_size": self.spec["reference"]["chunk_terminal_states"],
                   "scaled_delta_target": self.spec["targets"]["weight_scaled_delta_absolute"],
                   "path_evaluation_paths": self.spec["training"]["evaluation_paths"],
                   "reference_usable": usable, "runs": runs}
        config = study._config(self.spec["development_seeds"][0], self.spec["reference"]["sobol_powers"][0],
                               self.spec["reference"]["check_pairs"][0])
        _write_json(directory/"design.json", {"problem": json.loads(case.to_json()),
                    "config": asdict(config), "greek_options": options})
        if reference is not None:
            _write_json(directory/"reference.json", reference)
        outcome = study._task(ArithmeticPilot(config), "greeks", directory, self.spec["resources"])
        if outcome["status"] == "completed":
            outcome["result"] = json.loads((directory/"greeks.json").read_text())
            outcome["status"] = outcome["result"]["status"]
        return outcome

    def run(self, directory: Path, *, geometric_value: Path = Path("scratch/runs/geometric_five_fbsnn"),
            geometric_terminal: Path = Path("scratch/runs/geometric_five_terminal_bsde")) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        report = {"method": "greek_stress_development_study", "specification_sha256": self.design.sha256,
                  "specification": self.spec, "cases": [], "scheduled_fits": len(self.cases)*len(self.spec["development_seeds"])*2,
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "scope": "development seeds only; four single-factor stresses; existing asymmetric analytical fits; no tuning"}
        _write_json(directory/"design.json", report)
        for name in self.cases:
            study = ArithmeticDevelopmentStudy(name, self.design)
            case_dir = directory/name.replace("-", "_")
            result = study.run(case_dir)
            runs = [{"seed": r["seed"], **{k: f"../{r['directory']}/{k}" for k in self.spec["solvers"]}}
                    for r in result["runs"]]
            diagnostic = self._diagnostics(case_dir/"greeks", study, runs, result["reference"], result["reference_usable"])
            report["cases"].append({"case": name, "directory": case_dir.name, "study": result, "greeks": diagnostic})
            report.update(status="incomplete", wall_seconds=time.perf_counter()-started)
            _write_json(directory/"report.json", report)
        study = ArithmeticDevelopmentStudy("arithmetic-10", self.design)
        asym_dir = directory/"asymmetric"
        runs = [{"seed": 2026, "value_gradient": str(Path(geometric_value).resolve()),
                 "terminal_loss": str(Path(geometric_terminal).resolve())}]
        report["asymmetric"] = self._diagnostics(asym_dir, study, runs, None, True,
                                                  case=self.design.case("geometric-five"))
        report.update(status="completed" if all(c["study"]["status"] == c["greeks"]["status"] == "completed"
                      for c in report["cases"]) and report["asymmetric"]["status"] == "completed" else "incomplete",
                      wall_seconds=time.perf_counter()-started, controller_peak_resident_mib=_peak_rss_mib())
        _write_json(directory/"report.json", report)
        return report
