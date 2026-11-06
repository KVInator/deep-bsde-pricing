"""Saved-model price and risk comparison on common GBM states."""

import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .bsde import ControlRecovery, PricingBSDE
from .geometric_basket import reference_deltas_on_grid
from .neural import value_derivatives
from .terminal_bsde import TerminalBSDESolver
from .training import FBSNNSolver, sample_training_paths


def _recovery_summary(recovery: ControlRecovery) -> dict[str, float]:
    return {"max_absolute_reconstruction_residual": float(np.max(recovery.absolute_residual)),
            "max_relative_reconstruction_residual": float(np.max(recovery.relative_residual)),
            "condition_number_min": float(np.min(recovery.condition_number)),
            "condition_number_max": float(np.max(recovery.condition_number)),
            "smallest_diffusion_singular_value": float(np.min(recovery.smallest_singular_value)),
            "largest_inverse_diffusion_norm": float(np.max(recovery.inverse_norm))}


@dataclass(frozen=True)
class RiskComparisonResult:
    times: np.ndarray
    spots: np.ndarray
    reference_deltas: np.ndarray
    value_deltas: np.ndarray
    terminal_deltas: np.ndarray
    reference_controls: np.ndarray
    value_controls: np.ndarray
    terminal_controls: np.ndarray
    condition_numbers: np.ndarray
    inverse_norms: np.ndarray
    report: dict[str, object]

    def __post_init__(self):
        dimension = len(self.report["problem"]["initial_spots"])
        asset_shape = () if dimension == 1 else (dimension,)
        if (self.times.ndim != 1 or len(self.times) < 2 or self.times[0] != 0
                or np.any(np.diff(self.times) <= 0) or self.spots.ndim != 2+len(asset_shape)
                or self.spots.shape[0] < 2 or self.spots.shape[1:] != (len(self.times), *asset_shape)
                or np.any(self.spots <= 0)):
            raise ValueError("Inconsistent comparison times or states.")
        if self.report["n_paths"] != len(self.spots) or self.report["n_steps"] != len(self.times)-1:
            raise ValueError("Comparison arrays disagree with report dimensions.")
        for field in fields(self):
            if field.name == "report": continue
            array = getattr(self, field.name)
            if field.name not in ("times", "spots"):
                shape = (len(self.spots), len(self.times)-1)
                if field.name not in ("condition_numbers", "inverse_norms"):
                    shape += asset_shape
                if array.shape != shape:
                    raise ValueError("Comparison risk arrays disagree with the path grid or dimension.")
            if array.dtype != np.float64 or not np.isfinite(array).all():
                raise ValueError("Comparison arrays must be finite float64 data.")
            array.setflags(write=False)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/"paths.npz", **{
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "report"})
        (directory/"report.json").write_text(json.dumps(self.report, indent=2, sort_keys=True, allow_nan=False)+"\n")

    @classmethod
    def load(cls, directory: Path) -> "RiskComparisonResult":
        directory = Path(directory)
        report = json.loads((directory/"report.json").read_text())
        if report.get("method") not in (
            "saved_scalar_solver_risk_comparison", "saved_geometric_solver_risk_comparison",
        ):
            raise ValueError("Unsupported risk-comparison report.")
        with np.load(directory/"paths.npz", allow_pickle=False) as data:
            arrays = {f.name: data[f.name].copy() for f in fields(cls) if f.name != "report"}
        return cls(**arrays, report=report)


class SolverRiskComparison:
    """Evaluate selected vanilla or geometric models on a common grid.

    The terminal controls are undiscounted before diffusion inversion. The
    value-gradient delta is differentiated from its value map. A separate
    initial gamma diagnostic is available for the scalar value-gradient model.
    """

    def __init__(self, value_solver: FBSNNSolver, terminal_solver: TerminalBSDESolver):
        a, b = value_solver.case, terminal_solver.case
        if a.model != b.model or a.contract != b.contract or a.initial_spots != b.initial_spots:
            raise ValueError("The comparison requires identical financial inputs and starting states.")
        if value_solver.config.n_steps != terminal_solver.config.n_steps:
            raise ValueError("The comparison requires identical time grids.")
        self.value, self.terminal = value_solver, terminal_solver
        self.case = a
        self.problem = PricingBSDE(a.model, a.contract)
        self.reference = value_solver.reference
        self.n_steps = value_solver.config.n_steps
        self.dimension = a.model.dimension

    def _coordinate_shape(self, array: np.ndarray) -> np.ndarray:
        """Keep the published scalar report and archive shapes."""
        return array[..., 0] if self.dimension == 1 else array

    def initial_risk(self, kind: str) -> dict[str, object]:
        """Physical initial price, delta and control without a reference call."""
        if kind not in ("value_gradient", "terminal_loss"):
            raise ValueError("Unknown solver kind.")
        spots = np.array(self.case.initial_spots, dtype=np.float64)
        if kind == "value_gradient":
            point = torch.tensor([[0, *self.case.initial_spots]], dtype=torch.float64, device="cpu")
            prediction = value_derivatives(self.value.network, point, compute_hessian=False)
            price = float(prediction.values[0, 0])
            delta = prediction.gradient[0, 1:].numpy()
            control = self.problem.control(spots, delta)
        else:
            network = self.terminal.network
            price = float((network.value_scale*network.initial_value).detach())
            control = (network.control_scale*network.initial_control).detach().numpy().reshape(self.dimension)
            delta = None
        recovery = self.problem.recover_deltas(spots, control)
        if delta is None:
            delta = recovery.deltas
        return {"price": price, "delta": self._coordinate_shape(delta).tolist(),
                "brownian_control": self._coordinate_shape(control).tolist(),
                "recovered_delta": self._coordinate_shape(recovery.deltas).tolist(),
                "reconstruction": _recovery_summary(recovery)}

    def _path_risk(self, kind: str, times: np.ndarray, states: torch.Tensor):
        if kind == "value_gradient":
            n_paths = len(states)
            inputs = torch.cat((torch.from_numpy(times[:-1]).expand(n_paths, -1)[..., None], states[:, :-1]), dim=2)
            prediction = value_derivatives(self.value.network, inputs.reshape(-1, self.dimension+1), compute_hessian=False)
            deltas = prediction.gradient[:, 1:].numpy().reshape(n_paths, self.n_steps, self.dimension)
            controls = self.problem.control(states[:, :-1].numpy(), deltas)
        else:
            with torch.no_grad():
                discounted = self.terminal.network.discounted_controls(states).numpy()
            controls = discounted*np.exp(self.case.model.rate*times[None, :-1, None])
            deltas = None
        recovery = self.problem.recover_deltas(states[:, :-1].numpy(), controls)
        if deltas is None:
            deltas = recovery.deltas
        return deltas, controls, recovery

    def _gamma(self) -> dict[str, object]:
        spot = self.case.initial_spots[0]
        point = torch.tensor([[0, spot]], dtype=torch.float64, device="cpu")
        started = time.perf_counter()
        gamma = float(value_derivatives(self.value.network, point).hessian[0, 1, 1])
        gamma_seconds = time.perf_counter()-started
        target = self.reference.evaluate(spot).gamma
        bumps = []
        for fraction in (0.01, 0.005, 0.001):
            bump = spot*fraction
            inputs = torch.tensor([[0, spot-bump], [0, spot+bump]], dtype=torch.float64, device="cpu")
            deltas = value_derivatives(self.value.network, inputs, compute_hessian=False).gradient[:, 1]
            estimate = float((deltas[1]-deltas[0])/(2*bump))
            bumps.append({"fraction": fraction, "size": bump, "central_delta_bump_gamma": estimate,
                          "error_against_autograd": estimate-gamma})
        return {"gamma": gamma, "reference_gamma": target, "signed_error": gamma-target,
                "hessian_seconds": gamma_seconds, "bump_checks": bumps,
                "scope": "initial physical spot; bump checks verify the network derivative only"}

    def run(self, *, n_paths: int = 2048, seed: int = 2031) -> RiskComparisonResult:
        if self.reference is None:
            raise ValueError("Path-risk errors require an analytical reference; use arithmetic-pilot for initial numerical checks.")
        if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 2:
            raise ValueError("n_paths must be an integer of at least two.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        times = np.linspace(0, self.case.contract.maturity, self.n_steps+1)
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4,)))
        states, _ = sample_training_paths(self.case, self.n_steps, rng, n_paths)
        spots = states.numpy()
        left_spots = spots[:, :-1]
        reference_deltas = reference_deltas_on_grid(self.reference, spots, times)
        reference_controls = self.problem.control(left_spots, reference_deltas)
        tau = self.case.contract.maturity-times[:-1]
        if self.dimension == 1:
            target = self.reference.evaluate(self.case.initial_spots[0])
            initial_reference_delta = target.call_delta
            upper_delta = np.exp(-self.case.model.dividend_yields[0]*tau)[None, :, None]
        else:
            target = self.reference.evaluate(self.case.initial_spots)
            initial_reference_delta = list(target.call_deltas)
            basket = self.case.contract.basket_value(left_spots)
            upper_delta = ((np.exp(-target.effective_dividend_yield*tau)*basket)[..., None]
                           * np.asarray(self.case.contract.weights)/left_spots)
        models, arrays = {}, {}
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for kind, solver in (("value_gradient", self.value), ("terminal_loss", self.terminal)):
                for _ in range(2): self.initial_risk(kind)
                durations = []
                for _ in range(5):
                    started = time.perf_counter()
                    initial = self.initial_risk(kind)
                    durations.append(time.perf_counter()-started)
                initial.update(reference_price=target.call_price, reference_delta=initial_reference_delta,
                               price_error=initial["price"]-target.call_price,
                               delta_error=(np.asarray(initial["delta"])-initial_reference_delta).tolist())
                started = time.perf_counter()
                deltas, controls, recovery = self._path_risk(kind, times, states)
                path_seconds = time.perf_counter()-started
                errors = deltas-reference_deltas
                means = errors.reshape(n_paths, -1).mean(axis=1)
                violations = (deltas < -1e-10) | (deltas > upper_delta+1e-10)
                arrays[kind] = tuple(self._coordinate_shape(a).copy() for a in (deltas, controls))
                models[kind] = {
                    "initial": initial, "parameter_count": sum(p.numel() for p in solver.network.parameters()),
                    "path_delta_rmse": float(np.sqrt(np.mean(errors**2))),
                    "path_delta_mean_error": float(means.mean()),
                    "path_delta_mean_error_se": float(means.std(ddof=1)/math.sqrt(n_paths)),
                    "path_delta_min": float(np.min(deltas)), "path_delta_max": float(np.max(deltas)),
                    "max_absolute_path_delta_error": float(np.max(np.abs(errors))),
                    "delta_bound_violation_count": int(np.count_nonzero(violations)),
                    "path_risk_evaluation_count": int(deltas.size), "delta_bound_tolerance": 1e-10,
                    "path_reconstruction": _recovery_summary(recovery),
                    "timing": {"initial_seconds": durations, "initial_median_seconds": float(np.median(durations)),
                               "warmups": 2, "repeats": 5, "path_risk_seconds": path_seconds,
                               "initial_workload": "price_delta_control_and_diffusion_diagnostics_no_reference"}}
                if self.dimension > 1:
                    models[kind].update(
                        path_delta_rmse_by_coordinate=np.sqrt(np.mean(errors**2, axis=(0, 1))).tolist(),
                        max_absolute_path_delta_error_by_coordinate=np.max(np.abs(errors), axis=(0, 1)).tolist(),
                        delta_bound_violations_by_coordinate=np.count_nonzero(violations, axis=(0, 1)).tolist(),
                    )
            gamma = self._gamma() if self.dimension == 1 else None
        finally:
            torch.set_num_threads(previous_threads)
        value, terminal = arrays["value_gradient"], arrays["terminal_loss"]
        diffusion = self.problem.recover_deltas(left_spots, reference_controls)
        kind = "scalar" if self.dimension == 1 else "geometric"
        report = {"method": f"saved_{kind}_solver_risk_comparison", "problem": json.loads(self.case.to_json()),
                  "n_paths": n_paths, "n_steps": self.n_steps, "seed": seed, "seed_sequence_branch": 4,
                  "bit_generator": "PCG64", "trained_during_comparison": False,
                  "device": "cpu", "dtype": "torch.float64", "cpu_threads": 1,
                  "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                  "system": platform.platform(), "models": models, "value_gradient_gamma": gamma,
                  "terminal_gamma": None, "terminal_gamma_reason": "fixed-start control representation; no global price Hessian",
                  "inspection_targets": {"price_absolute_error": 0.25, "delta_absolute_error": 0.02},
                  "targets_met": {k: {"price": abs(v["initial"]["price_error"]) <= 0.25,
                                      "delta": bool(np.max(np.abs(v["initial"]["delta_error"])) <= 0.02)}
                                  for k, v in models.items()}}
        return RiskComparisonResult(times, self._coordinate_shape(spots).copy(), self._coordinate_shape(reference_deltas),
                                    value[0], terminal[0], self._coordinate_shape(reference_controls),
                                    value[1], terminal[1], diffusion.condition_number.copy(),
                                    diffusion.inverse_norm.copy(), report)


def compare_saved_runs(value_directory: Path, terminal_directory: Path, output: Path,
                       *, n_paths: int = 2048, seed: int = 2031) -> RiskComparisonResult:
    """Load declared selected checkpoints and retain their configuration/cost provenance."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Comparison directory already exists: {output}")
    directories = {"value_gradient": Path(value_directory), "terminal_loss": Path(terminal_directory)}
    value = FBSNNSolver.load(directories["value_gradient"]/"model.pt")
    terminal = TerminalBSDESolver.load(directories["terminal_loss"]/"model.pt")
    comparison = SolverRiskComparison(value, terminal)
    provenance = {}
    for kind, directory in directories.items():
        source = json.loads((directory/"report.json").read_text())
        initial = comparison.initial_risk(kind)
        keys = (("price", "price"), ("delta", "delta")) if kind == "value_gradient" else (
            ("price", "price"), ("brownian_control", "discounted_initial_control"))
        for result_key, source_key in keys:
            actual = np.asarray(initial[result_key])
            recorded = np.asarray(source["evaluation"]["initial"][source_key])
            if actual.shape != recorded.shape or not np.allclose(actual, recorded, rtol=0, atol=1e-10):
                raise ValueError("Saved model and source evaluation disagree.")
        solver = value if kind == "value_gradient" else terminal
        if source["config"] != solver.config.__dict__ or source["problem"] != json.loads(solver.case.to_json()):
            raise ValueError("Saved model and source metadata disagree.")
        provenance[kind] = {"directory": str(directory), "model_sha256": hashlib.sha256((directory/"model.pt").read_bytes()).hexdigest(),
                            "report_sha256": hashlib.sha256((directory/"report.json").read_bytes()).hexdigest(),
                            "config": source["config"], "completed_updates": source["completed_updates"],
                            "status": source["status"], "checkpoint_rule": source["checkpoint_rule"],
                            "selected_step": source.get("selected_step", source["completed_updates"]),
                            "training_seconds": source["training_seconds"], "evaluation_seconds": source["evaluation_seconds"],
                            "training_runtime": {k: source[k] for k in ("python", "torch", "numpy", "system", "device", "dtype")}}
    result = comparison.run(n_paths=n_paths, seed=seed)
    result.report["source_runs"] = provenance
    result.save(output)
    return result
