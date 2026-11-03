"""Bounded one-asset value-gradient training and checkpoint evaluation."""

import json
import math
import platform
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .black_scholes import BlackScholesReference
from .analytical_torch import AnalyticalCall
from .bsde import PricingBSDE
from .bsde_loss import LocalBSDELoss
from .neural import ValueNetwork, value_derivatives
from .problems import EuropeanCall, GBMModel, PricingCase


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 3000
    n_steps: int = 16
    batch_size: int = 128
    width: int = 16
    learning_rate: float = 0.001
    seed: int = 2026
    log_every: int = 100
    validation_paths: int = 256
    evaluation_paths: int = 2048
    max_seconds: float = 180
    terminal_weight: float = 1
    patience: int | None = None

    def __post_init__(self):
        for name in ("iterations", "n_steps", "batch_size", "width", "log_every",
                     "validation_paths", "evaluation_paths"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.evaluation_paths < 2:
            raise ValueError("evaluation_paths must be at least two.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("learning_rate and max_seconds must be positive and finite.")
        if not math.isfinite(self.terminal_weight) or self.terminal_weight <= 0:
            raise ValueError("terminal_weight must be positive and finite.")
        if self.patience is not None and (isinstance(self.patience, bool)
                or not isinstance(self.patience, int) or self.patience < 1):
            raise ValueError("patience must be a positive number of validation checks or None.")


@dataclass(frozen=True)
class PathEvaluation:
    times: np.ndarray
    spots: np.ndarray
    values: np.ndarray
    deltas: np.ndarray
    reference_values: np.ndarray
    reference_deltas: np.ndarray
    summary: dict[str, object]


class FBSNNSolver:
    """Fixed-start scalar call experiment, using CPU float64 throughout.

    Training, checkpoint selection and evaluation use disjoint random streams.
    Selection minimises the fixed validation objective at scheduled checks.
    An inference checkpoint and a separate continuation checkpoint are saved.
    """

    def __init__(self, case: PricingCase, config: TrainingConfig = TrainingConfig()):
        self.reference = BlackScholesReference(case.model, case.contract)
        if case.model.volatilities[0] <= 0 or case.contract.maturity <= 0:
            raise ValueError("The training pilot requires positive maturity and volatility.")
        self.case, self.config = case, config
        self.network = ValueNetwork(case.initial_spots, case.contract.maturity,
                                    case.contract.strike, width=config.width, seed=config.seed)
        self.loss = LocalBSDELoss(PricingBSDE(case.model, case.contract),
                                  value_scale=case.contract.strike, terminal_weight=config.terminal_weight)
        self.times = torch.from_numpy(np.linspace(0, case.contract.maturity, config.n_steps+1))
        self._has_fit = False
        self._training = None
        self._continuation_model = None

    def _batch(self, rng: np.random.Generator, n_paths: int):
        increments = rng.standard_normal((n_paths, self.config.n_steps, 1))*math.sqrt(
            self.case.contract.maturity/self.config.n_steps,
        )
        states = self.case.model.path_from_increments(
            self.case.initial_spots, self.case.contract.maturity, increments,
        )
        return torch.from_numpy(states), torch.from_numpy(increments)

    def initial_value(self) -> dict[str, float]:
        point = torch.tensor([[0, *self.case.initial_spots]], dtype=torch.float64, device="cpu")
        result = value_derivatives(self.network, point, compute_hessian=False)
        target = self.reference.evaluate(self.case.initial_spots[0])
        price, delta = float(result.values[0, 0]), float(result.gradient[0, 1])
        return {"price": price, "delta": delta, "reference_price": target.call_price,
                "reference_delta": target.call_delta, "price_error": price-target.call_price,
                "delta_error": delta-target.call_delta}

    def _evaluate(self, rng: np.random.Generator, n_paths: int) -> PathEvaluation:
        states, increments = self._batch(rng, n_paths)
        n_steps = self.config.n_steps
        inputs = torch.cat((self.times[:-1].expand(n_paths, -1)[..., None], states[:, :-1]), dim=2)
        prediction = value_derivatives(self.network, inputs.reshape(-1, 2), compute_hessian=False)
        with torch.no_grad():
            terminal = self.network(torch.cat((self.times[-1].expand(n_paths, 1), states[:, -1]), dim=1))
        values = np.column_stack((prediction.values.numpy().reshape(n_paths, n_steps), terminal.numpy()))
        deltas = prediction.gradient[:, 1].numpy().reshape(n_paths, n_steps)
        spots = states[:, :, 0].numpy()
        reference_values = np.empty_like(values)
        reference_deltas = np.empty_like(deltas)
        for index, time_point in enumerate(self.times.numpy()):
            results = [self.reference.evaluate(float(s), float(time_point)) for s in spots[:, index]]
            reference_values[:, index] = [r.call_price for r in results]
            if index < n_steps:
                reference_deltas[:, index] = [r.call_delta for r in results]
        loss = self.loss(self.network, self.times, states, increments, parameter_graph=False)
        analytical_loss = self.loss(AnalyticalCall(self.case.model, self.case.contract),
                                    self.times, states, increments, parameter_graph=False)
        value_error = values[:, :-1]-reference_values[:, :-1]
        delta_error = deltas-reference_deltas
        terminal_error = values[:, -1]-reference_values[:, -1]
        path_value_means = value_error.mean(axis=1)
        path_delta_means = delta_error.mean(axis=1)
        summary = {
            "n_paths": n_paths, "initial": self.initial_value(),
            "price_rmse_before_expiry": float(np.sqrt(np.mean(value_error**2))),
            "delta_rmse_before_expiry": float(np.sqrt(np.mean(delta_error**2))),
            "terminal_payoff_rmse": float(np.sqrt(np.mean(terminal_error**2))),
            "mean_price_error": float(path_value_means.mean()),
            "mean_price_error_se": float(path_value_means.std(ddof=1)/math.sqrt(n_paths)),
            "mean_delta_error": float(path_delta_means.mean()),
            "mean_delta_error_se": float(path_delta_means.std(ddof=1)/math.sqrt(n_paths)),
            "local_loss": float(loss.local), "terminal_loss": float(loss.terminal),
            "total_loss": float(loss.total),
            "unweighted_total_loss": float(loss.local+loss.terminal),
            "analytical_local_loss": float(analytical_loss.local),
        }
        arrays = [self.times.numpy().copy(), spots.copy(), values, deltas, reference_values, reference_deltas]
        for array in arrays: array.setflags(write=False)
        return PathEvaluation(*arrays, summary)

    def evaluate(self, *, n_paths: int = 512, seed: int = 2027) -> PathEvaluation:
        """Fresh descriptive paths for a loaded model; no checkpoint selection."""
        if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 2:
            raise ValueError("n_paths must be an integer of at least two.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        return self._evaluate(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,))), n_paths)

    def _start_training(self) -> None:
        children = np.random.SeedSequence(self.config.seed).spawn(3)
        train_rng, validation_rng, evaluation_rng = (np.random.default_rng(s) for s in children)
        self._training = {
            "completed": 0, "status": "paused", "history": [], "window": [],
            "training_seconds": 0.0, "best_loss": math.inf, "best_step": 0,
            "best_state": None, "checks_without_improvement": 0,
            "train_rng_state": train_rng.bit_generator.state,
            "evaluation_rng_state": evaluation_rng.bit_generator.state,
            "validation": self._batch(validation_rng, self.config.validation_paths),
            "torch_rng_state": torch.get_rng_state().clone(),
            "optimizer_state": None,
        }

    def _record(self, elapsed_seconds: float) -> None:
        state = self._training
        result = self.loss(self.network, self.times, *state["validation"], parameter_graph=False)
        total = float(result.total)
        if not math.isfinite(total):
            raise ArithmeticError("Nonfinite validation objective.")
        if total < state["best_loss"]:
            state.update(best_loss=total, best_step=state["completed"],
                         best_state=deepcopy(self.network.state_dict()), checks_without_improvement=0)
        else:
            state["checks_without_improvement"] += 1
        state["history"].append({
            "step": state["completed"], "elapsed_seconds": elapsed_seconds,
            "training_loss_window_mean": float(np.mean(state["window"])) if state["window"] else None,
            "validation_local": float(result.local), "validation_terminal": float(result.terminal),
            "validation_total": total, **self.initial_value(),
        })
        state["window"].clear()

    def fit(self, *, stop_after: int | None = None) -> dict[str, object]:
        """Fit to an absolute update count; a pause does not inspect evaluation paths.

        Validation is checked at update zero, log_every updates and the full
        configured budget. Pausing between checks preserves the loss window.
        max_seconds limits this invocation; training_seconds accumulates across
        invocations. The learning-rate schedule always uses config.iterations.
        """
        if self._has_fit:
            raise RuntimeError("Load resume.pt to continue a paused or time-limited run.")
        if stop_after is not None and (isinstance(stop_after, bool)
                or not isinstance(stop_after, int) or not 1 <= stop_after <= self.config.iterations):
            raise ValueError("stop_after must be an absolute update count within the configured budget.")
        target = self.config.iterations if stop_after is None else stop_after
        if self._training is not None and target <= self._training["completed"]:
            raise ValueError("stop_after must exceed the completed update count.")
        self._has_fit = True
        if self._training is None:
            self._start_training()
        state = self._training
        rng = np.random.default_rng()
        rng.bit_generator.state = state["train_rng_state"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        if state["optimizer_state"] is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        old_threads = torch.get_num_threads()
        started = time.perf_counter()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(state["torch_rng_state"])
                if not state["history"]:
                    analytical = self.loss(AnalyticalCall(self.case.model, self.case.contract),
                                           self.times, *state["validation"], parameter_graph=False)
                    state["analytical_validation_local"] = float(analytical.local)
                    self._record(state["training_seconds"]+time.perf_counter()-started)
                status = "paused"
                for step in range(state["completed"]+1, target+1):
                    if time.perf_counter()-started >= self.config.max_seconds:
                        status = "time_budget_reached"
                        break
                    learning_rate = self.config.learning_rate*(0.3 if step > 0.7*self.config.iterations else 1)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    states, increments = self._batch(rng, self.config.batch_size)
                    optimizer.zero_grad(set_to_none=True)
                    result = self.loss(self.network, self.times, states, increments)
                    result.total.backward()
                    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                               for p in self.network.parameters()):
                        raise ArithmeticError("Nonfinite or missing training gradient.")
                    optimizer.step()
                    if not all(bool(torch.isfinite(p).all()) for p in self.network.parameters()):
                        raise ArithmeticError("Nonfinite updated parameter.")
                    state["completed"] = step
                    state["window"].append(float(result.total.detach()))
                    if step % self.config.log_every == 0 or step == self.config.iterations:
                        self._record(state["training_seconds"]+time.perf_counter()-started)
                        if (self.config.patience is not None
                                and state["checks_without_improvement"] >= self.config.patience):
                            status = "early_stopped"
                            break
                if state["completed"] == self.config.iterations:
                    status = "completed"
                state["torch_rng_state"] = torch.get_rng_state().clone()
            state["training_seconds"] += time.perf_counter()-started
            state["status"] = status
            state["train_rng_state"] = rng.bit_generator.state
            state["optimizer_state"] = deepcopy(optimizer.state_dict())
            self._continuation_model = deepcopy(self.network.state_dict())
            evaluation, last_evaluation = None, None
            evaluation_seconds = 0.0
            if status != "paused":
                evaluation_started = time.perf_counter()
                evaluation_rng = np.random.default_rng()
                evaluation_rng.bit_generator.state = state["evaluation_rng_state"]
                last_evaluation = self._evaluate(evaluation_rng, self.config.evaluation_paths).summary
                # Rewind only the independent evaluation stream for a paired comparison.
                evaluation_rng.bit_generator.state = state["evaluation_rng_state"]
                self.network.load_state_dict(state["best_state"])
                evaluation = self._evaluate(evaluation_rng, self.config.evaluation_paths).summary
                evaluation_seconds = time.perf_counter()-evaluation_started
        finally:
            torch.set_num_threads(old_threads)
        initial = evaluation["initial"] if evaluation is not None else None
        return {
            "method": "value_gradient_fbsnn", "status": status,
            "problem": json.loads(self.case.to_json()), "config": asdict(self.config),
            "device": "cpu", "dtype": "torch.float64", "cpu_threads": 1,
            "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "system": platform.platform(), "completed_updates": state["completed"],
            "training_paths_drawn": state["completed"]*self.config.batch_size,
            "stream_children": {"training": 0, "validation": 1, "evaluation": 2},
            "checkpoint_rule": "minimum_validation_total_at_scheduled_checks",
            "selected_step": state["best_step"], "selected_validation_total": state["best_loss"],
            "history": deepcopy(state["history"]),
            "validation_analytical_local_loss": state["analytical_validation_local"],
            "training_seconds": state["training_seconds"], "evaluation_seconds": evaluation_seconds,
            "evaluation": evaluation, "last_evaluation": last_evaluation,
            "inspection_targets": {"price_absolute_error": 0.25, "delta_absolute_error": 0.02},
            "targets_met": None if initial is None else {
                "price": abs(initial["price_error"]) <= 0.25, "delta": abs(initial["delta_error"]) <= 0.02},
        }

    def save(self, directory: Path, report: dict[str, object]) -> None:
        """Save selected inference weights and current optimiser/RNG continuation."""
        if self._training is None or self._continuation_model is None:
            raise RuntimeError("Fit a run before saving it.")
        if (report.get("config") != asdict(self.config)
                or report.get("completed_updates") != self._training["completed"]
                or report.get("status") != self._training["status"]):
            raise ValueError("Report does not describe the current training state.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        common = {"format_version": 2, "problem": self.case.to_json(), "config": asdict(self.config)}
        torch.save({**common, "model_state": self._training["best_state"]}, directory/"model.pt")
        torch.save({**common, "model_state": self._continuation_model,
                    "training_state": self._training}, directory/"resume.pt")
        (directory/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")

    @classmethod
    def _from_payload(cls, payload: dict) -> "FBSNNSolver":
        if payload["format_version"] not in (1, 2):
            raise ValueError("Unsupported checkpoint format.")
        data = json.loads(payload["problem"])
        case = PricingCase(data["name"], GBMModel(**data["model"]),
                           EuropeanCall(**data["contract"]), data["initial_spots"])
        solver = cls(case, TrainingConfig(**payload["config"]))
        for name in ("input_scales", "value_scale"):
            if not torch.equal(payload["model_state"][name], getattr(solver.network, name)):
                raise ValueError("Checkpoint scales disagree with its financial metadata.")
        solver.network.load_state_dict(payload["model_state"], strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in solver.network.parameters()):
            raise ValueError("Checkpoint has nonfinite parameters.")
        return solver

    @classmethod
    def load(cls, checkpoint: Path) -> "FBSNNSolver":
        """Load version 1 or 2 inference weights, without fitting."""
        solver = cls._from_payload(torch.load(checkpoint, map_location="cpu", weights_only=True))
        solver.network.eval()
        solver._has_fit = True
        return solver

    @classmethod
    def resume(cls, checkpoint: Path) -> "FBSNNSolver":
        """Continue the latest update, rather than the selected inference weights."""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 2 or "training_state" not in payload:
            raise ValueError("Continuation requires a version 2 resume.pt checkpoint.")
        solver = cls._from_payload(payload)
        state = payload["training_state"]
        if state["status"] not in ("paused", "time_budget_reached"):
            raise ValueError("Only paused or time-limited runs can continue.")
        if not 0 <= state["completed"] < solver.config.iterations:
            raise ValueError("Invalid completed update count in continuation checkpoint.")
        solver._training = state
        return solver


def train_and_save(case: PricingCase, config: TrainingConfig, directory: Path,
                   *, stop_after: int | None = None) -> dict[str, object]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Run directory already exists: {directory}")
    solver = FBSNNSolver(case, config)
    try:
        report = solver.fit(stop_after=stop_after)
    except (RuntimeError, ArithmeticError, ValueError) as exc:
        if not directory.exists():
            directory.mkdir(parents=True)
            (directory/"failure.json").write_text(json.dumps({
                "status": "failed", "problem": json.loads(case.to_json()), "config": asdict(config),
                "error_type": type(exc).__name__, "error": str(exc),
            }, indent=2, allow_nan=False)+"\n")
        raise
    solver.save(directory, report)
    return report
