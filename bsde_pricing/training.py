"""Bounded one-asset value-gradient training and checkpoint evaluation."""

import json
import math
import platform
import time
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

    Training, monitoring and evaluation have disjoint SeedSequence children.
    The final iteration is retained; monitoring does not select a checkpoint.
    Saved checkpoints support inference, not optimiser continuation.
    """

    def __init__(self, case: PricingCase, config: TrainingConfig = TrainingConfig()):
        self.reference = BlackScholesReference(case.model, case.contract)
        if case.model.volatilities[0] <= 0 or case.contract.maturity <= 0:
            raise ValueError("The training pilot requires positive maturity and volatility.")
        self.case, self.config = case, config
        self.network = ValueNetwork(case.initial_spots, case.contract.maturity,
                                    case.contract.strike, width=config.width, seed=config.seed)
        self.loss = LocalBSDELoss(PricingBSDE(case.model, case.contract), value_scale=case.contract.strike)
        self.times = torch.from_numpy(np.linspace(0, case.contract.maturity, config.n_steps+1))
        self._has_fit = False

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

    def fit(self) -> dict[str, object]:
        if self._has_fit:
            raise RuntimeError("Create a new solver for a new fit; these checkpoints are for inference.")
        self._has_fit = True
        children = np.random.SeedSequence(self.config.seed).spawn(3)
        train_rng, validation_rng, evaluation_rng = (np.random.default_rng(s) for s in children)
        validation = self._batch(validation_rng, self.config.validation_paths)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        history = []
        old_threads = torch.get_num_threads()
        completed = 0
        window = []
        started = time.perf_counter()
        try:
            torch.set_num_threads(1)
            oracle = AnalyticalCall(self.case.model, self.case.contract)
            analytical_validation = self.loss(oracle, self.times, *validation, parameter_graph=False)
            def record(step):
                result = self.loss(self.network, self.times, *validation, parameter_graph=False)
                history.append({"step": step, "elapsed_seconds": time.perf_counter()-started,
                                "training_loss_window_mean": float(np.mean(window)) if window else None,
                                "validation_local": float(result.local), "validation_terminal": float(result.terminal),
                                "validation_total": float(result.total), **self.initial_value()})
                window.clear()
            record(0)
            for step in range(1, self.config.iterations+1):
                if time.perf_counter()-started >= self.config.max_seconds:
                    break
                learning_rate = self.config.learning_rate*(0.3 if step > 0.7*self.config.iterations else 1)
                for group in optimizer.param_groups: group["lr"] = learning_rate
                states, increments = self._batch(train_rng, self.config.batch_size)
                optimizer.zero_grad(set_to_none=True)
                result = self.loss(self.network, self.times, states, increments)
                result.total.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.network.parameters()):
                    raise ArithmeticError("Nonfinite or missing training gradient.")
                optimizer.step()
                completed = step
                window.append(float(result.total.detach()))
                if step % self.config.log_every == 0 or step == self.config.iterations:
                    record(step)
            if history[-1]["step"] != completed: record(completed)
            training_seconds = time.perf_counter()-started
            evaluation_started = time.perf_counter()
            evaluation = self._evaluate(evaluation_rng, self.config.evaluation_paths)
            evaluation_seconds = time.perf_counter()-evaluation_started
        finally:
            torch.set_num_threads(old_threads)
        initial = evaluation.summary["initial"]
        return {
            "method": "value_gradient_fbsnn", "status": "completed" if completed == self.config.iterations else "time_budget_reached",
            "problem": json.loads(self.case.to_json()), "config": asdict(self.config),
            "device": "cpu", "dtype": "torch.float64", "cpu_threads": 1,
            "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
            "system": platform.platform(), "completed_updates": completed,
            "training_paths_drawn": completed*self.config.batch_size,
            "stream_children": {"training": 0, "validation": 1, "evaluation": 2},
            "checkpoint_rule": "last_completed_update", "history": history,
            "validation_analytical_local_loss": float(analytical_validation.local),
            "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
            "evaluation": evaluation.summary,
            "inspection_targets": {"price_absolute_error": 0.25, "delta_absolute_error": 0.02},
            "targets_met": {"price": abs(initial["price_error"]) <= 0.25,
                            "delta": abs(initial["delta_error"]) <= 0.02},
        }

    def save(self, directory: Path, report: dict[str, object]) -> None:
        """Create a new run directory; existing runs are never overwritten."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        torch.save({"format_version": 1, "problem": self.case.to_json(), "config": asdict(self.config),
                    "model_state": self.network.state_dict()}, directory/"model.pt")
        (directory/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")

    @classmethod
    def load(cls, checkpoint: Path) -> "FBSNNSolver":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1:
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
        solver.network.eval()
        solver._has_fit = True
        return solver


def train_and_save(case: PricingCase, config: TrainingConfig, directory: Path) -> dict[str, object]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Run directory already exists: {directory}")
    solver = FBSNNSolver(case, config)
    try:
        report = solver.fit()
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
