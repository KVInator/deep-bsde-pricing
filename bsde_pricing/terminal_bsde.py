"""Fixed-start scalar Deep BSDE with learned discounted Brownian controls."""

import json
import math
import platform
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .black_scholes import BlackScholesReference
from .bsde import PricingBSDE
from .problems import EuropeanCall, GBMModel, PricingCase
from .training import PathTrainingConfig, sample_training_paths


class TerminalControlNetwork(nn.Module):
    """Learn Y0, discounted Z0 and one state-only network per later time.

    Each control uses its current spot only. No terminal state or future
    increment enters a control. All parameters and path data are CPU float64.
    The initial price is a free scalar, not a function of initial spot.
    """

    def __init__(self, case: PricingCase, config: PathTrainingConfig):
        super().__init__()
        self.n_steps = config.n_steps
        for name, value in (("spot_scale", case.initial_spots[0]), ("value_scale", case.contract.strike),
                            ("control_scale", case.contract.strike/math.sqrt(case.contract.maturity))):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64, device="cpu"))
        self.initial_value = nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))
        self.initial_control = nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config.seed)
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(1, config.width, dtype=torch.float64, device="cpu"), nn.Tanh(),
                              nn.Linear(config.width, config.width, dtype=torch.float64, device="cpu"), nn.Tanh(),
                              nn.Linear(config.width, 1, dtype=torch.float64, device="cpu"))
                for _ in range(config.n_steps-1)
            ])

    def discounted_controls(self, states: torch.Tensor) -> torch.Tensor:
        if (states.dtype != torch.float64 or states.device.type != "cpu" or states.ndim != 3
                or states.shape[0] < 1 or states.shape[1:] != (self.n_steps+1, 1)
                or states.requires_grad or not bool(torch.isfinite(states).all()) or bool((states <= 0).any())):
            raise ValueError("States must be fixed positive CPU float64 data with shape (N,M+1,1).")
        controls = [self.initial_control.expand(states.shape[0], 1)]
        controls.extend(layer(states[:, step]/self.spot_scale-1)
                        for step, layer in enumerate(self.layers, start=1))
        return torch.stack(controls, dim=1)*self.control_scale

    def forward(self, states: torch.Tensor, increments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        controls = self.discounted_controls(states)
        if (increments.shape != controls.shape or increments.dtype != torch.float64
                or increments.device.type != "cpu" or increments.requires_grad
                or not bool(torch.isfinite(increments).all())):
            raise ValueError("Increments must be fixed finite CPU float64 data with shape (N,M,1).")
        initial = (self.value_scale*self.initial_value).expand(states.shape[0], 1)
        values = torch.cat((initial, initial+torch.cumsum((controls*increments).sum(dim=2), dim=1)), dim=1)
        if not bool(torch.isfinite(values).all()):
            raise ArithmeticError("Nonfinite propagated terminal-BSDE values.")
        return values, controls


@dataclass(frozen=True)
class TerminalEvaluation:
    times: np.ndarray
    spots: np.ndarray
    discounted_values: np.ndarray
    discounted_controls: np.ndarray
    discounted_payoffs: np.ndarray
    terminal_residuals: np.ndarray
    analytical_terminal_residuals: np.ndarray
    summary: dict[str, object]


class TerminalBSDESolver:
    """Terminal-payoff fitting for one scalar GBM call, without price labels."""

    def __init__(self, case: PricingCase, config: PathTrainingConfig = PathTrainingConfig()):
        if type(config) is not PathTrainingConfig:
            raise ValueError("Terminal-BSDE training uses PathTrainingConfig without local-loss weighting or patience.")
        self.reference = BlackScholesReference(case.model, case.contract)
        if case.contract.maturity <= 0 or case.model.volatilities[0] <= 0:
            raise ValueError("Terminal-BSDE training requires positive maturity and volatility.")
        self.case, self.config = case, config
        self.network = TerminalControlNetwork(case, config)
        self.times = np.linspace(0, case.contract.maturity, config.n_steps+1)
        self.discount = np.exp(-case.model.rate*self.times)
        if not np.isfinite(self.discount).all() or np.any(self.discount <= 0):
            raise ValueError("Discount factors must be positive and finite.")
        self._has_fit = False

    def _batch(self, rng: np.random.Generator, n_paths: int):
        return sample_training_paths(self.case, self.config.n_steps, rng, n_paths)

    def _payoffs(self, states: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.discount[-1]*self.case.contract.payoff(states[:, -1].numpy()))

    def _loss(self, states: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
        values, _ = self.network(states, increments)
        return ((values[:, -1]-self._payoffs(states))/self.case.contract.strike).square().mean()

    def initial_value(self) -> dict[str, float]:
        price = float((self.network.value_scale*self.network.initial_value).detach())
        target = self.reference.evaluate(self.case.initial_spots[0]).call_price
        return {"price": price, "reference_price": target, "price_error": price-target,
                "discounted_initial_control": float((self.network.control_scale*self.network.initial_control).detach())}

    def _evaluate(self, rng: np.random.Generator, n_paths: int) -> TerminalEvaluation:
        states, increments = self._batch(rng, n_paths)
        with torch.no_grad():
            values, controls = self.network(states, increments)
            payoffs = self._payoffs(states)
        residuals = (values[:, -1]-payoffs).numpy()
        problem = PricingBSDE(self.case.model, self.case.contract)
        # This matched-path oracle is descriptive and never enters training or selection.
        analytical_controls = np.stack([
            problem.control(states[:, m].numpy(), self.reference.call_delta_array(states[:, m].numpy(), t))
            for m, t in enumerate(self.times[:-1])
        ], axis=1)*self.discount[None, :-1, None]
        reference_price = self.reference.evaluate(self.case.initial_spots[0]).call_price
        analytical_terminal = reference_price+(analytical_controls*increments.numpy()).sum(axis=(1, 2))
        analytical_residuals = analytical_terminal-payoffs.numpy()
        stochastic_integral = values[:, -1].numpy()-self.initial_value()["price"]
        rmse = float(np.sqrt(np.mean(residuals**2)))
        summary = {
            "n_paths": n_paths, "initial": self.initial_value(),
            "discounted_terminal_rmse": rmse,
            "undiscounted_terminal_rmse": rmse/float(self.discount[-1]),
            "normalized_terminal_loss": float(np.mean((residuals/self.case.contract.strike)**2)),
            "terminal_residual_mean": float(np.mean(residuals)),
            "terminal_residual_mean_se": float(np.std(residuals, ddof=1)/math.sqrt(n_paths)),
            "terminal_residual_quantiles": dict(zip(("0.05", "0.50", "0.95"), map(float, np.quantile(residuals, (0.05, 0.5, 0.95))))),
            "stochastic_integral_mean": float(np.mean(stochastic_integral)),
            "stochastic_integral_mean_se": float(np.std(stochastic_integral, ddof=1)/math.sqrt(n_paths)),
            "analytical_discounted_terminal_rmse": float(np.sqrt(np.mean(analytical_residuals**2))),
        }
        arrays = [self.times.copy(), states[:, :, 0].numpy().copy(), values.numpy(), controls[:, :, 0].numpy(),
                  payoffs.numpy(), residuals, analytical_residuals]
        for array in arrays:
            array.setflags(write=False)
        return TerminalEvaluation(*arrays, summary)

    def evaluate(self, *, n_paths: int = 512, seed: int = 2027) -> TerminalEvaluation:
        if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 2:
            raise ValueError("n_paths must be an integer of at least two.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        return self._evaluate(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3,))), n_paths)

    def fit(self) -> dict[str, object]:
        if self._has_fit:
            raise RuntimeError("Create a new solver for a new terminal-BSDE fit.")
        self._has_fit = True
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(self.config.seed).spawn(3)]
        training_rng, validation_rng, evaluation_rng = streams
        validation = self._batch(validation_rng, self.config.validation_paths)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        history, window = [], []
        best_loss, best_step, best_state = math.inf, 0, None
        completed = 0
        old_threads = torch.get_num_threads()
        started = time.perf_counter()
        try:
            torch.set_num_threads(1)
            def record():
                nonlocal best_loss, best_step, best_state
                with torch.no_grad():
                    validation_loss = float(self._loss(*validation))
                if not math.isfinite(validation_loss):
                    raise ArithmeticError("Nonfinite terminal validation loss.")
                if validation_loss < best_loss:
                    best_loss, best_step = validation_loss, completed
                    best_state = deepcopy(self.network.state_dict())
                history.append({"step": completed, "elapsed_seconds": time.perf_counter()-started,
                                "training_loss_window_mean": float(np.mean(window)) if window else None,
                                "validation_terminal": validation_loss, **self.initial_value()})
                window.clear()
            record()
            for step in range(1, self.config.iterations+1):
                if time.perf_counter()-started >= self.config.max_seconds:
                    break
                for group in optimizer.param_groups:
                    group["lr"] = self.config.learning_rate*(0.3 if step > 0.7*self.config.iterations else 1)
                batch = self._batch(training_rng, self.config.batch_size)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(*batch)
                loss.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.network.parameters()):
                    raise ArithmeticError("Nonfinite or missing terminal-BSDE gradient.")
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in self.network.parameters()):
                    raise ArithmeticError("Nonfinite terminal-BSDE parameter.")
                completed = step
                window.append(float(loss.detach()))
                if step % self.config.log_every == 0 or step == self.config.iterations:
                    record()
            training_seconds = time.perf_counter()-started
            evaluation_started = time.perf_counter()
            evaluation_state = deepcopy(evaluation_rng.bit_generator.state)
            last_evaluation = self._evaluate(evaluation_rng, self.config.evaluation_paths).summary
            self._last_state = deepcopy(self.network.state_dict())
            self.network.load_state_dict(best_state)
            evaluation_rng.bit_generator.state = evaluation_state
            evaluation = self._evaluate(evaluation_rng, self.config.evaluation_paths).summary
            evaluation_seconds = time.perf_counter()-evaluation_started
        finally:
            torch.set_num_threads(old_threads)
        return {"method": "terminal_loss_deep_bsde", "problem": json.loads(self.case.to_json()),
                "config": asdict(self.config), "status": "completed" if completed == self.config.iterations else "time_budget_reached",
                "device": "cpu", "dtype": "torch.float64", "cpu_threads": 1,
                "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
                "system": platform.platform(), "parameter_count": sum(p.numel() for p in self.network.parameters()),
                "stream_children": {"training": 0, "validation": 1, "evaluation": 2},
                "completed_updates": completed, "training_paths_drawn": completed*self.config.batch_size,
                "checkpoint_rule": "minimum_validation_terminal_at_scheduled_checks",
                "selected_step": best_step, "selected_validation_terminal": best_loss,
                "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                "history": history, "evaluation": evaluation, "last_evaluation": last_evaluation,
                "inspection_targets": {"price_absolute_error": 0.25},
                "targets_met": {"price": abs(evaluation["initial"]["price_error"]) <= 0.25},
                "checkpoint_support": "inference_only; no optimiser continuation"}

    def save(self, directory: Path, report: dict[str, object]) -> None:
        if not hasattr(self, "_last_state"):
            raise RuntimeError("Fit a terminal-BSDE run before saving it.")
        if (report.get("method") != "terminal_loss_deep_bsde" or report.get("config") != asdict(self.config)
                or report.get("evaluation", {}).get("initial") != self.initial_value()):
            raise ValueError("Report does not describe the current terminal-BSDE model.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        common = {"format_version": 1, "method": "terminal_loss_deep_bsde",
                  "problem": self.case.to_json(), "config": asdict(self.config)}
        torch.save({**common, "model_state": self.network.state_dict()}, directory/"model.pt")
        torch.save({**common, "model_state": self._last_state}, directory/"last_model.pt")
        (directory/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")

    @classmethod
    def load(cls, checkpoint: Path) -> "TerminalBSDESolver":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or payload.get("method") != "terminal_loss_deep_bsde":
            raise ValueError("Unsupported terminal-BSDE inference checkpoint.")
        data = json.loads(payload["problem"])
        case = PricingCase(data["name"], GBMModel(**data["model"]), EuropeanCall(**data["contract"]), data["initial_spots"])
        solver = cls(case, PathTrainingConfig(**payload["config"]))
        for name in ("spot_scale", "value_scale", "control_scale"):
            if not torch.equal(payload["model_state"][name], getattr(solver.network, name)):
                raise ValueError("Terminal checkpoint scales disagree with financial metadata.")
        solver.network.load_state_dict(payload["model_state"], strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in solver.network.parameters()):
            raise ValueError("Nonfinite terminal-BSDE checkpoint parameters.")
        solver.network.eval()
        solver._has_fit = True
        return solver


def train_terminal_and_save(case: PricingCase, config: PathTrainingConfig, directory: Path) -> dict[str, object]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Run directory already exists: {directory}")
    solver = TerminalBSDESolver(case, config)
    try:
        report = solver.fit()
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        directory.mkdir(parents=True, exist_ok=False)
        (directory/"failure.json").write_text(json.dumps({"status": "failed", "config": asdict(config),
            "problem": json.loads(case.to_json()), "error_type": type(exc).__name__, "error": str(exc)}, indent=2)+"\n")
        raise
    solver.save(directory, report)
    return report
