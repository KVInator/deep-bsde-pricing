"""Paired seed and terminal-weight comparison for the scalar FBSNN."""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .problems import PricingCase
from .training import FBSNNSolver, TrainingConfig, train_and_save


def _same_state(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right))
    return left == right


class TrainingStudy:
    """Three predeclared seeds, two loss weights and a fixed update budget.

    The weight-one final update reproduces the first experiment. Validation
    selects checkpoints independently within each weight. Held-out analytical
    errors describe the resulting models; they never select weights or updates.
    """

    seeds = (2026, 2027, 2028)
    terminal_weights = (1, 5)

    def __init__(self, case: PricingCase, config: TrainingConfig = TrainingConfig()):
        if config.patience is not None:
            raise ValueError("The paired study uses the full update budget without early stopping.")
        self.case, self.config = case, config

    def _verify_resume(self, directory: Path) -> dict[str, object]:
        """Pause off the logging grid and cross the scheduled learning-rate change."""
        config = replace(self.config, iterations=80, n_steps=4, batch_size=16, width=4,
                         log_every=20, validation_paths=16, evaluation_paths=32)
        full = FBSNNSolver(self.case, config)
        full_report = full.fit()
        full.save(directory/"uninterrupted", full_report)
        partial = FBSNNSolver(self.case, config)
        paused = partial.fit(stop_after=37)
        partial.save(directory/"paused", paused)
        continued = FBSNNSolver.resume(directory/"paused"/"resume.pt")
        continued_report = continued.fit()
        continued.save(directory/"continued", continued_report)
        a = torch.load(directory/"uninterrupted"/"resume.pt", weights_only=True)
        b = torch.load(directory/"continued"/"resume.pt", weights_only=True)
        checks = {"current_weights": _same_state(a["model_state"], b["model_state"])}
        for name in ("optimizer_state", "train_rng_state", "evaluation_rng_state", "torch_rng_state",
                     "validation", "best_state", "best_step", "checks_without_improvement", "window"):
            checks[name] = _same_state(a["training_state"][name], b["training_state"][name])
        histories = [[{k: v for k, v in h.items() if k != "elapsed_seconds"}
                      for h in report["history"]] for report in (full_report, continued_report)]
        checks["history_excluding_timing"] = histories[0] == histories[1]
        checks["held_out_evaluation"] = full_report["evaluation"] == continued_report["evaluation"]
        checks["no_evaluation_at_pause"] = paused["evaluation"] is None
        checks["full_budget"] = full_report["completed_updates"] == continued_report["completed_updates"] == 80
        result = {"passed": all(checks.values()), "checks": checks,
                  "pause_after": 37, "config": asdict(config),
                  "comparison": "bitwise on this runtime; elapsed times excluded"}
        (directory/"resume_check.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        return result

    @staticmethod
    def _summary(runs: list[dict]) -> list[dict]:
        summaries = []
        for weight in TrainingStudy.terminal_weights:
            group = [r for r in runs if r["terminal_weight"] == weight]
            usable = [r for r in group if r["status"] in ("completed", "time_budget_reached")]
            for checkpoint in ("selected", "last"):
                row = {"terminal_weight": weight, "checkpoint": checkpoint,
                       "runs": len(group), "usable_runs": len(usable),
                       "completed_runs": sum(r["status"] == "completed" for r in group),
                       "failures": sum(r["status"] == "failed" for r in group)}
                for key in ("price_error", "delta_error", "price_rmse_before_expiry",
                            "delta_rmse_before_expiry", "terminal_payoff_rmse", "unweighted_total_loss",
                            "training_seconds", "evaluation_seconds"):
                    values = [r[key] if key.endswith("seconds") else r[checkpoint][key] for r in usable]
                    row[key] = {"mean": float(np.mean(values)) if values else None,
                                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                "min": min(values) if values else None, "max": max(values) if values else None}
                summaries.append(row)
        return summaries

    def run(self, directory: Path) -> dict[str, object]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        # Persist the design before fitting or inspecting any held-out result.
        design = {"method": "paired_fbsnn_seed_and_terminal_weight_study",
                  "problem": json.loads(self.case.to_json()), "base_config": asdict(self.config),
                  "seeds": self.seeds, "terminal_weights": self.terminal_weights,
                  "paired_streams_within_seed": True, "selection": "minimum fixed validation objective",
                  "early_stopping": False}
        (directory/"design.json").write_text(json.dumps(design, indent=2, allow_nan=False)+"\n")
        resume_check = self._verify_resume(directory/"continuation")
        if not resume_check["passed"]:
            raise ArithmeticError("Continuation check failed; inspect continuation/resume_check.json.")
        runs = []
        for seed in self.seeds:
            for weight in self.terminal_weights:
                name = f"seed_{seed}_weight_{weight}"
                config = replace(self.config, seed=seed, terminal_weight=weight)
                row = {"seed": seed, "terminal_weight": weight, "directory": name}
                try:
                    report = train_and_save(self.case, config, directory/name)
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                else:
                    row.update(status=report["status"], completed_updates=report["completed_updates"],
                               selected_step=report["selected_step"], targets_met=report["targets_met"],
                               training_seconds=report["training_seconds"],
                               evaluation_seconds=report["evaluation_seconds"])
                    for label, key in (("selected", "evaluation"), ("last", "last_evaluation")):
                        evaluation = report[key]
                        row[label] = {**evaluation["initial"], **{k: evaluation[k] for k in (
                            "price_rmse_before_expiry", "delta_rmse_before_expiry", "terminal_payoff_rmse",
                            "local_loss", "terminal_loss", "total_loss", "unweighted_total_loss")}}
                runs.append(row)
                result = {**design, "resume_check": resume_check, "runs": runs,
                          "summary": self._summary(runs), "wall_seconds": time.perf_counter()-started,
                          "status": "completed" if len(runs) == 6 and all(r["status"] == "completed" for r in runs)
                                    else "incomplete"}
                (directory/"study.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        return result
