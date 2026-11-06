"""Arithmetic payoff training, numerical uncertainty and isolated pilot costs."""

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from bsde_pricing.arithmetic_pilot import (
    ArithmeticPilot, ArithmeticPilotConfig, _error_assessment, _peak_rss_mib,
)
from bsde_pricing.basket_reference import BasketReferencePricer
from bsde_pricing.problems import arithmetic_ten_case
from bsde_pricing.risk_comparison import SolverRiskComparison
from bsde_pricing.terminal_bsde import TerminalBSDESolver
from bsde_pricing.training import FBSNNSolver, PathTrainingConfig, TrainingConfig


class ArithmeticPilotTests(unittest.TestCase):
    def setUp(self):
        self.case = arithmetic_ten_case()
        self.training = PathTrainingConfig(iterations=4, n_steps=2, batch_size=8, width=4,
                                           log_every=2, validation_paths=8, evaluation_paths=8)
        self.config = ArithmeticPilotConfig(training=self.training, coarse_power=6,
                                            reference_power=8, scrambles=4, pilot_pairs=64, check_pairs=256)
        Path("scratch/tests").mkdir(parents=True, exist_ok=True)

    def test_ten_asset_economics_and_independent_pathwise_delta_bumps(self):
        case = self.case
        np.testing.assert_allclose(np.linalg.eigvalsh(case.model.correlation), [0.7]*9+[3.7], atol=2e-15)
        self.assertAlmostEqual(sum(case.contract.weights), 1)
        np.testing.assert_allclose(case.contract.payoff([[90]*10, [110]*10]), [0, 10], atol=1e-13)
        pricer = BasketReferencePricer(case.model, case.contract)
        normals = np.random.default_rng(31).standard_normal((128, 10))
        spots = np.array(case.initial_spots)
        contributions, _ = pricer._contributions(spots, 1, normals)
        for j in range(10):
            bump = np.eye(10)[j]*1e-3
            up = case.model.terminal_from_normals(spots+bump, 1, normals)
            down = case.model.terminal_from_normals(spots-bump, 1, normals)
            independent_delta = np.exp(-0.03)*(case.contract.payoff(up)-case.contract.payoff(down))/2e-3
            np.testing.assert_allclose(contributions[:, j+1], independent_delta, rtol=0, atol=2e-10)

    def test_reference_free_fits_selection_and_round_trip(self):
        value = FBSNNSolver(self.case, TrainingConfig(**self.training.__dict__))
        terminal = TerminalBSDESolver(self.case, self.training)
        with patch("bsde_pricing.training.AnalyticalCall", side_effect=AssertionError("No arithmetic oracle")), \
             patch.object(BasketReferencePricer, "sobol", side_effect=AssertionError("No reference labels")):
            a, b = value.fit(), terminal.fit()
        for solver, report, objective in ((value, a, "validation_total"), (terminal, b, "validation_terminal")):
            self.assertIsNone(solver.reference)
            self.assertIsNone(report["targets_met"])
            self.assertIsNone(report["evaluation"]["initial"]["price_error"])
            self.assertEqual(report["selected_step"], min(report["history"], key=lambda h: h[objective])["step"])
        self.assertIsNone(value.evaluate(n_paths=8).reference_deltas)
        self.assertIsNone(terminal.evaluate(n_paths=8).analytical_terminal_residuals)
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp:
            root = Path(temp)
            value.save(root/"value", a)
            terminal.save(root/"terminal", b)
            for model, path in ((value, root/"value/model.pt"), (terminal, root/"terminal/model.pt")):
                restored = type(model).load(path)
                self.assertEqual(restored.initial_value(), model.initial_value())
                for left, right in zip(model.network.parameters(), restored.network.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        comparison = SolverRiskComparison(value, terminal)
        self.assertEqual(len(comparison.initial_risk("terminal_loss")["delta"]), 10)
        with self.assertRaisesRegex(ValueError, "analytical reference"):
            comparison.run(n_paths=8)

    def test_reference_interval_propagation_and_weight_scaled_units(self):
        stat = {"estimate": 0.06, "confidence_low": 0.0599, "confidence_high": 0.0601}
        result = _error_assessment(0.061, stat, 0.02, scale=0.1)
        self.assertAlmostEqual(result["signed_error"], 0.01)
        np.testing.assert_allclose(result["signed_error_interval"], [0.009, 0.011], atol=1e-15)
        self.assertEqual(result["classification"], "within_target")
        self.assertEqual(_error_assessment(0.061, stat, 0.01, 0.1)["classification"], "unresolved_at_reference_precision")
        self.assertEqual(_error_assessment(0.065, stat, 0.02, 0.1)["classification"], "outside_target")
        missing = {**stat, "confidence_low": None, "confidence_high": None}
        self.assertIsNone(_error_assessment(0.06, missing, 0.01)["signed_error_interval"])

    def test_isolated_pipeline_retains_reference_costs_and_protected_outputs(self):
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp:
            directory = Path(temp)/"pilot"
            result = ArithmeticPilot(self.config).run(directory)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(json.loads((directory/"report.json").read_text()), result)
            reference = result["reference"]
            self.assertEqual([r["evaluation_terminal_states"] for r in reference["sobol_levels"]], [256, 1024])
            self.assertEqual(reference["sobol_levels"][-1]["sampling_unit"], "scramble_mean")
            for kind in ("reference", "value_gradient", "terminal_loss"):
                self.assertGreater(result["resources"][kind]["peak_resident_mib"], 0)
                self.assertGreater(result["resources"][kind]["process_wall_seconds"], 0)
            for kind, model in result["models"].items():
                self.assertEqual(len(model["scaled_delta_checks"]), 10)
                self.assertIsNone(model["path_delta_rmse"])
                self.assertEqual(len(model["source"]["model_sha256"]), 64)
                self.assertEqual(model["source"]["completed_updates"], 4)
                self.assertLess(model["initial"]["reconstruction"]["max_absolute_reconstruction_residual"], 1e-12)
            with patch.object(ArithmeticPilot, "_run_worker") as worker:
                with self.assertRaises(FileExistsError):
                    ArithmeticPilot(self.config).run(directory)
                worker.assert_not_called()

    def test_worker_failure_keeps_design_and_identifies_task(self):
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp:
            directory = Path(temp)/"failed"
            with patch.object(ArithmeticPilot, "_run_worker", side_effect=subprocess.CalledProcessError(1, "worker")):
                with self.assertRaises(RuntimeError):
                    ArithmeticPilot(self.config).run(directory)
            failure = json.loads((directory/"failure.json").read_text())
            self.assertEqual(failure["task"], "reference")
            self.assertEqual(failure["design"]["config"]["training"]["iterations"], 4)
            self.assertTrue((directory/"design.json").exists())

    def test_resource_units_and_invalid_designs(self):
        with patch("bsde_pricing.arithmetic_pilot.platform.system", return_value="Darwin"), \
             patch("bsde_pricing.arithmetic_pilot.resource.getrusage", return_value=SimpleNamespace(ru_maxrss=256*1024**2)):
            self.assertEqual(_peak_rss_mib(), 256)
        with patch("bsde_pricing.arithmetic_pilot.platform.system", return_value="Linux"), \
             patch.object(Path, "read_text", return_value="Name: python\nVmHWM: 262144 kB\n"):
            self.assertEqual(_peak_rss_mib(), 256)
        for changes in ({"reference_power": 6}, {"scrambles": 3}, {"coarse_power": True},
                        {"reference_seed": 2026}, {"check_seed": 2032}, {"reference_fraction": 2},
                        {"price_target": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)


if __name__ == "__main__":
    unittest.main()
