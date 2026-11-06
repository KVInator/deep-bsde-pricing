"""Streaming uncertainty, resource guards and paired development evidence."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bsde_pricing.arithmetic_pilot import _guarded_worker, _child_rss_mib, WorkerLimitError
from bsde_pricing.basket_reference import BasketReferencePricer
from bsde_pricing.development_study import ArithmeticDevelopmentStudy, TimeGridStudy, profile_update_components
from bsde_pricing.training import PathTrainingConfig
from bsde_pricing.experiment_design import ExperimentDesign
from bsde_pricing.problems import correlated_case


class DevelopmentStudyTests(unittest.TestCase):
    def test_component_profile_restores_threads_and_retains_all_workloads(self):
        import torch
        before = torch.get_num_threads()
        config = PathTrainingConfig(iterations=2, n_steps=2, batch_size=8, width=4)
        profile = profile_update_components(correlated_case(), config, repeats=3)
        self.assertEqual(torch.get_num_threads(), before)
        for model in profile["models"].values():
            self.assertEqual(set(model["seconds"]), {"sample_paths", "objective_forward", "backward", "optimizer_step"})
            self.assertTrue(all(len(v) == 3 and min(v) >= 0 for v in model["seconds"].values()))

    def setUp(self):
        Path("scratch/tests").mkdir(parents=True, exist_ok=True)

    def test_chunked_references_preserve_samples_and_uncertainty(self):
        c = correlated_case()
        pricer = BasketReferencePricer(c.model, c.contract)
        for method, options in ((pricer.antithetic, {"n_pairs": 1027}),
                                (pricer.sobol, {"power": 10, "scrambles": 4})):
            full = method(c.initial_spots, pilot_pairs=64, **options)
            chunked = method(c.initial_spots, pilot_pairs=64, chunk_size=128, **options)
            self.assertEqual(full.control_coefficients, chunked.control_coefficients)
            self.assertEqual(full.n_units, chunked.n_units)
            for name in ("plain", "controlled"):
                a, b = getattr(full, name), getattr(chunked, name)
                for x, y in zip((a.price, *a.deltas), (b.price, *b.deltas)):
                    for field in ("estimate", "sample_standard_deviation", "standard_error", "confidence_low", "confidence_high"):
                        self.assertAlmostEqual(getattr(x, field), getattr(y, field), delta=2e-13)
            if full.replicate_controlled is not None:
                np.testing.assert_allclose(full.replicate_controlled, chunked.replicate_controlled, atol=2e-13, rtol=0)
        with self.assertRaises(ValueError):
            pricer.sobol(c.initial_spots, chunk_size=100)
        with patch.object(BasketReferencePricer, "_contributions", wraps=pricer._contributions) as contributions:
            pricer.sobol(c.initial_spots, power=10, scrambles=4, pilot_pairs=64, chunk_size=128)
            self.assertLessEqual(max(len(call.args[2]) for call in contributions.call_args_list), 128)

    def test_real_workers_record_time_and_memory_limit_failures(self):
        with patch("bsde_pricing.arithmetic_pilot.platform.system", return_value="Darwin"), \
             patch("bsde_pricing.arithmetic_pilot.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "262144\n", "")):
            self.assertEqual(_child_rss_mib(123), 256)
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp:
            directory = Path(temp)
            for name, limits in (("time", {"worker_wall_seconds": 0.05, "worker_peak_resident_mib": 1e6}),
                                 ("memory", {"worker_wall_seconds": 5, "worker_peak_resident_mib": 0.001})):
                with (directory/f"{name}.log").open("w") as log, self.assertRaises(WorkerLimitError):
                    _guarded_worker([sys.executable, "-c", "import time; time.sleep(2)"], os.environ.copy(),
                                    log, directory, name, limits)
                guard = json.loads((directory/f"{name}_guard.json").read_text())
                self.assertEqual(guard["status"], f"{name}_limit")
                self.assertLess(guard["process_wall_seconds"], 2)

    def test_small_study_retains_all_seeds_models_and_development_provenance(self):
        data = ExperimentDesign.load().data
        data["training"].update(iterations=2, base_steps=2, batch_size=8, width=4,
                                validation_paths=8, evaluation_paths=8, log_every=1)
        data["development_seeds"] = [2026, 2027]
        data["reference"].update(sobol_powers=[6, 8], check_pairs=[256, 1024], scrambles=4,
                                 pilot_pairs=64, chunk_terminal_states=64)
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp:
            directory = Path(temp)/"study"
            result = ArithmeticDevelopmentStudy("arithmetic-100", design=ExperimentDesign(data)).run(directory)
            self.assertEqual([r["seed"] for r in result["runs"]], [2026, 2027])
            self.assertEqual(result, json.loads((directory/"report.json").read_text()))
            for row in result["runs"]:
                self.assertEqual(set(row["models"]), {"value_gradient", "terminal_loss"})
                self.assertEqual(len(row["models"]["terminal_loss"]["initial"]["delta"]), 100)
                self.assertEqual(len(row["models"]["value_gradient"]["scaled_delta_checks"]), 100)
                self.assertEqual(row["tasks"]["value_gradient"]["completed_updates"], 2)
            with patch("bsde_pricing.arithmetic_pilot.ArithmeticPilot._run_worker") as run:
                with self.assertRaises(FileExistsError):
                    ArithmeticDevelopmentStudy("arithmetic-100", design=ExperimentDesign(data)).run(directory)
                run.assert_not_called()

    def test_failed_tasks_remain_in_study(self):
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp, \
             patch("bsde_pricing.arithmetic_pilot.ArithmeticPilot._run_worker", side_effect=WorkerLimitError("limit")):
            result = ArithmeticDevelopmentStudy().run(Path(temp)/"failed")
            self.assertFalse(result["reference_usable"])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["runs"]), 3)
            self.assertEqual(result["summary"]["value_gradient"]["compared_runs"], 0)
            self.assertTrue(all(t["status"] == "failed" for r in result["runs"] for t in r["tasks"].values()))

    def test_grid_study_shares_reference_and_saves_grid_specific_models(self):
        data = ExperimentDesign.load().data
        data["training"].update(iterations=2, base_steps=2, batch_size=8, width=4,
                                validation_paths=8, evaluation_paths=8, log_every=1)
        data["development_seeds"] = [2026]
        data["grid_steps"] = [1, 3]
        data["reference"].update(sobol_powers=[6, 8], check_pairs=[256, 1024], scrambles=4,
                                 pilot_pairs=64, chunk_terminal_states=64)
        study = TimeGridStudy(ExperimentDesign(data))
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp:
            directory = Path(temp)/"grid"
            with patch.object(study, "_reference", wraps=study._reference) as reference:
                result = study.run(directory)
            reference.assert_called_once()
            self.assertEqual(result, json.loads((directory/"report.json").read_text()))
            self.assertEqual(result["scheduled_fits"], 4)
            self.assertEqual(result["specification_sha256"], study.design.sha256)
            counts = {k: [] for k in data["solvers"]}
            for grid in result["grids"]:
                run = grid["runs"][0]
                self.assertEqual(run["seed"], 2026)
                run_dir = directory/grid["directory"]/run["directory"]
                self.assertEqual(json.loads((run_dir/"reference.json").read_text()), result["reference"])
                for kind in data["solvers"]:
                    source = json.loads((run_dir/kind/"report.json").read_text())
                    self.assertEqual(source["config"]["n_steps"], grid["n_steps"])
                    self.assertEqual(source["completed_updates"], 2)
                    self.assertTrue(np.isfinite(run["tasks"][kind]["selected_validation_objective"]))
                    self.assertEqual(len(run["models"][kind]["scaled_delta_checks"]), 10)
                    counts[kind].append(run["models"][kind]["parameter_count"])
            self.assertEqual(counts["value_gradient"][0], counts["value_gradient"][1])
            self.assertGreater(counts["terminal_loss"][1], counts["terminal_loss"][0])
            with patch.object(study, "_reference") as reference, self.assertRaises(FileExistsError):
                study.run(directory)
            reference.assert_not_called()

    def test_grid_failures_keep_every_planned_row(self):
        data = ExperimentDesign.load().data
        data["development_seeds"] = [2026]
        data["grid_steps"] = [10, 20]
        with tempfile.TemporaryDirectory(dir="scratch/tests") as temp, \
             patch("bsde_pricing.arithmetic_pilot.ArithmeticPilot._run_worker", side_effect=WorkerLimitError("limit")):
            result = TimeGridStudy(ExperimentDesign(data)).run(Path(temp)/"failed_grid")
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["reference_usable"])
            self.assertEqual([g["n_steps"] for g in result["grids"]], [10, 20])
            self.assertTrue(all(t["status"] == "failed" for g in result["grids"]
                                for r in g["runs"] for t in r["tasks"].values()))


if __name__ == "__main__":
    unittest.main()
