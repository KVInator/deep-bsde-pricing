"""Declared workloads, financial cases and reserved-stream separation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from bsde_pricing.experiment_design import ExperimentDesign


class ExperimentDesignTests(unittest.TestCase):
    def test_enumeration_does_not_sample_train_or_mutate_the_specification(self):
        state = torch.get_rng_state().clone()
        with patch("numpy.random.default_rng", side_effect=AssertionError("No final sampling")), \
             patch("bsde_pricing.training.FBSNNSolver.fit", side_effect=AssertionError("No final fitting")):
            design = ExperimentDesign.load()
            summary = design.summary()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(summary["final_samples_drawn"], 0)
        jobs = design.jobs()
        self.assertEqual(len(jobs), len({tuple(row.values()) for row in jobs}))
        self.assertEqual(len(jobs), 78)
        external = design.data
        external["training"]["width"] = 999
        self.assertEqual(design.data["training"]["width"], 16)
        self.assertEqual(design.sha256, ExperimentDesign(design.data).sha256)

    def test_case_inputs_and_grid_comparisons(self):
        design = ExperimentDesign.load()
        case = design.case("arithmetic-100")
        self.assertEqual(case.model.dimension, 100)
        self.assertAlmostEqual(sum(case.contract.weights), 1)
        np.testing.assert_allclose(np.linalg.eigvalsh(case.model.correlation), [0.7]*99+[30.7], atol=2e-14)
        self.assertEqual(design.case("arithmetic-10-strike-110").contract.strike, 110)
        self.assertEqual(design.case("arithmetic-10-short").contract.maturity, 0.25)
        grids = {r["n_steps"] for r in design.jobs() if r["case"] == "arithmetic-10"}
        self.assertEqual(grids, {10, 16, 20, 40})
        with self.assertRaises(ValueError):
            design.case("unknown")

    def test_invalid_streams_cases_and_budget_contracts(self):
        original = ExperimentDesign.load()
        changes = [lambda d: d["final_training_seeds"].__setitem__(0, 2026),
                   lambda d: d["cases"][0].__setitem__("strike", 200),
                   lambda d: d["final_streams"].__setitem__("reference_seed_base", 41001),
                   lambda d: d["cases"].append(d["cases"][0]),
                   lambda d: d["cases"][0].__setitem__("dimension", 2),
                   lambda d: d["resources"].__setitem__("worker_wall_seconds", 100),
                   lambda d: d["tuning"].__setitem__("final_accuracy_selects_checkpoint", True),
                   lambda d: d["reference"]["sobol_powers"].__setitem__(0, -1),
                   lambda d: d["targets"].pop("price_absolute")]
        for change in changes:
            data = original.data
            change(data)
            with self.assertRaises(ValueError):
                ExperimentDesign(data)


if __name__ == "__main__":
    unittest.main()
