"""Bounded fitting, independent evaluation and inference checkpoint contracts."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bsde_pricing.analytical_torch import AnalyticalCall
from bsde_pricing.problems import vanilla_case, geometric_case
from bsde_pricing.training import FBSNNSolver, TrainingConfig, train_and_save


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(iterations=12, n_steps=4, batch_size=8, width=4,
                                     log_every=5, validation_paths=8, evaluation_paths=8)
        Path('scratch/tests').mkdir(parents=True, exist_ok=True)

    def test_fit_changes_value_preserves_rng_and_retains_missed_targets(self):
        before = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        solver = FBSNNSolver(vanilla_case(), self.config)
        point_before = solver.initial_value()['price']
        report = solver.fit()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['completed_updates'], 12)
        self.assertEqual(report['training_paths_drawn'], 96)
        self.assertNotEqual(report['evaluation']['initial']['price'], point_before)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual([h['step'] for h in report['history']], [0, 5, 10, 12])
        self.assertFalse(all(report['targets_met'].values()))
        self.assertTrue(np.isfinite(report['evaluation']['total_loss']))
        with self.assertRaises(RuntimeError): solver.fit()

    def test_inference_round_trip_and_existing_run_protection(self):
        solver = FBSNNSolver(vanilla_case(), self.config)
        report = solver.fit()
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)/'run'
            solver.save(directory, report)
            restored = FBSNNSolver.load(directory/'model.pt')
            np.testing.assert_allclose(list(solver.initial_value().values()),
                                       list(restored.initial_value().values()), atol=1e-12)
            original_eval = solver.evaluate(n_paths=8, seed=19)
            loaded_eval = restored.evaluate(n_paths=8, seed=19)
            np.testing.assert_allclose(original_eval.values, loaded_eval.values, atol=1e-12)
            np.testing.assert_allclose(original_eval.deltas, loaded_eval.deltas, atol=1e-12)
            self.assertFalse(loaded_eval.spots.flags.writeable)
            old_report = (directory/'report.json').read_bytes()
            with self.assertRaises(FileExistsError): solver.save(directory, report)
            with patch.object(FBSNNSolver, 'fit') as fit:
                with self.assertRaises(FileExistsError): train_and_save(vanilla_case(), self.config, directory)
                fit.assert_not_called()
            self.assertEqual((directory/'report.json').read_bytes(), old_report)
            payload = torch.load(directory/'model.pt', weights_only=True)
            payload['model_state']['value_scale'] += 1
            torch.save(payload, Path(temp)/'inconsistent.pt')
            with self.assertRaises(ValueError): FBSNNSolver.load(Path(temp)/'inconsistent.pt')
            with self.assertRaises(RuntimeError): restored.fit()

    def test_evaluation_stream_does_not_change_training(self):
        first = FBSNNSolver(vanilla_case(), self.config)
        second = FBSNNSolver(vanilla_case(), replace(self.config, evaluation_paths=16))
        a, b = first.fit(), second.fit()
        for left, right in zip(first.network.parameters(), second.network.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(a['history'][-1]['validation_total'], b['history'][-1]['validation_total'])
        self.assertEqual(a['evaluation']['n_paths'], 8)
        self.assertEqual(b['evaluation']['n_paths'], 16)

    def test_evaluation_metrics_against_analytical_value_map(self):
        solver = FBSNNSolver(vanilla_case(), self.config)
        solver.network = AnalyticalCall(solver.case.model, solver.case.contract)
        result = solver.evaluate(n_paths=16, seed=7)
        self.assertLess(result.summary['price_rmse_before_expiry'], 1e-12)
        self.assertLess(result.summary['delta_rmse_before_expiry'], 1e-13)
        self.assertEqual(result.summary['terminal_payoff_rmse'], 0)
        self.assertAlmostEqual(result.summary['local_loss'], result.summary['analytical_local_loss'], delta=1e-15)
        self.assertEqual(result.deltas.shape, (16, self.config.n_steps))
        self.assertEqual(result.values.shape, (16, self.config.n_steps+1))

    def test_budget_invalid_configuration_and_failure_record(self):
        report = FBSNNSolver(vanilla_case(), replace(self.config, max_seconds=1e-12)).fit()
        self.assertEqual(report['status'], 'time_budget_reached')
        self.assertEqual(report['completed_updates'], 0)
        for kwargs in ({'iterations': True}, {'n_steps': 0}, {'evaluation_paths': 1},
                       {'learning_rate': float('nan')}, {'max_seconds': -1}, {'seed': -1}):
            with self.assertRaises(ValueError): TrainingConfig(**kwargs)
        with self.assertRaises(ValueError): FBSNNSolver(geometric_case(), self.config)
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)/'failed_run'
            with patch.object(FBSNNSolver, 'fit', side_effect=ArithmeticError('nonfinite loss')):
                with self.assertRaises(ArithmeticError): train_and_save(vanilla_case(), self.config, directory)
            failure = json.loads((directory/'failure.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['error'], 'nonfinite loss')
            self.assertFalse((directory/'model.pt').exists())


if __name__ == '__main__':
    unittest.main()
