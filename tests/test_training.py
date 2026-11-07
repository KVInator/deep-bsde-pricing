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
from bsde_pricing.basket_reference import BasketReferencePricer
from bsde_pricing.problems import arithmetic_ten_case, correlated_case, vanilla_case
from bsde_pricing.terminal_bsde import TerminalBSDESolver
from bsde_pricing.training import FBSNNSolver, PathTrainingConfig, TrainingConfig, train_and_save


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
        self.assertIsNone(FBSNNSolver(correlated_case(), self.config).reference)
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)/'failed_run'
            with patch.object(FBSNNSolver, 'fit', side_effect=ArithmeticError('nonfinite loss')):
                with self.assertRaises(ArithmeticError): train_and_save(vanilla_case(), self.config, directory)
            failure = json.loads((directory/'failure.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertEqual(failure['error'], 'nonfinite loss')
            self.assertFalse((directory/'model.pt').exists())

    def test_resume_matches_uninterrupted_across_schedule_and_log_boundaries(self):
        full = FBSNNSolver(vanilla_case(), self.config)
        full_report = full.fit()
        partial = FBSNNSolver(vanilla_case(), self.config)
        paused = partial.fit(stop_after=7)
        self.assertIsNone(paused['evaluation'])
        self.assertEqual([h['step'] for h in paused['history']], [0, 5])
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)
            partial.save(directory/'pause', paused)
            resumed = FBSNNSolver.resume(directory/'pause'/'resume.pt')
            before = torch.get_rng_state().clone()
            resumed_report = resumed.fit()
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            resumed.save(directory/'resumed', resumed_report)
            full.save(directory/'full', full_report)
            a = torch.load(directory/'full'/'resume.pt', weights_only=True)
            b = torch.load(directory/'resumed'/'resume.pt', weights_only=True)
            for key in a['model_state']:
                self.assertTrue(torch.equal(a['model_state'][key], b['model_state'][key]))
            left, right = a['training_state'], b['training_state']
            self.assertEqual(left['train_rng_state'], right['train_rng_state'])
            self.assertEqual(left['window'], right['window'])
            for pid, fields in left['optimizer_state']['state'].items():
                for name, value in fields.items():
                    self.assertTrue(torch.equal(value, right['optimizer_state']['state'][pid][name]))
            histories = [[{k: v for k, v in h.items() if k != 'elapsed_seconds'} for h in r['history']]
                         for r in (full_report, resumed_report)]
            self.assertEqual(histories[0], histories[1])
            self.assertEqual(full_report['evaluation'], resumed_report['evaluation'])
            self.assertEqual(full_report['last_evaluation'], resumed_report['last_evaluation'])
            with self.assertRaises(ValueError): FBSNNSolver.resume(directory/'full'/'resume.pt')
            with self.assertRaises(ValueError): FBSNNSolver.resume(directory/'full'/'model.pt')

    def test_selection_uses_validation_objective_and_optional_patience(self):
        solver = FBSNNSolver(vanilla_case(), replace(self.config, terminal_weight=5))
        report = solver.fit()
        best = min(report['history'], key=lambda h: h['validation_total'])
        self.assertEqual(report['selected_step'], best['step'])
        self.assertEqual(report['selected_validation_total'], best['validation_total'])
        self.assertAlmostEqual(report['evaluation']['total_loss'],
                               report['evaluation']['local_loss']+5*report['evaluation']['terminal_loss'])
        stopped = FBSNNSolver(vanilla_case(), replace(self.config, patience=1))
        original = stopped.loss.forward
        def constant_validation(*args, **kwargs):
            result = original(*args, **kwargs)
            return replace(result, total=result.total*0+1) if kwargs.get('parameter_graph') is False else result
        with patch.object(stopped.loss, 'forward', side_effect=constant_validation):
            result = stopped.fit()
        self.assertEqual(result['status'], 'early_stopped')
        self.assertEqual(result['completed_updates'], 5)
        self.assertEqual(result['selected_step'], 0)

    def test_legacy_checkpoint_and_invalid_continuation_controls(self):
        solver = FBSNNSolver(vanilla_case(), self.config)
        legacy_config = {k: v for k, v in self.config.__dict__.items() if k not in ('terminal_weight', 'patience')}
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            path = Path(temp)/'legacy.pt'
            torch.save({'format_version': 1, 'problem': solver.case.to_json(),
                        'config': legacy_config, 'model_state': solver.network.state_dict()}, path)
            restored = FBSNNSolver.load(path)
            self.assertEqual(restored.initial_value(), solver.initial_value())
            with self.assertRaises(ValueError): FBSNNSolver.resume(path)
        for kwargs in ({'terminal_weight': 0}, {'terminal_weight': float('nan')},
                       {'patience': 0}, {'patience': True}):
            with self.assertRaises(ValueError): TrainingConfig(**kwargs)
        for count in (0, True, 13):
            with self.assertRaises(ValueError): solver.fit(stop_after=count)

    def test_reference_free_fits_selection_and_round_trip(self):
        case = arithmetic_ten_case()
        config = PathTrainingConfig(iterations=4, n_steps=4, batch_size=8, width=4,
                                    log_every=2, validation_paths=8, evaluation_paths=8)
        value = FBSNNSolver(case, TrainingConfig(**vars(config)))
        terminal = TerminalBSDESolver(case, config)
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


if __name__ == '__main__':
    unittest.main()
