"""Terminal-BSDE propagation, adapted controls, independent errors and saving."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bsde_pricing.problems import correlated_case, vanilla_case
from bsde_pricing.terminal_bsde import TerminalBSDESolver, train_terminal_and_save
from bsde_pricing.training import PathTrainingConfig, TrainingConfig


class TerminalBSDETests(unittest.TestCase):
    def setUp(self):
        self.config = PathTrainingConfig(iterations=20, n_steps=3, batch_size=8, width=4,
                                         log_every=5, validation_paths=16, evaluation_paths=16)
        Path('scratch/tests').mkdir(parents=True, exist_ok=True)

    def test_hand_worked_discounted_propagation_loss_and_scalar_gradient(self):
        solver = TerminalBSDESolver(vanilla_case(), replace(self.config, n_steps=2))
        with torch.no_grad():
            for p in solver.network.parameters(): p.zero_()
            solver.network.initial_value.fill_(0.1)
            solver.network.initial_control.fill_(0.2)
            solver.network.layers[0][-1].bias.fill_(0.3)
        states = torch.tensor([[[100.], [105.], [110.]], [[100.], [95.], [90.]]], dtype=torch.float64)
        increments = torch.tensor([[[0.1], [-0.2]], [[-0.1], [0.2]]], dtype=torch.float64)
        values, controls = solver.network(states, increments)
        torch.testing.assert_close(controls, torch.tensor([[[20.], [30.]], [[20.], [30.]]], dtype=torch.float64))
        torch.testing.assert_close(values, torch.tensor([[10., 12., 6.], [10., 8., 14.]], dtype=torch.float64))
        errors = np.array([6.-math.exp(-0.03)*10, 14.])
        loss = solver._loss(states, increments)
        self.assertAlmostEqual(float(loss.detach()), float(np.mean(errors**2)/10000), places=15)
        loss.backward()
        self.assertAlmostEqual(float(solver.network.initial_value.grad), 2*float(errors.mean())/100, places=14)
        self.assertIsNotNone(solver.network.initial_control.grad)

    def test_controls_are_adapted_and_do_not_couple_paths(self):
        solver = TerminalBSDESolver(vanilla_case(), self.config)
        states, increments = solver._batch(np.random.default_rng(17), 8)
        with torch.no_grad():
            values, controls = solver.network(states, increments)
            changed = states.clone()
            changed[:, 2:] *= 2
            future_values, future_controls = solver.network(changed, increments)
            torch.testing.assert_close(controls[:, :2], future_controls[:, :2], rtol=0, atol=0)
            torch.testing.assert_close(values[:, :3], future_values[:, :3], rtol=0, atol=0)
            other_paths = states.clone()
            other_paths[1:] *= 3
            _, changed_controls = solver.network(other_paths, increments)
            torch.testing.assert_close(controls[0], changed_controls[0], rtol=0, atol=0)
        sample = solver.evaluate(n_paths=8192, seed=19).summary
        self.assertLess(abs(sample['stochastic_integral_mean']), 5*sample['stochastic_integral_mean_se'])

    def test_fit_selection_and_evaluation_stream_ownership(self):
        before, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        first = TerminalBSDESolver(vanilla_case(), self.config)
        report = first.fit()
        second = TerminalBSDESolver(vanilla_case(), replace(self.config, evaluation_paths=32))
        other = second.fit()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['training_paths_drawn'], 160)
        self.assertNotEqual(report['evaluation']['initial']['price'], 0)
        self.assertEqual(report['selected_step'], min(report['history'], key=lambda h: h['validation_terminal'])['step'])
        self.assertEqual(report['selected_step'], other['selected_step'])
        for a, b in zip(first.network.parameters(), second.network.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(RuntimeError): first.fit()
        n = 16
        self.assertEqual(report['parameter_count'], 2+(self.config.n_steps-1)*(self.config.width**2+4*self.config.width+1))
        self.assertEqual(report['evaluation']['n_paths'], n)

    def test_report_metrics_and_checkpoint_round_trip(self):
        solver = TerminalBSDESolver(vanilla_case(), self.config)
        report = solver.fit()
        result = solver.evaluate(n_paths=32, seed=7)
        errors = result.discounted_values[:, -1]-result.discounted_payoffs
        np.testing.assert_array_equal(errors, result.terminal_residuals)
        self.assertAlmostEqual(result.summary['normalized_terminal_loss'], float(np.mean(errors**2)/10000), places=15)
        self.assertAlmostEqual(result.summary['terminal_residual_mean_se'], float(errors.std(ddof=1)/np.sqrt(32)), places=15)
        self.assertFalse(result.discounted_values.flags.writeable)
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)/'run'
            with self.assertRaises(ValueError): solver.save(directory, {**report, 'config': {}})
            self.assertFalse(directory.exists())
            solver.save(directory, report)
            restored = TerminalBSDESolver.load(directory/'model.pt')
            self.assertEqual(solver.initial_value(), restored.initial_value())
            np.testing.assert_array_equal(result.discounted_values, restored.evaluate(n_paths=32, seed=7).discounted_values)
            last = TerminalBSDESolver.load(directory/'last_model.pt')
            self.assertEqual(last.initial_value(), report['last_evaluation']['initial'])
            with patch.object(TerminalBSDESolver, 'fit') as fit:
                with self.assertRaises(FileExistsError): train_terminal_and_save(vanilla_case(), self.config, directory)
                fit.assert_not_called()
            with self.assertRaises(FileExistsError): solver.save(directory, report)
            payload = torch.load(directory/'model.pt', weights_only=True)
            payload['model_state']['control_scale'] += 1
            torch.save(payload, Path(temp)/'bad.pt')
            with self.assertRaises(ValueError): TerminalBSDESolver.load(Path(temp)/'bad.pt')

    def test_scope_limits_invalid_paths_time_cap_and_failure_record(self):
        with self.assertRaises(ValueError): TerminalBSDESolver(correlated_case(), self.config)
        with self.assertRaises(ValueError): TerminalBSDESolver(vanilla_case(), TrainingConfig())
        solver = TerminalBSDESolver(vanilla_case(), self.config)
        states, increments = solver._batch(np.random.default_rng(1), 8)
        for bad in (states.float(), states[:, :-1], states*0, states.clone().requires_grad_()):
            with self.assertRaises(ValueError): solver.network(bad, increments)
        with self.assertRaises(ValueError): solver.network(states, increments[:, :-1])
        with self.assertRaises(ValueError): solver.evaluate(n_paths=1)
        capped = TerminalBSDESolver(vanilla_case(), replace(self.config, max_seconds=1e-12)).fit()
        self.assertEqual(capped['status'], 'time_budget_reached')
        self.assertEqual(capped['completed_updates'], 0)
        one_step = TerminalBSDESolver(vanilla_case(), replace(self.config, n_steps=1))
        self.assertEqual(sum(p.numel() for p in one_step.network.parameters()), 2)
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)/'failed'
            with patch.object(TerminalBSDESolver, 'fit', side_effect=ArithmeticError('nonfinite')):
                with self.assertRaises(ArithmeticError): train_terminal_and_save(vanilla_case(), self.config, directory)
            self.assertEqual(json.loads((directory/'failure.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
