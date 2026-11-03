"""Control inversion, physical Greeks and saved-model comparison contracts."""

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bsde_pricing.analytical_torch import AnalyticalCall
from bsde_pricing.bsde import PricingBSDE
from bsde_pricing.problems import GBMModel, EuropeanCall, vanilla_case
from bsde_pricing.risk_comparison import RiskComparisonResult, SolverRiskComparison, compare_saved_runs
from bsde_pricing.terminal_bsde import TerminalBSDESolver
from bsde_pricing.training import FBSNNSolver, PathTrainingConfig, TrainingConfig


class ControlRecoveryTests(unittest.TestCase):
    def test_correlated_hand_worked_orientation_and_batched_recovery(self):
        model = GBMModel(0.03, (0.2, 0.3), (0, 0), ((1, 0.6), (0.6, 1)))
        problem = PricingBSDE(model, EuropeanCall(100, 1, (0.5, 0.5)))
        result = problem.recover_deltas([100, 80], [18.08, 13.44])
        np.testing.assert_allclose(result.deltas, [0.4, 0.7], atol=2e-15, rtol=0)
        self.assertLess(float(result.absolute_residual), 1e-14)
        spots = np.array([[[100, 80], [95, 90]], [[110, 85], [130, 100]]], dtype=float)
        deltas = np.array([[[0.4, 0.7], [0.2, 0.3]], [[0.8, 0.1], [0, 0]]])
        recovered = problem.recover_deltas(spots, problem.control(spots, deltas))
        np.testing.assert_allclose(recovered.deltas, deltas, atol=3e-16, rtol=0)
        self.assertEqual(recovered.relative_residual[-1, -1], 0)
        self.assertFalse(recovered.deltas.flags.writeable)

    def test_scalar_condition_number_does_not_hide_inverse_amplification(self):
        case = vanilla_case()
        problem = PricingBSDE(replace(case.model, volatilities=(1e-10,)), case.contract)
        a = problem.recover_deltas([100], [1.0])
        b = problem.recover_deltas([100], [1.0+1e-8])
        self.assertEqual(float(a.condition_number), 1)
        self.assertAlmostEqual(float(a.inverse_norm), 1e8)
        self.assertAlmostEqual(float(b.deltas[0]-a.deltas[0]), 1, places=5)

    def test_near_singular_correlation_reports_condition_without_repair(self):
        rho = 0.999999
        model = GBMModel(0.03, (0.2, 0.2), (0, 0), ((1, rho), (rho, 1)))
        problem = PricingBSDE(model, EuropeanCall(100, 1, (0.5, 0.5)))
        result = problem.recover_deltas([100, 100], problem.control([100, 100], [0.3, 0.2]))
        self.assertAlmostEqual(float(result.condition_number), math.sqrt((1+rho)/(1-rho)), delta=1e-7)
        self.assertAlmostEqual(float(result.inverse_norm), 1/(20*math.sqrt(1-rho)), delta=1e-8)
        np.testing.assert_allclose(result.deltas, [0.3, 0.2], atol=1e-12)

    def test_invalid_recovery_contract_and_singular_diffusion(self):
        case = vanilla_case();problem = PricingBSDE(case.model, case.contract)
        for spots, controls in (([0], [1]), ([100], [[1]]), ([100], [float('nan')])):
            with self.assertRaises(ValueError): problem.recover_deltas(spots, controls)
        with self.assertRaises(ValueError):
            PricingBSDE(replace(case.model, volatilities=(0,)), case.contract).recover_deltas([100], [0])


class SolverRiskComparisonTests(unittest.TestCase):
    def setUp(self):
        self.config = PathTrainingConfig(iterations=8, n_steps=4, batch_size=8, width=4,
                                         log_every=4, validation_paths=8, evaluation_paths=8)
        self.value = FBSNNSolver(vanilla_case(), TrainingConfig(**self.config.__dict__))
        self.terminal = TerminalBSDESolver(vanilla_case(), self.config)
        Path('scratch/tests').mkdir(parents=True, exist_ok=True)

    def test_analytical_price_delta_gamma_and_control_recovery(self):
        case = vanilla_case();target = self.value.reference.evaluate(100)
        self.value.network = AnalyticalCall(case.model, case.contract)
        with torch.no_grad():
            self.terminal.network.initial_value.fill_(target.call_price/100)
            self.terminal.network.initial_control.fill_(20*target.call_delta/100)
        before, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        result = SolverRiskComparison(self.value, self.terminal).run(n_paths=32, seed=18)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads)
        for kind in ('value_gradient', 'terminal_loss'):
            initial = result.report['models'][kind]['initial']
            self.assertLess(abs(initial['price_error']), 2e-14)
            self.assertLess(abs(initial['delta_error']), 3e-16)
            self.assertLess(initial['reconstruction']['max_absolute_reconstruction_residual'], 2e-14)
        # Independent normal-CDF/derivative expressions need a roundoff allowance
        # at all sampled states; this is far below the financial error target.
        np.testing.assert_allclose(result.value_deltas, result.reference_deltas, rtol=0, atol=2e-14)
        self.assertEqual(result.report['models']['value_gradient']['delta_bound_violation_count'], 0)
        gamma = result.report['value_gradient_gamma']
        self.assertLess(abs(gamma['signed_error']), 1e-16)
        self.assertLess(abs(gamma['bump_checks'][-1]['error_against_autograd']),
                        abs(gamma['bump_checks'][0]['error_against_autograd']))
        self.assertIsNone(result.report['terminal_gamma'])

    def test_matched_paths_discount_unwinding_and_archive_round_trip(self):
        result = SolverRiskComparison(self.value, self.terminal).run(n_paths=16, seed=29)
        states = torch.from_numpy(result.spots.copy()[..., None])
        with torch.no_grad(): discounted = self.terminal.network.discounted_controls(states).numpy()[:, :, 0]
        expected = discounted*np.exp(0.03*result.times[None, :-1])/(0.2*result.spots[:, :-1])
        np.testing.assert_allclose(result.terminal_deltas, expected, atol=2e-15, rtol=2e-15)
        self.assertEqual(result.terminal_deltas.shape, result.value_deltas.shape)
        for kind, deltas in (('value_gradient', result.value_deltas), ('terminal_loss', result.terminal_deltas)):
            expected_rmse = np.sqrt(np.mean((deltas-result.reference_deltas)**2))
            self.assertEqual(result.report['models'][kind]['path_delta_rmse'], expected_rmse)
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)/'comparison';result.save(directory)
            restored = RiskComparisonResult.load(directory)
            np.testing.assert_array_equal(result.terminal_deltas, restored.terminal_deltas)
            self.assertFalse(restored.terminal_deltas.flags.writeable)
            with self.assertRaises(FileExistsError): result.save(directory)
        mismatch = TerminalBSDESolver(replace(vanilla_case(), initial_spots=(101,)), self.config)
        with self.assertRaises(ValueError): SolverRiskComparison(self.value, mismatch)
        with self.assertRaises(ValueError): SolverRiskComparison(self.value, TerminalBSDESolver(vanilla_case(), replace(self.config, n_steps=5)))

    def test_saved_run_provenance_and_no_retraining_or_overwrite(self):
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp);a,b,out = directory/'value',directory/'terminal',directory/'comparison'
            self.value.save(a, self.value.fit());self.terminal.save(b, self.terminal.fit())
            with patch.object(FBSNNSolver, 'fit') as value_fit, patch.object(TerminalBSDESolver, 'fit') as terminal_fit:
                result = compare_saved_runs(a, b, out, n_paths=8, seed=22)
                value_fit.assert_not_called();terminal_fit.assert_not_called()
            self.assertEqual(result.report['source_runs']['terminal_loss']['completed_updates'], 8)
            self.assertEqual(len(result.report['source_runs']['value_gradient']['model_sha256']), 64)
            with patch.object(FBSNNSolver, 'load') as load:
                with self.assertRaises(FileExistsError): compare_saved_runs(a, b, out)
                load.assert_not_called()
            data = json.loads((a/'report.json').read_text());data['evaluation']['initial']['price'] += 1
            (a/'report.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError): compare_saved_runs(a,b,directory/'bad',n_paths=8)


if __name__ == '__main__':
    unittest.main()
