"""Independent curvature checks and bounded stress-study execution."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from bsde_pricing.analytical_torch import AnalyticalCall
from bsde_pricing.arithmetic_pilot import WorkerLimitError
from bsde_pricing.black_scholes import BlackScholesReference
from bsde_pricing.experiment_design import ExperimentDesign
from bsde_pricing.geometric_basket import GeometricBasketReference
from bsde_pricing.greek_study import GreekDiagnostics, GreekStressStudy, scramble_statistics
from bsde_pricing.neural import value_derivatives
from bsde_pricing.problems import geometric_five_case, vanilla_case
from bsde_pricing.terminal_bsde import train_terminal_and_save
from bsde_pricing.training import PathTrainingConfig, TrainingConfig, train_and_save


class GreekStudyTests(unittest.TestCase):
    def setUp(self):
        Path("scratch/tests").mkdir(parents=True, exist_ok=True)

    def test_geometric_hessian_matches_torch_and_coordinate_delta_bumps(self):
        import torch
        case = geometric_five_case()
        reference = GeometricBasketReference(case.model, case.contract)
        expected = reference.call_hessian(case.initial_spots)
        point = torch.tensor([[0, *case.initial_spots]], dtype=torch.float64)
        actual = value_derivatives(AnalyticalCall(case.model, case.contract), point).hessian[0, 1:, 1:].numpy()
        np.testing.assert_allclose(expected, actual, rtol=0, atol=2e-17)
        for i in range(5):
            shift = np.eye(5)[i]*0.001
            bumped = (reference.call_delta_array(np.asarray(case.initial_spots)+shift)-
                      reference.call_delta_array(np.asarray(case.initial_spots)-shift))/0.002
            np.testing.assert_allclose(bumped, expected[:, i], rtol=0, atol=2e-12)
        with self.assertRaises(ValueError):
            reference.call_hessian(case.initial_spots, case.contract.maturity)
        # Negative cross terms from the geometric basket must not be discarded.
        self.assertLess(float(np.linalg.eigvalsh(expected)[0]), 0)

    def test_numerical_curvature_retains_paired_bumps_and_detects_bias(self):
        case = vanilla_case()
        diagnostic = GreekDiagnostics(case)
        options = dict(bumps=[1, 0.5, 0.1], power=6, scrambles=4, pilot_pairs=64, seed=2035, chunk_size=64)
        r = diagnostic.curvature_references(options)
        target = BlackScholesReference(case.model, case.contract).evaluate(100).gamma
        errors = [abs(x['statistics']['estimate']-target) for x in r['parallel']['bumps']]
        self.assertGreater(errors[0], errors[-1]*50)
        self.assertLess(errors[-1], 1e-7)
        # Here the geometric control equals the scalar payoff. Matched plus/minus
        # scramble estimates recover the independent analytical delta secant.
        for row in r['parallel']['bumps']:
            h = row['bump']
            oracle = BlackScholesReference(case.model, case.contract)
            secant = (oracle.evaluate(100+h).call_delta-oracle.evaluate(100-h).call_delta)/(2*h)
            self.assertAlmostEqual(row['statistics']['estimate'], secant, delta=1e-14)
        options['bumps'] = [100]
        with self.assertRaises(ValueError): diagnostic.curvature_references(options)

    def test_directional_uncertainty_uses_covariance_between_coordinates(self):
        rows = np.array([[1, -1], [2, -2], [3, -3], [4, -4]], dtype=float)
        stat = scramble_statistics(rows@np.ones(2))
        self.assertEqual(stat['standard_error'], 0)
        stat = scramble_statistics(rows@np.array([1, 0]))
        se = np.std(rows[:, 0], ddof=1)/2
        self.assertEqual(stat['standard_error'], se)
        self.assertAlmostEqual(stat['confidence_high']-stat['estimate'], t.ppf(0.975, 3)*se)
        with self.assertRaises(ValueError): scramble_statistics([1, 2, 3])

    def small_design(self):
        data = ExperimentDesign.load().data
        data['cases'] = [c for c in data['cases'] if c['name'] in ('arithmetic-10', 'geometric-five', 'arithmetic-10-short')]
        data['training'].update(iterations=2, batch_size=8, width=4, base_steps=2, log_every=1,
                                validation_paths=8, evaluation_paths=8)
        data['development_seeds'] = [2026]
        data['reference'].update(sobol_powers=[6], check_pairs=[256], pilot_pairs=64, scrambles=4, chunk_terminal_states=64)
        return ExperimentDesign(data)

    def test_small_real_study_reuses_workers_and_preserves_asymmetric_models(self):
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)
            case = geometric_five_case()
            config = PathTrainingConfig(iterations=2, batch_size=8, width=4, n_steps=2,
                                         log_every=1, validation_paths=8, evaluation_paths=8)
            train_and_save(case, TrainingConfig(**asdict(config)), directory/'value')
            train_terminal_and_save(case, config, directory/'terminal')
            study = GreekStressStudy(self.small_design())
            result = study.run(directory/'study', geometric_value=directory/'value', geometric_terminal=directory/'terminal')
            self.assertEqual(result, json.loads((directory/'study/report.json').read_text()))
            self.assertEqual(result['scheduled_fits'], 2)
            self.assertEqual(len(result['cases']), 1)
            r = result['cases'][0]
            self.assertEqual(r['greeks']['status'], 'completed')
            row = r['greeks']['result']['runs'][0]
            self.assertEqual(len(row['value_gradient_curvature']['parallel']['bump_checks']), 3)
            self.assertIsNone(row['terminal_curvature'])
            asym = result['asymmetric']['result']['runs'][0]
            self.assertEqual(asym['analytical_path_risk']['n_paths'], 8)
            self.assertEqual(len(asym['models']['terminal_loss']['initial']['delta']), 5)
            self.assertEqual(len(asym['models']['value_gradient']['scaled_delta_checks']), 5)
            self.assertEqual(asym['value_gradient_curvature']['coordinate_1']['exact_gamma_error'],
                             asym['value_gradient_curvature']['coordinate_1']['gamma']-
                             result['asymmetric']['result']['curvature_reference']['coordinate_1']['exact_gamma'])
            with self.assertRaises(FileExistsError): study.run(directory/'study')

    def test_failed_workers_keep_declared_cases_and_asymmetric_failure(self):
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp, \
             patch('bsde_pricing.arithmetic_pilot.ArithmeticPilot._run_worker', side_effect=WorkerLimitError('limit')):
            result = GreekStressStudy(self.small_design()).run(Path(temp)/'failed')
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['cases'][0]['greeks']['status'], 'failed')
            self.assertEqual(result['asymmetric']['status'], 'failed')
            self.assertEqual(len(result['cases'][0]['study']['runs']), 1)
