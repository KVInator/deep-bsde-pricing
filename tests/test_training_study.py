"""Paired study persistence, failed runs and cross-weight summary contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bsde_pricing.problems import vanilla_case
from bsde_pricing.training import TrainingConfig
from bsde_pricing.training_study import TrainingStudy


class TrainingStudyTests(unittest.TestCase):
    def test_full_matrix_keeps_failure_and_preserves_existing_results(self):
        config = TrainingConfig(iterations=3, n_steps=2, batch_size=4, width=4,
                                log_every=2, validation_paths=4, evaluation_paths=4)
        Path('scratch/tests').mkdir(parents=True, exist_ok=True)
        from bsde_pricing.training import train_and_save
        def fail_one(case, config, directory):
            if config.seed == 2027 and config.terminal_weight == 5:
                raise ArithmeticError('deliberate failed run')
            return train_and_save(case, config, directory)
        with tempfile.TemporaryDirectory(dir='scratch/tests') as temp:
            directory = Path(temp)/'study'
            with patch('bsde_pricing.training_study.train_and_save', side_effect=fail_one):
                report = TrainingStudy(vanilla_case(), config).run(directory)
            self.assertEqual(report['status'], 'incomplete')
            self.assertTrue(report['resume_check']['passed'])
            self.assertEqual(len(report['runs']), 6)
            failed = [r for r in report['runs'] if r['status'] == 'failed']
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]['seed'], 2027)
            summary = [r for r in report['summary'] if r['terminal_weight'] == 5][0]
            self.assertEqual((summary['runs'], summary['usable_runs'], summary['failures']), (3, 2, 1))
            saved = json.loads((directory/'study.json').read_text())
            self.assertEqual(len(saved['runs']), 6)
            self.assertTrue((directory/'design.json').exists())
            with self.assertRaises(FileExistsError): TrainingStudy(vanilla_case(), config).run(directory)


if __name__ == '__main__':
    unittest.main()
