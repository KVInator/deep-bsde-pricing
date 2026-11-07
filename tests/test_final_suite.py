"""Ledger coverage, retained worker failures and immutable resume."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bsde_pricing._workers import WorkerLimitError, _child_rss_mib, _guarded_worker
from bsde_pricing.final_suite import FinalSuite, _error_assessment


class FinalSuiteTests(unittest.TestCase):
    def test_exhausted_budget_preserves_the_schedule_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            suite = FinalSuite(directory)
            suite.limit = 0
            summary = suite.run()
            self.assertEqual(summary["fit_status"], {"not_attempted_budget": 78})
            self.assertEqual(summary["reference_status"], {"exact": 2, "inconclusive": 8})
            self.assertEqual(summary["estimator_status"], {"dependency_unavailable": 48})

            ledger = json.loads((directory/"report.json").read_text())
            self.assertEqual([row["job"] for row in ledger["fits"]], suite.design.jobs())
            self.assertEqual(FinalSuite(directory).run(), summary)
            changed = FinalSuite(directory)
            changed.manifest["source_sha256"]["final_suite.py"] = "different"
            with self.assertRaisesRegex(ValueError, "original source"):
                changed.run()

    def test_guarded_worker_failure_is_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as work:
            suite = FinalSuite(Path(work))
            suite.report = {"elapsed_seconds": 0.0}
            row = {"status": "pending"}
            task = {"kind": "reference", "case": "arithmetic-10"}

            def exceed_limit(command, environment, log, directory, kind, limits):
                self.assertEqual(limits["worker_wall_seconds"], 180)
                (directory/"task_guard.json").write_text('{"status":"time_limit"}')
                raise WorkerLimitError("reference worker time limit")

            with patch("bsde_pricing.final_suite._guarded_worker", side_effect=exceed_limit) as worker, \
                    contextlib.redirect_stderr(io.StringIO()):
                suite._execute(row, Path("reference"), task, 180)
                suite._execute(row, Path("reference"), task, 180)

            self.assertEqual(row["status"], "time_limit")
            self.assertEqual(row["attempts"][0]["guard"]["status"], "time_limit")
            self.assertEqual(row["attempts"][0]["directory"], "reference/attempt-1")
            worker.assert_called_once()


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

    def test_real_workers_record_time_and_memory_limit_failures(self):
        with patch("bsde_pricing._workers.platform.system", return_value="Darwin"), \
             patch("bsde_pricing._workers.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "262144\n", "")):
            self.assertEqual(_child_rss_mib(123), 256)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, limits in (("time", {"worker_wall_seconds": 0.05, "worker_peak_resident_mib": 1e6}),
                                 ("memory", {"worker_wall_seconds": 5, "worker_peak_resident_mib": 0.001})):
                with (directory/f"{name}.log").open("w") as log, self.assertRaises(WorkerLimitError):
                    _guarded_worker([sys.executable, "-c", "import time; time.sleep(2)"], os.environ.copy(),
                                    log, directory, name, limits)
                guard = json.loads((directory/f"{name}_guard.json").read_text())
                self.assertEqual(guard["status"], f"{name}_limit")
                self.assertLess(guard["process_wall_seconds"], 2)


if __name__ == "__main__":
    unittest.main()
