"""Ledger coverage, retained worker failures and immutable resume."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bsde_pricing.arithmetic_pilot import WorkerLimitError
from bsde_pricing.final_suite import FinalSuite


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


if __name__ == "__main__":
    unittest.main()
