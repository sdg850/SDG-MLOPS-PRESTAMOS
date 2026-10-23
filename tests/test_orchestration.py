"""A failing stage must leave evidence and stop downstream work."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from loan_renewal.orchestration import run_stages


class OrchestrationTests(unittest.TestCase):
    def test_missing_data_preserves_failure_report_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("loan_renewal.orchestration.train") as training:
                with self.assertRaises(FileNotFoundError):
                    run_stages(Path(folder) / "absent.csv", folder)
                training.assert_not_called()
            result = json.loads((Path(folder) / "stage_report.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["stages"]), 1)
            self.assertEqual(result["stages"][0]["stage"], "ingestion")
