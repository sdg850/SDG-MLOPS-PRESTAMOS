"""Exercise real training/persistence with synthetic data and separate outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
from sklearn.tree import DecisionTreeClassifier

from loan_renewal.orchestration import run_stages
from loan_renewal.pipeline import train
from loan_renewal.predict import predict_frame
from loan_renewal.schema import FEATURE_COLUMNS
from test_contracts import synthetic_frame


class CookiecutterPersistenceTests(unittest.TestCase):
    def test_training_keeps_raw_source_model_and_reports_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw" / "invented.csv"
            raw.parent.mkdir(parents=True)
            frame = synthetic_frame()
            frame.to_csv(raw, sep=";", index=False)
            original_bytes = raw.read_bytes()
            models = root / "models"
            reports = root / "reports" / "training"
            specifications = [("synthetic_tree", DecisionTreeClassifier(max_depth=2, random_state=42), "none")]
            with patch("loan_renewal.pipeline.candidate_specs", return_value=specifications):
                report = train(raw, models, include_cluster=False, report_dir=reports)
            self.assertEqual(raw.read_bytes(), original_bytes)
            self.assertEqual(report["manifest"]["dataset_sha256"], hashlib.sha256(original_bytes).hexdigest())
            self.assertEqual({path.name for path in models.iterdir()}, {"model.joblib"})
            self.assertEqual({path.name for path in reports.iterdir()}, {"metrics.json", "manifest.json", "comparison.csv"})
            self.assertEqual(Path(report["model_path"]), models / "model.joblib")
            self.assertEqual(Path(report["metrics_path"]), reports / "metrics.json")
            self.assertEqual(Path(report["comparison_path"]), reports / "comparison.csv")
            metrics = json.loads((reports / "metrics.json").read_text())
            self.assertTrue(metrics["reload_prediction_parity"])
            bundle = joblib.load(models / "model.joblib")
            predictions = predict_frame(bundle, frame.loc[:, FEATURE_COLUMNS].iloc[-2:])
            self.assertEqual(len(predictions), 2)
            self.assertEqual({path.name for path in raw.parent.iterdir()}, {"invented.csv"})

    def test_stage_failure_writes_evidence_to_reports_without_touching_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models, reports = root / "models", root / "reports" / "training"
            missing = root / "data" / "raw" / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                run_stages(missing, models, report_dir=reports)
            evidence = json.loads((reports / "stage_report.json").read_text())
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["stages"][0]["stage"], "ingestion")
            self.assertEqual(evidence["stages"][0]["error_type"], "FileNotFoundError")
            self.assertEqual(list(models.iterdir()), [])
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
