"""Held-out test is scored once after global selection, on invented data only."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from loan_renewal import course, final_evaluation, pipeline
from loan_renewal.modeling import tune_random_forest
from loan_renewal.schema import NUMERIC_COLUMNS


def temporal_fixture():
    rows = 72
    frame = pd.DataFrame({name: np.arange(rows, dtype=float) + 2 for name in NUMERIC_COLUMNS})
    frame["SEXO"] = ["F", "M"] * (rows // 2)
    frame["EST_CIVIL"] = "SYNTHETIC_STATUS"
    frame["REGION"] = "SYNTHETIC_REGION"
    frame["FLAG_LIMA_PROVINCIA"] = np.arange(rows) % 2
    frame["MES"] = np.repeat(np.arange(201501, 201510), 8)
    frame["CLIENTE"] = np.arange(rows)
    frame["FLAG_VENTA"] = np.arange(rows) % 2
    return frame


def cheap_candidates(*_args):
    return [("synthetic_tree", DecisionTreeClassifier(max_depth=2, random_state=42), "none")]


class DeferredTrainingTests(unittest.TestCase):
    def test_experiment_training_never_predicts_on_test_or_reports_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "invented.csv"
            temporal_fixture().to_csv(data, sep=";", index=False)
            original = pipeline.positive_probability

            def only_validation(model, frame):
                self.assertTrue(set(frame.index) <= set(range(48, 64)))
                return original(model, frame)

            with patch.object(pipeline, "candidate_specs", side_effect=cheap_candidates), \
                    patch.object(pipeline, "positive_probability", side_effect=only_validation):
                report = pipeline.train(data, root / "model", include_cluster=False, evaluate_test=False)
            self.assertIsNone(report["test"])
            self.assertEqual(report["manifest"]["test_evaluation"], "deferred_until_global_selection")
            self.assertNotIn("positive_rate", report["manifest"]["partitions"]["test"])
            self.assertEqual(report["manifest"]["dataset_validation"]["rows"], 64)

    def test_registry_refuses_missing_or_stale_final_evaluation(self):
        selected = {"run_id": "selected", "model_uri": "models:/exact", "threshold": 0.4,
                    "dataset_sha256": "a" * 64, "protocol_id": "b" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "best_run.json").write_text(json.dumps(selected))
            with patch("loan_renewal.experimentos.registrar_modelo") as register:
                with self.assertRaisesRegex(FileNotFoundError, "evaluate before registry"):
                    course.main("registry", ["--reports", directory, "--output", str(root / "models")])
                (root / "final_evaluation.json").write_text(json.dumps({"status": "completed", "selection": {}}))
                with self.assertRaisesRegex(ValueError, "does not match"):
                    course.main("registry", ["--reports", directory, "--output", str(root / "models")])
                register.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("mlflow"), "Optional MLflow dependency required")
class GlobalWinnerIntegration(unittest.TestCase):
    def test_all_logs_three_validation_runs_and_scores_test_only_for_global_winner(self):
        import mlflow
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "invented.csv"
            temporal_fixture().to_csv(data, sep=";", index=False)
            uri = "sqlite:///" + str(root / "tracking.db")
            args = ["--data", str(data), "--output", str(root / "models"),
                    "--reports", str(root / "reports"), "--tracking-uri", uri,
                    "--experiment", "synthetic-final-evaluation", "--registry-name", "SyntheticFinalWinner"]
            scores = final_evaluation.positive_probability
            phases = []

            def final_scores(model, frame):
                phases.append("test")
                self.assertEqual(set(frame.index), set(range(64, 72)))
                self.assertTrue((root / "reports" / "best_run.json").exists())
                return scores(model, frame)

            def cheap_search(frame, include_cluster, random_state, method):
                return tune_random_forest(frame, include_cluster=False, method=method, n_iter=1,
                                         param_grid={"model__estimator__n_estimators": [3],
                                                     "model__estimator__max_depth": [2]})

            validation_scores = pipeline.positive_probability

            def no_test_during_training(model, frame):
                self.assertFalse(set(frame.index) & set(range(64, 72)))
                phases.append("validation")
                return validation_scores(model, frame)

            with patch.object(pipeline, "candidate_specs", side_effect=cheap_candidates), \
                    patch.object(pipeline, "tune_random_forest", side_effect=cheap_search), \
                    patch.object(pipeline, "positive_probability", side_effect=no_test_during_training), \
                    patch.object(final_evaluation, "positive_probability", side_effect=final_scores), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = course.main("all", args)
                self.assertEqual(phases.count("test"), 1)
                self.assertEqual(phases[-1], "test")
                self.assertTrue(result["registry"]["alias_reload_prediction_parity"])
                receipt = course.main("evaluate", args)
                self.assertEqual(receipt, result["final_evaluation"])
                self.assertEqual(phases.count("test"), 1)
                with self.assertRaises(FileExistsError):
                    course.main("all", args)
                with self.assertRaises(FileExistsError):
                    course.main("best", args)
                data.write_text(data.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "dataset differs"):
                    course.main("evaluate", args)
                self.assertEqual(phases.count("test"), 1)

            client = mlflow.tracking.MlflowClient(tracking_uri=uri)
            experiment = client.get_experiment_by_name("synthetic-final-evaluation")
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(len(runs), 3)
            selected = result["best"]["run_id"]
            for run in runs:
                self.assertIn("validation_f1", run.data.metrics)
                self.assertEqual("test_report_f1" in run.data.metrics, run.info.run_id == selected)
            for name in ("base", "grid", "random"):
                report = json.loads((root / "reports" / name / "metrics.json").read_text())
                self.assertIsNone(report["test"])
            with self.assertRaisesRegex(ValueError, "does not match"):
                final_evaluation.require_final_evaluation(dict(result["best"], threshold=0.99), receipt)


if __name__ == "__main__":
    unittest.main()
