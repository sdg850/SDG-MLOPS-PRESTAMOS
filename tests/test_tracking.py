"""Tracking contracts, validation-only ranking and registry lineage checks."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from loan_renewal.experimentos import compare_runs, registrar_modelo
from loan_renewal.modeling import build_model, tune_random_forest
from loan_renewal.tracking import (
    protocol_id, resolve_tracking_uri, synthetic_probe, track_report,
)


def make_artifact(root):
    frame = pd.concat([synthetic_probe()] * 4, ignore_index=True)
    labels = np.tile([0, 1], 4)
    pipeline = build_model(DecisionTreeClassifier(random_state=42), include_cluster=False)
    pipeline.fit(frame, labels)
    manifest = {
        "task": "commercial_loan_renewal_propensity_educational",
        "dataset_sha256": "d" * 64,
        "source_sha256": {"pipeline.py": "a" * 64},
        "target": "FLAG_VENTA", "positive_class": 1,
        "raw_features": list(frame.columns), "random_state": 42,
        "partitions": {"train": {"months": [201501, 201502, 201503, 201504, 201505, 201506]}, "validation": {"months": [201507, 201508]}, "test": {"months": [201509]}},
        "selection": {"partition": "validation", "primary_metric": "f1_class_1", "threshold_grid": [0.25, 0.5, 0.75], "refit_on_validation": False},
        "sampling": "none", "include_cluster": False,
        "versions": {},
    }
    path = root / "model.joblib"
    joblib.dump({"pipeline": pipeline, "threshold": 0.55, "manifest": manifest}, path)
    report = {
        "model_path": str(path), "manifest": manifest,
        "winner": "synthetic_tree", "threshold": 0.55,
        "validation": {"f1": 0.7, "average_precision": 0.8},
        "test": {"f1": 0.99, "average_precision": 0.99},
        "candidates": [],
    }
    return pipeline, report


def fake_mlflow(pipeline):
    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="7", lifecycle_stage="active")
    module = SimpleNamespace(
        active_run=MagicMock(return_value=None),
        get_tracking_uri=MagicMock(return_value="sqlite:////tmp/previous.db"),
        get_registry_uri=MagicMock(return_value="sqlite:////tmp/previous.db"),
        set_tracking_uri=MagicMock(), set_registry_uri=MagicMock(),
        tracking=SimpleNamespace(MlflowClient=MagicMock(return_value=client)),
        start_run=MagicMock(), log_params=MagicMock(), log_metrics=MagicMock(),
        log_dict=MagicMock(), log_artifact=MagicMock(), set_tags=MagicMock(),
        models=SimpleNamespace(infer_signature=MagicMock(return_value="signature"), get_model_info=MagicMock()),
        exceptions=SimpleNamespace(MlflowException=RuntimeError),
        register_model=MagicMock(return_value=SimpleNamespace(version="2")),
    )
    module.start_run.return_value.__enter__.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-proof"))
    capture = {}

    def log_model(sk_model, name=None, **kwargs):
        capture.update({"sk_model": sk_model, "name": name, **kwargs})
        return SimpleNamespace(model_uri="models:/m-exact-returned-id")

    module.sklearn = SimpleNamespace(log_model=log_model, load_model=MagicMock(return_value=pipeline))
    module.capture = capture
    return module, client


def fake_run(identifier, f1, test_f1, dataset="d" * 64, protocol="a" * 64, status="FINISHED"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=identifier, status=status, start_time=1),
        data=SimpleNamespace(
            tags={"dataset_sha256": dataset, "protocol_id": protocol, "selection_partition": "validation", "model_uri": f"models:/m-{identifier}"},
            metrics={"validation_f1": f1, "validation_average_precision": 0.2, "test_report_f1": test_f1},
            params={"threshold": "0.55"},
        ),
    )


class TrackingContracts(unittest.TestCase):
    def test_absolute_sqlite_only_and_protocol_stable_under_test_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            uri = resolve_tracking_uri(output_dir=directory)
            self.assertEqual(uri, "sqlite:///" + str(Path(directory).resolve() / "mlflow.db"))
            for invalid in ("file:///tmp/mlruns", "sqlite:///relative.db", "http://example.org:5000"):
                with self.assertRaises(ValueError):
                    resolve_tracking_uri(invalid)
            _, report = make_artifact(Path(directory))
            original = protocol_id(report["manifest"])
            changed = dict(report["manifest"], tuning={"method": "RandomizedSearchCV"}, test_f1=0.1)
            self.assertEqual(protocol_id(changed), original)
            changed = dict(report["manifest"], random_state=1)
            self.assertNotEqual(protocol_id(changed), original)

    def test_tracker_preserves_exact_model_uri_threshold_and_prefixed_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline, report = make_artifact(root)
            module, _ = fake_mlflow(pipeline)
            with patch("loan_renewal.tracking.require_mlflow", return_value=module):
                result = track_report(report, root)
            self.assertEqual(result["model_uri"], "models:/m-exact-returned-id")
            self.assertEqual(result["run_id"], "run-proof")
            self.assertEqual(result["dataset_sha256"], "d" * 64)
            self.assertTrue(result["reload_prediction_parity"])
            self.assertEqual(module.capture["name"], "loan_pipeline")
            self.assertEqual(module.capture["serialization_format"], "cloudpickle")
            self.assertEqual(module.capture["metadata"]["threshold"], 0.55)
            self.assertEqual(module.capture["pyfunc_predict_fn"], "predict_proba")
            metrics = module.log_metrics.call_args.args[0]
            self.assertEqual(metrics["validation_f1"], 0.7)
            self.assertEqual(metrics["test_report_f1"], 0.99)
            self.assertNotIn("f1", metrics)
            self.assertEqual(json.loads((root / "tracking.json").read_text())["model_uri"], result["model_uri"])
            self.assertNotIn("CLIENTE", module.capture["input_example"].columns)
            self.assertTrue((module.capture["input_example"]["SEXO"] == "__synthetic_tracking_probe__").all())

    def test_comparison_excludes_failed_incompatible_missing_and_nan_runs(self):
        module, client = fake_mlflow(None)
        client.search_runs.return_value = [
            fake_run("validation-winner", 0.8, 0.1),
            fake_run("test-winner", 0.2, 0.99),
            fake_run("failed", 0.99, 0.99, status="FAILED"),
            fake_run("other-data", 0.99, 0.99, dataset="e" * 64),
            fake_run("other-protocol", 0.99, 0.99, protocol="b" * 64),
            fake_run("missing-score", None, 0.99),
            fake_run("nan-score", math.nan, 0.99),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch("loan_renewal.tracking.require_mlflow", return_value=module):
                rows = compare_runs(resolve_tracking_uri(output_dir=directory), dataset_sha256="d" * 64, protocol_id="a" * 64)
        self.assertEqual([row["run_id"] for row in rows], ["validation-winner", "test-winner"])
        self.assertNotIn("test_report_f1", client.search_runs.call_args.kwargs["order_by"][0])
        with self.assertRaises(ValueError):
            compare_runs("sqlite:////tmp/unused.db", dataset_sha256="d" * 64)

    def test_registry_keeps_alias_previous_version_and_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline, _ = make_artifact(Path(directory))
            module, client = fake_mlflow(pipeline)
            module.models.get_model_info.return_value = SimpleNamespace(metadata={"selection_partition": "validation", "threshold": 0.55, "dataset_sha256": "d" * 64, "protocol_id": "a" * 64})
            client.get_registered_model.return_value = SimpleNamespace(aliases={"candidato": "1"})
            client.get_model_version_by_alias.return_value = SimpleNamespace(version="2")
            with patch("loan_renewal.tracking.require_mlflow", return_value=module):
                result = registrar_modelo("models:/m-exact-returned-id", "loan-test", tracking_uri=resolve_tracking_uri(output_dir=directory), threshold=0.55)
            self.assertEqual(result["previous_version"], "1")
            self.assertEqual(result["threshold"], 0.55)
            self.assertEqual(result["model_uri"], "models:/loan-test@candidato")
            client.set_registered_model_alias.assert_called_once_with("loan-test", "candidato", "2")
            tags = {call.args[2]: call.args[3] for call in client.set_model_version_tag.call_args_list}
            self.assertEqual(tags["threshold"], "0.55")
            self.assertEqual(tags["previous_alias_version"], "1")
            self.assertEqual(tags["registration_is_not_deployment"], "true")
            with patch("loan_renewal.tracking.require_mlflow", return_value=module):
                with self.assertRaises(ValueError):
                    registrar_modelo("models:/m-exact-returned-id", "loan-test", tracking_uri=resolve_tracking_uri(output_dir=directory), threshold=0.9)


class RandomizedTemporalSearchTests(unittest.TestCase):
    def test_seeded_random_search_uses_train_only_expanding_folds(self):
        raw = pd.concat([synthetic_probe()] * 24, ignore_index=True)
        raw["MES"] = np.repeat(np.arange(201501, 201507), 8)
        raw["FLAG_VENTA"] = np.tile([0, 1], 24)
        raw["CLIENTE"] = np.arange(len(raw))
        raw = raw.sample(frac=1, random_state=8).reset_index(drop=True)
        model, summary = tune_random_forest(
            raw, include_cluster=False,
            param_grid={"model__estimator__n_estimators": [3], "model__estimator__max_depth": [2]},
            method="random", n_iter=1,
        )
        self.assertEqual(summary["method"], "RandomizedSearchCV")
        self.assertEqual(summary["candidate"], "forest_random_under")
        self.assertEqual(summary["cv_fit_count"], 3)
        self.assertEqual(summary["refit_count"], 1)
        self.assertEqual([fold["train_rows"] for fold in summary["fold_protocol"]], [24, 32, 40])
        self.assertTrue(all(max(fold["train_months"]) < fold["validation_month"] for fold in summary["fold_protocol"]))
        self.assertEqual(model.named_steps["model"].strategy, "under")
        self.assertNotIn("MES", model.named_steps["features"].named_steps["raw"].feature_names_in_)


@unittest.skipUnless(importlib.util.find_spec("mlflow") is not None, "Optional MLflow dependency is not installed")
class SQLiteIntegration(unittest.TestCase):
    def test_real_sqlite_log_compare_and_registry_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, report = make_artifact(root)
            result = track_report(report, root, experiment="synthetic-tracking-test")
            rows = compare_runs(result["tracking_uri"], "synthetic-tracking-test", result["dataset_sha256"], result["protocol_id"])
            self.assertEqual(rows[0]["model_uri"], result["model_uri"])
            registered = registrar_modelo(result["model_uri"], "synthetic-loan-test", tracking_uri=result["tracking_uri"], threshold=result["threshold"])
            self.assertTrue(registered["alias_reload_prediction_parity"])
            self.assertEqual(registered["threshold"], 0.55)


if __name__ == "__main__":
    unittest.main()
