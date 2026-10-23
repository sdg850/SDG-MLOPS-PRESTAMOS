"""Optional MLflow tracking backed by local SQLite and explicit local artifacts.

MLflow is imported only for tracking operations, so ordinary training remains
usable without the optional dependency. Only trusted, locally produced joblib
artifacts may be logged; loading pickle/joblib can execute arbitrary code.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .modeling import positive_probability
from .schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def require_mlflow():
    """Load MLflow lazily with a useful optional-dependency failure."""
    try:
        module = importlib.import_module("mlflow")
        importlib.import_module("mlflow.sklearn")
        return module
    except ModuleNotFoundError as error:
        raise ImportError("MLflow tracking requires the optional dependency: pip install -e '.[tracking]'.") from error


def resolve_tracking_uri(tracking_uri=None, output_dir=None) -> str:
    """Require an absolute local SQLite backend, never an implicit file store."""
    if tracking_uri is None:
        from .config import TRACKING_DIR
        root = Path(output_dir).expanduser().resolve() if output_dir is not None else TRACKING_DIR
        root.mkdir(parents=True, exist_ok=True)
        return "sqlite:///" + (root / "mlflow.db").as_posix()
    prefix = "sqlite:///"
    if not isinstance(tracking_uri, str) or not tracking_uri.startswith(prefix):
        raise ValueError("Use a local SQLite URI, e.g. sqlite:////absolute/path/mlflow.db.")
    filename = tracking_uri[len(prefix):]
    if not filename.startswith("/") or "?" in filename or "#" in filename or filename == "/":
        raise ValueError("SQLite tracking requires an absolute database file path.")
    path = Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.as_posix()


@contextmanager
def tracking_context(tracking_uri):
    """Temporarily select explicit tracking/registry backends without nesting runs."""
    mlflow = require_mlflow()
    if mlflow.active_run() is not None:
        raise RuntimeError("Finish the active MLflow run before this operation; implicit nested tracking is disabled.")
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    try:
        yield mlflow
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


def protocol_details(manifest) -> dict:
    """Define comparability by data partitions/schema/selection, not test scores.

    Search method, model family and tuned parameters intentionally do not enter
    the protocol ID: they are alternatives evaluated under the same protocol.
    """
    selection = manifest.get("selection", {})
    return {
        "task": manifest.get("task"),
        "target": manifest.get("target"),
        "positive_class": manifest.get("positive_class", 1),
        "partitions": manifest.get("partitions"),
        "raw_features": manifest.get("raw_features"),
        "random_state": manifest.get("random_state"),
        "include_cluster": manifest.get("include_cluster"),
        "source_sha256": manifest.get("source_sha256"),
        "primary_metric": selection.get("primary_metric"),
        "threshold_grid": selection.get("threshold_grid"),
        "refit_on_validation": selection.get("refit_on_validation", False),
    }


def protocol_id(manifest) -> str:
    """Hash the exact evaluation protocol, independently of candidate outcomes."""
    encoded = json.dumps(protocol_details(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def synthetic_probe() -> pd.DataFrame:
    """Public synthetic raw inputs solely for serialization/parity checks.

    These rows are not examples drawn from customer data and never substantiate
    predictive performance. Unknown string categories exercise encoder safety.
    """
    rows = []
    for value in (1.0, 2.0):
        row = {column: value for column in NUMERIC_COLUMNS}
        row.update({column: "__synthetic_tracking_probe__" for column in CATEGORICAL_COLUMNS})
        row.update({"EDAD": 35.0 + value, "PLAZO_RENOVADO": 12.0, "FLAG_LIMA_PROVINCIA": 0.0})
        rows.append(row)
    return pd.DataFrame(rows).loc[:, FEATURE_COLUMNS]


def _finite_metrics(metrics, prefix):
    output = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            output[f"{prefix}_{name}"] = float(value)
    return output


def _requirements(manifest):
    result = []
    versions = dict(manifest.get("versions", {}))
    for package in ("scikit-learn", "numpy", "pandas", "joblib", "cloudpickle"):
        version = versions.get(package)
        if not version:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
        result.append(f"{package}=={version}")
    return result


def track_report(report, output_dir, tracking_uri=None, experiment="loan-renewal") -> dict:
    """Log one completed training execution, its complete pipeline and evidence.

    Selection metrics are prefixed validation_; final test metrics are prefixed
    test_report_ and never used for ranking. The model's pyfunc returns class
    probabilities; the separately preserved validation threshold is mandatory
    when converting those probabilities into a commercial propensity label.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    uri = resolve_tracking_uri(tracking_uri, output)
    manifest = report.get("manifest")
    if not isinstance(manifest, dict) or not manifest.get("dataset_sha256"):
        raise ValueError("Tracking requires a manifest with the exact dataset hash.")
    if not manifest.get("partitions") or not manifest.get("source_sha256"):
        raise ValueError("Tracking requires split evidence and executed source hashes.")
    if manifest.get("selection", {}).get("partition") != "validation":
        raise ValueError("Only validation-selected reports are eligible for this tracker.")
    validation = report.get("validation", {})
    if not isinstance(validation.get("f1"), (int, float)) or not math.isfinite(validation["f1"]):
        raise ValueError("A finite validation F1 metric is required.")
    model_path = Path(report.get("model_path", output / "model.joblib")).expanduser().resolve()
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not {"pipeline", "threshold", "manifest"} <= bundle.keys():
        raise ValueError("Expected the complete local pipeline/threshold/manifest artifact.")
    threshold = float(bundle["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The artifact threshold must be finite and between 0 and 1.")
    if bundle["manifest"].get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("Report and artifact refer to different dataset hashes.")
    if not np.isclose(float(report.get("threshold", threshold)), threshold, rtol=0, atol=0):
        raise ValueError("Report and artifact thresholds differ.")
    protocol = protocol_id(manifest)
    probe = synthetic_probe()
    original_scores = positive_probability(bundle["pipeline"], probe)
    with tracking_context(uri) as mlflow:
        client = mlflow.tracking.MlflowClient(tracking_uri=uri, registry_uri=uri)
        existing = client.get_experiment_by_name(experiment)
        if existing is None:
            artifact_root = Path(uri[len("sqlite:///"):]).parent / "mlflow-artifacts"
            artifact_root.mkdir(parents=True, exist_ok=True)
            experiment_id = client.create_experiment(experiment, artifact_location=artifact_root.as_uri())
        else:
            if existing.lifecycle_stage != "active":
                raise ValueError("The named experiment is deleted; choose an active experiment explicitly.")
            experiment_id = existing.experiment_id
        tags = {
            "dataset_sha256": manifest["dataset_sha256"],
            "protocol_id": protocol,
            "selection_partition": "validation",
            "task": manifest.get("task", "commercial_loan_renewal_propensity_educational"),
            "probe_origin": "synthetic_not_customer_data",
        }
        with mlflow.start_run(experiment_id=experiment_id, run_name=report.get("winner", "loan-renewal"), tags=tags) as run:
            mlflow.log_params({
                "winner": report.get("winner", manifest.get("winner", "unknown")),
                "threshold": threshold,
                "sampling": manifest.get("sampling", "none"),
                "random_state": manifest.get("random_state", 42),
                "include_cluster": manifest.get("include_cluster", False),
                "candidate_count": manifest.get("selection", {}).get("candidate_count", len(report.get("candidates", []))),
                "search_method": manifest.get("tuning", {}).get("method", "disabled"),
            })
            mlflow.log_metrics({**_finite_metrics(validation, "validation"), **_finite_metrics(report.get("test", {}), "test_report")})
            mlflow.log_dict(manifest, "evidence/manifest.json")
            mlflow.log_dict(report, "evidence/training_report.json")
            mlflow.log_dict(protocol_details(manifest), "evidence/selection_protocol.json")
            mlflow.log_dict({"threshold": threshold, "positive_class": 1, "selection_partition": "validation", "pyfunc_output": "predict_proba; apply threshold to class 1"}, "evidence/threshold.json")
            if (output / "comparison.csv").exists():
                mlflow.log_artifact(str(output / "comparison.csv"), artifact_path="evidence")
            metadata = {
                "threshold": threshold,
                "positive_class": 1,
                "dataset_sha256": manifest["dataset_sha256"],
                "protocol_id": protocol,
                "selection_partition": "validation",
                "task": tags["task"],
                "probabilities_calibrated": False,
            }
            signature = mlflow.models.infer_signature(probe, bundle["pipeline"].predict_proba(probe))
            kwargs = {
                "sk_model": bundle["pipeline"],
                "serialization_format": "cloudpickle",
                "signature": signature,
                "input_example": probe,
                "pyfunc_predict_fn": "predict_proba",
                "metadata": metadata,
                "code_paths": [str(Path(__file__).resolve().parent)],
                "pip_requirements": _requirements(manifest),
            }
            # MLflow 3 uses name; retain explicit compatibility with a 2.x API.
            parameter_names = inspect.signature(mlflow.sklearn.log_model).parameters
            kwargs["name" if "name" in parameter_names else "artifact_path"] = "loan_pipeline"
            model_info = mlflow.sklearn.log_model(**kwargs)
            model_uri = model_info.model_uri
            if not model_uri:
                raise RuntimeError("MLflow did not return an exact model URI.")
            reloaded = mlflow.sklearn.load_model(model_uri)
            reloaded_scores = positive_probability(reloaded, probe)
            np.testing.assert_allclose(original_scores, reloaded_scores, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(original_scores >= threshold, reloaded_scores >= threshold)
            mlflow.set_tags({"model_uri": model_uri, "reload_prediction_parity": "true", "threshold": str(threshold)})
            result = {
                "run_id": run.info.run_id,
                "model_uri": model_uri,
                "experiment_id": str(experiment_id),
                "tracking_uri": uri,
                "protocol_id": protocol,
                "dataset_sha256": manifest["dataset_sha256"],
                "threshold": threshold,
                "reload_prediction_parity": True,
                "probe_origin": "synthetic_not_customer_data",
            }
            mlflow.log_dict(result, "evidence/tracking.json")
    (output / "tracking.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
