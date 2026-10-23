"""Evaluate the frozen validation winner once, retaining an identity-bound receipt."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .data import dataset_sha256, load_dataset, split_temporal
from .evaluation import evaluate_binary
from .modeling import positive_probability
from .pipeline import _partition_manifest
from .schema import FEATURE_COLUMNS
from .tracking import protocol_id, resolve_tracking_uri, tracking_context


def selection_identity(selected):
    """Bind evaluation to an exact run/model/data/protocol/threshold tuple."""
    identity = {key: selected[key] for key in (
        "run_id", "model_uri", "dataset_sha256", "protocol_id",
    )}
    threshold = float(selected["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid selected threshold.")
    identity["threshold"] = threshold
    return identity


def require_final_evaluation(selected, receipt):
    """Reject missing/stale evidence before any Registry mutation."""
    if (receipt.get("selection") != selection_identity(selected)
            or receipt.get("status") != "completed"
            or receipt.get("evaluation_count") != 1
            or not receipt.get("test_partition", {}).get("row_indices_sha256")):
        raise ValueError("Final evaluation does not match the selected winner.")
    for key in ("f1", "precision", "recall", "roc_auc", "average_precision"):
        value = receipt.get("test", {}).get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Final evaluation metrics are missing or invalid.")
    return receipt


def evaluate_selected_run(selected, data_path, receipt_path, tracking_uri):
    """Load the exact logged winner; never fit or retune against held-out test.

    Matching completed evidence is reused without predicting again. A different
    winner cannot overwrite an existing receipt. The lock prevents concurrent
    callers from independently evaluating the same experiment directory.
    """
    identity = selection_identity(selected)
    if dataset_sha256(data_path) != identity["dataset_sha256"]:
        raise ValueError("Evaluation dataset differs from the selected run.")
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        return require_final_evaluation(selected, json.loads(receipt_path.read_text()))
    lock = receipt_path.with_suffix(".lock")
    with lock.open("x") as stream:
        stream.write(json.dumps(identity))
    try:
        if receipt_path.exists():
            return require_final_evaluation(selected, json.loads(receipt_path.read_text()))
        with tracking_context(resolve_tracking_uri(tracking_uri)) as mlflow:
            info = mlflow.models.get_model_info(identity["model_uri"])
            metadata = info.metadata or {}
            for key in ("dataset_sha256", "protocol_id", "threshold"):
                value = float(metadata[key]) if key == "threshold" else metadata[key]
                if value != identity[key]:
                    raise ValueError("Logged model differs from the selected winner.")
            manifest = mlflow.artifacts.load_dict(
                f"runs:/{identity['run_id']}/evidence/manifest.json"
            )
            if protocol_id(manifest) != identity["protocol_id"]:
                raise ValueError("Logged run has a different evaluation protocol.")
            model = mlflow.sklearn.load_model(identity["model_uri"])
            parts = split_temporal(load_dataset(data_path))
            actual = _partition_manifest(parts, include_test_labels=False)
            if actual != manifest["partitions"]:
                raise ValueError("Evaluation partitions differ from the selected run.")
            test = parts["test"]
            scores = positive_probability(model, test.loc[:, FEATURE_COLUMNS])
            metrics = evaluate_binary(test["FLAG_VENTA"], scores, identity["threshold"])
            receipt = {
                "status": "completed", "evaluation_count": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection": identity, "test_partition": actual["test"],
                "test": metrics,
            }
            # Persist before remote logging: retries reuse the scores' evidence.
            temporary = receipt_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False))
            temporary.replace(receipt_path)
            return receipt
    finally:
        lock.unlink()


def publish_final_evaluation(selected, receipt, tracking_uri):
    """Attach final metrics to the selected run only, independently of ranking."""
    require_final_evaluation(selected, receipt)
    with tracking_context(resolve_tracking_uri(tracking_uri)) as mlflow:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        for name, value in receipt["test"].items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                client.log_metric(selected["run_id"], f"test_report_{name}", float(value))
        client.set_tag(selected["run_id"], "test_evaluation", "completed_global_winner")
