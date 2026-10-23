"""Unit 04-style stages with durable success/failure evidence.

The features stage declares/checks raw inputs; learned transformations remain
inside training/CV. Evaluation consumes final metrics rather than refitting.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from .data import load_dataset, validate_dataset, split_temporal, dataset_sha256
from .schema import FEATURE_COLUMNS, LOG_COLUMNS, CLUSTER_COLUMNS, validate_features
from .pipeline import train

LOGGER = logging.getLogger(__name__)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_stages(data_path, output_dir, tune=False, search_method="grid", track=False, report_dir=None):
    """Stop on a failed stage and retain stage_report.json, including failures."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = Path(report_dir).resolve() if report_dir is not None else output
    reports.mkdir(parents=True, exist_ok=True)
    state, stages = {}, []
    result = {"status": "running", "stages": stages}

    def ingest():
        state["data"] = load_dataset(data_path)
        profile = validate_dataset(state["data"])
        profile["dataset_sha256"] = dataset_sha256(data_path)
        _write(reports / "data_profile.json", profile)
        return {"rows": profile["rows"], "dataset_sha256": profile["dataset_sha256"]}

    def features():
        parts = split_temporal(state["data"])
        for frame in parts.values():
            validate_features(frame)
        contract = {"raw_features": FEATURE_COLUMNS, "log_columns": LOG_COLUMNS,
                    "cluster_columns": CLUSTER_COLUMNS,
                    "fit_policy": "Each training/CV fold fits its own full transformer.",
                    "partition_rows": {k: len(v) for k, v in parts.items()}}
        _write(reports / "feature_contract.json", contract)
        return {"raw_feature_count": len(FEATURE_COLUMNS), "partition_rows": contract["partition_rows"]}

    def training():
        state["report"] = train(data_path, output, tune=tune, search_method=search_method, report_dir=reports)
        return {"winner": state["report"]["winner"], "threshold": state["report"]["threshold"]}

    def evaluate():
        report = state["report"]
        for section in ("validation", "test"):
            for metric in ("f1", "precision", "recall", "roc_auc", "average_precision"):
                value = report[section][metric]
                if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {section} metric: {metric}")
        if not report["reload_prediction_parity"]:
            raise ValueError("Artifact reload parity did not pass.")
        return {"validation_f1": report["validation"]["f1"], "test_f1": report["test"]["f1"],
                "reload_prediction_parity": True,
                "production_approval": False}

    operations = [("ingestion", ingest), ("features_contract", features),
                  ("training_and_final_predictions", training), ("evaluation_checks", evaluate)]
    if track:
        def tracking():
            from .tracking import track_report
            from .config import tracking_uri
            return track_report(state["report"], reports, tracking_uri=tracking_uri())
        operations.append(("mlflow_tracking", tracking))
    for name, action in operations:
        started = time.perf_counter()
        try:
            details = action()
            stages.append({"stage": name, "status": "ok", "seconds": round(time.perf_counter() - started, 4), "details": details})
        except Exception as error:
            stages.append({"stage": name, "status": "failed", "seconds": round(time.perf_counter() - started, 4), "error_type": type(error).__name__})
            result["status"] = "failed"
            _write(reports / "stage_report.json", result)
            raise
        _write(reports / "stage_report.json", result)
    result["status"] = "ok"
    _write(reports / "stage_report.json", result)
    return result
