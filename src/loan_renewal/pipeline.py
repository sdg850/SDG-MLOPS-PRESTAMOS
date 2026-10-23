"""Reproducible temporal training, validation selection, and final test evaluation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import load_dataset, split_temporal, validate_dataset
from .evaluation import choose_threshold, evaluate_binary
from .modeling import build_model, candidate_specs, positive_probability, tune_random_forest
from .schema import FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "imbalanced-learn", "xgboost", "mlflow", "dvc", "fastapi", "pydantic", "uvicorn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_state():
    try:
        root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5, check=True).stdout.strip())
        return {"commit": commit, "worktree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "worktree_dirty": None}


def _partition_manifest(partitions):
    result = {}
    for name, frame in partitions.items():
        result[name] = {
            "rows": int(len(frame)),
            "months": sorted(int(month) for month in frame["MES"].unique()),
            "positive_count": int(frame["FLAG_VENTA"].sum()),
            "positive_rate": float(frame["FLAG_VENTA"].mean()),
            "row_indices_sha256": hashlib.sha256(np.asarray(frame.index, dtype="int64").tobytes()).hexdigest(),
        }
    return result


def fit_candidates(train_frame, validation_frame, random_state=42, include_cluster=True, include_smote=False, include_xgboost=False, tune=False, search_method="grid"):
    """Fit/rank on train/validation only, with optional train-internal temporal CV."""
    specifications = candidate_specs(random_state, include_smote, include_xgboost)
    results = []
    winner = None
    winner_key = None

    def score_candidate(name, model, strategy, elapsed_fit, tuning=None):
        nonlocal winner, winner_key
        probability = positive_probability(model, validation_frame.loc[:, FEATURE_COLUMNS])
        threshold = choose_threshold(validation_frame["FLAG_VENTA"], probability)
        metrics = evaluate_binary(validation_frame["FLAG_VENTA"], probability, threshold)
        sampler = model.named_steps["model"]
        result = {
            "candidate": name,
            "sampling": strategy,
            "validation_threshold": threshold,
            **{f"validation_{key}": value for key, value in metrics.items() if key not in {"threshold", "confusion_matrix"}},
            "fit_and_validation_seconds": round(time.perf_counter() - elapsed_fit, 6),
            "train_class_counts": sampler.original_class_counts_,
            "sampled_train_class_counts": sampler.sampled_class_counts_,
            "estimator_params": sampler.estimator.get_params(deep=False),
        }
        if tuning is not None:
            result["tuning"] = tuning
        # Deterministic tie-break: higher AP, then the earlier declared candidate.
        key = (metrics["f1"], metrics["average_precision"] or 0.0, -len(results))
        results.append(result)
        if winner_key is None or key > winner_key:
            winner_key = key
            winner = {"name": name, "pipeline": model, "threshold": threshold, "metrics": metrics, "sampling": strategy}
        LOGGER.info("%s validation F1=%.4f AP=%.4f threshold=%.2f (%.1fs)", name, metrics["f1"], metrics["average_precision"] or 0.0, threshold, result["fit_and_validation_seconds"])

    for name, estimator, strategy in specifications:
        started = time.perf_counter()
        model = build_model(estimator, strategy, include_cluster, random_state)
        model.fit(train_frame.loc[:, FEATURE_COLUMNS], train_frame["FLAG_VENTA"])
        score_candidate(name, model, strategy, started)
    if tune:
        started = time.perf_counter()
        tuned_model, tuning = tune_random_forest(train_frame, include_cluster, random_state, method=search_method)
        score_candidate(tuning["candidate"], tuned_model, "under", started, tuning)
    return winner, results


def train(data_path, output_dir, random_state=42, include_cluster=True, include_smote=False, include_xgboost=False, tune=False, search_method="grid", report_dir=None):
    """Train a complete raw-input artifact and evaluate the selected candidate once.

    Validation and threshold search use July/August. September is reserved for
    the final report of the selected candidate. The winning pipeline remains
    fitted on Jan-Jun, without an undisclosed train+validation refit.
    """
    started = time.perf_counter()
    data_path, output_dir = Path(data_path), Path(output_dir)
    frame = load_dataset(data_path)
    validation_report = validate_dataset(frame)
    partitions = split_temporal(frame)
    winner, results = fit_candidates(
        partitions["train"], partitions["validation"], random_state,
        include_cluster, include_smote, include_xgboost, tune, search_method,
    )
    # Only the chosen estimator sees the test features, and only for inference.
    test_probability = positive_probability(winner["pipeline"], partitions["test"].loc[:, FEATURE_COLUMNS])
    test_metrics = evaluate_binary(partitions["test"]["FLAG_VENTA"], test_probability, winner["threshold"])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "commercial_loan_renewal_propensity_educational",
        "target": "FLAG_VENTA",
        "positive_class": 1,
        "dataset_sha256": file_sha256(data_path),
        "dataset_filename": data_path.name,
        "dataset_validation": validation_report,
        "partitions": _partition_manifest(partitions),
        "raw_features": list(FEATURE_COLUMNS),
        "excluded_columns": ["FLAG_VENTA", "MES", "CLIENTE"],
        "random_state": int(random_state),
        "include_cluster": bool(include_cluster),
        "winner": winner["name"],
        "sampling": winner["sampling"],
        "threshold": float(winner["threshold"]),
        "selection": {
            "partition": "validation",
            "primary_metric": "f1_class_1",
            "threshold_grid": [round(float(x), 2) for x in np.arange(0.05, 1, 0.05)],
            "tie_break": "average_precision then declared candidate order",
            "candidate_count": len(results),
            "refit_on_validation": False,
        },
        "tuning": next((result["tuning"] for result in results if "tuning" in result), {"enabled": False}),
        "versions": _versions(),
        "git": _git_state(),
        "source_sha256": {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "limitations": [
            "Educational commercial-propensity model; no lending eligibility decisions.",
            "Scores are not calibrated probabilities of renewal.",
            "Feature availability at campaign decision time requires business confirmation.",
            "Age/sex/marital/geographic features retained for fidelity; no fairness assessment or production approval is claimed.",
            "Repeated use of the same September test during later development would invalidate independent final evaluation.",
            "SMOTE, when explicitly enabled, interpolates transformed one-hot features; validate appropriateness before business use.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(report_dir) if report_dir is not None else output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": winner["pipeline"], "threshold": winner["threshold"], "manifest": manifest}
    model_path = output_dir / "model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    # Persistence is verified against the exact raw-input path.
    reloaded = joblib.load(model_path)
    probe = partitions["test"].loc[:, FEATURE_COLUMNS].head(16)
    np.testing.assert_allclose(positive_probability(reloaded["pipeline"], probe), positive_probability(winner["pipeline"], probe), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(positive_probability(reloaded["pipeline"], probe) >= winner["threshold"], positive_probability(winner["pipeline"], probe) >= winner["threshold"])
    manifest["artifact_sha256"] = file_sha256(model_path)
    report = {
        "winner": winner["name"],
        "threshold": winner["threshold"],
        "validation": winner["metrics"],
        "test": test_metrics,
        "reload_prediction_parity": True,
        "reload_probability_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "candidates": results,
        "manifest": manifest,
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (report_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    scalar_rows = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in results]
    pd.DataFrame(scalar_rows).to_csv(report_dir / "comparison.csv", index=False)
    LOGGER.info("Winner %s; final test F1=%.4f; artifact %s", winner["name"], test_metrics["f1"], model_path)
    return {"model_path": str(model_path), "metrics_path": str(report_dir / "metrics.json"), "comparison_path": str(report_dir / "comparison.csv"), **report}
