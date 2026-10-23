"""Metrics calculated from predictions, without embedded notebook results."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def _binary_arrays(y_true, probabilities):
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal, nonzero lengths.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    return y.astype(int), p


def evaluate_binary(y_true, probabilities, threshold: float = 0.5) -> dict:
    """Evaluate positive class 1; AUC uses scores, never thresholded labels."""
    y, p = _binary_arrays(y_true, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    predicted = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    return {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": float(predicted.mean()),
    }


def choose_threshold(y_true, probabilities, thresholds=None) -> float:
    """Choose threshold on validation only; prefer higher threshold on an F1 tie."""
    y, p = _binary_arrays(y_true, probabilities)
    candidates = np.asarray(list(thresholds) if thresholds is not None else np.arange(0.05, 1, 0.05), dtype=float)
    if len(candidates) == 0 or not np.isfinite(candidates).all() or ((candidates < 0) | (candidates > 1)).any():
        raise ValueError("At least one valid candidate threshold is required.")
    scored = [(float(f1_score(y, p >= t, zero_division=0)), float(t)) for t in candidates]
    return max(scored)[1]
