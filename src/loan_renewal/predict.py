"""Apply a trusted complete artifact to raw feature rows."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .modeling import positive_probability
from .schema import validate_features


def load_artifact(path):
    """Load only trusted local joblib artifacts; pickle can execute arbitrary code."""
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or not {"pipeline", "threshold", "manifest"} <= bundle.keys():
        raise ValueError("Expected an artifact containing pipeline, threshold and manifest.")
    if not np.isfinite(bundle["threshold"]) or not 0 <= bundle["threshold"] <= 1:
        raise ValueError("Invalid artifact decision threshold.")
    return bundle


def predict_frame(bundle, raw_frame):
    """Return commercial propensity scores, without fitting or mutating input."""
    ordered = validate_features(raw_frame)
    probabilities = positive_probability(bundle["pipeline"], ordered)
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite scores.")
    threshold = float(bundle["threshold"])
    return pd.DataFrame({
        "probabilidad_renovacion": probabilities,
        "prediccion_flag_venta": (probabilities >= threshold).astype(int),
        "umbral": threshold,
    }, index=raw_frame.index)


def predict_file(model_path, input_path, output_path):
    frame = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    prediction = predict_frame(load_artifact(model_path), frame)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(destination, index=False)
    return prediction
