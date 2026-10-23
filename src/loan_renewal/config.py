"""Shared locations for course adapters; no import-time filesystem mutations."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("LOAN_DATA", ROOT / "data/raw/Dataset Renovacion_prestamo.csv"))
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TRACKING_DIR = ROOT / "mlruns"
ARTIFACTS = MODELS_DIR  # Compatibility for experiment output callers
EXPERIMENT = "loan-renewal"


def tracking_uri():
    return "sqlite:///" + str((TRACKING_DIR / "mlflow.db").resolve())
