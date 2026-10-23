"""Dataset loading, validation and chronological partitions, without mutation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .schema import (
    ID_COLUMN, REQUIRED_COLUMNS, TARGET, TIME_COLUMN, validate_features,
)


def validate_dataset(frame: pd.DataFrame) -> dict:
    """Validate the source contract and return only aggregate diagnostics.

    This case contains one observation per customer. Repeated customers fail
    explicitly instead of silently leaking a customer across temporal splits.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    validate_features(frame)
    target = frame[TARGET]
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("FLAG_VENTA must contain non-null binary values 0 or 1.")
    month = pd.to_numeric(frame[TIME_COLUMN], errors="coerce")
    if month.isna().any() or (month % 1 != 0).any():
        raise ValueError("MES must contain integer YYYYMM campaign months.")
    if not ((month % 100).between(1, 12) & (month // 100).between(1900, 2200)).all():
        raise ValueError("MES must contain valid YYYYMM campaign months.")
    if frame[ID_COLUMN].isna().any():
        raise ValueError("CLIENTE must not be null.")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("Repeated CLIENTE requires an explicit group-aware protocol.")
    if frame.duplicated().any():
        raise ValueError("Duplicate observations are not permitted.")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "unique_customers": int(frame[ID_COLUMN].nunique()),
        "target_counts": {str(k): int(v) for k, v in target.value_counts().sort_index().items()},
        "positive_rate": float(target.mean()),
        "missing_counts": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "months": sorted(int(v) for v in month.unique()),
    }


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the original semicolon-delimited CSV and validate its contract."""
    frame = pd.read_csv(Path(path), sep=";")
    validate_dataset(frame)
    return frame


def dataset_sha256(path: str | Path) -> str:
    """Identify exact source bytes without placing records in logs."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_temporal(
    frame: pd.DataFrame, train_end: int = 201506, validation_end: int = 201508,
) -> dict[str, pd.DataFrame]:
    """Separate campaigns before learning any feature transformation.

    Defaults: Jan-Jun train, Jul-Aug validation, Sep 2015 test for the supplied
    file. Any future campaigns join test, and the actual months are reported by
    the training manifest. No row is sampled away or reused across partitions.
    """
    validate_dataset(frame)
    if train_end >= validation_end:
        raise ValueError("train_end must be earlier than validation_end.")
    months = pd.to_numeric(frame[TIME_COLUMN])
    partitions = {
        "train": frame.loc[months <= train_end].copy(),
        "validation": frame.loc[(months > train_end) & (months <= validation_end)].copy(),
        "test": frame.loc[months > validation_end].copy(),
    }
    for name, part in partitions.items():
        if part.empty:
            raise ValueError(f"Temporal partition {name} is empty; inspect campaign dates.")
        if part[TARGET].nunique() != 2:
            raise ValueError(f"Temporal partition {name} must contain both target classes.")
    return partitions


# Stable convenience aliases for the CLI and notebook callers.
load_data = load_dataset
file_sha256 = dataset_sha256


def temporal_split(
    frame: pd.DataFrame, train_end: int = 201506, validation_end: int = 201508,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tuple interface: (train, validation, test), preserving source row indices."""
    parts = split_temporal(frame, train_end, validation_end)
    return parts["train"], parts["validation"], parts["test"]
