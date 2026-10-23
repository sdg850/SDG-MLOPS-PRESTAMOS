"""Raw input contract for the educational loan-renewal marketing case.

Column names follow the CSV; the dictionary's USO_TRIM_LINEA_Bco corresponds
here to USO_TRIM_LINEA_BBVA. IDs, campaign month and target never enter a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "FLAG_VENTA"
TIME_COLUMN = "MES"
ID_COLUMN = "CLIENTE"
CATEGORICAL_COLUMNS = ["SEXO", "EST_CIVIL", "REGION"]
NUMERIC_COLUMNS = [
    "LINEA_RENOVADO", "PLAZO_RENOVADO", "USO_LINEA_TOTAL_TC_T2",
    "USO_TRIM_LINEA_BBVA", "NR_ENTIDADES_TOTAL_T2",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12", "SDO_CONSUMO_T2",
    "RESENCIA_OFERTA_PLD_RENOVADO", "Ahorro_Sldo_Bco_T1",
    "PConsumo_Sldo_Bco_T1", "SDO_BCO_tot_sm_pasivo_Bco_6M", "EDAD",
    "ANTIGUEDAD_MES", "FLAG_LIMA_PROVINCIA", "SUELDO_ESTIMADO",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = [TIME_COLUMN, ID_COLUMN, TARGET] + FEATURE_COLUMNS
LOG_COLUMNS = [
    "USO_LINEA_TOTAL_TC_T2", "USO_TRIM_LINEA_BBVA", "SDO_CONSUMO_T2",
    "SUELDO_ESTIMADO", "ANTIGUEDAD_MES", "LINEA_RENOVADO",
    "Ahorro_Sldo_Bco_T1", "PConsumo_Sldo_Bco_T1",
    "SDO_BCO_tot_sm_pasivo_Bco_6M", "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
]
CLUSTER_COLUMNS = [
    "USO_TRIM_LINEA_BBVA", "PConsumo_Sldo_Bco_T1", "USO_LINEA_TOTAL_TC_T2",
]


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an independent, ordered raw feature frame or fail explicitly.

    Null values are permitted for train-fitted imputers. Required columns must
    exist even if null; absent features are never silently filled with zero.
    Metadata columns MES/CLIENTE/FLAG_VENTA are allowed but excluded. Unknown
    columns are rejected to catch typos and manually preprocessed inputs.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame with raw column names.")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not permitted.")
    if frame.empty:
        raise ValueError("Input has no observations.")
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected columns: {unexpected}")
    result = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        series = result[column]
        converted = pd.to_numeric(series, errors="coerce")
        if (series.notna() & converted.isna()).any():
            raise ValueError(f"Non-numeric values in {column}.")
        values = converted.astype(float)
        if np.isinf(values.to_numpy()).any():
            raise ValueError(f"Infinite values in {column}.")
        result[column] = values
    for column in CATEGORICAL_COLUMNS:
        series = result[column]
        if not series.dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Categorical {column} must contain strings or null.")
        # Normalize missing markers to np.nan for SimpleImputer, preserve labels.
        result[column] = series.astype(object).where(series.notna(), np.nan)
    return result
