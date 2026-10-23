"""Serializable train-fitted preprocessing for raw loan-renewal records."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .schema import (
    CATEGORICAL_COLUMNS, CLUSTER_COLUMNS, FEATURE_COLUMNS, LOG_COLUMNS,
    NUMERIC_COLUMNS, validate_features,
)


class RawFeatures(TransformerMixin, BaseEstimator):
    """Validate/select raw columns and log1p clipped numeric features.

    Clipping at zero extends the original notebook's treatment of three signed
    balances consistently to the ten non-negative log-domain variables. It is
    a fixed business transformation, not a quantile learned from all data.
    Signed entity-count differences are not clipped or logarithmized.
    """

    def fit(self, X, y=None):
        frame = validate_features(X)
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        frame = validate_features(X)
        frame.loc[:, LOG_COLUMNS] = np.log1p(frame.loc[:, LOG_COLUMNS].clip(lower=0))
        return frame

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()


def build_preprocessor() -> ColumnTransformer:
    """Impute/scale logged numeric inputs; impute/encode raw categories.

    Call through build_features so RawFeatures performs raw validation/logs.
    Statistics, missing indicators and category vocabularies are train-fitted.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], remainder="drop", verbose_feature_names_out=True)


class ClusterAppender(TransformerMixin, BaseEstimator):
    """Append three one-hot KMeans memberships fitted only on training rows.

    ColumnTransformer emits numeric columns first, in NUMERIC_COLUMNS order.
    Clustering uses the same three log features as the source notebook, now
    imputed/scaled using train state. A cluster ID has no ordinal meaning.
    """

    def __init__(self, random_state=42, n_clusters=3):
        self.random_state = random_state
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Clustering requires a finite two-dimensional feature array.")
        if len(values) < self.n_clusters:
            raise ValueError("Training rows must be at least n_clusters.")
        self.n_features_in_ = values.shape[1]
        self.cluster_indices_ = [NUMERIC_COLUMNS.index(name) for name in CLUSTER_COLUMNS]
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state,
        ).fit(values[:, self.cluster_indices_])
        return self

    def transform(self, X):
        check_is_fitted(self, "kmeans_")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Transformed feature count differs from the training contract.")
        if not np.isfinite(values).all():
            raise ValueError("Transformed features must be finite.")
        labels = self.kmeans_.predict(values[:, self.cluster_indices_])
        memberships = np.eye(self.n_clusters, dtype=float)[labels]
        return np.column_stack([values, memberships])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "kmeans_")
        names = np.asarray(
            input_features if input_features is not None
            else [f"x{i}" for i in range(self.n_features_in_)], dtype=object,
        )
        return np.concatenate([names, [f"cluster_{i}" for i in range(self.n_clusters)]])


def build_features(include_cluster=True, random_state=42) -> Pipeline:
    """Build the complete cloneable transformer consumed by the model pipeline."""
    steps = [("raw", RawFeatures()), ("preprocessor", build_preprocessor())]
    if include_cluster:
        steps.append(("cluster", ClusterAppender(random_state=random_state)))
    return Pipeline(steps)
