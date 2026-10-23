"""Fitted preprocessing and train-only balancing kept in a serializable estimator."""
from __future__ import annotations

import importlib.util
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from .features import build_features


def _counts(y) -> dict:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


class SampledClassifier(ClassifierMixin, BaseEstimator):
    """Resample inside fit; prediction never resamples or learns transformations.

    Under/oversampling use row selection, retaining valid encoded categories.
    Optional SMOTE is an educational alternative in the transformed numeric
    space; interpolation can produce fractional one-hot values. It is not used
    by default and requires an explicit --smote request and imbalanced-learn.
    """

    def __init__(self, estimator=None, strategy="none", random_state=42):
        self.estimator = estimator
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X, y):
        values = np.asarray(X)
        labels = np.asarray(y).reshape(-1)
        if len(labels) != len(values) or len(np.unique(labels)) != 2:
            raise ValueError("Training requires aligned rows and both target classes.")
        self.original_class_counts_ = _counts(labels)
        rng = np.random.default_rng(self.random_state)
        if self.strategy in {"under", "over"}:
            rows = [np.flatnonzero(labels == label) for label in sorted(self.original_class_counts_)]
            sizes = [len(indices) for indices in rows]
            size = min(sizes) if self.strategy == "under" else max(sizes)
            selected = np.concatenate([
                rng.choice(indices, size=size, replace=len(indices) < size)
                if len(indices) != size else indices.copy()
                for indices in rows
            ])
            rng.shuffle(selected)
            values, labels = values[selected], labels[selected]
        elif self.strategy == "smote":
            if importlib.util.find_spec("imblearn") is None:
                raise ImportError("--smote requires the optional imbalanced-learn dependency.")
            from imblearn.over_sampling import SMOTE
            minority = min(self.original_class_counts_.values())
            if minority < 2:
                raise ValueError("SMOTE requires at least two minority training rows.")
            values, labels = SMOTE(k_neighbors=min(5, minority - 1), random_state=self.random_state).fit_resample(values, labels)
        elif self.strategy != "none":
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
        self.sampled_class_counts_ = _counts(labels)
        estimator = self.estimator if self.estimator is not None else DecisionTreeClassifier(random_state=self.random_state)
        self.estimator_ = clone(estimator).fit(values, labels)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = values.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)


def build_model(estimator, strategy="none", include_cluster=True, random_state=42):
    """Return an unfitted raw-input pipeline with all learned transformations."""
    return Pipeline([
        ("features", build_features(include_cluster=include_cluster, random_state=random_state)),
        ("model", SampledClassifier(estimator=estimator, strategy=strategy, random_state=random_state)),
    ])


def candidate_specs(random_state=42, include_smote=False, include_xgboost=False):
    """Explicit fixed candidates; these are not claimed to be GridSearch winners."""
    specs = [
        ("dummy_prior", DummyClassifier(strategy="prior"), "none"),
        ("tree_original", DecisionTreeClassifier(random_state=random_state), "none"),
        ("tree_under", DecisionTreeClassifier(random_state=random_state), "under"),
        ("tree_over", DecisionTreeClassifier(random_state=random_state), "over"),
        ("forest_baseline", RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=2, random_state=random_state), "none"),
        ("forest_regularized_under", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=2, random_state=random_state), "under"),
    ]
    if include_smote:
        if importlib.util.find_spec("imblearn") is None:
            raise ImportError("Requested SMOTE but imbalanced-learn is not installed.")
        specs.append(("tree_smote", DecisionTreeClassifier(random_state=random_state), "smote"))
    if include_xgboost:
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError("Requested XGBoost but the optional xgboost dependency is not installed.")
        from xgboost import XGBClassifier
        specs.append(("xgboost_under", XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.1, subsample=0.8, n_jobs=2, random_state=random_state, eval_metric="logloss"), "under"))
    return specs


def positive_probability(pipeline, frame):
    """Locate class 1 explicitly rather than assuming predict_proba column order."""
    classes = np.asarray(pipeline.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("Model must expose exactly one positive target class 1.")
    return np.asarray(pipeline.predict_proba(frame))[:, int(positive[0])]


def temporal_cv_splits(train_frame):
    """Expanding-window folds inside Jan-Jun only, never validation/test months."""
    months = np.asarray(train_frame["MES"], dtype=int)
    labels = np.asarray(train_frame["FLAG_VENTA"], dtype=int)
    if (months > 201506).any():
        raise ValueError("Tuning input must contain training campaigns only (MES <= 201506).")
    folds, protocol = [], []
    for cutoff, holdout in ((201503, 201504), (201504, 201505), (201505, 201506)):
        fit_rows = np.flatnonzero(months <= cutoff)
        validation_rows = np.flatnonzero(months == holdout)
        if len(fit_rows) == 0 or len(validation_rows) == 0:
            raise ValueError(f"Temporal CV requires training campaigns through {cutoff} and validation campaign {holdout}.")
        if len(np.unique(labels[fit_rows])) != 2 or len(np.unique(labels[validation_rows])) != 2:
            raise ValueError(f"Both target classes are required in temporal CV fold ending {holdout}.")
        folds.append((fit_rows, validation_rows))
        protocol.append({
            "train_months": sorted(int(month) for month in np.unique(months[fit_rows])),
            "validation_month": int(holdout),
            "train_rows": int(len(fit_rows)),
            "validation_rows": int(len(validation_rows)),
        })
    return folds, protocol


def tune_random_forest(train_frame, include_cluster=True, random_state=42, param_grid=None, method="grid", n_iter=4):
    """Run a bounded grid/random search with 3 temporal folds + train-only refit.

    Defaults: grid 4 combinations or random 4 draws, each 12 CV fits plus one
    refit. Grid tests the endpoints; random samples integer values inside the
    same bounds. These are distinct search strategies, not equal candidate sets.

    GridSearchCV clones the complete raw-input pipeline in every fold, so
    imputation, scaling, encoding, clustering and undersampling are refitted
    solely on the corresponding earlier campaigns. The held-out July/August
    selection and September test data are not arguments to this function.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from .schema import FEATURE_COLUMNS
    if method not in {"grid", "random"}:
        raise ValueError("Search method must be grid or random.")
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer.")
    folds, protocol = temporal_cv_splits(train_frame)
    estimator = build_model(
        RandomForestClassifier(min_samples_leaf=10, n_jobs=2, random_state=random_state),
        strategy="under", include_cluster=include_cluster, random_state=random_state,
    )
    common = {
        "estimator": estimator, "cv": folds, "scoring": "f1", "refit": True,
        "n_jobs": 1, "error_score": "raise", "return_train_score": False,
    }
    if method == "grid":
        search_space = param_grid if param_grid is not None else {
            "model__estimator__n_estimators": [50, 100],
            "model__estimator__max_depth": [4, 8],
        }
        search = GridSearchCV(param_grid=search_space, **common)
        documented_space = search_space
    else:
        from scipy.stats import randint
        search_space = param_grid if param_grid is not None else {
            "model__estimator__n_estimators": randint(50, 101),
            "model__estimator__max_depth": randint(4, 9),
        }
        search = RandomizedSearchCV(param_distributions=search_space, n_iter=n_iter, random_state=random_state, **common)
        documented_space = param_grid if param_grid is not None else {
            "model__estimator__n_estimators": {"distribution": "randint", "low": 50, "high_exclusive": 101},
            "model__estimator__max_depth": {"distribution": "randint", "low": 4, "high_exclusive": 9},
        }
    search.fit(train_frame.loc[:, FEATURE_COLUMNS], train_frame["FLAG_VENTA"])
    summary = {
        "enabled": True,
        "method": "GridSearchCV" if method == "grid" else "RandomizedSearchCV",
        "candidate": "forest_grid_under" if method == "grid" else "forest_random_under",
        "scoring": "f1_class_1_at_estimator_threshold_0.5",
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "best_cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "search_space": documented_space,
        "parameter_grid": documented_space if method == "grid" else None,
        "requested_n_iter": n_iter if method == "random" else None,
        "combinations": int(len(search.cv_results_["params"])),
        "fold_count": len(folds),
        "cv_fit_count": int(len(search.cv_results_["params"]) * len(folds)),
        "refit_count": 1,
        "outer_n_jobs": 1,
        "fold_protocol": protocol,
        "cv_results": [
            {"params": params, "mean_f1": float(mean), "std_f1": float(std)}
            for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])
        ],
    }
    return search.best_estimator_, summary


# Short alias for programmatic notebook callers and focused smoke tests.
tune_forest = tune_random_forest
