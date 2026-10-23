"""Contract and leakage regressions using invented records, never customer rows."""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from loan_renewal.data import split_temporal, validate_dataset
from loan_renewal.evaluation import choose_threshold, evaluate_binary
from loan_renewal.features import RawFeatures, build_features
from loan_renewal.modeling import SampledClassifier, build_model, temporal_cv_splits, tune_forest
from loan_renewal.predict import load_artifact, predict_frame
from loan_renewal.schema import (
    CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, validate_features,
)


def synthetic_frame(rows=24):
    """Explicitly synthetic, schema-shaped fixture without source data values."""
    frame = pd.DataFrame({name: np.arange(rows, dtype=float) + 1 for name in NUMERIC_COLUMNS})
    frame["REGION"] = ["REGION_A" if i % 2 else "REGION_B" for i in range(rows)]
    frame["SEXO"] = ["F" if i % 2 else "M" for i in range(rows)]
    frame["EST_CIVIL"] = ["C" if i % 3 else "S" for i in range(rows)]
    frame["FLAG_LIMA_PROVINCIA"] = np.arange(rows) % 2
    frame["MES"] = np.repeat([201506, 201507, 201509], rows // 3)
    frame["CLIENTE"] = np.arange(1, rows + 1)
    frame["FLAG_VENTA"] = np.arange(rows) % 2
    return frame


class InputContractTests(unittest.TestCase):
    def test_metadata_is_excluded_and_feature_order_is_stable(self):
        frame = synthetic_frame()
        reversed_frame = frame.loc[:, list(reversed(frame.columns))]
        actual = validate_features(reversed_frame)
        self.assertEqual(list(actual), FEATURE_COLUMNS)
        self.assertFalse({"FLAG_VENTA", "CLIENTE", "MES"} & set(actual.columns))
        actual.iloc[0, 0] = -999
        self.assertNotEqual(frame.iloc[0, 0], -999)

    def test_missing_feature_is_not_silently_zero_filled(self):
        with self.assertRaisesRegex(ValueError, "Missing raw feature"):
            validate_features(synthetic_frame().drop(columns=[NUMERIC_COLUMNS[0]]))

    def test_manual_preprocessing_and_invalid_numeric_input_are_rejected(self):
        frame = synthetic_frame()
        frame["Cluster"] = 1
        with self.assertRaisesRegex(ValueError, "Unexpected columns"):
            validate_features(frame)
        frame = synthetic_frame()
        frame[NUMERIC_COLUMNS[0]] = frame[NUMERIC_COLUMNS[0]].astype(object)
        frame.loc[0, NUMERIC_COLUMNS[0]] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "Non-numeric"):
            validate_features(frame)
        frame.loc[0, NUMERIC_COLUMNS[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "Infinite"):
            validate_features(frame)

    def test_invalid_target_and_repeated_customer_are_rejected(self):
        frame = synthetic_frame()
        frame.loc[0, "FLAG_VENTA"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_dataset(frame)
        frame = synthetic_frame()
        frame.loc[0, "CLIENTE"] = frame.loc[1, "CLIENTE"]
        with self.assertRaisesRegex(ValueError, "Repeated CLIENTE"):
            validate_dataset(frame)

    def test_temporal_split_is_disjoint_complete_and_ordered(self):
        frame = synthetic_frame()
        parts = split_temporal(frame.sample(frac=1, random_state=42))
        train, validation, test = [parts[key] for key in ["train", "validation", "test"]]
        self.assertLess(train.MES.max(), validation.MES.min())
        self.assertLess(validation.MES.max(), test.MES.min())
        self.assertEqual(sum(map(len, parts.values())), len(frame))
        ids = [set(part.CLIENTE) for part in parts.values()]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


class LearnedStateTests(unittest.TestCase):
    def test_transform_does_not_learn_validation_statistics_or_categories(self):
        train = synthetic_frame().iloc[:8].copy()
        train.loc[0, "EDAD"] = np.nan
        transformer = build_features(include_cluster=True, random_state=42)
        transformer.fit(train)
        pre = transformer.named_steps["preprocessor"]
        imputer = pre.named_transformers_["numeric"].named_steps["imputer"]
        scaler = pre.named_transformers_["numeric"].named_steps["scaler"]
        encoder = pre.named_transformers_["categorical"].named_steps["encoder"]
        stats = imputer.statistics_.copy()
        means = scaler.mean_.copy()
        centers = transformer.named_steps["cluster"].kmeans_.cluster_centers_.copy()
        vocab = [values.copy() for values in encoder.categories_]
        validation = synthetic_frame().iloc[8:12].copy()
        validation.loc[:, "EDAD"] = 1_000_000
        validation.loc[:, "REGION"] = "UNSEEN_REGION"
        transformed = transformer.transform(validation)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape[1], transformer.transform(train).shape[1])
        assert_allclose(stats, imputer.statistics_)
        assert_allclose(means, scaler.mean_)
        assert_allclose(centers, transformer.named_steps["cluster"].kmeans_.cluster_centers_)
        for before, after in zip(vocab, encoder.categories_):
            self.assertEqual(before.tolist(), after.tolist())
        age_index = NUMERIC_COLUMNS.index("EDAD")
        self.assertEqual(imputer.statistics_[age_index], train.EDAD.median())

    def test_serialized_pipeline_preserves_raw_input_predictions(self):
        frame = synthetic_frame()
        model = build_model(DecisionTreeClassifier(max_depth=3, random_state=42), include_cluster=True)
        model.fit(frame.iloc[:16], frame.FLAG_VENTA.iloc[:16])
        raw = frame.iloc[16:].drop(columns=["FLAG_VENTA", "CLIENTE", "MES"])
        expected = model.predict_proba(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            assert_allclose(expected, restored.predict_proba(raw), rtol=0, atol=0)
            reordered = raw.loc[:, list(reversed(raw.columns))]
            assert_allclose(expected, restored.predict_proba(reordered), rtol=0, atol=0)


    def test_prediction_bundle_uses_saved_threshold_and_raw_input_contract(self):
        frame = synthetic_frame()
        model = build_model(DecisionTreeClassifier(max_depth=3, random_state=42), include_cluster=False)
        model.fit(frame.iloc[:16], frame.FLAG_VENTA.iloc[:16])
        bundle = {"pipeline": model, "threshold": 0.8, "manifest": {"task": "synthetic-test"}}
        raw = frame.iloc[16:].drop(columns=["FLAG_VENTA", "MES", "CLIENTE"])
        expected = predict_frame(bundle, raw)
        self.assertEqual(expected.index.tolist(), raw.index.tolist())
        self.assertTrue((expected["umbral"] == 0.8).all())
        self.assertEqual(expected["prediccion_flag_venta"].tolist(),
                         (expected["probabilidad_renovacion"] >= 0.8).astype(int).tolist())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            joblib.dump(bundle, path)
            restored = load_artifact(path)
            actual = predict_frame(restored, raw)
            assert_allclose(expected["probabilidad_renovacion"], actual["probabilidad_renovacion"], rtol=1e-12, atol=1e-12)
            self.assertEqual(expected["prediccion_flag_venta"].tolist(), actual["prediccion_flag_venta"].tolist())
        with self.assertRaisesRegex(ValueError, "Missing raw feature"):
            predict_frame(bundle, raw.drop(columns=[FEATURE_COLUMNS[0]]))


class SamplingAndMetricTests(unittest.TestCase):
    def test_over_sampling_grows_minority_and_preserves_majority(self):
        X = np.arange(40).reshape(20, 2)
        y = np.array([0] * 16 + [1] * 4)
        model = SampledClassifier(DecisionTreeClassifier(random_state=42), strategy="over")
        model.fit(X, y)
        self.assertEqual(model.original_class_counts_, {0: 16, 1: 4})
        self.assertEqual(model.sampled_class_counts_, {0: 16, 1: 16})
        model.predict_proba(X[:3])
        self.assertEqual(model.sampled_class_counts_, {0: 16, 1: 16})
        self.assertEqual(len(y), 20)

    def test_under_sampling_reduces_majority_only(self):
        X = np.arange(40).reshape(20, 2)
        y = np.array([0] * 16 + [1] * 4)
        model = SampledClassifier(DecisionTreeClassifier(random_state=42), strategy="under")
        model.fit(X, y)
        self.assertEqual(model.sampled_class_counts_, {0: 4, 1: 4})

    def test_auc_uses_scores_and_is_independent_of_decision_threshold(self):
        y = [0, 1, 0, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        low = evaluate_binary(y, scores, threshold=0.2)
        high = evaluate_binary(y, scores, threshold=0.9)
        self.assertEqual(low["roc_auc"], roc_auc_score(y, scores))
        self.assertEqual(low["roc_auc"], high["roc_auc"])
        self.assertNotEqual(low["confusion_matrix"], high["confusion_matrix"])
        self.assertNotEqual(low["roc_auc"], roc_auc_score(y, np.array(scores) >= 0.9))

    def test_threshold_selection_and_invalid_probabilities(self):
        self.assertEqual(choose_threshold([0, 1], [0.1, 0.9], thresholds=[0.2, 0.8]), 0.8)
        with self.assertRaisesRegex(ValueError, "Probabilities"):
            evaluate_binary([0, 1], [0.1, np.nan])
        with self.assertRaisesRegex(ValueError, "Probabilities"):
            evaluate_binary([0, 1], [-0.1, 0.9])


class TemporalTuningTests(unittest.TestCase):
    def test_gridsearch_refits_features_and_sampler_inside_expanding_folds(self):
        frame = synthetic_frame(rows=48)
        frame["MES"] = np.repeat([201501, 201502, 201503, 201504, 201505, 201506], 8)
        frame["FLAG_VENTA"] = (np.arange(len(frame)) % 4 == 0).astype(int)
        # Positional CV indices must remain correct even when source rows are shuffled.
        frame = frame.sample(frac=1, random_state=42)
        folds, protocol = temporal_cv_splits(frame)
        prior_train = set()
        expected_fit_indices = []
        for (train_positions, validation_positions), declaration in zip(folds, protocol):
            earlier = frame.iloc[train_positions]
            later = frame.iloc[validation_positions]
            self.assertLess(earlier.MES.max(), later.MES.min())
            self.assertFalse(set(earlier.index) & set(later.index))
            self.assertTrue(prior_train <= set(earlier.index))
            self.assertGreater(len(earlier), len(prior_train))
            self.assertEqual(sorted(earlier.MES.unique()), declaration["train_months"])
            prior_train = set(earlier.index)
            expected_fit_indices.append(set(earlier.index))
        expected_fit_indices.append(set(frame.index))
        feature_fit_indices = []
        sampler_row_counts = []
        original_feature_fit = RawFeatures.fit
        original_sampler_fit = SampledClassifier.fit

        def feature_fit_spy(instance, X, y=None):
            feature_fit_indices.append(set(X.index))
            self.assertFalse({"MES", "CLIENTE", "FLAG_VENTA"} & set(X.columns))
            return original_feature_fit(instance, X, y)

        def sampler_fit_spy(instance, X, y):
            sampler_row_counts.append(len(y))
            return original_sampler_fit(instance, X, y)

        with patch.object(RawFeatures, "fit", new=feature_fit_spy), patch.object(SampledClassifier, "fit", new=sampler_fit_spy):
            fitted, summary = tune_forest(
                frame, include_cluster=False,
                param_grid={"model__estimator__n_estimators": [3], "model__estimator__max_depth": [2]},
            )
        self.assertEqual(feature_fit_indices, expected_fit_indices)
        self.assertEqual(sampler_row_counts, [24, 32, 40, 48])
        self.assertEqual(summary["combinations"], 1)
        self.assertEqual(summary["fold_count"], 3)
        self.assertEqual(summary["cv_fit_count"], 3)
        self.assertEqual(summary["refit_count"], 1)
        self.assertEqual(fitted.named_steps["model"].strategy, "under")
        self.assertIn("raw", fitted.named_steps["features"].named_steps)
        self.assertIn("preprocessor", fitted.named_steps["features"].named_steps)
        self.assertEqual(fitted.named_steps["model"].sampled_class_counts_, {0: 12, 1: 12})
        self.assertTrue(np.isfinite(fitted.predict_proba(frame.iloc[:2])).all())

    def test_tuning_rejects_external_validation_months(self):
        frame = synthetic_frame()
        with self.assertRaisesRegex(ValueError, "training campaigns only"):
            temporal_cv_splits(frame)


if __name__ == "__main__":
    unittest.main()
