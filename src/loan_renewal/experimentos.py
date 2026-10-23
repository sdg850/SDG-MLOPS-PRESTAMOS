"""Compare compatible validation-selected runs and register candidate aliases.

Registry state is evidence of registration, not deployment or approval for
lending decisions. Deprecated model stages are deliberately not used.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .modeling import positive_probability
from .tracking import resolve_tracking_uri, synthetic_probe, tracking_context


def _hash_filter(value, field):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{field} must be an exact 64-character SHA-256 identifier.")
    return value.lower()


def compare_runs(tracking_uri, experiment="loan-renewal", dataset_sha256=None, protocol_id=None):
    """Rank FINISHED runs from the same data and evaluation protocol by validation.

    Require exact data/protocol hashes rather than silently mixing campaigns,
    labels or split protocols. Missing/NaN scores and unlogged model artifacts
    are excluded; test_report_* metrics never participate in sorting.
    """
    dataset = _hash_filter(dataset_sha256, "dataset_sha256")
    protocol = _hash_filter(protocol_id, "protocol_id")
    uri = resolve_tracking_uri(tracking_uri)
    with tracking_context(uri) as mlflow:
        client = mlflow.tracking.MlflowClient(tracking_uri=uri, registry_uri=uri)
        selected = client.get_experiment_by_name(experiment)
        if selected is None or selected.lifecycle_stage != "active":
            return []
        filter_string = (
            "attributes.status = 'FINISHED' "
            f"and tags.dataset_sha256 = '{dataset}' "
            f"and tags.protocol_id = '{protocol}' "
            "and tags.selection_partition = 'validation'"
        )
        output, token = [], None
        while True:
            page = client.search_runs(
                experiment_ids=[selected.experiment_id], filter_string=filter_string,
                order_by=["metrics.validation_f1 DESC", "attributes.start_time DESC"],
                max_results=1000, page_token=token,
            )
            for run in page:
                tags, metrics = run.data.tags, run.data.metrics
                f1 = metrics.get("validation_f1")
                if (run.info.status != "FINISHED"
                    or tags.get("dataset_sha256") != dataset
                    or tags.get("protocol_id") != protocol
                    or tags.get("selection_partition") != "validation"
                    or not tags.get("model_uri")
                    or not isinstance(f1, (int, float)) or not math.isfinite(f1)):
                    continue
                ap = metrics.get("validation_average_precision")
                output.append({
                    "run_id": run.info.run_id,
                    "run_name": tags.get("mlflow.runName", ""),
                    "model_uri": tags["model_uri"],
                    "validation_f1": float(f1),
                    "validation_average_precision": float(ap) if isinstance(ap, (int, float)) and math.isfinite(ap) else None,
                    "threshold": run.data.params.get("threshold", tags.get("threshold")),
                    "start_time": run.info.start_time or 0,
                    "dataset_sha256": dataset,
                    "protocol_id": protocol,
                })
            token = getattr(page, "token", None)
            if not token:
                break
        output.sort(key=lambda row: (row["validation_f1"], row["validation_average_precision"] if row["validation_average_precision"] is not None else -math.inf, row["start_time"]), reverse=True)
        return output


def registrar_modelo(model_uri, registry_name, alias="candidato", tracking_uri=None, threshold=None):
    """Register the exact logged model and set a versioned, reversible alias.

    The validation-selected threshold must be preserved in model metadata and
    version tags. Consumers loading sklearn models must apply it to class-1
    predict_proba output; the pipeline's default predict threshold is 0.5.
    """
    if not isinstance(model_uri, str) or not model_uri.startswith(("models:/", "runs:/")):
        raise ValueError("Register an exact models:/ or runs:/ URI returned by tracking.")
    if not registry_name or not alias:
        raise ValueError("A registered model name and explicit alias are required.")
    uri = resolve_tracking_uri(tracking_uri)
    with tracking_context(uri) as mlflow:
        client = mlflow.tracking.MlflowClient(tracking_uri=uri, registry_uri=uri)
        model_info = mlflow.models.get_model_info(model_uri)
        metadata = model_info.metadata or {}
        if metadata.get("selection_partition") != "validation":
            raise ValueError("Model metadata must document validation-based selection.")
        recorded_threshold = metadata.get("threshold")
        if recorded_threshold is None:
            raise ValueError("The logged model is missing the selected threshold metadata.")
        recorded_threshold = float(recorded_threshold)
        if not math.isfinite(recorded_threshold) or not 0 <= recorded_threshold <= 1:
            raise ValueError("The logged model threshold is invalid.")
        if threshold is not None and not np.isclose(float(threshold), recorded_threshold, rtol=0, atol=0):
            raise ValueError("The requested threshold differs from the logged model metadata.")
        dataset = _hash_filter(metadata.get("dataset_sha256"), "dataset_sha256")
        protocol = _hash_filter(metadata.get("protocol_id"), "protocol_id")
        before = mlflow.sklearn.load_model(model_uri)
        probe = synthetic_probe()
        before_scores = positive_probability(before, probe)
        previous_version = None
        try:
            registered = client.get_registered_model(registry_name)
            previous_version = registered.aliases.get(alias)
        except mlflow.exceptions.MlflowException as error:
            if getattr(error, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
                raise
        version = mlflow.register_model(model_uri, registry_name, await_registration_for=30)
        version_tags = {
            "threshold": str(recorded_threshold),
            "positive_class": "1",
            "dataset_sha256": dataset,
            "protocol_id": protocol,
            "selection_partition": "validation",
            "source_model_uri": model_uri,
            "task": metadata.get("task", "commercial_loan_renewal_propensity_educational"),
            "registration_is_not_deployment": "true",
        }
        if previous_version is not None:
            version_tags["previous_alias_version"] = str(previous_version)
        for key, value in version_tags.items():
            client.set_model_version_tag(registry_name, version.version, key, value)
        client.set_registered_model_alias(registry_name, alias, version.version)
        alias_version = client.get_model_version_by_alias(registry_name, alias)
        if str(alias_version.version) != str(version.version):
            raise RuntimeError("Registry alias does not resolve to the newly registered version.")
        alias_uri = f"models:/{registry_name}@{alias}"
        after = mlflow.sklearn.load_model(alias_uri)
        after_scores = positive_probability(after, probe)
        np.testing.assert_allclose(before_scores, after_scores, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(before_scores >= recorded_threshold, after_scores >= recorded_threshold)
        return {
            "name": registry_name,
            "version": str(version.version),
            "alias": alias,
            "previous_version": str(previous_version) if previous_version is not None else None,
            "source_model_uri": model_uri,
            "model_uri": alias_uri,
            "tracking_uri": uri,
            "threshold": recorded_threshold,
            "dataset_sha256": dataset,
            "protocol_id": protocol,
            "alias_reload_prediction_parity": True,
            "state": "registered_candidate_not_deployed",
        }
