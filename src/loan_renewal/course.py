"""Unit 06 entry points adapted to loan renewal, with one shared implementation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DATA_PATH, ARTIFACTS, REPORTS_DIR, EXPERIMENT, tracking_uri as default_tracking_uri


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(action, argv=None):
    """Execute a course action; paths default to the project, never the cwd."""
    parser = argparse.ArgumentParser(description=f"Loan renewal Unit 06: {action}")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "experiments")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR / "experiments")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--registry-name", default="LoanRenewal-Propensity")
    parser.add_argument("--alias", default="candidato")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    uri = args.tracking_uri or default_tracking_uri()

    def execute_experiment(method):
        from .pipeline import train
        from .tracking import track_report
        destination = output / method
        report_destination = reports / method
        report = train(args.data, destination, report_dir=report_destination, tune=method != "base", search_method="random" if method == "random" else "grid")
        record = track_report(report, report_destination, tracking_uri=uri, experiment=args.experiment)
        _write(report_destination / "tracking.json", record)
        return record

    def compare():
        from .experimentos import compare_runs
        source = next((reports / name / "tracking.json" for name in ("grid", "random", "base")
                       if (reports / name / "tracking.json").exists()), None)
        if source is None:
            raise FileNotFoundError("Run a tracked grid/random experiment before comparison.")
        identity = json.loads(source.read_text())
        records = compare_runs(uri, experiment=args.experiment,
                               dataset_sha256=identity["dataset_sha256"], protocol_id=identity["protocol_id"])
        if not records:
            raise ValueError("No finished comparable runs with valid validation metrics.")
        _write(reports / "comparison_runs.json", records)
        return records

    def best():
        record = compare()[0]
        _write(reports / "best_run.json", record)
        return record

    def registry():
        from .experimentos import registrar_modelo
        source = reports / "best_run.json"
        if not source.exists():
            raise FileNotFoundError("Run experiments best before registry.")
        selected = json.loads(source.read_text())
        record = registrar_modelo(selected["model_uri"], args.registry_name,
                                  alias=args.alias, tracking_uri=uri, threshold=selected["threshold"])
        _write(reports / "registry.json", record)
        return record

    if action in {"base", "grid", "random"}:
        result = execute_experiment(action)
    elif action == "compare":
        result = compare()
    elif action == "best":
        result = best()
    elif action == "registry":
        result = registry()
    elif action == "all":
        result = {method: execute_experiment(method) for method in ("base", "grid", "random")}
        result["best"] = best()
        result["registry"] = registry()
    else:
        raise ValueError(f"Unknown course action: {action}")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result
