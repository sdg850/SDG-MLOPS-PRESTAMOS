"""Unidad 01: comprobar el entorno sin modificar dependencias ni credenciales."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unidad 01: verificar Python y dependencias del caso loan")
    parser.add_argument("--include-dev", action="store_true", help="Comprobar también ejecución de notebooks")
    args = parser.parse_args(argv)
    packages = ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib"]
    if args.include_dev:
        packages += ["ipykernel", "nbformat", "nbclient"]
    versions = {}
    missing = []
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            missing.append(package)
    supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    print(json.dumps({"python": platform.python_version(), "python_supported": supported,
                      "packages": versions, "missing": missing}, ensure_ascii=False, indent=2))
    return 0 if supported and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
