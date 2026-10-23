"""Unidad 03: identificar datos; DVC init/add solo mediante --dvc explícito.

No configura remotos ni ejecuta git add, commit, push o dvc push. Los bytes del
CSV permanecen privados; versionar un puntero no transporta los datos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import importlib.util
import subprocess

from .config import ROOT


def describe_data(data_path: Path) -> dict:
    source = data_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"No existe el CSV: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"filename": source.name, "sha256": digest.hexdigest(), "size_bytes": source.stat().st_size}


def add_to_dvc(data_path: Path) -> dict:
    source = data_path.resolve()
    try:
        relative = source.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("DVC exige un archivo dentro de este repositorio.") from error
    if not (ROOT / ".git").exists():
        raise RuntimeError("Inicializa o abre el repositorio Git del proyecto antes de usar --dvc.")
    executable = shutil.which("dvc")
    command = [sys.executable, "-m", "dvc"] if importlib.util.find_spec("dvc") else ([executable] if executable else None)
    if command is None:
        raise RuntimeError("DVC no está instalado. Instala el extra de versionamiento documentado.")
    if not (ROOT / ".dvc").is_dir():
        subprocess.run(command + ["init"], cwd=ROOT, check=True)
    subprocess.run(command + ["add", str(relative)], cwd=ROOT, check=True)
    pointer = relative.with_name(relative.name + ".dvc")
    ignored = subprocess.run(["git", "check-ignore", "--quiet", str(pointer)], cwd=ROOT, check=False).returncode == 0
    return {"pointer": pointer.as_posix(), "pointer_ignored_by_git": ignored,
            "note": "El puntero identifica los datos; confirmar caché/remoto antes de prometer recuperación."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unidad 03: hash de datos y versionamiento DVC opt-in")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/Dataset Renovacion_prestamo.csv")
    parser.add_argument("--dvc", action="store_true", help="Ejecutar dvc init si falta y dvc add; no publicar datos")
    args = parser.parse_args(argv)
    result = describe_data(args.data)
    if args.dvc:
        result["dvc"] = add_to_dvc(args.data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.dvc:
        if result["dvc"]["pointer_ignored_by_git"]:
            print("Añade una excepción para *.dvc en data/raw a .gitignore antes de versionar el puntero.")
        print("Revisa y versiona solo metadatos .dvc y configuración sin secretos en Git. No añadas el CSV.")
        print("dvc push requiere un remoto privado previamente configurado; este script no lo configura ni ejecuta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
