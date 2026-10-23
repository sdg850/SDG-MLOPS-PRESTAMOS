"""Move the project's existing SQLite MLflow history into mlruns/ once.

No backend conversion and no experiment deletion. Back up the complete old store
before rebasing its own absolute paths. Run while no training/tracking is active.
"""
from pathlib import Path
import json
import shutil
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def migrate(root=ROOT):
    root = Path(root).resolve()
    source, target = root / "reports/tracking", root / "mlruns"
    database = source / "mlflow.db"
    if not database.exists():
        return {"status": "not_needed", "destination": str(target)}
    target.mkdir(exist_ok=True)
    conflicts = [p.name for p in source.iterdir() if p.name != ".gitkeep" and (target / p.name).exists()]
    if conflicts:
        raise FileExistsError("Destination already contains tracking state; refusing to merge.")
    backup = Path(tempfile.mkdtemp(prefix="loan-mlflow-backup-")) / "tracking"
    shutil.copytree(source, backup)
    # Capture a consistent database snapshot, including any committed WAL state.
    with sqlite3.connect(database) as old, sqlite3.connect(backup / "mlflow.db") as saved:
        old.backup(saved)
        assert old.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    old.close()
    saved.close()
    for item in source.iterdir():
        if item.name != ".gitkeep":
            shutil.move(str(item), target / item.name)
    old_prefix, new_prefix = str(source), str(target)
    def quote(name):
        return '"' + name.replace('"', '""') + '"'
    updated = 0
    with sqlite3.connect(target / "mlflow.db") as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'") if not row[0].startswith("sqlite_")]
        counts = {t: connection.execute(f"SELECT count(*) FROM {quote(t)}").fetchone()[0] for t in tables}
        for table in tables:
            columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quote(table)})")]
            for column in columns:
                cursor = connection.execute(
                    f"UPDATE {quote(table)} SET {quote(column)}=replace({quote(column)},?,?) "
                    f"WHERE typeof({quote(column)})='text' AND instr({quote(column)},?)>0",
                    (old_prefix, new_prefix, old_prefix),
                )
                updated += cursor.rowcount
        assert counts == {t: connection.execute(f"SELECT count(*) FROM {quote(t)}").fetchone()[0] for t in tables}
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()
    # MLmodel YAML and local reports also contain absolute URIs. Serialized models
    # are untouched; only text documents containing this exact prefix are rebased.
    for base in (target, root / "reports"):
        for path in base.rglob("*"):
            if path.is_file() and (path.suffix in {".json", ".yaml", ".yml"} or path.name == "MLmodel"):
                text = path.read_text(encoding="utf-8")
                if old_prefix in text:
                    path.write_text(text.replace(old_prefix, new_prefix), encoding="utf-8")
    remaining = list(source.iterdir())
    if remaining == []:
        source.rmdir()
    return {"status": "migrated", "destination": str(target), "backup": str(backup),
            "rebased_values": updated, "table_row_counts_preserved": True}


if __name__ == "__main__":
    print(json.dumps(migrate(), indent=2))
