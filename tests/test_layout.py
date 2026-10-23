"""Separate entry points and preservation of existing tracking identities."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_numbered_pipeline_scripts_work_outside_repository(self):
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT)
        with tempfile.TemporaryDirectory() as cwd:
            for path in sorted((ROOT / 'pipelines').glob('*.py')):
                with self.subTest(path=path.name):
                    result = subprocess.run([sys.executable, str(path), '--help'], cwd=cwd,
                                            env=environment, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)
            self.assertEqual(list(Path(cwd).iterdir()), [])

    def test_tracking_migration_preserves_records_artifacts_and_backup(self):
        spec = importlib.util.spec_from_file_location('migrate_tracking', ROOT / 'scripts/migrate_tracking.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'reports/tracking'
            source.mkdir(parents=True)
            artifact = source / 'mlflow-artifacts' / 'synthetic-model'
            artifact.mkdir(parents=True)
            (artifact / 'model.pkl').write_bytes(b'opaque synthetic bytes')
            (artifact / 'MLmodel').write_text('artifact_path: '+artifact.as_uri())
            with sqlite3.connect(source / 'mlflow.db') as db:
                db.execute('CREATE TABLE runs (id TEXT PRIMARY KEY, artifact_uri TEXT)')
                db.execute('INSERT INTO runs VALUES (?,?)', ('synthetic-id', artifact.as_uri()))
            report = root / 'reports/tracking.json'
            report.write_text(json.dumps({'tracking_uri':'sqlite:///'+str(source / 'mlflow.db')}))
            result = module.migrate(root)
            self.assertEqual(result['status'], 'migrated')
            self.assertTrue(result['table_row_counts_preserved'])
            self.assertTrue((Path(result['backup']) / 'mlflow.db').exists())
            target = root / 'mlruns'
            with sqlite3.connect(target / 'mlflow.db') as db:
                self.assertEqual(db.execute('SELECT * FROM runs').fetchall(), [('synthetic-id',(target / 'mlflow-artifacts/synthetic-model').as_uri())])
            self.assertEqual((target / 'mlflow-artifacts/synthetic-model/model.pkl').read_bytes(), b'opaque synthetic bytes')
            self.assertNotIn(str(source), (target / 'mlflow-artifacts/synthetic-model/MLmodel').read_text())
            self.assertIn(str(target), report.read_text())
            self.assertFalse(source.exists())
            self.assertEqual(module.migrate(root)['status'], 'not_needed')

    def test_existing_tracking_destination_is_never_overwritten(self):
        spec = importlib.util.spec_from_file_location('migrate_tracking', ROOT / 'scripts/migrate_tracking.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'reports/tracking', root / 'mlruns'
            source.mkdir(parents=True)
            target.mkdir()
            (source / 'mlflow.db').write_bytes(b'source')
            (target / 'mlflow.db').write_bytes(b'destination')
            with self.assertRaises(FileExistsError):module.migrate(root)
            self.assertEqual((source / 'mlflow.db').read_bytes(), b'source')
            self.assertEqual((target / 'mlflow.db').read_bytes(), b'destination')
