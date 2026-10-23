"""Unified CLI contracts from outside the repository; no customer data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def cli(arguments, cwd):
    environment = dict(os.environ)
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "loan_renewal", *arguments], cwd=cwd,
        env=environment, capture_output=True, text=True, timeout=30,
    )


class UnifiedEntryTests(unittest.TestCase):
    def test_commands_offer_help_from_an_unrelated_working_directory(self):
        entries = [[name, "--help"] for name in (
            "train", "predict", "pipeline", "version-data", "check-env", "experiments",
        )]
        entries.extend(["experiments", action, "--help"] for action in (
            "base", "grid", "random", "compare", "best", "registry", "all",
        ))
        with tempfile.TemporaryDirectory() as directory:
            for arguments in entries:
                with self.subTest(arguments=arguments):
                    completed = cli(arguments, directory)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("usage:", completed.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_versioning_default_only_reports_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "invented.csv"
            contents = b"synthetic;value\n1;2\n"
            fixture.write_bytes(contents)
            completed = cli(["version-data", "--data", str(fixture)], directory)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["filename"], "invented.csv")
            self.assertEqual(report["sha256"], hashlib.sha256(contents).hexdigest())
            self.assertEqual(report["size_bytes"], len(contents))
            self.assertEqual(fixture.read_bytes(), contents)
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["invented.csv"])
            self.assertNotIn("dvc", report)

    def test_unknown_training_argument_fails_before_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = cli(["train", "--invented-option"], directory)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("unrecognized arguments", completed.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
