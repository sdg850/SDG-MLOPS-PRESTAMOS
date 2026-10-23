"""A Git-managed source remains available without a DVC data remote."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from loan_renewal import versioning


class GitSourceTests(unittest.TestCase):
    def test_git_source_is_not_added_to_dvc_or_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            source = root / 'data/raw/reference.csv'
            source.parent.mkdir(parents=True)
            original = b'synthetic;value\n1;2\n'
            source.write_bytes(original)
            subprocess.run(['git', 'add', 'data/raw/reference.csv'], cwd=root, check=True)
            with patch.object(versioning, 'ROOT', root), \
                    patch.object(versioning.importlib.util, 'find_spec') as find_spec:
                result = versioning.add_to_dvc(source)
            self.assertEqual(result['method'], 'git')
            self.assertEqual(result['path'], 'data/raw/reference.csv')
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(source.with_suffix('.csv.dvc').exists())
            self.assertFalse((root / '.dvc').exists())
            find_spec.assert_not_called()


if __name__ == '__main__':
    unittest.main()
