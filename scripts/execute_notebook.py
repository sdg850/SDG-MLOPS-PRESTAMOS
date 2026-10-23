"""Execute a notebook in a fresh kernel; fail on any cell error."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
args = parser.parse_args()
path = args.notebook.resolve()
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=600, kernel_name=notebook.metadata.kernelspec.name, resources={"metadata": {"path": str(path.parent)}})
client.execute()
# Execution outputs are intentionally kept private (model metrics and no rows).
out = Path(__file__).resolve().parents[1] / "reports/notebook.executed.ipynb"
out.parent.mkdir(exist_ok=True, parents=True)
nbformat.write(notebook, out)
print(f"Executed {len(notebook.cells)} cells without errors: {out}")
