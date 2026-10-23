"""Run the staged training/validation/tracking workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loan_renewal.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["pipeline", *sys.argv[1:]]))
