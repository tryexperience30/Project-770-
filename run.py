#!/usr/bin/env python3
"""Run the droplet-formation reference study from the repository root.

    python -m pip install -r requirements.txt
    python run.py

Results are written to ``results/``. Open ``results/index.html`` in a browser
to view the KaTeX report, figures, and tabulated outputs.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from droplet_formation.study import run_reference_study


def main() -> int:
    output_dir = ROOT / "results"
    print("Droplet-formation solver")
    print("Running the analytical field, Chebyshev verification, and force balance.")
    print(f"Output directory: {output_dir}")
    print()
    try:
        artefacts = run_reference_study(output_dir)
    except KeyboardInterrupt:
        print("\nInterrupted before the calculation finished.", file=sys.stderr)
        return 130
    except Exception as exc:
        print("\nCalculation failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    figures = sorted(path.name for path in artefacts.figures_dir.glob("*.png"))
    tables = sorted(path.name for path in artefacts.data_dir.glob("*.csv"))
    print()
    print("=" * 64)
    print("Calculation completed successfully.")
    print()
    print("Figures")
    print(f"  {artefacts.figures_dir}")
    for name in figures:
        print(f"    {name}")
    print()
    print("Numerical tables (CSV)")
    print(f"  {artefacts.data_dir}")
    for name in tables:
        print(f"    {name}")
    print()
    print("HTML report (open in a browser)")
    print(f"  {artefacts.html_path}")
    print()
    print("Text summary")
    print(f"  {artefacts.summary_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
