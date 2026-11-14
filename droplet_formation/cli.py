"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from droplet_formation.study import repository_root, run_reference_study


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the microfluidic droplet-formation reference study."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repository_root() / "results",
        help="Directory for figures, tables, and the HTML report.",
    )
    args = parser.parse_args(argv)
    run_reference_study(args.output)
