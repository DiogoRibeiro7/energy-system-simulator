from __future__ import annotations

import argparse
from pathlib import Path

from energy_system_simulator.reporting import compare_output_directories


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two or more simulation output folders.")
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "comparison.md")
    args = parser.parse_args()
    table = compare_output_directories(args.outputs, args.output)
    print(f"Comparison report written: {args.output}")
    print(f"Compared metrics: {table['metric'].nunique() if not table.empty else 0}")


if __name__ == "__main__":
    main()
