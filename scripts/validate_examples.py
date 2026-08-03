from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "configs/example.yaml",
    "configs/portfolio_hydro.yaml",
    "configs/portfolio_demand_response.yaml",
    "configs/portfolio_nodal_three_bus.yaml",
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    for config in CONFIGS:
        run([sys.executable, "-m", "energy_system_simulator", "validate", "--config", config])
        run(
            [
                sys.executable,
                "-m",
                "energy_system_simulator",
                "simulate",
                "--config",
                config,
                "--dry-run",
            ]
        )
    run([sys.executable, "examples/capacity_expansion.py"])
    print("examples validation ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
