from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from energy_system_simulator.config import load_config
from energy_system_simulator.data import load_input_data
from energy_system_simulator.exceptions import EnergySystemError
from energy_system_simulator.metadata import get_package_version
from energy_system_simulator.reporting import write_outputs
from energy_system_simulator.simulation import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="energy-sim",
        description="Simulate a hybrid electricity system.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate configuration and input data")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit validation errors as JSON",
    )

    simulate = subparsers.add_parser("simulate", help="Run the configured simulation")
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    json_output = bool(getattr(args, "json_output", False))
    try:
        config = load_config(args.config)

        if args.command == "validate":
            data = load_input_data(config.paths.input_csv, config.simulation.time_step_hours)
            if json_output:
                print(json.dumps({"ok": True, "periods": len(data)}, sort_keys=True))
            else:
                print(f"Configuration valid. Input contains {len(data)} periods.")
            return

        if args.command == "simulate":
            result = SimulationEngine(config).run()
            write_outputs(
                result,
                config.paths.output_directory,
                config=config,
                config_path=args.config,
                create_plots=not args.no_plots,
            )
            print(f"Simulation complete: {config.paths.output_directory}")
            print(f"Objective: EUR {result.objective_eur:,.2f}")
            print(f"Unserved energy: {result.summary['unserved_energy_mwh']:.3f} MWh")
            print(f"Renewable share: {result.summary['renewable_share_of_primary_generation']:.2%}")
            return
    except EnergySystemError as error:
        if json_output:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "type": error.__class__.__name__,
                            "message": str(error),
                        },
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"Validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    raise RuntimeError(f"Unsupported command: {args.command}")
