from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from energy_system_simulator.config import load_config
from energy_system_simulator.data import load_input_data
from energy_system_simulator.reporting import write_outputs
from energy_system_simulator.simulation import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="energy-sim",
        description="Simulate a hybrid electricity system.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate configuration and input data")
    validate.add_argument("--config", type=Path, required=True)

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
    config = load_config(args.config)

    if args.command == "validate":
        data = load_input_data(config.paths.input_csv, config.simulation.time_step_hours)
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

    raise RuntimeError(f"Unsupported command: {args.command}")
