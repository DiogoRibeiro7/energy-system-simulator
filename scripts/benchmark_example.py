from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from energy_system_simulator.config import load_config
from energy_system_simulator.data import load_input_data
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.generation import SolarPlant, WindFarm
from energy_system_simulator.network import DistributionNetwork


def run_benchmark(config_path: Path) -> dict[str, Any]:
    """Build and solve the example dispatch model, returning compact metrics."""
    config = load_config(config_path)
    data = load_input_data(config.paths.input_csv, config.simulation.time_step_hours)
    solar = SolarPlant(config.solar).output_mw(
        data["irradiance_w_m2"].to_numpy(),
        data["ambient_temperature_c"].to_numpy(),
    )
    wind = WindFarm(config.wind).output_mw(data["wind_speed_m_s"].to_numpy())
    renewable = solar + wind
    demand = DistributionNetwork(config.network).prepare_demand(data["demand_mw"].to_numpy())

    model = UnitCommitment(config)
    build_started = perf_counter()
    formulation = model.build_formulation(renewable, demand.gross_demand_mw)
    build_time_seconds = perf_counter() - build_started

    solve_started = perf_counter()
    result = model.solve_formulation(formulation)
    solve_time_seconds = perf_counter() - solve_started

    return {
        "periods": len(data),
        **asdict(formulation.statistics),
        "build_time_seconds": build_time_seconds,
        "solve_time_seconds": solve_time_seconds,
        "solver_termination_status": result.solver_message,
        "objective_eur": result.objective_eur,
        "mip_gap": result.mip_gap,
    }


def main() -> None:
    """Run the committed example benchmark."""
    root = Path(__file__).resolve().parents[1]
    metrics = run_benchmark(root / "configs" / "example.yaml")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
