from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from energy_system_simulator.config import (
    BusConfig,
    DemandConfig,
    ImportConfig,
    ImportResourceConfig,
    ModelConfig,
    NetworkConfig,
    StorageUnitConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "benchmarks"


def run_scaling_benchmarks(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict[str, Any]]:
    """Run deterministic CI-size scaling cases and write table outputs."""
    rows: list[dict[str, Any]] = []
    for periods in (4, 8, 16):
        rows.append(_run_case("periods", periods=periods))
    for units in (1, 2, 4):
        rows.append(_run_case("thermal_units", thermal_units=units))
    for storage_assets in (0, 1, 2):
        rows.append(_run_case("storage_assets", storage_assets=storage_assets))
    for buses in (1, 2, 3):
        rows.append(_run_case("buses", buses=buses))
    for scenarios in (1, 2, 4):
        rows.append(_run_case("scenarios", scenarios=scenarios))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "scaling.md").write_text(_markdown_table(rows), encoding="utf-8")
    return rows


def _run_case(
    axis: str,
    *,
    periods: int = 8,
    thermal_units: int = 1,
    storage_assets: int = 0,
    buses: int = 1,
    scenarios: int = 1,
) -> dict[str, Any]:
    config = _config(thermal_units=thermal_units, storage_assets=storage_assets, buses=buses)
    renewable = np.zeros(periods, dtype=np.float64)
    demand = np.full(periods, 40.0, dtype=np.float64)
    kwargs = _nodal_inputs(periods, buses) if buses > 1 else {}
    total_build = 0.0
    total_solve = 0.0
    statistics = None
    objective = 0.0
    status = ""
    for _ in range(scenarios):
        model = UnitCommitment(config)
        build_started = perf_counter()
        formulation = model.build_formulation(renewable, demand, **kwargs)
        total_build += perf_counter() - build_started
        solve_started = perf_counter()
        result = model.solve_formulation(formulation)
        total_solve += perf_counter() - solve_started
        statistics = formulation.statistics
        objective += result.objective_eur
        status = result.solver_status
    assert statistics is not None
    return {
        "axis": axis,
        "periods": periods,
        "thermal_units": thermal_units,
        "storage_assets": storage_assets,
        "buses": buses,
        "scenarios": scenarios,
        "continuous_variables": statistics.continuous_variables,
        "binary_variables": statistics.binary_variables,
        "linear_constraints": statistics.linear_constraints,
        "matrix_nonzeros": statistics.matrix_nonzeros,
        "build_time_seconds": total_build,
        "solve_time_seconds": total_solve,
        "solver_status": status,
        "objective_eur": objective,
    }


def _config(*, thermal_units: int, storage_assets: int, buses: int) -> ModelConfig:
    base = load_config(ROOT / "configs" / "example.yaml")
    imports = ImportConfig(
        maximum_power_mw=0.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    thermal_configs = tuple(
        _thermal(base.thermal, unit_index=index, maximum_output_mw=60.0)
        for index in range(thermal_units)
    )
    storage_config = replace(
        base.battery,
        energy_capacity_mwh=10.0 if storage_assets else 0.0,
        power_capacity_mw=10.0 if storage_assets else 0.0,
        charge_power_capacity_mw=None,
        discharge_power_capacity_mw=None,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=10.0 if storage_assets else 0.0,
        initial_soc_mwh=0.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode="free",
        throughput_cost_eur_per_mwh=0.0,
        degradation_bands=(),
    )
    bus_ids = tuple(f"bus_{index}" for index in range(buses))
    network = (
        NetworkConfig(loss_fraction=0.0, transfer_capacity_mw=1_000.0)
        if buses == 1
        else NetworkConfig(
            loss_fraction=0.0,
            transfer_capacity_mw=1_000.0,
            network_mode="nodal",
            slack_bus_id=bus_ids[0],
        )
    )
    return replace(
        base,
        thermal=thermal_configs[0],
        battery=storage_config,
        imports=imports,
        network=network,
        penalties=replace(
            base.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
        portfolio=replace(
            base.portfolio,
            buses=tuple(BusConfig(id=bus_id, zone_id="zone") for bus_id in bus_ids),
            lines=_lines(bus_ids),
            renewable_generators=(),
            thermal_generators=tuple(
                ThermalGeneratorConfig(
                    id=f"thermal_{index}",
                    bus_id=bus_ids[0],
                    fuel_id="gas",
                    config=thermal_config,
                )
                for index, thermal_config in enumerate(thermal_configs)
            ),
            storage_units=tuple(
                StorageUnitConfig(id=f"battery_{index}", bus_id=bus_ids[0], config=storage_config)
                for index in range(storage_assets)
            ),
            hydro_units=(),
            imports=(ImportResourceConfig(id="imports", bus_id=bus_ids[0], config=imports),),
            demand=tuple(
                DemandConfig(id=f"load_{index}", bus_id=bus_id, time_series_key=f"load_{index}_mw")
                for index, bus_id in enumerate(bus_ids)
            ),
        ),
    )


def _thermal(base: ThermalConfig, *, unit_index: int, maximum_output_mw: float) -> ThermalConfig:
    return replace(
        base,
        name=f"thermal {unit_index}",
        minimum_output_mw=0.0,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=1_000.0,
        ramp_down_mw_per_hour=1_000.0,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=20.0 + unit_index,
        no_load_cost_eur_per_hour=0.0,
        startup_cost_eur=0.0,
        shutdown_cost_eur=0.0,
        emission_factor_tonnes_per_mwh=0.0,
        minimum_up_hours=1.0,
        minimum_down_hours=1.0,
        initial_on=False,
        initial_output_mw=0.0,
        initial_up_time_hours=0.0,
        initial_down_time_hours=10.0,
        terminal_commitment_mode="forbid_incomplete_transitions",
        terminal_on=None,
        heat_rate_segments=(),
        startup_categories=(),
    )


def _lines(bus_ids: tuple[str, ...]) -> tuple[TransmissionLineConfig, ...]:
    if len(bus_ids) == 1:
        return ()
    return tuple(
        TransmissionLineConfig(
            id=f"{bus_ids[index]}_{bus_ids[index + 1]}",
            from_bus_id=bus_ids[index],
            to_bus_id=bus_ids[index + 1],
            susceptance=10.0,
            capacity_mw=1_000.0,
        )
        for index in range(len(bus_ids) - 1)
    )


def _nodal_inputs(periods: int, buses: int) -> dict[str, dict[str, np.ndarray]]:
    demand_share = np.full(periods, 40.0 / buses, dtype=np.float64)
    return {"demand_profiles_mw": {f"load_{index}": demand_share.copy() for index in range(buses)}}


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    rows = run_scaling_benchmarks()
    print(_markdown_table(rows), end="")


if __name__ == "__main__":
    main()
