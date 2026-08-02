from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
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
    RenewableGeneratorConfig,
    StorageUnitConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
    load_config,
)
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.dispatch import UnitCommitment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "verification"


def run_verification_benchmarks(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Run deterministic verification benchmarks and write JSON and CSV summaries."""
    benchmarks = [
        _dispatch_case(
            "single_period_merit_order",
            _single_period_config(),
            np.array([0.0]),
            np.array([80.0]),
        ),
        _dispatch_case(
            "battery_arbitrage",
            _battery_arbitrage_config(),
            np.zeros(2),
            np.array([10.0, 10.0]),
            import_price_series=np.array([10.0, 100.0]),
        ),
        _dispatch_case(
            "three_bus_loop_flow",
            _three_bus_config(),
            np.array([90.0]),
            np.array([90.0]),
            demand_profiles_mw={"load_b": np.array([45.0]), "load_c": np.array([45.0])},
            renewable_availability_by_asset_mw={"solar": np.array([90.0])},
        ),
    ]
    payload = {
        "schema_version": 1,
        "tolerance_policy": {
            "power_abs_tolerance_mw": DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw,
            "energy_abs_tolerance_mwh": DEFAULT_NUMERICAL_POLICY.energy_reconciliation_mwh,
            "objective_abs_tolerance_eur": DEFAULT_NUMERICAL_POLICY.objective_reconciliation_eur,
        },
        "benchmarks": benchmarks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benchmarks[0]))
        writer.writeheader()
        writer.writerows(benchmarks)
    return payload


def _dispatch_case(
    case_id: str,
    config: ModelConfig,
    renewable_mw: np.ndarray,
    demand_mw: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    model = UnitCommitment(config)
    build_started = perf_counter()
    formulation = model.build_formulation(renewable_mw, demand_mw, **kwargs)
    build_time_seconds = perf_counter() - build_started

    solve_started = perf_counter()
    result = model.solve_formulation(formulation)
    solve_time_seconds = perf_counter() - solve_started

    return {
        "case_id": case_id,
        "periods": len(demand_mw),
        **asdict(formulation.statistics),
        "build_time_seconds": build_time_seconds,
        "solve_time_seconds": solve_time_seconds,
        "solver_status": result.solver_status,
        "objective_eur": result.objective_eur,
    }


def _base_config() -> ModelConfig:
    base = load_config(ROOT / "configs" / "example.yaml")
    battery = replace(
        base.battery,
        energy_capacity_mwh=0.0,
        power_capacity_mw=0.0,
        charge_power_capacity_mw=None,
        discharge_power_capacity_mw=None,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=0.0,
        initial_soc_mwh=0.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode="free",
        degradation_bands=(),
    )
    thermal = replace(
        base.thermal,
        minimum_output_mw=0.0,
        maximum_output_mw=0.0,
        ramp_up_mw_per_hour=1_000.0,
        ramp_down_mw_per_hour=1_000.0,
        startup_ramp_mw=0.0,
        shutdown_ramp_mw=0.0,
        variable_cost_eur_per_mwh=0.0,
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
        minimum_fuel_input_mwh_per_hour=0.0,
        heat_rate_segments=(),
        startup_categories=(),
    )
    imports = ImportConfig(
        maximum_power_mw=0.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    return replace(
        base,
        thermal=thermal,
        battery=battery,
        imports=imports,
        network=replace(base.network, loss_fraction=0.0, transfer_capacity_mw=1_000.0),
        penalties=replace(
            base.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
        portfolio=replace(
            base.portfolio,
            buses=(BusConfig(id="system", zone_id="zone"),),
            lines=(),
            renewable_generators=(),
            thermal_generators=(
                ThermalGeneratorConfig(
                    id="offline",
                    bus_id="system",
                    fuel_id="gas",
                    config=thermal,
                ),
            ),
            storage_units=(),
            hydro_units=(),
            imports=(ImportResourceConfig(id="imports", bus_id="system", config=imports),),
            demand=(DemandConfig(id="load", bus_id="system", time_series_key="load_mw"),),
        ),
    )


def _single_period_config() -> ModelConfig:
    base = _base_config()
    cheap = replace(
        base.thermal,
        name="cheap",
        maximum_output_mw=50.0,
        startup_ramp_mw=50.0,
        shutdown_ramp_mw=50.0,
        variable_cost_eur_per_mwh=5.0,
    )
    peaker = replace(
        base.thermal,
        name="peaker",
        maximum_output_mw=100.0,
        startup_ramp_mw=100.0,
        shutdown_ramp_mw=100.0,
        variable_cost_eur_per_mwh=50.0,
    )
    return replace(
        base,
        thermal=cheap,
        portfolio=replace(
            base.portfolio,
            thermal_generators=(
                ThermalGeneratorConfig(id="cheap", bus_id="system", fuel_id="gas", config=cheap),
                ThermalGeneratorConfig(id="peaker", bus_id="system", fuel_id="gas", config=peaker),
            ),
        ),
    )


def _battery_arbitrage_config() -> ModelConfig:
    base = _base_config()
    battery = replace(
        base.battery,
        energy_capacity_mwh=10.0,
        power_capacity_mw=10.0,
        maximum_soc_mwh=10.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        throughput_cost_eur_per_mwh=0.0,
    )
    imports = ImportConfig(
        maximum_power_mw=20.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    return replace(
        base,
        battery=battery,
        imports=imports,
        portfolio=replace(
            base.portfolio,
            storage_units=(StorageUnitConfig(id="battery", bus_id="system", config=battery),),
            imports=(ImportResourceConfig(id="imports", bus_id="system", config=imports),),
        ),
    )


def _three_bus_config() -> ModelConfig:
    base = _base_config()
    return replace(
        base,
        network=NetworkConfig(
            loss_fraction=0.0,
            transfer_capacity_mw=1_000.0,
            network_mode="nodal",
            slack_bus_id="a",
        ),
        portfolio=replace(
            base.portfolio,
            buses=(
                BusConfig(id="a", zone_id="zone"),
                BusConfig(id="b", zone_id="zone"),
                BusConfig(id="c", zone_id="zone"),
            ),
            lines=(
                TransmissionLineConfig(
                    id="ab",
                    from_bus_id="a",
                    to_bus_id="b",
                    susceptance=10.0,
                    capacity_mw=100.0,
                ),
                TransmissionLineConfig(
                    id="bc",
                    from_bus_id="b",
                    to_bus_id="c",
                    susceptance=10.0,
                    capacity_mw=100.0,
                ),
                TransmissionLineConfig(
                    id="ac",
                    from_bus_id="a",
                    to_bus_id="c",
                    susceptance=10.0,
                    capacity_mw=100.0,
                ),
            ),
            renewable_generators=(
                RenewableGeneratorConfig(
                    id="solar",
                    kind="solar",
                    bus_id="a",
                    capacity_mw=1_000.0,
                    time_series_key="solar_mw",
                ),
            ),
            demand=(
                DemandConfig(id="load_b", bus_id="b", time_series_key="load_b_mw"),
                DemandConfig(id="load_c", bus_id="c", time_series_key="load_c_mw"),
            ),
        ),
    )


def main() -> None:
    payload = run_verification_benchmarks()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
