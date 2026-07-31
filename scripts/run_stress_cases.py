from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from energy_system_simulator.config import (
    BatteryConfig,
    ModelConfig,
    ThermalConfig,
    load_config,
    validate_config,
)
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.exceptions import ConfigurationError
from energy_system_simulator.network import DistributionNetwork

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "examples" / "stress" / "cases.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "stress" / "summary.csv"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """Load committed stress-case definitions."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise AssertionError("Stress case schema_version must be 1")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AssertionError("Stress case file must contain cases")
    return cases


def run_stress_cases(output_path: Path = DEFAULT_OUTPUT) -> list[dict[str, Any]]:
    """Run all committed stress cases and write a compact comparison table."""
    base = load_config(ROOT / "configs" / "example.yaml")
    rows: list[dict[str, Any]] = []
    coverage = _empty_coverage()
    for case in load_cases():
        row = _run_case(base, case)
        rows.append(row)
        _update_coverage(coverage, row)

    missing = sorted(metric for metric, seen in coverage.items() if not seen)
    if missing:
        raise AssertionError(f"Stress suite did not exercise metrics: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _run_case(base: ModelConfig, case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case["id"])
    try:
        config = _case_config(base, case)
        validate_config(config)
    except ConfigurationError as error:
        expected_error = case.get("expect_error")
        if expected_error is None or str(expected_error) not in str(error):
            raise
        return {
            "case": case_id,
            "status": "expected_error",
            "error": str(error),
            **_zero_metrics(),
        }

    if case.get("expect_error") is not None:
        raise AssertionError(f"{case_id} expected a configuration error")

    dt = config.simulation.time_step_hours
    demand = np.asarray(case["demand_mw"], dtype=np.float64)
    renewable = np.asarray(case["renewable_mw"], dtype=np.float64)
    distribution = DistributionNetwork(config.network).prepare_demand(demand)
    dispatch = UnitCommitment(config).solve(renewable, distribution.gross_demand_mw)
    frame = dispatch.frame
    network_shed_mwh = float(distribution.network_capacity_shed_mw.sum() * dt)
    network_shed_cost = network_shed_mwh * config.penalties.lost_load_eur_per_mwh
    total_objective = dispatch.objective_eur + network_shed_cost
    cost_sum = sum(dispatch.cost_components_eur.values()) + network_shed_cost
    if abs(total_objective - cost_sum) > DEFAULT_NUMERICAL_POLICY.objective_reconciliation_eur:
        raise AssertionError(f"{case_id} cost components do not reconcile")

    row = {
        "case": case_id,
        "status": dispatch.solver_status,
        "error": "",
        "objective_eur": total_objective,
        "renewable_curtailed_mwh": _energy(frame["renewable_curtailed_mw"], dt),
        "network_capacity_shed_mwh": network_shed_mwh,
        "dispatch_load_shed_mwh": _energy(frame["source_load_shed_mw"], dt),
        "battery_charge_mwh": _energy(frame["battery_charge_mw"], dt),
        "battery_discharge_mwh": _energy(frame["battery_discharge_mw"], dt),
        "imports_mwh": _energy(frame["imports_mw"], dt),
        "thermal_generation_mwh": _energy(frame["thermal_output_mw"], dt),
        "thermal_starts": float(frame["thermal_startup"].sum()),
        "thermal_shutdowns": float(frame["thermal_shutdown"].sum()),
        "startup_cost_eur": dispatch.cost_components_eur["startup_cost_eur"],
        "shutdown_cost_eur": dispatch.cost_components_eur["shutdown_cost_eur"],
        "import_carbon_cost_eur": dispatch.cost_components_eur["import_carbon_cost_eur"],
        "renewable_curtailment_cost_eur": dispatch.cost_components_eur[
            "renewable_curtailment_cost_eur"
        ],
        "dispatch_load_shedding_cost_eur": dispatch.cost_components_eur[
            "dispatch_load_shedding_cost_eur"
        ],
        "network_capacity_load_shedding_cost_eur": network_shed_cost,
        "terminal_residual_up_hours": (
            dispatch.terminal_commitment_state.residual_minimum_up_hours
        ),
        "terminal_residual_down_hours": (
            dispatch.terminal_commitment_state.residual_minimum_down_hours
        ),
    }
    _check_expectations(case_id, row, case.get("expect_min", {}))
    return row


def _case_config(base: ModelConfig, case: dict[str, Any]) -> ModelConfig:
    mode = str(case["thermal_mode"])
    battery_mode = str(case["battery_mode"])
    return replace(
        base,
        simulation=replace(base.simulation, time_step_hours=1.0),
        solar=replace(base.solar, capacity_mw=0.0),
        wind=replace(base.wind, capacity_mw=0.0),
        thermal=_thermal(base.thermal, mode),
        battery=_battery(base.battery, battery_mode),
        network=replace(
            base.network,
            loss_fraction=0.0,
            transfer_capacity_mw=float(case.get("network_transfer_capacity_mw", 1_000.0)),
        ),
        imports=replace(
            base.imports,
            maximum_power_mw=float(case.get("imports_max_mw", 0.0)),
            price_eur_per_mwh=50.0,
            emission_factor_tonnes_per_mwh=0.1,
        ),
        penalties=replace(
            base.penalties,
            renewable_curtailment_eur_per_mwh=3.0,
            lost_load_eur_per_mwh=1_000.0,
            carbon_price_eur_per_tonne=float(case.get("carbon_price_eur_per_tonne", 0.0)),
        ),
    )


def _thermal(base: ThermalConfig, mode: str) -> ThermalConfig:
    if mode == "off":
        return replace(
            base,
            minimum_output_mw=0.0,
            maximum_output_mw=0.0,
            startup_ramp_mw=0.0,
            shutdown_ramp_mw=0.0,
            variable_cost_eur_per_mwh=0.0,
            no_load_cost_eur_per_hour=100.0,
            emission_factor_tonnes_per_mwh=0.0,
            minimum_up_hours=1.0,
            minimum_down_hours=1.0,
            initial_on=False,
            initial_output_mw=0.0,
            initial_up_time_hours=0.0,
            initial_down_time_hours=10.0,
        )
    if mode == "flexible":
        return replace(
            base,
            minimum_output_mw=10.0,
            maximum_output_mw=100.0,
            ramp_up_mw_per_hour=100.0,
            ramp_down_mw_per_hour=100.0,
            startup_ramp_mw=100.0,
            shutdown_ramp_mw=100.0,
            variable_cost_eur_per_mwh=20.0,
            no_load_cost_eur_per_hour=10.0,
            startup_cost_eur=500.0,
            shutdown_cost_eur=100.0,
            emission_factor_tonnes_per_mwh=0.2,
            minimum_up_hours=1.0,
            minimum_down_hours=1.0,
            initial_on=False,
            initial_output_mw=0.0,
            initial_up_time_hours=0.0,
            initial_down_time_hours=10.0,
        )
    if mode == "dirty":
        return replace(
            _thermal(base, "flexible"),
            minimum_output_mw=0.0,
            variable_cost_eur_per_mwh=10.0,
            emission_factor_tonnes_per_mwh=1.0,
        )
    if mode == "invalid_startup":
        return replace(_thermal(base, "flexible"), startup_ramp_mw=5.0)
    if mode == "terminal_carry":
        return replace(
            _thermal(base, "flexible"),
            minimum_up_hours=3.0,
            minimum_down_hours=1.0,
            terminal_commitment_mode="carry_residual_obligations",
        )
    raise AssertionError(f"Unknown thermal mode: {mode}")


def _battery(base: BatteryConfig, mode: str) -> BatteryConfig:
    if mode == "empty":
        return _battery_values(base, capacity=0.0, power=0.0, initial=0.0)
    if mode == "initial_40":
        return _battery_values(base, capacity=40.0, power=40.0, initial=40.0)
    if mode == "cycle_50":
        return _battery_values(base, capacity=50.0, power=50.0, initial=0.0)
    raise AssertionError(f"Unknown battery mode: {mode}")


def _battery_values(
    base: BatteryConfig, *, capacity: float, power: float, initial: float
) -> BatteryConfig:
    return replace(
        base,
        energy_capacity_mwh=capacity,
        power_capacity_mw=power,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=capacity,
        initial_soc_mwh=initial,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        throughput_cost_eur_per_mwh=1.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode="free",
    )


def _check_expectations(case_id: str, row: dict[str, Any], expectations: dict[str, Any]) -> None:
    for metric, minimum in expectations.items():
        actual = float(row[metric])
        if actual + DEFAULT_NUMERICAL_POLICY.report_rounding < float(minimum):
            raise AssertionError(f"{case_id} expected {metric} >= {minimum}, got {actual}")


def _energy(values: Any, dt: float) -> float:
    return float(values.sum() * dt)


def _zero_metrics() -> dict[str, float]:
    return {
        "objective_eur": 0.0,
        "renewable_curtailed_mwh": 0.0,
        "network_capacity_shed_mwh": 0.0,
        "dispatch_load_shed_mwh": 0.0,
        "battery_charge_mwh": 0.0,
        "battery_discharge_mwh": 0.0,
        "imports_mwh": 0.0,
        "thermal_generation_mwh": 0.0,
        "thermal_starts": 0.0,
        "thermal_shutdowns": 0.0,
        "startup_cost_eur": 0.0,
        "shutdown_cost_eur": 0.0,
        "import_carbon_cost_eur": 0.0,
        "renewable_curtailment_cost_eur": 0.0,
        "dispatch_load_shedding_cost_eur": 0.0,
        "network_capacity_load_shedding_cost_eur": 0.0,
        "terminal_residual_up_hours": 0.0,
        "terminal_residual_down_hours": 0.0,
    }


def _empty_coverage() -> dict[str, bool]:
    return {
        "renewable_curtailed_mwh": False,
        "network_capacity_shed_mwh": False,
        "dispatch_load_shed_mwh": False,
        "battery_charge_mwh": False,
        "battery_discharge_mwh": False,
        "imports_mwh": False,
        "thermal_starts": False,
        "thermal_shutdowns": False,
        "startup_cost_eur": False,
        "shutdown_cost_eur": False,
        "import_carbon_cost_eur": False,
        "renewable_curtailment_cost_eur": False,
        "dispatch_load_shedding_cost_eur": False,
        "network_capacity_load_shedding_cost_eur": False,
        "terminal_residual_up_hours": False,
    }


def _update_coverage(coverage: dict[str, bool], row: dict[str, Any]) -> None:
    for metric in coverage:
        coverage[metric] = coverage[metric] or float(row[metric]) > 0.0


def main() -> int:
    rows = run_stress_cases()
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"stress summary: {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
