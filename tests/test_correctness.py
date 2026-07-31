from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np

from energy_system_simulator.config import (
    BatteryConfig,
    ModelConfig,
    ThermalConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.simulation import SimulationEngine


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _dispatch_test_config(
    *,
    time_step_hours: float,
    thermal: ThermalConfig,
    battery: BatteryConfig,
) -> ModelConfig:
    config = _example_config()
    return replace(
        config,
        simulation=replace(config.simulation, time_step_hours=time_step_hours),
        thermal=thermal,
        battery=battery,
        imports=replace(config.imports, maximum_power_mw=0.0),
    )


def test_battery_charge_and_discharge_are_mutually_exclusive() -> None:
    result = SimulationEngine(_example_config()).run()
    product = result.timeseries["battery_charge_mw"] * result.timeseries["battery_discharge_mw"]
    assert np.allclose(product, 0.0, atol=1e-7)


def test_cost_and_energy_reconciliation_are_reported() -> None:
    result = SimulationEngine(_example_config()).run()
    summary = result.summary
    assert summary["objective_reconciliation_error_eur"] <= 1e-4
    assert summary["energy_reconciliation"]["max_abs_source_balance_residual_mw"] <= 1e-6
    assert summary["energy_reconciliation"]["max_abs_delivered_demand_residual_mw"] <= 1e-6
    assert summary["energy_reconciliation"]["max_abs_battery_energy_residual_mwh"] <= 1e-6
    assert result.solver_status == "optimal"


def test_exact_terminal_battery_state_of_charge_is_enforced() -> None:
    config = _example_config()
    target_soc = config.battery.initial_soc_mwh
    exact_battery = replace(
        config.battery,
        terminal_soc_mode=cast(Literal["minimum", "exact", "cyclic", "free"], "exact"),
        minimum_final_soc_mwh=target_soc,
    )
    result = SimulationEngine(replace(config, battery=exact_battery)).run()
    assert result.summary["final_battery_soc_mwh"] == target_soc


def test_subhourly_minimum_up_duration_uses_ceiling_periods() -> None:
    config = _example_config()
    thermal = replace(
        config.thermal,
        minimum_output_mw=20.0,
        maximum_output_mw=100.0,
        ramp_up_mw_per_hour=200.0,
        ramp_down_mw_per_hour=200.0,
        startup_ramp_mw=100.0,
        shutdown_ramp_mw=100.0,
        minimum_up_hours=1.5,
        minimum_down_hours=0.5,
        initial_on=False,
        initial_output_mw=0.0,
        initial_up_time_hours=0.0,
        initial_down_time_hours=2.0,
    )
    battery = replace(
        config.battery,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=100.0,
        initial_soc_mwh=0.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode=cast(Literal["minimum", "exact", "cyclic", "free"], "free"),
    )
    test_config = _dispatch_test_config(
        time_step_hours=0.5,
        thermal=thermal,
        battery=battery,
    )

    result = UnitCommitment(test_config).solve(
        renewable_available_mw=np.zeros(4),
        gross_demand_mw=np.array([60.0, 0.0, 0.0, 0.0]),
    )

    assert result.frame["thermal_on"].iloc[:3].tolist() == [1, 1, 1]


def test_residual_initial_up_time_obligation_is_enforced() -> None:
    config = _example_config()
    thermal = replace(
        config.thermal,
        minimum_output_mw=20.0,
        maximum_output_mw=100.0,
        ramp_up_mw_per_hour=200.0,
        ramp_down_mw_per_hour=200.0,
        startup_ramp_mw=100.0,
        shutdown_ramp_mw=100.0,
        minimum_up_hours=3.0,
        initial_on=True,
        initial_output_mw=50.0,
        initial_up_time_hours=1.0,
        initial_down_time_hours=0.0,
    )
    battery = replace(
        config.battery,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=100.0,
        initial_soc_mwh=0.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode=cast(Literal["minimum", "exact", "cyclic", "free"], "free"),
    )
    test_config = _dispatch_test_config(
        time_step_hours=1.0,
        thermal=thermal,
        battery=battery,
    )

    result = UnitCommitment(test_config).solve(
        renewable_available_mw=np.zeros(3),
        gross_demand_mw=np.array([50.0, 0.0, 0.0]),
    )

    assert result.frame["thermal_on"].iloc[:2].tolist() == [1, 1]
