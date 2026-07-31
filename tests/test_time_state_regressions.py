from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from energy_system_simulator.config import BatteryConfig, ModelConfig, ThermalConfig, load_config
from energy_system_simulator.dispatch import UnitCommitment


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _direct_config(
    *,
    time_step_hours: float,
    thermal: ThermalConfig,
    battery: BatteryConfig,
    import_limit_mw: float = 0.0,
    import_price_eur_per_mwh: float = 0.0,
    import_emission_factor_tonnes_per_mwh: float = 0.0,
    lost_load_eur_per_mwh: float = 10_000.0,
    curtailment_eur_per_mwh: float = 0.0,
    carbon_price_eur_per_tonne: float = 0.0,
) -> ModelConfig:
    config = _example_config()
    return replace(
        config,
        simulation=replace(config.simulation, time_step_hours=time_step_hours),
        thermal=thermal,
        battery=battery,
        network=replace(config.network, loss_fraction=0.0),
        imports=replace(
            config.imports,
            maximum_power_mw=import_limit_mw,
            price_eur_per_mwh=import_price_eur_per_mwh,
            emission_factor_tonnes_per_mwh=import_emission_factor_tonnes_per_mwh,
        ),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=curtailment_eur_per_mwh,
            lost_load_eur_per_mwh=lost_load_eur_per_mwh,
            carbon_price_eur_per_tonne=carbon_price_eur_per_tonne,
        ),
    )


def _thermal_unit(
    *,
    initial_on: bool,
    initial_output_mw: float,
    minimum_output_mw: float = 0.0,
    maximum_output_mw: float = 100.0,
    ramp_mw_per_hour: float = 1_000.0,
    variable_cost_eur_per_mwh: float = 0.0,
    no_load_cost_eur_per_hour: float = 0.0,
    emission_factor_tonnes_per_mwh: float = 0.0,
    minimum_up_hours: float = 0.25,
    minimum_down_hours: float = 0.25,
    initial_up_time_hours: float = 10.0,
    initial_down_time_hours: float = 10.0,
) -> ThermalConfig:
    config = _example_config()
    return replace(
        config.thermal,
        minimum_output_mw=minimum_output_mw,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=ramp_mw_per_hour,
        ramp_down_mw_per_hour=ramp_mw_per_hour,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=variable_cost_eur_per_mwh,
        no_load_cost_eur_per_hour=no_load_cost_eur_per_hour,
        startup_cost_eur=0.0,
        shutdown_cost_eur=0.0,
        emission_factor_tonnes_per_mwh=emission_factor_tonnes_per_mwh,
        minimum_up_hours=minimum_up_hours,
        minimum_down_hours=minimum_down_hours,
        initial_on=initial_on,
        initial_output_mw=initial_output_mw,
        initial_up_time_hours=initial_up_time_hours if initial_on else 0.0,
        initial_down_time_hours=0.0 if initial_on else initial_down_time_hours,
        terminal_commitment_mode="forbid_incomplete_transitions",
    )


def _off_thermal() -> ThermalConfig:
    return _thermal_unit(
        initial_on=False,
        initial_output_mw=0.0,
        maximum_output_mw=0.0,
        no_load_cost_eur_per_hour=1_000.0,
        initial_up_time_hours=0.0,
    )


def _battery(
    *,
    capacity_mwh: float = 0.0,
    power_mw: float = 0.0,
    initial_soc_mwh: float = 0.0,
    minimum_final_soc_mwh: float = 0.0,
    terminal_soc_mode: Literal["minimum", "exact", "cyclic", "free"] = "free",
    throughput_cost_eur_per_mwh: float = 0.0,
) -> BatteryConfig:
    config = _example_config()
    return replace(
        config.battery,
        energy_capacity_mwh=capacity_mwh,
        power_capacity_mw=power_mw,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=capacity_mwh,
        initial_soc_mwh=initial_soc_mwh,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        throughput_cost_eur_per_mwh=throughput_cost_eur_per_mwh,
        minimum_final_soc_mwh=minimum_final_soc_mwh,
        terminal_soc_mode=terminal_soc_mode,
    )


def _energy_mwh(values: np.ndarray, time_step_hours: float) -> float:
    return float(values.sum() * time_step_hours)


@pytest.mark.parametrize("time_step_hours", [0.25, 0.5, 1.0])
def test_equivalent_thermal_case_has_consistent_energy_and_cost(
    time_step_hours: float,
) -> None:
    periods = round(1.0 / time_step_hours)
    thermal = _thermal_unit(
        initial_on=True,
        initial_output_mw=40.0,
        variable_cost_eur_per_mwh=20.0,
        no_load_cost_eur_per_hour=5.0,
        emission_factor_tonnes_per_mwh=0.5,
    )
    config = _direct_config(
        time_step_hours=time_step_hours,
        thermal=thermal,
        battery=_battery(),
        carbon_price_eur_per_tonne=10.0,
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(periods),
        gross_demand_mw=np.full(periods, 40.0),
    )

    # One simulated hour at 40 MW gives 40 MWh, EUR 800 variable,
    # EUR 5 no-load, and EUR 200 carbon cost.
    assert _energy_mwh(result.frame["thermal_output_mw"].to_numpy(), time_step_hours) == 40.0
    assert result.cost_components_eur["thermal_variable_cost_eur"] == pytest.approx(800.0)
    assert result.cost_components_eur["thermal_no_load_cost_eur"] == pytest.approx(5.0)
    assert result.cost_components_eur["thermal_carbon_cost_eur"] == pytest.approx(200.0)
    assert result.objective_eur == pytest.approx(1_005.0)


@pytest.mark.parametrize("time_step_hours", [0.25, 0.5, 1.0])
def test_ramp_limits_are_scaled_from_mw_per_hour(time_step_hours: float) -> None:
    ramp_mw_per_hour = 40.0
    reachable_mw = 20.0 + ramp_mw_per_hour * time_step_hours
    thermal = _thermal_unit(
        initial_on=True,
        initial_output_mw=20.0,
        ramp_mw_per_hour=ramp_mw_per_hour,
    )
    config = _direct_config(
        time_step_hours=time_step_hours,
        thermal=thermal,
        battery=_battery(),
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(2),
        gross_demand_mw=np.array([20.0, reachable_mw + 1.0]),
    )

    assert result.frame["thermal_output_mw"].iloc[1] == pytest.approx(reachable_mw)
    assert result.frame["source_load_shed_mw"].iloc[1] == pytest.approx(1.0)


@pytest.mark.parametrize("time_step_hours", [0.25, 0.5, 1.0])
def test_battery_state_and_throughput_cost_scale_by_time_step(
    time_step_hours: float,
) -> None:
    fixed_mwh = 10.0
    power_mw = fixed_mwh / time_step_hours
    config = _direct_config(
        time_step_hours=time_step_hours,
        thermal=_off_thermal(),
        battery=_battery(
            capacity_mwh=fixed_mwh,
            power_mw=power_mw,
            throughput_cost_eur_per_mwh=2.0,
        ),
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.array([power_mw, 0.0]),
        gross_demand_mw=np.array([0.0, power_mw]),
    )

    # Charging and discharging 10 MWh at EUR 2/MWh in each direction costs EUR 40.
    assert result.frame["battery_soc_mwh"].tolist() == pytest.approx([10.0, 0.0])
    assert _energy_mwh(result.frame["battery_charge_mw"].to_numpy(), time_step_hours) == 10.0
    assert _energy_mwh(result.frame["battery_discharge_mw"].to_numpy(), time_step_hours) == 10.0
    assert result.cost_components_eur["battery_throughput_cost_eur"] == pytest.approx(40.0)


@pytest.mark.parametrize("time_step_hours", [0.25, 0.5, 1.0])
def test_import_cost_and_emissions_scale_by_time_step(time_step_hours: float) -> None:
    fixed_mwh = 10.0
    power_mw = fixed_mwh / time_step_hours
    config = _direct_config(
        time_step_hours=time_step_hours,
        thermal=_off_thermal(),
        battery=_battery(),
        import_limit_mw=power_mw,
        import_price_eur_per_mwh=50.0,
        import_emission_factor_tonnes_per_mwh=0.2,
        carbon_price_eur_per_tonne=100.0,
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([power_mw]),
    )

    assert result.cost_components_eur["import_energy_cost_eur"] == pytest.approx(500.0)
    assert result.cost_components_eur["import_carbon_cost_eur"] == pytest.approx(200.0)


@pytest.mark.parametrize("time_step_hours", [0.25, 0.5, 1.0])
def test_curtailment_and_lost_load_costs_scale_by_time_step(
    time_step_hours: float,
) -> None:
    fixed_mwh = 10.0
    power_mw = fixed_mwh / time_step_hours
    curtailment_config = _direct_config(
        time_step_hours=time_step_hours,
        thermal=_off_thermal(),
        battery=_battery(),
        curtailment_eur_per_mwh=3.0,
    )
    shedding_config = _direct_config(
        time_step_hours=time_step_hours,
        thermal=_off_thermal(),
        battery=_battery(),
        lost_load_eur_per_mwh=1_000.0,
    )

    curtailment = UnitCommitment(curtailment_config).solve(
        renewable_available_mw=np.array([power_mw]),
        gross_demand_mw=np.zeros(1),
    )
    shedding = UnitCommitment(shedding_config).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([power_mw]),
    )

    assert curtailment.cost_components_eur["renewable_curtailment_cost_eur"] == pytest.approx(30.0)
    assert shedding.cost_components_eur["dispatch_load_shedding_cost_eur"] == pytest.approx(
        10_000.0
    )


def test_initially_off_residual_minimum_down_obligation_is_enforced() -> None:
    thermal = _thermal_unit(
        initial_on=False,
        initial_output_mw=0.0,
        minimum_output_mw=10.0,
        minimum_up_hours=1.0,
        minimum_down_hours=3.0,
        initial_up_time_hours=0.0,
        initial_down_time_hours=1.0,
        variable_cost_eur_per_mwh=1.0,
    )
    config = _direct_config(time_step_hours=1.0, thermal=thermal, battery=_battery())

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(3),
        gross_demand_mw=np.array([60.0, 60.0, 60.0]),
    )

    assert result.frame["thermal_on"].tolist() == [0, 0, 1]
    assert result.frame["source_load_shed_mw"].iloc[:2].tolist() == pytest.approx([60.0, 60.0])


@pytest.mark.parametrize(
    ("minimum_up_hours", "expected_on"),
    [
        (1.0, [1, 1, 0]),
        (1.1, [1, 1, 1]),
    ],
)
def test_minimum_duration_uses_exact_and_ceiling_period_counts(
    minimum_up_hours: float,
    expected_on: list[int],
) -> None:
    thermal = _thermal_unit(
        initial_on=False,
        initial_output_mw=0.0,
        minimum_output_mw=10.0,
        minimum_up_hours=minimum_up_hours,
        minimum_down_hours=0.5,
        initial_up_time_hours=0.0,
        initial_down_time_hours=10.0,
        variable_cost_eur_per_mwh=1.0,
        no_load_cost_eur_per_hour=1.0,
    )
    config = _direct_config(
        time_step_hours=0.5,
        thermal=thermal,
        battery=_battery(capacity_mwh=20.0, power_mw=20.0),
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(3),
        gross_demand_mw=np.array([60.0, 0.0, 0.0]),
    )

    assert result.frame["thermal_on"].tolist() == expected_on


@pytest.mark.parametrize(
    ("mode", "expected_final_soc_mwh", "expected_shed_mw"),
    [
        ("free", 0.0, 0.0),
        ("minimum", 4.0, 4.0),
        ("exact", 4.0, 4.0),
        ("cyclic", 5.0, 5.0),
    ],
)
def test_battery_terminal_modes_are_enforced(
    mode: Literal["minimum", "exact", "cyclic", "free"],
    expected_final_soc_mwh: float,
    expected_shed_mw: float,
) -> None:
    config = _direct_config(
        time_step_hours=1.0,
        thermal=_off_thermal(),
        battery=_battery(
            capacity_mwh=5.0,
            power_mw=5.0,
            initial_soc_mwh=5.0,
            minimum_final_soc_mwh=4.0,
            terminal_soc_mode=cast(Literal["minimum", "exact", "cyclic", "free"], mode),
        ),
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([5.0]),
    )

    assert result.frame["battery_soc_mwh"].iloc[-1] == pytest.approx(expected_final_soc_mwh)
    assert result.frame["source_load_shed_mw"].iloc[-1] == pytest.approx(expected_shed_mw)
