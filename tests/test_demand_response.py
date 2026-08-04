from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_system_simulator.config import DemandConfig, ModelConfig, ThermalConfig, load_config
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.simulation import SimulationEngine
from energy_system_simulator.simulation.assets import DemandAsset


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal() -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        minimum_output_mw=0.0,
        maximum_output_mw=0.0,
        startup_ramp_mw=0.0,
        shutdown_ramp_mw=0.0,
        initial_on=False,
        initial_output_mw=0.0,
        initial_up_time_hours=0.0,
        initial_down_time_hours=10.0,
        terminal_commitment_mode="forbid_incomplete_transitions",
        terminal_on=None,
    )


def _config(*demand: DemandConfig) -> ModelConfig:
    config = _example_config()
    battery = replace(
        config.battery,
        energy_capacity_mwh=0.0,
        power_capacity_mw=0.0,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=0.0,
        initial_soc_mwh=0.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode="free",
    )
    return replace(
        config,
        thermal=_thermal(),
        battery=battery,
        portfolio=replace(config.portfolio, demand=demand),
        imports=replace(config.imports, maximum_power_mw=0.0),
        network=replace(config.network, loss_fraction=0.0),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
    )


def _demand(asset_id: str, **kwargs: object) -> DemandConfig:
    return DemandConfig(id=asset_id, bus_id="system", time_series_key=f"{asset_id}_mw", **kwargs)


def test_shiftable_demand_conserves_energy_over_horizon() -> None:
    demand = _demand(
        "industrial",
        kind="shiftable",
        shift_up_capacity_mw=10.0,
        shift_down_capacity_mw=10.0,
    )

    result = UnitCommitment(_config(demand)).solve(
        renewable_available_mw=np.array([10.0, 0.0]),
        gross_demand_mw=np.array([0.0, 10.0]),
        demand_profiles_mw={"industrial": np.array([0.0, 10.0])},
    )

    frame = result.frame
    assert frame["demand_shift_up_mw__industrial"].tolist() == pytest.approx([10.0, 0.0])
    assert frame["demand_shift_down_mw__industrial"].tolist() == pytest.approx([0.0, 10.0])
    assert frame["source_load_shed_mw"].sum() == pytest.approx(0.0)


def test_ev_charging_task_completes_before_deadline() -> None:
    demand = _demand(
        "ev",
        kind="ev_charging",
        task_power_capacity_mw=10.0,
        task_required_energy_mwh=20.0,
        task_start_period=0,
        task_end_period=2,
        task_unserved_penalty_eur_per_mwh=20_000.0,
    )

    result = UnitCommitment(_config(demand)).solve(
        renewable_available_mw=np.array([10.0, 10.0, 10.0]),
        gross_demand_mw=np.zeros(3),
        demand_profiles_mw={"ev": np.zeros(3)},
    )

    assert result.frame["demand_task_charge_mw__ev"].tolist() == pytest.approx([10.0, 10.0, 0.0])
    assert result.frame["demand_task_unserved_mwh__ev"].sum() == pytest.approx(0.0)


def test_task_reports_unserved_energy_when_window_is_physically_insufficient() -> None:
    demand = _demand(
        "ev",
        kind="ev_charging",
        task_power_capacity_mw=5.0,
        task_required_energy_mwh=20.0,
        task_start_period=0,
        task_end_period=2,
        task_unserved_penalty_eur_per_mwh=20_000.0,
    )

    result = UnitCommitment(_config(demand)).solve(
        renewable_available_mw=np.array([5.0, 5.0, 5.0]),
        gross_demand_mw=np.zeros(3),
        demand_profiles_mw={"ev": np.zeros(3)},
    )

    assert result.frame["demand_task_charge_mw__ev"].sum() == pytest.approx(10.0)
    assert result.frame["demand_task_unserved_mwh__ev"].sum() == pytest.approx(10.0)


def test_ev_fleet_availability_limits_charging_and_reports_unmet_departure_energy() -> None:
    demand = _demand(
        "ev",
        kind="ev_charging",
        task_power_capacity_mw=10.0,
        task_required_energy_mwh=10.0,
        task_start_period=0,
        task_end_period=1,
        task_unserved_penalty_eur_per_mwh=20_000.0,
        ev_energy_capacity_mwh=20.0,
        ev_initial_energy_mwh=0.0,
        ev_required_departure_energy_mwh=10.0,
        ev_arrival_period=0,
        ev_departure_period=1,
        ev_availability_fraction=0.5,
    )

    result = UnitCommitment(_config(demand)).solve(
        renewable_available_mw=np.array([10.0]),
        gross_demand_mw=np.zeros(1),
        demand_profiles_mw={"ev": np.zeros(1)},
    )

    assert result.frame["demand_task_charge_mw__ev"].iloc[0] == pytest.approx(5.0)
    assert result.frame["ev_energy_mwh__ev"].iloc[0] == pytest.approx(4.75)
    assert result.frame["demand_task_unserved_mwh__ev"].sum() == pytest.approx(5.0)


def test_ev_v2g_exports_only_when_vehicle_is_available() -> None:
    fixed = _demand("load")
    ev = _demand(
        "ev",
        kind="ev_charging",
        task_power_capacity_mw=5.0,
        task_required_energy_mwh=1.0,
        task_start_period=0,
        task_end_period=1,
        task_unserved_penalty_eur_per_mwh=1.0,
        ev_energy_capacity_mwh=10.0,
        ev_initial_energy_mwh=5.0,
        ev_required_departure_energy_mwh=0.0,
        ev_v2g_power_capacity_mw=5.0,
        ev_v2g_efficiency=1.0,
        ev_degradation_cost_eur_per_mwh=1.0,
    )

    result = UnitCommitment(_config(fixed, ev)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([5.0]),
        demand_profiles_mw={"load": np.array([5.0]), "ev": np.zeros(1)},
    )

    assert result.frame["ev_v2g_discharge_mw__ev"].iloc[0] == pytest.approx(5.0)
    assert result.frame["demand_involuntary_shed_mw__load"].iloc[0] == pytest.approx(0.0)
    assert result.cost_components_eur["ev_v2g_degradation_cost_eur"] == pytest.approx(5.0)


def test_sector_specific_lost_load_cost_prioritises_scarce_supply() -> None:
    residential = _demand("residential", value_of_lost_load_eur_per_mwh=1_000.0)
    hospital = _demand("hospital", value_of_lost_load_eur_per_mwh=20_000.0)

    result = UnitCommitment(_config(residential, hospital)).solve(
        renewable_available_mw=np.array([10.0]),
        gross_demand_mw=np.array([20.0]),
        demand_profiles_mw={
            "residential": np.array([10.0]),
            "hospital": np.array([10.0]),
        },
    )

    assert result.frame["demand_involuntary_shed_mw__residential"].iloc[0] == pytest.approx(10.0)
    assert result.frame["demand_involuntary_shed_mw__hospital"].iloc[0] == pytest.approx(0.0)


def test_voluntary_curtailment_is_reported_separately_from_involuntary_shedding() -> None:
    demand = _demand(
        "commercial",
        kind="curtailable",
        maximum_curtailment_fraction=0.5,
        voluntary_curtailment_cost_eur_per_mwh=100.0,
    )

    result = UnitCommitment(_config(demand)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([10.0]),
        demand_profiles_mw={"commercial": np.array([10.0])},
    )

    assert result.frame["demand_voluntary_curtailment_mw__commercial"].iloc[0] == pytest.approx(5.0)
    assert result.frame["demand_involuntary_shed_mw__commercial"].iloc[0] == pytest.approx(5.0)
    assert result.cost_components_eur["demand_voluntary_curtailment_cost_eur"] == pytest.approx(
        500.0
    )
    assert result.cost_components_eur["dispatch_load_shedding_cost_eur"] == pytest.approx(50_000.0)


def test_temperature_sensitive_demand_uses_degree_terms() -> None:
    asset = DemandAsset.from_config(
        _demand(
            "residential",
            temperature_time_series_key="temperature_c",
            heating_base_temperature_c=18.0,
            cooling_base_temperature_c=22.0,
            heating_sensitivity_mw_per_c=2.0,
            cooling_sensitivity_mw_per_c=3.0,
        )
    )
    data = pd.DataFrame(
        {
            "residential_mw": [10.0, 10.0],
            "temperature_c": [15.0, 25.0],
        }
    )

    assert asset.demand_mw(data).tolist() == pytest.approx([16.0, 19.0])


def test_heat_pump_uses_temperature_dependent_cop_and_thermal_storage() -> None:
    heat = _demand(
        "heat",
        kind="heat_pump",
        task_power_capacity_mw=10.0,
        heat_pump_cop_base=2.0,
        heat_pump_cop_temperature_coefficient_per_c=0.1,
        heat_pump_cop_min=1.0,
        heat_pump_thermal_storage_capacity_mwh=10.0,
        heat_pump_initial_thermal_storage_mwh=0.0,
        heat_pump_comfort_min_mwh=0.0,
        heat_pump_comfort_max_mwh=10.0,
        heat_pump_backup_heat_capacity_mw=0.0,
        heat_pump_comfort_violation_penalty_eur_per_mwh=1_000.0,
        temperature_time_series_key="temperature_c",
    )
    config = _config(heat)
    data = pd.DataFrame({"heat_mw": [6.0], "temperature_c": [10.0]})
    asset = DemandAsset.from_config(heat)

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.array([10.0]),
        gross_demand_mw=np.zeros(1),
        demand_profiles_mw={"heat": asset.demand_mw(data)},
        heat_pump_cop_profiles={"heat": asset.heat_pump_cop(data)},
    )

    assert asset.heat_pump_cop(data).tolist() == pytest.approx([3.0])
    assert result.frame["heat_pump_electric_mw__heat"].iloc[0] == pytest.approx(2.0)
    assert result.frame["heat_pump_comfort_violation_mwh__heat"].iloc[0] == pytest.approx(0.0)


def test_heat_pump_backup_heat_reports_cost_and_emissions_when_electric_supply_is_scarce() -> None:
    heat = _demand(
        "heat",
        kind="heat_pump",
        task_power_capacity_mw=0.0,
        heat_pump_cop_base=3.0,
        heat_pump_thermal_storage_capacity_mwh=10.0,
        heat_pump_initial_thermal_storage_mwh=0.0,
        heat_pump_comfort_min_mwh=0.0,
        heat_pump_comfort_max_mwh=10.0,
        heat_pump_backup_heat_capacity_mw=6.0,
        heat_pump_backup_heat_cost_eur_per_mwh=50.0,
        heat_pump_backup_heat_emission_tonnes_per_mwh=0.2,
        heat_pump_comfort_violation_penalty_eur_per_mwh=1_000.0,
    )

    result = UnitCommitment(_config(heat)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.zeros(1),
        demand_profiles_mw={"heat": np.array([6.0])},
    )

    assert result.frame["heat_pump_backup_heat_mw__heat"].iloc[0] == pytest.approx(6.0)
    assert result.frame["heat_pump_backup_heat_cost_eur"].iloc[0] == pytest.approx(300.0)
    assert result.frame["heat_pump_backup_heat_emissions_tonnes"].iloc[0] == pytest.approx(1.2)


def test_demand_response_portfolio_example_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "portfolio_demand_response.yaml")

    result = SimulationEngine(config).run()

    assert set(result.summary["demand_assets"]) == {"residential", "industrial", "ev-fleet"}
    assert result.summary["demand_task_charge_mwh"] > 0.0
    assert result.summary["demand_task_unserved_mwh"] == pytest.approx(0.0)
    assert result.summary["energy_reconciliation"]["max_abs_source_balance_residual_mw"] <= 1e-6
    demand_rows = result.asset_timeseries[
        result.asset_timeseries["asset_id"].isin({"residential", "industrial", "ev-fleet"})
    ]
    assert {"baseline_mw_source", "adjusted_mw_source", "involuntary_shed_mw_source"} <= set(
        demand_rows["variable"]
    )
