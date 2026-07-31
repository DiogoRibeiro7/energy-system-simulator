from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from energy_system_simulator.config import HydroUnitConfig, ModelConfig, ThermalConfig, load_config
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.simulation import SimulationEngine


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal(
    *,
    maximum_output_mw: float = 0.0,
    variable_cost_eur_per_mwh: float = 0.0,
) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        minimum_output_mw=0.0,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=1_000.0,
        ramp_down_mw_per_hour=1_000.0,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=variable_cost_eur_per_mwh,
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
        minimum_fuel_input_mwh_per_hour=0.0,
        heat_rate_segments=(),
        startup_categories=(),
    )


def _hydro(
    *,
    kind: Literal["reservoir", "run_of_river"] = "reservoir",
    turbine_capacity_mw: float = 20.0,
    turbine_efficiency: float = 1.0,
    minimum_reservoir_mwh: float = 0.0,
    maximum_reservoir_mwh: float = 100.0,
    initial_reservoir_mwh: float = 0.0,
    minimum_final_reservoir_mwh: float = 0.0,
    terminal_reservoir_mode: Literal["minimum", "exact", "cyclic", "free"] = "free",
    spill_capacity_mw: float | None = None,
    minimum_release_mw: float = 0.0,
    evaporation_rate_per_hour: float = 0.0,
    water_value_eur_per_mwh: float = 0.0,
) -> HydroUnitConfig:
    return HydroUnitConfig(
        id="hydro",
        bus_id="system",
        kind=kind,
        inflow_time_series_key="hydro_inflow_mw",
        turbine_capacity_mw=turbine_capacity_mw,
        turbine_efficiency=turbine_efficiency,
        minimum_reservoir_mwh=minimum_reservoir_mwh,
        maximum_reservoir_mwh=maximum_reservoir_mwh,
        initial_reservoir_mwh=initial_reservoir_mwh,
        minimum_final_reservoir_mwh=minimum_final_reservoir_mwh,
        terminal_reservoir_mode=terminal_reservoir_mode,
        spill_capacity_mw=spill_capacity_mw,
        minimum_release_mw=minimum_release_mw,
        evaporation_rate_per_hour=evaporation_rate_per_hour,
        water_value_eur_per_mwh=water_value_eur_per_mwh,
    )


def _config(
    hydro: HydroUnitConfig,
    *,
    thermal: ThermalConfig | None = None,
    lost_load_eur_per_mwh: float = 10_000.0,
) -> ModelConfig:
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
    resolved_thermal = thermal if thermal is not None else _thermal()
    return replace(
        config,
        thermal=resolved_thermal,
        portfolio=replace(config.portfolio, hydro_units=(hydro,)),
        battery=battery,
        imports=replace(config.imports, maximum_power_mw=0.0),
        network=replace(config.network, loss_fraction=0.0),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=lost_load_eur_per_mwh,
            carbon_price_eur_per_tonne=0.0,
        ),
    )


def _solve(
    hydro: HydroUnitConfig,
    demand_mw: np.ndarray,
    inflow_mw: np.ndarray,
    *,
    thermal: ThermalConfig | None = None,
    lost_load_eur_per_mwh: float = 10_000.0,
):
    return UnitCommitment(
        _config(hydro, thermal=thermal, lost_load_eur_per_mwh=lost_load_eur_per_mwh)
    ).solve(
        renewable_available_mw=np.zeros_like(demand_mw),
        gross_demand_mw=demand_mw,
        hydro_inflows_mw={"hydro": inflow_mw},
    )


def test_reservoir_water_conservation_reconciles() -> None:
    result = _solve(
        _hydro(initial_reservoir_mwh=10.0, maximum_reservoir_mwh=20.0),
        demand_mw=np.array([10.0, 5.0]),
        inflow_mw=np.array([5.0, 0.0]),
    )

    frame = result.frame
    assert frame["hydro_generation_mw__hydro"].sum() == pytest.approx(15.0)
    assert frame["hydro_reservoir_mwh__hydro"].iloc[-1] == pytest.approx(0.0)
    assert frame["hydro_water_balance_residual_mwh__hydro"].abs().max() <= 1e-6


def test_full_reservoir_spills_excess_inflow() -> None:
    result = _solve(
        _hydro(
            turbine_capacity_mw=0.0,
            initial_reservoir_mwh=10.0,
            maximum_reservoir_mwh=10.0,
            minimum_final_reservoir_mwh=10.0,
            terminal_reservoir_mode="minimum",
        ),
        demand_mw=np.array([0.0]),
        inflow_mw=np.array([5.0]),
    )

    assert result.frame["hydro_spill_mw__hydro"].iloc[0] == pytest.approx(5.0)
    assert result.frame["hydro_reservoir_mwh__hydro"].iloc[0] == pytest.approx(10.0)


def test_drought_dispatch_depletes_available_reservoir_and_sheds_shortfall() -> None:
    result = _solve(
        _hydro(initial_reservoir_mwh=5.0, maximum_reservoir_mwh=5.0),
        demand_mw=np.array([5.0, 5.0]),
        inflow_mw=np.array([0.0, 0.0]),
    )

    assert result.frame["hydro_generation_mw__hydro"].sum() == pytest.approx(5.0)
    assert result.frame["source_load_shed_mw"].sum() == pytest.approx(5.0)


def test_environmental_release_can_be_satisfied_by_spill() -> None:
    result = _solve(
        _hydro(minimum_release_mw=3.0, maximum_reservoir_mwh=10.0),
        demand_mw=np.array([0.0]),
        inflow_mw=np.array([3.0]),
    )

    assert result.frame["hydro_release_mw__hydro"].iloc[0] == pytest.approx(0.0)
    assert result.frame["hydro_spill_mw__hydro"].iloc[0] == pytest.approx(3.0)


def test_terminal_water_value_can_prefer_stored_water_to_serving_load() -> None:
    result = _solve(
        _hydro(
            initial_reservoir_mwh=10.0,
            maximum_reservoir_mwh=10.0,
            water_value_eur_per_mwh=20_000.0,
        ),
        demand_mw=np.array([10.0]),
        inflow_mw=np.array([0.0]),
    )

    assert result.frame["hydro_generation_mw__hydro"].iloc[0] == pytest.approx(0.0)
    assert result.frame["hydro_reservoir_mwh__hydro"].iloc[-1] == pytest.approx(10.0)
    assert result.frame["source_load_shed_mw"].iloc[0] == pytest.approx(10.0)
    assert result.cost_components_eur["hydro_terminal_value_eur"] == pytest.approx(-200_000.0)


def test_hydro_displaces_more_expensive_thermal_generation() -> None:
    result = _solve(
        _hydro(initial_reservoir_mwh=10.0, maximum_reservoir_mwh=10.0),
        demand_mw=np.array([10.0]),
        inflow_mw=np.array([0.0]),
        thermal=_thermal(maximum_output_mw=20.0, variable_cost_eur_per_mwh=100.0),
    )

    assert result.frame["hydro_generation_mw__hydro"].iloc[0] == pytest.approx(10.0)
    assert result.frame["thermal_output_mw"].iloc[0] == pytest.approx(0.0)


def test_run_of_river_cannot_shift_inflow_across_periods() -> None:
    result = _solve(
        _hydro(
            kind=cast(Literal["reservoir", "run_of_river"], "run_of_river"),
            turbine_capacity_mw=10.0,
            turbine_efficiency=0.5,
            maximum_reservoir_mwh=0.0,
        ),
        demand_mw=np.array([10.0, 10.0]),
        inflow_mw=np.array([20.0, 0.0]),
    )

    assert result.frame["hydro_generation_mw__hydro"].tolist() == pytest.approx([10.0, 0.0])
    assert result.frame["source_load_shed_mw"].tolist() == pytest.approx([0.0, 10.0])
    assert result.frame["hydro_reservoir_mwh__hydro"].tolist() == pytest.approx([0.0, 0.0])


def test_hydro_portfolio_example_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "portfolio_hydro.yaml")

    result = SimulationEngine(config).run()

    assert result.summary["hydro_generation_mwh"] > 0.0
    assert set(result.summary["hydro_assets"]) == {"alpine-reservoir", "river-run"}
    assert result.summary["energy_reconciliation"]["max_abs_source_balance_residual_mw"] <= 1e-6
    assert (
        result.summary["energy_reconciliation"]["hydro_assets"]["alpine-reservoir"][
            "max_abs_residual"
        ]
        <= 1e-6
    )
    assert {"generation_mw", "reservoir_mwh_water_equivalent", "spill_mw_water_equivalent"} <= set(
        result.asset_timeseries["variable"]
    )
