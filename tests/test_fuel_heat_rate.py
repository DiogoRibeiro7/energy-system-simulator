from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from energy_system_simulator.config import (
    FuelConfig,
    HeatRateSegmentConfig,
    ModelConfig,
    StartupCategoryConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal(
    name: str,
    *,
    maximum_output_mw: float,
    variable_cost_eur_per_mwh: float = 0.0,
    emission_factor_tonnes_per_mwh: float = 0.0,
    minimum_fuel_input_mwh_per_hour: float = 0.0,
    heat_rate_segments: tuple[HeatRateSegmentConfig, ...] = (),
    startup_categories: tuple[StartupCategoryConfig, ...] = (),
    initial_down_time_hours: float = 10.0,
) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        name=name,
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
        emission_factor_tonnes_per_mwh=emission_factor_tonnes_per_mwh,
        minimum_up_hours=1.0,
        minimum_down_hours=1.0,
        initial_on=False,
        initial_output_mw=0.0,
        initial_up_time_hours=0.0,
        initial_down_time_hours=initial_down_time_hours,
        terminal_commitment_mode="forbid_incomplete_transitions",
        terminal_on=None,
        minimum_fuel_input_mwh_per_hour=minimum_fuel_input_mwh_per_hour,
        heat_rate_segments=heat_rate_segments,
        startup_categories=startup_categories,
    )


def _generator(asset_id: str, fuel_id: str, thermal: ThermalConfig) -> ThermalGeneratorConfig:
    return ThermalGeneratorConfig(
        id=asset_id,
        bus_id="system",
        fuel_id=fuel_id,
        config=thermal,
    )


def _config(
    generators: tuple[ThermalGeneratorConfig, ...],
    fuels: tuple[FuelConfig, ...],
    *,
    carbon_price_eur_per_tonne: float = 0.0,
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
    return replace(
        config,
        thermal=generators[0].config,
        portfolio=replace(config.portfolio, fuels=fuels, thermal_generators=generators),
        battery=battery,
        imports=replace(config.imports, maximum_power_mw=0.0),
        network=replace(config.network, loss_fraction=0.0),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=carbon_price_eur_per_tonne,
        ),
    )


def test_segment_dispatch_and_fuel_accounting_reconcile() -> None:
    fuel = FuelConfig(
        id="gas",
        price_eur_per_mwh_thermal=10.0,
        co2_factor_tonnes_per_mwh_thermal=0.2,
        methane_factor_tonnes_per_mwh_thermal=0.01,
        nox_factor_kg_per_mwh_thermal=0.1,
        sox_factor_kg_per_mwh_thermal=0.02,
    )
    thermal = _thermal(
        "ccgt",
        maximum_output_mw=100.0,
        heat_rate_segments=(
            HeatRateSegmentConfig(
                id="efficient",
                capacity_mw=40.0,
                heat_rate_mwh_thermal_per_mwh=2.0,
            ),
            HeatRateSegmentConfig(
                id="duct_firing",
                capacity_mw=60.0,
                heat_rate_mwh_thermal_per_mwh=3.0,
            ),
        ),
    )
    result = UnitCommitment(_config((_generator("ccgt", "gas", thermal),), (fuel,))).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([70.0]),
    )

    frame = result.frame
    assert frame["thermal_segment_output_mw__ccgt__efficient"].iloc[0] == pytest.approx(40.0)
    assert frame["thermal_segment_output_mw__ccgt__duct_firing"].iloc[0] == pytest.approx(30.0)
    assert frame["thermal_fuel_input_mwh_thermal__ccgt"].iloc[0] == pytest.approx(170.0)
    assert frame["thermal_efficiency__ccgt"].iloc[0] == pytest.approx(70.0 / 170.0)
    assert frame["thermal_fuel_cost_eur__ccgt"].iloc[0] == pytest.approx(1_700.0)
    assert frame["thermal_carbon_cost_eur__ccgt"].iloc[0] == pytest.approx(0.0)
    assert result.cost_components_eur["thermal_variable_cost_eur"] == pytest.approx(1_700.0)


def test_carbon_price_changes_merit_order_between_fuels() -> None:
    coal = FuelConfig(
        id="coal",
        price_eur_per_mwh_thermal=5.0,
        co2_factor_tonnes_per_mwh_thermal=0.9,
    )
    gas = FuelConfig(
        id="gas",
        price_eur_per_mwh_thermal=15.0,
        co2_factor_tonnes_per_mwh_thermal=0.1,
    )
    segment = (
        HeatRateSegmentConfig(
            id="all",
            capacity_mw=100.0,
            heat_rate_mwh_thermal_per_mwh=2.0,
        ),
    )
    coal_unit = _generator(
        "coal_unit",
        "coal",
        _thermal("coal", maximum_output_mw=100.0, heat_rate_segments=segment),
    )
    gas_unit = _generator(
        "gas_unit",
        "gas",
        _thermal("gas", maximum_output_mw=100.0, heat_rate_segments=segment),
    )

    no_carbon = UnitCommitment(
        _config((coal_unit, gas_unit), (coal, gas), carbon_price_eur_per_tonne=0.0)
    ).solve(np.zeros(1), np.array([50.0]))
    high_carbon = UnitCommitment(
        _config((coal_unit, gas_unit), (coal, gas), carbon_price_eur_per_tonne=100.0)
    ).solve(np.zeros(1), np.array([50.0]))

    assert no_carbon.frame["thermal_output_mw__coal_unit"].iloc[0] == pytest.approx(50.0)
    assert high_carbon.frame["thermal_output_mw__gas_unit"].iloc[0] == pytest.approx(50.0)


def test_startup_category_uses_prior_downtime_and_startup_fuel() -> None:
    fuel = FuelConfig(
        id="gas",
        price_eur_per_mwh_thermal=10.0,
        co2_factor_tonnes_per_mwh_thermal=0.5,
    )
    thermal = _thermal(
        "peaker",
        maximum_output_mw=50.0,
        variable_cost_eur_per_mwh=1.0,
        startup_categories=(
            StartupCategoryConfig(id="hot", minimum_down_time_hours=0.0, startup_cost_eur=10.0),
            StartupCategoryConfig(
                id="warm",
                minimum_down_time_hours=2.0,
                startup_cost_eur=20.0,
                startup_fuel_input_mwh_thermal=1.0,
            ),
            StartupCategoryConfig(
                id="cold",
                minimum_down_time_hours=6.0,
                startup_cost_eur=30.0,
                startup_fuel_input_mwh_thermal=3.0,
            ),
        ),
        initial_down_time_hours=3.0,
    )
    result = UnitCommitment(
        _config((_generator("peaker", "gas", thermal),), (fuel,), carbon_price_eur_per_tonne=85.0)
    ).solve(np.zeros(1), np.array([50.0]))

    assert result.frame["thermal_startup_category__peaker__warm"].iloc[0] == 1
    assert result.frame["thermal_startup_fuel_input_mwh_thermal__peaker"].iloc[0] == pytest.approx(
        1.0
    )
    assert result.cost_components_eur["startup_cost_eur"] == pytest.approx(30.0)
    assert result.cost_components_eur["thermal_carbon_cost_eur"] == pytest.approx(0.5 * 85.0)


def test_time_varying_fuel_price_changes_dispatch_by_period() -> None:
    gas = FuelConfig(
        id="gas",
        price_eur_per_mwh_thermal=5.0,
        co2_factor_tonnes_per_mwh_thermal=0.0,
        price_time_series_key="gas_price",
    )
    oil = FuelConfig(
        id="oil",
        price_eur_per_mwh_thermal=20.0,
        co2_factor_tonnes_per_mwh_thermal=0.0,
    )
    segment = (
        HeatRateSegmentConfig(
            id="all",
            capacity_mw=100.0,
            heat_rate_mwh_thermal_per_mwh=1.0,
        ),
    )
    gas_unit = _generator(
        "gas_unit",
        "gas",
        _thermal("gas", maximum_output_mw=100.0, heat_rate_segments=segment),
    )
    oil_unit = _generator(
        "oil_unit",
        "oil",
        _thermal("oil", maximum_output_mw=100.0, heat_rate_segments=segment),
    )

    result = UnitCommitment(_config((gas_unit, oil_unit), (gas, oil))).solve(
        np.zeros(2),
        np.array([50.0, 50.0]),
        fuel_price_series={"gas": np.array([5.0, 50.0])},
    )

    assert result.frame["thermal_output_mw__gas_unit"].tolist() == pytest.approx([50.0, 0.0])
    assert result.frame["thermal_output_mw__oil_unit"].tolist() == pytest.approx([0.0, 50.0])
