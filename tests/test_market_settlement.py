from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from energy_system_simulator.config import (
    BusConfig,
    DemandConfig,
    ImportConfig,
    ImportResourceConfig,
    ModelConfig,
    NetworkConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.market import MarketAnalyzer


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal(
    *,
    unit_name: str,
    maximum_output_mw: float,
    variable_cost_eur_per_mwh: float,
    no_load_cost_eur_per_hour: float = 0.0,
) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        name=unit_name,
        minimum_output_mw=0.0,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=1_000.0,
        ramp_down_mw_per_hour=1_000.0,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=variable_cost_eur_per_mwh,
        no_load_cost_eur_per_hour=no_load_cost_eur_per_hour,
        startup_cost_eur=0.0,
        shutdown_cost_eur=0.0,
        emission_factor_tonnes_per_mwh=0.0,
        minimum_up_hours=0.25,
        minimum_down_hours=0.25,
        initial_on=True,
        initial_output_mw=0.0,
        initial_up_time_hours=10.0,
        initial_down_time_hours=0.0,
        terminal_commitment_mode="forbid_incomplete_transitions",
        terminal_on=None,
        minimum_fuel_input_mwh_per_hour=0.0,
        heat_rate_segments=(),
        startup_categories=(),
    )


def _imports_off() -> ImportConfig:
    return ImportConfig(
        maximum_power_mw=0.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )


def _aggregate_config(
    *,
    thermal: ThermalConfig,
    lost_load_eur_per_mwh: float = 1_000.0,
) -> ModelConfig:
    config = _example_config()
    imports = _imports_off()
    return replace(
        config,
        thermal=thermal,
        battery=replace(
            config.battery,
            energy_capacity_mwh=0.0,
            power_capacity_mw=0.0,
            minimum_soc_mwh=0.0,
            maximum_soc_mwh=0.0,
            initial_soc_mwh=0.0,
            minimum_final_soc_mwh=0.0,
            terminal_soc_mode="free",
        ),
        imports=imports,
        network=NetworkConfig(loss_fraction=0.0, transfer_capacity_mw=1_000.0),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=lost_load_eur_per_mwh,
            carbon_price_eur_per_tonne=0.0,
        ),
        portfolio=replace(
            config.portfolio,
            buses=(BusConfig(id="system", zone_id="system"),),
            lines=(),
            renewable_generators=(),
            thermal_generators=(
                ThermalGeneratorConfig(
                    id="thermal",
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


def _nodal_config() -> ModelConfig:
    config = _example_config()
    imports = _imports_off()
    cheap = _thermal(
        unit_name="north cheap",
        maximum_output_mw=100.0,
        variable_cost_eur_per_mwh=10.0,
    )
    expensive = _thermal(
        unit_name="south expensive",
        maximum_output_mw=100.0,
        variable_cost_eur_per_mwh=100.0,
    )
    return replace(
        _aggregate_config(thermal=cheap, lost_load_eur_per_mwh=10_000.0),
        thermal=cheap,
        imports=imports,
        network=NetworkConfig(
            loss_fraction=0.0,
            transfer_capacity_mw=1_000.0,
            network_mode=cast(Literal["nodal"], "nodal"),
            slack_bus_id="north",
        ),
        portfolio=replace(
            config.portfolio,
            buses=(
                BusConfig(id="north", zone_id="zone"),
                BusConfig(id="south", zone_id="zone"),
            ),
            lines=(
                TransmissionLineConfig(
                    id="north_south",
                    from_bus_id="north",
                    to_bus_id="south",
                    susceptance=10.0,
                    capacity_mw=40.0,
                ),
            ),
            renewable_generators=(),
            thermal_generators=(
                ThermalGeneratorConfig(id="cheap", bus_id="north", fuel_id="gas", config=cheap),
                ThermalGeneratorConfig(
                    id="expensive",
                    bus_id="south",
                    fuel_id="gas",
                    config=expensive,
                ),
            ),
            storage_units=(),
            hydro_units=(),
            imports=(ImportResourceConfig(id="imports", bus_id="north", config=imports),),
            demand=(DemandConfig(id="load", bus_id="south", time_series_key="load_mw"),),
        ),
    )


def _settle_aggregate(config: ModelConfig, *, demand_mw: float):
    model = UnitCommitment(config)
    problem = model.build_formulation(
        renewable_available_mw=np.array([0.0], dtype=float),
        gross_demand_mw=np.array([demand_mw], dtype=float),
    )
    dispatch = model.solve_formulation(problem)
    return MarketAnalyzer(config).settle(problem, dispatch), dispatch


def test_uncongested_dispatch_has_uniform_marginal_price() -> None:
    config = _aggregate_config(
        thermal=_thermal(
            unit_name="marginal thermal",
            maximum_output_mw=100.0,
            variable_cost_eur_per_mwh=20.0,
        )
    )

    settlement, dispatch = _settle_aggregate(config, demand_mw=50.0)

    assert settlement.prices["energy_price_eur_per_mwh"].iloc[0] == pytest.approx(20.0)
    assert settlement.consumer_payment_eur == pytest.approx(1_000.0)
    assert settlement.generator_energy_revenue_eur == pytest.approx(1_000.0)
    assert settlement.generator_settlements[0].variable_cost_eur == pytest.approx(1_000.0)
    assert settlement.generator_settlements[0].make_whole_payment_eur == pytest.approx(0.0)
    assert dispatch.frame["thermal_output_mw__thermal"].iloc[0] == pytest.approx(50.0)


def test_scarcity_price_uses_explicit_lost_load_cap() -> None:
    config = _aggregate_config(
        thermal=_thermal(
            unit_name="unavailable",
            maximum_output_mw=0.0,
            variable_cost_eur_per_mwh=20.0,
        ),
        lost_load_eur_per_mwh=1_000.0,
    )

    settlement, dispatch = _settle_aggregate(config, demand_mw=10.0)

    assert dispatch.frame["source_load_shed_mw"].iloc[0] == pytest.approx(10.0)
    assert settlement.prices["energy_price_eur_per_mwh"].iloc[0] == pytest.approx(1_000.0)
    assert settlement.prices["scarcity_price_eur_per_mwh"].iloc[0] == pytest.approx(1_000.0)
    assert settlement.scarcity_rent_eur == pytest.approx(10_000.0)


def test_fixed_commitment_pricing_triggers_make_whole_payment() -> None:
    config = _aggregate_config(
        thermal=_thermal(
            unit_name="committed thermal",
            maximum_output_mw=100.0,
            variable_cost_eur_per_mwh=20.0,
            no_load_cost_eur_per_hour=100.0,
        )
    )

    settlement, _dispatch = _settle_aggregate(config, demand_mw=10.0)
    generator = settlement.generator_settlements[0]

    assert settlement.prices["energy_price_eur_per_mwh"].iloc[0] == pytest.approx(20.0)
    assert generator.energy_revenue_eur == pytest.approx(200.0)
    assert generator.committed_cost_eur == pytest.approx(300.0)
    assert generator.make_whole_payment_eur == pytest.approx(100.0)
    assert settlement.uplift_eur == pytest.approx(100.0)


def test_congested_nodal_dispatch_has_lmp_separation_and_congestion_rent() -> None:
    config = _nodal_config()
    model = UnitCommitment(config)
    problem = model.build_formulation(
        renewable_available_mw=np.array([0.0], dtype=float),
        gross_demand_mw=np.array([100.0], dtype=float),
        demand_profiles_mw={"load": np.array([100.0], dtype=float)},
    )
    dispatch = model.solve_formulation(problem)

    settlement = MarketAnalyzer(config).settle(problem, dispatch)

    north = settlement.nodal_prices.loc[
        settlement.nodal_prices["bus_id"] == "north",
        "lmp_eur_per_mwh",
    ].iloc[0]
    south = settlement.nodal_prices.loc[
        settlement.nodal_prices["bus_id"] == "south",
        "lmp_eur_per_mwh",
    ].iloc[0]
    assert north == pytest.approx(10.0)
    assert south == pytest.approx(100.0)
    assert dispatch.frame["line_flow_mw__north_south"].iloc[0] == pytest.approx(40.0)
    assert settlement.congestion_rent_eur == pytest.approx(3_600.0)
    assert settlement.consumer_payment_eur == pytest.approx(10_000.0)
