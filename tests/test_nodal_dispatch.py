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
    RenewableGeneratorConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
    load_config,
    validate_config,
)
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.exceptions import ConfigurationError


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal(maximum_output_mw: float) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        minimum_output_mw=0.0,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=1_000.0,
        ramp_down_mw_per_hour=1_000.0,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=1.0,
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
    )


def _config(
    *,
    mode: Literal["aggregate", "nodal"] = "nodal",
    buses: tuple[str, ...] = ("north", "south"),
    lines: tuple[TransmissionLineConfig, ...] = (
        TransmissionLineConfig(
            id="north_south",
            from_bus_id="north",
            to_bus_id="south",
            susceptance=10.0,
            capacity_mw=40.0,
        ),
    ),
    renewable_bus: str = "north",
    thermal_bus: str = "north",
    thermal_capacity_mw: float = 0.0,
    demand_bus: str = "south",
) -> ModelConfig:
    config = _example_config()
    thermal = _thermal(thermal_capacity_mw)
    imports = ImportConfig(
        maximum_power_mw=0.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
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
        network=NetworkConfig(
            loss_fraction=0.0,
            transfer_capacity_mw=1_000.0,
            network_mode=cast(Literal["aggregate", "nodal"], mode),
            slack_bus_id=buses[0] if mode == "nodal" else None,
        ),
        imports=imports,
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
        portfolio=replace(
            config.portfolio,
            buses=tuple(BusConfig(id=bus_id, zone_id="zone") for bus_id in buses),
            lines=lines if mode == "nodal" else (),
            renewable_generators=(
                RenewableGeneratorConfig(
                    id="solar",
                    kind="solar",
                    bus_id=renewable_bus,
                    capacity_mw=100.0,
                    time_series_key="solar_mw",
                ),
                RenewableGeneratorConfig(
                    id="wind",
                    kind="wind",
                    bus_id=renewable_bus,
                    capacity_mw=100.0,
                    time_series_key="wind_mw",
                ),
            ),
            thermal_generators=(
                ThermalGeneratorConfig(
                    id="thermal",
                    bus_id=thermal_bus,
                    fuel_id="gas",
                    config=thermal,
                ),
            ),
            storage_units=(),
            hydro_units=(),
            imports=(ImportResourceConfig(id="imports", bus_id=buses[0], config=imports),),
            demand=(DemandConfig(id="load", bus_id=demand_bus, time_series_key="load_mw"),),
        ),
    )


def _solve(
    config: ModelConfig,
    *,
    renewable: float,
    demand: float,
    line_availability: float | None = None,
):
    line_factors = (
        {"north_south": np.array([line_availability], dtype=float)}
        if line_availability is not None
        else None
    )
    return UnitCommitment(config).solve(
        renewable_available_mw=np.array([renewable], dtype=float),
        gross_demand_mw=np.array([demand], dtype=float),
        demand_profiles_mw={"load": np.array([demand], dtype=float)}
        if config.network.network_mode == "nodal"
        else None,
        renewable_availability_by_asset_mw={
            "solar": np.array([renewable], dtype=float),
            "wind": np.array([0.0], dtype=float),
        },
        line_availability_factors=line_factors,
    )


def test_constrained_two_bus_dispatch_sheds_load_at_sink() -> None:
    result = _solve(_config(), renewable=100.0, demand=100.0)
    frame = result.frame

    assert frame["renewable_used_mw"].iloc[0] == pytest.approx(40.0)
    assert frame["source_load_shed_mw"].iloc[0] == pytest.approx(60.0)
    assert frame["line_flow_mw__north_south"].iloc[0] == pytest.approx(40.0)
    assert frame["line_abs_utilisation__north_south"].iloc[0] == pytest.approx(1.0)
    assert frame["bus_balance_residual_mw"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_unconstrained_nodal_dispatch_matches_aggregate_dispatch() -> None:
    nodal = _solve(
        _config(
            lines=(
                TransmissionLineConfig(
                    id="north_south",
                    from_bus_id="north",
                    to_bus_id="south",
                    susceptance=10.0,
                    capacity_mw=200.0,
                ),
            )
        ),
        renewable=100.0,
        demand=100.0,
    )
    aggregate = _solve(_config(mode="aggregate"), renewable=100.0, demand=100.0)

    assert nodal.objective_eur == pytest.approx(aggregate.objective_eur)
    assert nodal.frame["source_load_shed_mw"].iloc[0] == pytest.approx(
        aggregate.frame["source_load_shed_mw"].iloc[0]
    )
    assert nodal.frame["renewable_used_mw"].iloc[0] == pytest.approx(
        aggregate.frame["renewable_used_mw"].iloc[0]
    )


def test_three_bus_loop_exports_angles_flows_and_residuals() -> None:
    config = _config(
        buses=("a", "b", "c"),
        renewable_bus="a",
        demand_bus="b",
        lines=(
            TransmissionLineConfig(
                id="ab", from_bus_id="a", to_bus_id="b", susceptance=10.0, capacity_mw=100.0
            ),
            TransmissionLineConfig(
                id="bc", from_bus_id="b", to_bus_id="c", susceptance=10.0, capacity_mw=100.0
            ),
            TransmissionLineConfig(
                id="ac", from_bus_id="a", to_bus_id="c", susceptance=10.0, capacity_mw=100.0
            ),
        ),
    )
    config = replace(
        config,
        portfolio=replace(
            config.portfolio,
            demand=(
                DemandConfig(id="load_b", bus_id="b", time_series_key="load_b_mw"),
                DemandConfig(id="load_c", bus_id="c", time_series_key="load_c_mw"),
            ),
        ),
    )
    result = UnitCommitment(config).solve(
        renewable_available_mw=np.array([90.0], dtype=float),
        gross_demand_mw=np.array([90.0], dtype=float),
        demand_profiles_mw={
            "load_b": np.array([45.0], dtype=float),
            "load_c": np.array([45.0], dtype=float),
        },
        renewable_availability_by_asset_mw={
            "solar": np.array([90.0], dtype=float),
            "wind": np.array([0.0], dtype=float),
        },
    )

    frame = result.frame
    assert {"bus_voltage_angle_rad__a", "bus_voltage_angle_rad__b", "line_flow_mw__ab"} <= set(
        frame.columns
    )
    assert frame["line_max_abs_utilisation"].iloc[0] <= 1.0 + 1e-7
    assert frame["bus_balance_residual_mw"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_disconnected_nodal_network_is_rejected() -> None:
    config = _config(
        buses=("a", "b", "c"),
        renewable_bus="a",
        demand_bus="c",
        lines=(
            TransmissionLineConfig(
                id="ab", from_bus_id="a", to_bus_id="b", susceptance=10.0, capacity_mw=100.0
            ),
        ),
    )

    with pytest.raises(ConfigurationError, match="disconnected"):
        validate_config(config)


def test_line_availability_multiplier_limits_capacity() -> None:
    config = _config(
        lines=(
            TransmissionLineConfig(
                id="north_south",
                from_bus_id="north",
                to_bus_id="south",
                susceptance=10.0,
                capacity_mw=100.0,
            ),
        )
    )
    result = _solve(config, renewable=100.0, demand=100.0, line_availability=0.25)

    assert result.frame["line_capacity_available_mw__north_south"].iloc[0] == pytest.approx(25.0)
    assert result.frame["line_flow_mw__north_south"].iloc[0] == pytest.approx(25.0)
    assert result.frame["source_load_shed_mw"].iloc[0] == pytest.approx(75.0)


def test_asset_relocation_changes_congestion_outcome() -> None:
    source_thermal = _solve(
        _config(thermal_bus="north", thermal_capacity_mw=100.0),
        renewable=0.0,
        demand=100.0,
    )
    sink_thermal = _solve(
        _config(thermal_bus="south", thermal_capacity_mw=100.0),
        renewable=0.0,
        demand=100.0,
    )

    assert source_thermal.frame["thermal_output_mw__thermal"].iloc[0] == pytest.approx(40.0)
    assert source_thermal.frame["source_load_shed_mw"].iloc[0] == pytest.approx(60.0)
    assert sink_thermal.frame["thermal_output_mw__thermal"].iloc[0] == pytest.approx(100.0)
    assert sink_thermal.frame["source_load_shed_mw"].iloc[0] == pytest.approx(0.0)
