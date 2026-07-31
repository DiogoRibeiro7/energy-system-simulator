from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from energy_system_simulator.config import (
    ModelConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _unit(
    name: str,
    *,
    minimum_output_mw: float = 0.0,
    maximum_output_mw: float = 100.0,
    ramp_mw_per_hour: float = 1_000.0,
    startup_ramp_mw: float | None = None,
    shutdown_ramp_mw: float | None = None,
    variable_cost_eur_per_mwh: float = 10.0,
    no_load_cost_eur_per_hour: float = 0.0,
    startup_cost_eur: float = 0.0,
    shutdown_cost_eur: float = 0.0,
    minimum_up_hours: float = 1.0,
    minimum_down_hours: float = 1.0,
    initial_on: bool = False,
    initial_output_mw: float = 0.0,
    initial_up_time_hours: float = 0.0,
    initial_down_time_hours: float = 10.0,
) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        name=name,
        minimum_output_mw=minimum_output_mw,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=ramp_mw_per_hour,
        ramp_down_mw_per_hour=ramp_mw_per_hour,
        startup_ramp_mw=startup_ramp_mw if startup_ramp_mw is not None else maximum_output_mw,
        shutdown_ramp_mw=shutdown_ramp_mw if shutdown_ramp_mw is not None else maximum_output_mw,
        variable_cost_eur_per_mwh=variable_cost_eur_per_mwh,
        no_load_cost_eur_per_hour=no_load_cost_eur_per_hour,
        startup_cost_eur=startup_cost_eur,
        shutdown_cost_eur=shutdown_cost_eur,
        emission_factor_tonnes_per_mwh=0.0,
        minimum_up_hours=minimum_up_hours,
        minimum_down_hours=minimum_down_hours,
        initial_on=initial_on,
        initial_output_mw=initial_output_mw,
        initial_up_time_hours=initial_up_time_hours,
        initial_down_time_hours=initial_down_time_hours,
        terminal_commitment_mode="forbid_incomplete_transitions",
    )


def _generator(
    asset_id: str,
    config: ThermalConfig,
    *,
    availability_factor: float = 1.0,
) -> ThermalGeneratorConfig:
    return ThermalGeneratorConfig(
        id=asset_id,
        bus_id="system",
        fuel_id="gas",
        config=config,
        availability_factor=availability_factor,
    )


def _multi_unit_config(*generators: ThermalGeneratorConfig) -> ModelConfig:
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
    portfolio = replace(config.portfolio, thermal_generators=tuple(generators))
    return replace(
        config,
        thermal=generators[0].config,
        portfolio=portfolio,
        battery=battery,
        imports=replace(config.imports, maximum_power_mw=0.0),
        network=replace(config.network, loss_fraction=0.0),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
    )


def test_merit_order_dispatch_uses_low_variable_cost_unit_first() -> None:
    cheap = _generator(
        "cheap",
        _unit("cheap", maximum_output_mw=50.0, variable_cost_eur_per_mwh=5.0),
    )
    peaker = _generator(
        "peaker",
        _unit("peaker", maximum_output_mw=100.0, variable_cost_eur_per_mwh=50.0),
    )
    result = UnitCommitment(_multi_unit_config(cheap, peaker)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([80.0]),
    )

    assert result.frame["thermal_output_mw__cheap"].iloc[0] == pytest.approx(50.0)
    assert result.frame["thermal_output_mw__peaker"].iloc[0] == pytest.approx(30.0)


def test_minimum_output_can_make_smaller_unit_optimal() -> None:
    baseload = _generator(
        "baseload",
        _unit(
            "baseload",
            minimum_output_mw=50.0,
            maximum_output_mw=100.0,
            variable_cost_eur_per_mwh=1.0,
        ),
    )
    flexible = _generator(
        "flexible",
        _unit("flexible", maximum_output_mw=40.0, variable_cost_eur_per_mwh=20.0),
    )
    result = UnitCommitment(_multi_unit_config(baseload, flexible)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([30.0]),
    )

    assert result.frame["thermal_on__baseload"].iloc[0] == 0
    assert result.frame["thermal_output_mw__flexible"].iloc[0] == pytest.approx(30.0)


def test_peaker_starts_when_baseload_is_ramp_constrained() -> None:
    baseload = _generator(
        "baseload",
        _unit(
            "baseload",
            maximum_output_mw=100.0,
            ramp_mw_per_hour=10.0,
            variable_cost_eur_per_mwh=1.0,
            initial_on=True,
            initial_output_mw=50.0,
            initial_up_time_hours=10.0,
        ),
    )
    peaker = _generator(
        "peaker",
        _unit(
            "peaker",
            maximum_output_mw=100.0,
            variable_cost_eur_per_mwh=100.0,
            no_load_cost_eur_per_hour=1.0,
        ),
    )
    result = UnitCommitment(_multi_unit_config(baseload, peaker)).solve(
        renewable_available_mw=np.zeros(2),
        gross_demand_mw=np.array([50.0, 90.0]),
    )

    assert result.frame["thermal_output_mw__baseload"].iloc[1] == pytest.approx(60.0)
    assert result.frame["thermal_startup__peaker"].iloc[1] == 1
    assert result.frame["thermal_output_mw__peaker"].iloc[1] == pytest.approx(30.0)


def test_high_startup_cost_keeps_unit_online_through_low_load_period() -> None:
    unit = _generator(
        "baseload",
        _unit(
            "baseload",
            minimum_output_mw=10.0,
            maximum_output_mw=100.0,
            variable_cost_eur_per_mwh=1.0,
            no_load_cost_eur_per_hour=1.0,
            startup_cost_eur=1_000.0,
            initial_on=True,
            initial_output_mw=50.0,
            initial_up_time_hours=10.0,
        ),
    )
    result = UnitCommitment(_multi_unit_config(unit)).solve(
        renewable_available_mw=np.zeros(3),
        gross_demand_mw=np.array([50.0, 10.0, 50.0]),
    )

    assert result.frame["thermal_on__baseload"].tolist() == [1, 1, 1]
    assert result.frame["thermal_startup__baseload"].sum() == 0


def test_residual_initial_down_time_is_enforced_per_generator() -> None:
    cheap = _generator(
        "cheap",
        _unit(
            "cheap",
            maximum_output_mw=50.0,
            variable_cost_eur_per_mwh=1.0,
            minimum_down_hours=2.0,
            initial_down_time_hours=0.0,
        ),
    )
    expensive = _generator(
        "expensive",
        _unit("expensive", maximum_output_mw=50.0, variable_cost_eur_per_mwh=100.0),
    )
    result = UnitCommitment(_multi_unit_config(cheap, expensive)).solve(
        renewable_available_mw=np.zeros(3),
        gross_demand_mw=np.array([50.0, 50.0, 50.0]),
    )

    assert result.frame["thermal_on__cheap"].tolist() == [0, 0, 1]
    assert result.frame["thermal_output_mw__expensive"].iloc[:2].tolist() == pytest.approx(
        [50.0, 50.0]
    )


def test_unavailable_unit_cannot_generate() -> None:
    unavailable = _generator(
        "unavailable",
        _unit("unavailable", maximum_output_mw=100.0, variable_cost_eur_per_mwh=1.0),
        availability_factor=0.0,
    )
    available = _generator(
        "available",
        _unit("available", maximum_output_mw=100.0, variable_cost_eur_per_mwh=20.0),
    )
    result = UnitCommitment(_multi_unit_config(unavailable, available)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([40.0]),
    )

    assert result.frame["thermal_output_mw__unavailable"].iloc[0] == pytest.approx(0.0)
    assert result.frame["thermal_output_mw__available"].iloc[0] == pytest.approx(40.0)


def test_dynamic_availability_factor_limits_capacity_by_period() -> None:
    baseload = _generator(
        "baseload",
        _unit("baseload", maximum_output_mw=100.0, variable_cost_eur_per_mwh=1.0),
    )
    peaker = _generator(
        "peaker",
        _unit("peaker", maximum_output_mw=100.0, variable_cost_eur_per_mwh=50.0),
    )
    result = UnitCommitment(_multi_unit_config(baseload, peaker)).solve(
        renewable_available_mw=np.zeros(2),
        gross_demand_mw=np.array([80.0, 80.0]),
        thermal_availability_factors={"baseload": np.array([1.0, 0.5])},
    )

    assert result.frame["thermal_capacity_available_mw__baseload"].tolist() == pytest.approx(
        [100.0, 50.0]
    )
    assert result.frame["thermal_output_mw__peaker"].iloc[1] == pytest.approx(30.0)


def test_identical_unit_symmetry_serves_load_with_one_unit() -> None:
    first = _generator(
        "first",
        _unit(
            "first",
            maximum_output_mw=50.0,
            variable_cost_eur_per_mwh=10.0,
            no_load_cost_eur_per_hour=1.0,
        ),
    )
    second = _generator(
        "second",
        _unit(
            "second",
            maximum_output_mw=50.0,
            variable_cost_eur_per_mwh=10.0,
            no_load_cost_eur_per_hour=1.0,
        ),
    )
    result = UnitCommitment(_multi_unit_config(first, second)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([50.0]),
    )

    assert result.frame["thermal_output_mw"].iloc[0] == pytest.approx(50.0)
    assert result.frame["thermal_on"].iloc[0] == 1
    assert result.frame["thermal_startup"].iloc[0] == 1
