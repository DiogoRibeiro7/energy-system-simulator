from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from energy_system_simulator.config import (
    DemandConfig,
    ImportConfig,
    ImportResourceConfig,
    ModelConfig,
    ReserveConfig,
    StorageUnitConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal(
    *,
    maximum_output_mw: float,
    ramp_up_mw_per_hour: float = 1_000.0,
    ramp_down_mw_per_hour: float = 1_000.0,
    variable_cost_eur_per_mwh: float = 1.0,
    no_load_cost_eur_per_hour: float = 0.0,
) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        minimum_output_mw=0.0,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=ramp_up_mw_per_hour,
        ramp_down_mw_per_hour=ramp_down_mw_per_hour,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=variable_cost_eur_per_mwh,
        no_load_cost_eur_per_hour=no_load_cost_eur_per_hour,
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
    thermal: tuple[ThermalGeneratorConfig, ...],
    reserves: ReserveConfig,
    storage: tuple[StorageUnitConfig, ...] = (),
    imports_mw: float = 0.0,
) -> ModelConfig:
    config = _example_config()
    imports = ImportConfig(
        maximum_power_mw=imports_mw,
        price_eur_per_mwh=100.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    return replace(
        config,
        thermal=thermal[0].config,
        battery=storage[0].config if storage else replace(config.battery, power_capacity_mw=0.0),
        imports=imports,
        reserves=reserves,
        network=replace(config.network, loss_fraction=0.0),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
        portfolio=replace(
            config.portfolio,
            thermal_generators=thermal,
            storage_units=storage,
            hydro_units=(),
            imports=(ImportResourceConfig(id="imports", bus_id="system", config=imports),),
            demand=(DemandConfig(id="load", bus_id="system", time_series_key="load_mw"),),
        ),
    )


def _unit(unit_id: str, config: ThermalConfig) -> ThermalGeneratorConfig:
    return ThermalGeneratorConfig(id=unit_id, bus_id="system", fuel_id="gas", config=config)


def test_reserve_requirement_commits_additional_thermal_unit() -> None:
    cheap = _unit("cheap", _thermal(maximum_output_mw=60.0, no_load_cost_eur_per_hour=1.0))
    peaker = _unit("peaker", _thermal(maximum_output_mw=60.0, no_load_cost_eur_per_hour=10.0))

    no_reserve = UnitCommitment(_config(thermal=(cheap, peaker), reserves=ReserveConfig())).solve(
        np.array([0.0]), np.array([50.0])
    )
    with_reserve = UnitCommitment(
        _config(thermal=(cheap, peaker), reserves=ReserveConfig(upward_fixed_mw=50.0))
    ).solve(np.array([0.0]), np.array([50.0]))

    assert no_reserve.frame["thermal_on__peaker"].iloc[0] == 0
    assert with_reserve.frame["thermal_on__peaker"].iloc[0] == 1
    assert with_reserve.frame["reserve_upward_procured_mw"].iloc[0] == pytest.approx(50.0)


def test_storage_upward_reserve_is_limited_by_state_of_charge() -> None:
    base = _example_config()
    battery = replace(
        base.battery,
        energy_capacity_mwh=100.0,
        power_capacity_mw=100.0,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=100.0,
        initial_soc_mwh=10.0,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode="free",
    )
    result = UnitCommitment(
        _config(
            thermal=(_unit("offline", _thermal(maximum_output_mw=0.0)),),
            storage=(StorageUnitConfig(id="battery", bus_id="system", config=battery),),
            reserves=ReserveConfig(upward_fixed_mw=30.0),
        )
    ).solve(np.array([0.0]), np.array([0.0]))

    assert result.frame["storage_upward_reserve_mw__battery"].iloc[0] == pytest.approx(10.0)
    assert result.frame["reserve_upward_shortfall_mw"].iloc[0] == pytest.approx(20.0)


def test_thermal_upward_reserve_is_ramp_limited() -> None:
    result = UnitCommitment(
        _config(
            thermal=(
                _unit(
                    "slow",
                    _thermal(maximum_output_mw=100.0, ramp_up_mw_per_hour=10.0),
                ),
            ),
            reserves=ReserveConfig(upward_fixed_mw=50.0),
        )
    ).solve(np.array([0.0]), np.array([0.0]))

    assert result.frame["thermal_upward_reserve_mw__slow"].iloc[0] == pytest.approx(10.0)
    assert result.frame["reserve_upward_shortfall_mw"].iloc[0] == pytest.approx(40.0)


def test_reserve_shortfall_reports_insufficient_capacity() -> None:
    result = UnitCommitment(
        _config(
            thermal=(_unit("small", _thermal(maximum_output_mw=20.0)),),
            reserves=ReserveConfig(upward_fixed_mw=50.0),
        )
    ).solve(np.array([0.0]), np.array([0.0]))

    assert result.frame["thermal_upward_reserve_mw__small"].iloc[0] == pytest.approx(20.0)
    assert result.frame["reserve_upward_shortfall_mw"].iloc[0] == pytest.approx(30.0)
    assert result.cost_components_eur["reserve_shortfall_cost_eur"] == pytest.approx(3_000_000.0)


def test_zero_reserve_configuration_reproduces_prior_solution() -> None:
    unit = _unit("thermal", _thermal(maximum_output_mw=100.0))
    config = _config(thermal=(unit,), reserves=ReserveConfig())

    result = UnitCommitment(config).solve(np.array([0.0]), np.array([50.0]))

    assert "reserve_upward_procured_mw" not in result.frame
    assert result.frame["thermal_output_mw__thermal"].iloc[0] == pytest.approx(50.0)
    assert result.objective_eur == pytest.approx(50.0)


def test_reserve_procurement_cost_reconciles_with_objective() -> None:
    result = UnitCommitment(
        _config(
            thermal=(_unit("thermal", _thermal(maximum_output_mw=100.0)),),
            reserves=ReserveConfig(
                upward_fixed_mw=10.0,
                thermal_upward_cost_eur_per_mw_hour=2.0,
            ),
        )
    ).solve(np.array([0.0]), np.array([50.0]))

    assert result.frame["reserve_upward_procured_mw"].iloc[0] == pytest.approx(10.0)
    assert result.cost_components_eur["reserve_procurement_cost_eur"] == pytest.approx(20.0)
    assert result.objective_eur == pytest.approx(70.0)
