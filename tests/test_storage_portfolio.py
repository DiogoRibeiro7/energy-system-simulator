from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from energy_system_simulator.config import (
    BatteryConfig,
    ModelConfig,
    StorageDegradationBandConfig,
    StorageUnitConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _storage(
    *,
    energy_capacity_mwh: float,
    charge_power_mw: float,
    discharge_power_mw: float,
    initial_soc_mwh: float = 0.0,
    minimum_final_soc_mwh: float = 0.0,
    terminal_soc_mode: Literal["minimum", "exact", "cyclic", "free"] = "free",
    charge_efficiency: float = 1.0,
    discharge_efficiency: float = 1.0,
    self_discharge_rate_per_hour: float = 0.0,
    throughput_cost_eur_per_mwh: float = 0.0,
    degradation_bands: tuple[StorageDegradationBandConfig, ...] = (),
    technology: Literal["battery", "pumped_storage"] = "battery",
) -> BatteryConfig:
    base = _example_config().battery
    return replace(
        base,
        technology=technology,
        energy_capacity_mwh=energy_capacity_mwh,
        power_capacity_mw=max(charge_power_mw, discharge_power_mw),
        charge_power_capacity_mw=charge_power_mw,
        discharge_power_capacity_mw=discharge_power_mw,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=energy_capacity_mwh,
        initial_soc_mwh=initial_soc_mwh,
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        self_discharge_rate_per_hour=self_discharge_rate_per_hour,
        throughput_cost_eur_per_mwh=throughput_cost_eur_per_mwh,
        minimum_final_soc_mwh=minimum_final_soc_mwh,
        terminal_soc_mode=terminal_soc_mode,
        degradation_bands=degradation_bands,
    )


def _storage_unit(asset_id: str, config: BatteryConfig) -> StorageUnitConfig:
    return StorageUnitConfig(id=asset_id, bus_id="system", config=config)


def _config(*storage_units: StorageUnitConfig) -> ModelConfig:
    config = _example_config()
    thermal = replace(
        config.thermal,
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
    return replace(
        config,
        thermal=thermal,
        portfolio=replace(config.portfolio, storage_units=storage_units),
        battery=storage_units[0].config,
        imports=replace(config.imports, maximum_power_mw=0.0),
        network=replace(config.network, loss_fraction=0.0),
        penalties=replace(
            config.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
    )


def test_multiple_storage_assets_operate_without_column_collisions() -> None:
    battery = _storage_unit(
        "short_battery",
        _storage(energy_capacity_mwh=50.0, charge_power_mw=50.0, discharge_power_mw=50.0),
    )
    pumped = _storage_unit(
        "pumped_storage",
        _storage(
            energy_capacity_mwh=200.0,
            charge_power_mw=100.0,
            discharge_power_mw=100.0,
            technology="pumped_storage",
        ),
    )

    result = UnitCommitment(_config(battery, pumped)).solve(
        renewable_available_mw=np.array([160.0, 0.0]),
        gross_demand_mw=np.array([0.0, 120.0]),
    )

    frame = result.frame
    assert frame["storage_charge_mw__short_battery"].iloc[0] > 0.0
    assert frame["storage_charge_mw__pumped_storage"].iloc[0] > 0.0
    assert frame["storage_discharge_mw__short_battery"].iloc[1] > 0.0
    assert frame["storage_discharge_mw__pumped_storage"].iloc[1] > 0.0
    assert frame["battery_discharge_mw"].iloc[1] == pytest.approx(120.0)


def test_storage_self_discharge_applies_over_time_step() -> None:
    storage = _storage_unit(
        "battery",
        _storage(
            energy_capacity_mwh=100.0,
            charge_power_mw=0.0,
            discharge_power_mw=0.0,
            initial_soc_mwh=100.0,
            self_discharge_rate_per_hour=0.1,
        ),
    )

    result = UnitCommitment(_config(storage)).solve(
        renewable_available_mw=np.zeros(2),
        gross_demand_mw=np.zeros(2),
    )

    assert result.frame["storage_soc_mwh__battery"].tolist() == pytest.approx([90.0, 81.0])


def test_exact_storage_modes_prevent_simultaneous_charge_and_discharge() -> None:
    storage = _storage_unit(
        "battery",
        _storage(
            energy_capacity_mwh=100.0,
            charge_power_mw=100.0,
            discharge_power_mw=100.0,
            initial_soc_mwh=100.0,
            throughput_cost_eur_per_mwh=-1.0,
        ),
    )

    result = UnitCommitment(_config(storage)).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.zeros(1),
    )

    product = (
        result.frame["storage_charge_mw__battery"] * result.frame["storage_discharge_mw__battery"]
    )
    assert product.max() == pytest.approx(0.0)


def test_storage_degradation_cost_reconciles_with_objective() -> None:
    storage = _storage_unit(
        "battery",
        _storage(
            energy_capacity_mwh=50.0,
            charge_power_mw=20.0,
            discharge_power_mw=20.0,
            degradation_bands=(
                StorageDegradationBandConfig(
                    id="throughput",
                    capacity_mwh=40.0,
                    cost_eur_per_mwh=5.0,
                ),
            ),
        ),
    )

    result = UnitCommitment(_config(storage)).solve(
        renewable_available_mw=np.array([20.0, 0.0]),
        gross_demand_mw=np.array([0.0, 20.0]),
    )

    assert result.frame["storage_degradation_cost_eur__battery"].sum() == pytest.approx(200.0)
    assert result.cost_components_eur["storage_degradation_cost_eur"] == pytest.approx(200.0)
    assert result.cost_components_eur["dispatch_load_shedding_cost_eur"] == pytest.approx(0.0)


def test_storage_terminal_modes_are_per_asset() -> None:
    exact = _storage_unit(
        "exact",
        _storage(
            energy_capacity_mwh=20.0,
            charge_power_mw=20.0,
            discharge_power_mw=20.0,
            initial_soc_mwh=5.0,
            minimum_final_soc_mwh=5.0,
            terminal_soc_mode=cast(Literal["minimum", "exact", "cyclic", "free"], "exact"),
        ),
    )
    free = _storage_unit(
        "free",
        _storage(
            energy_capacity_mwh=20.0,
            charge_power_mw=20.0,
            discharge_power_mw=20.0,
            terminal_soc_mode=cast(Literal["minimum", "exact", "cyclic", "free"], "free"),
        ),
    )

    result = UnitCommitment(_config(exact, free)).solve(
        renewable_available_mw=np.array([20.0, 0.0]),
        gross_demand_mw=np.array([0.0, 20.0]),
    )

    assert result.frame["storage_soc_mwh__exact"].iloc[-1] == pytest.approx(5.0)
