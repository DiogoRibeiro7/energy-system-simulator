from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

from energy_system_simulator.config import (
    FrequencyConfig,
    ModelConfig,
    StorageUnitConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    load_config,
)
from energy_system_simulator.frequency import evaluate_frequency_adequacy


@dataclass(frozen=True)
class _SolvedDispatch:
    timeseries: pd.DataFrame
    objective_eur: float = 0.0


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal_config(maximum_output_mw: float) -> ThermalConfig:
    base = _example_config().thermal
    return replace(base, minimum_output_mw=0.0, maximum_output_mw=maximum_output_mw)


def _thermal_unit(
    unit_id: str,
    *,
    maximum_output_mw: float,
    synchronous_inertia_mw_s: float,
    primary_response_mw: float = 0.0,
    primary_response_time_seconds: float = 10.0,
) -> ThermalGeneratorConfig:
    return ThermalGeneratorConfig(
        id=unit_id,
        bus_id="system",
        fuel_id="gas",
        config=_thermal_config(maximum_output_mw),
        synchronous_inertia_mw_s=synchronous_inertia_mw_s,
        primary_response_mw=primary_response_mw,
        primary_response_time_seconds=primary_response_time_seconds,
    )


def _config(
    *,
    thermal: tuple[ThermalGeneratorConfig, ...] = (),
    storage: tuple[StorageUnitConfig, ...] = (),
    frequency: FrequencyConfig,
) -> ModelConfig:
    base = _example_config()
    return replace(
        base,
        frequency=frequency,
        portfolio=replace(
            base.portfolio,
            thermal_generators=thermal,
            storage_units=storage,
            hydro_units=(),
        ),
    )


def test_hand_calculated_two_generator_inertia_case() -> None:
    unit_a = _thermal_unit(
        "a",
        maximum_output_mw=100.0,
        synchronous_inertia_mw_s=100.0,
        primary_response_mw=30.0,
    )
    unit_b = _thermal_unit(
        "b",
        maximum_output_mw=100.0,
        synchronous_inertia_mw_s=200.0,
        primary_response_mw=40.0,
    )
    config = _config(
        thermal=(unit_a, unit_b),
        frequency=FrequencyConfig(
            minimum_inertia_mw_s=250.0,
            maximum_rocof_hz_per_s=5.0,
            maximum_primary_response_time_seconds=15.0,
        ),
    )
    frame = pd.DataFrame(
        {
            "thermal_on__a": [1.0],
            "thermal_output_mw__a": [50.0],
            "thermal_capacity_available_mw__a": [100.0],
            "thermal_upward_reserve_mw__a": [30.0],
            "thermal_on__b": [1.0],
            "thermal_output_mw__b": [30.0],
            "thermal_capacity_available_mw__b": [100.0],
            "thermal_upward_reserve_mw__b": [40.0],
            "imports_mw": [0.0],
        }
    )

    evaluation = evaluate_frequency_adequacy(config, _SolvedDispatch(frame))
    row = evaluation.records.iloc[0]

    assert evaluation.adequate
    assert row["synchronous_inertia_mw_s"] == pytest.approx(300.0)
    assert row["largest_credible_loss_mw"] == pytest.approx(50.0)
    assert row["rocof_hz_per_s"] == pytest.approx(50.0 * 50.0 / (2.0 * 300.0))
    assert row["sustained_primary_response_mw"] == pytest.approx(70.0)


def test_largest_online_unit_changes_credible_loss() -> None:
    small = _thermal_unit("small", maximum_output_mw=100.0, synchronous_inertia_mw_s=100.0)
    large = _thermal_unit("large", maximum_output_mw=100.0, synchronous_inertia_mw_s=100.0)
    config = _config(
        thermal=(small, large),
        frequency=FrequencyConfig(maximum_rocof_hz_per_s=20.0),
    )
    frame = pd.DataFrame(
        {
            "thermal_on__small": [1.0, 1.0],
            "thermal_output_mw__small": [30.0, 30.0],
            "thermal_capacity_available_mw__small": [100.0, 100.0],
            "thermal_on__large": [0.0, 1.0],
            "thermal_output_mw__large": [0.0, 80.0],
            "thermal_capacity_available_mw__large": [100.0, 100.0],
            "imports_mw": [0.0, 0.0],
        }
    )

    evaluation = evaluate_frequency_adequacy(config, _SolvedDispatch(frame))

    assert evaluation.records["largest_credible_loss_mw"].tolist() == pytest.approx([30.0, 80.0])


def test_battery_fast_response_is_limited_by_power_and_state_of_charge() -> None:
    base = _example_config()
    storage = StorageUnitConfig(
        id="battery",
        bus_id="system",
        config=replace(
            base.battery,
            power_capacity_mw=20.0,
            discharge_power_capacity_mw=20.0,
            minimum_soc_mwh=0.0,
            maximum_soc_mwh=1.0,
        ),
        fast_frequency_response_mw=50.0,
        fast_frequency_response_duration_seconds=10.0,
        synthetic_inertia_mw_s=1_000.0,
    )
    config = _config(
        storage=(storage,),
        frequency=FrequencyConfig(
            credible_loss_mw=20.0,
            minimum_inertia_mw_s=0.0,
            maximum_rocof_hz_per_s=2.0,
        ),
    )
    frame = pd.DataFrame(
        {
            "storage_soc_mwh__battery": [0.02],
            "storage_discharge_mw__battery": [0.0],
            "imports_mw": [0.0],
        }
    )

    evaluation = evaluate_frequency_adequacy(config, _SolvedDispatch(frame))
    row = evaluation.records.iloc[0]

    assert row["fast_frequency_response_mw"] == pytest.approx(7.2)
    assert row["synthetic_inertia_mw_s"] == pytest.approx(72.0)
    assert row["response_shortfall_mw"] == pytest.approx(12.8)


def test_removing_synchronous_unit_violates_inertia_when_energy_is_feasible() -> None:
    unit = _thermal_unit(
        "sync",
        maximum_output_mw=100.0,
        synchronous_inertia_mw_s=200.0,
        primary_response_mw=20.0,
    )
    config = _config(
        thermal=(unit,),
        frequency=FrequencyConfig(
            credible_loss_mw=10.0,
            minimum_inertia_mw_s=100.0,
            maximum_rocof_hz_per_s=5.0,
        ),
    )
    frame = pd.DataFrame(
        {
            "thermal_on__sync": [1.0, 0.0],
            "thermal_output_mw__sync": [0.0, 0.0],
            "thermal_capacity_available_mw__sync": [100.0, 100.0],
            "thermal_upward_reserve_mw__sync": [20.0, 0.0],
            "imports_mw": [10.0, 10.0],
        }
    )

    evaluation = evaluate_frequency_adequacy(config, _SolvedDispatch(frame))

    assert evaluation.records["adequate"].tolist() == [True, False]
    assert evaluation.records["limitation"].iloc[1] == "inertia"
    assert evaluation.records["inertia_shortfall_mw_s"].iloc[1] == pytest.approx(100.0)
