from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from energy_system_simulator.config import SolarConfig, WindConfig, load_config
from energy_system_simulator.exceptions import DataValidationError
from energy_system_simulator.generation import SolarPlant, WindFarm
from energy_system_simulator.simulation import SimulationEngine


def test_example_simulation_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    result = SimulationEngine(config).run()
    assert len(result.timeseries) == 336
    assert result.summary["total_demand_mwh"] > 0.0
    assert result.summary["served_demand_mwh"] <= result.summary["total_demand_mwh"] + 1e-6
    assert result.summary["renewable_available_mwh"] > 0.0


def test_multiple_renewable_assets_reconcile_to_aggregate_dispatch(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "multi_renewable.csv"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "demand_mw": [80.0, 90.0, 100.0, 70.0],
            "irradiance_w_m2": [0.0, 0.0, 0.0, 0.0],
            "ambient_temperature_c": [12.0, 12.0, 12.0, 12.0],
            "wind_speed_m_s": [0.0, 0.0, 0.0, 0.0],
            "solar_a_w_m2": [0.0, 700.0, 900.0, 200.0],
            "solar_b_w_m2": [100.0, 500.0, 1000.0, 600.0],
            "ambient_a_c": [10.0, 11.0, 12.0, 13.0],
            "ambient_b_c": [15.0, 16.0, 17.0, 18.0],
            "wind_a_m_s": [4.0, 8.0, 12.0, 20.0],
            "wind_b_m_s": [5.0, 9.0, 13.0, 25.0],
        }
    )
    frame.to_csv(input_path, index=False)

    raw = yaml.safe_load((root / "configs" / "portfolio_two_thermal.yaml").read_text())
    raw["paths"]["input_csv"] = str(input_path)
    raw["paths"]["output_directory"] = str(tmp_path / "outputs")
    raw["renewable_generators"] = [
        {
            "id": "solar-a",
            "kind": "solar",
            "bus_id": "north-hub",
            "capacity_mw": 40.0,
            "performance_ratio": 0.86,
            "reference_irradiance_w_m2": 1000.0,
            "temperature_coefficient_per_c": -0.004,
            "nominal_operating_cell_temperature_c": 45.0,
            "time_series_key": "solar_a_w_m2",
            "ambient_temperature_key": "ambient_a_c",
        },
        {
            "id": "solar-b",
            "kind": "solar",
            "bus_id": "south-hub",
            "capacity_mw": 25.0,
            "performance_ratio": 0.9,
            "reference_irradiance_w_m2": 1000.0,
            "temperature_coefficient_per_c": -0.003,
            "nominal_operating_cell_temperature_c": 42.0,
            "time_series_key": "solar_b_w_m2",
            "ambient_temperature_key": "ambient_b_c",
        },
        {
            "id": "wind-a",
            "kind": "wind",
            "bus_id": "north-hub",
            "capacity_mw": 30.0,
            "cut_in_speed_m_s": 3.0,
            "rated_speed_m_s": 12.0,
            "cut_out_speed_m_s": 25.0,
            "time_series_key": "wind_a_m_s",
        },
        {
            "id": "wind-b",
            "kind": "wind",
            "bus_id": "south-hub",
            "capacity_mw": 45.0,
            "cut_in_speed_m_s": 3.0,
            "rated_speed_m_s": 12.0,
            "cut_out_speed_m_s": 25.0,
            "time_series_key": "wind_b_m_s",
        },
    ]
    config_path = tmp_path / "multi_renewable.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)

    result = SimulationEngine(config).run()
    asset_table = result.asset_timeseries
    available = asset_table[asset_table["variable"] == "available_mw"]
    used = asset_table[asset_table["variable"] == "used_mw"]
    curtailed = asset_table[asset_table["variable"] == "curtailed_mw"]

    expected_solar_a = SolarPlant(
        SolarConfig(
            capacity_mw=40.0,
            performance_ratio=0.86,
            reference_irradiance_w_m2=1000.0,
            temperature_coefficient_per_c=-0.004,
            nominal_operating_cell_temperature_c=45.0,
        )
    ).output_mw(frame["solar_a_w_m2"], frame["ambient_a_c"])
    expected_wind_b = WindFarm(
        WindConfig(
            capacity_mw=45.0,
            cut_in_speed_m_s=3.0,
            rated_speed_m_s=12.0,
            cut_out_speed_m_s=25.0,
        )
    ).output_mw(frame["wind_b_m_s"])

    assert set(available["asset_id"]) == {"solar-a", "solar-b", "wind-a", "wind-b"}
    assert np.allclose(
        available.groupby("timestamp")["value"].sum().to_numpy(),
        result.timeseries["renewable_available_mw"].to_numpy(),
    )
    assert np.allclose(
        used.groupby("timestamp")["value"].sum().to_numpy(),
        result.timeseries["renewable_used_mw"].to_numpy(),
    )
    assert np.allclose(
        curtailed.groupby("timestamp")["value"].sum().to_numpy(),
        result.timeseries["renewable_curtailed_mw"].to_numpy(),
    )
    assert np.allclose(
        result.timeseries["renewable_available_mw__solar-a"].to_numpy(),
        expected_solar_a,
    )
    assert np.allclose(
        result.timeseries["renewable_available_mw__wind-b"].to_numpy(),
        expected_wind_b,
    )
    assert set(result.summary["renewable_assets"]) == {
        "solar-a",
        "solar-b",
        "wind-a",
        "wind-b",
    }


def test_missing_asset_input_column_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "demand_mw": [20.0, 20.0],
            "irradiance_w_m2": [0.0, 0.0],
            "ambient_temperature_c": [10.0, 10.0],
            "wind_speed_m_s": [4.0, 4.0],
        }
    ).to_csv(input_path, index=False)
    raw = yaml.safe_load((root / "configs" / "portfolio_two_thermal.yaml").read_text())
    raw["paths"]["input_csv"] = str(input_path)
    raw["renewable_generators"][0]["time_series_key"] = "missing_irradiance"
    config_path = tmp_path / "missing-column.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(DataValidationError, match="missing_irradiance"):
        SimulationEngine(load_config(config_path)).run()
