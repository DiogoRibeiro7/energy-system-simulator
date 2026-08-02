from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from energy_system_simulator.config import (
    RenewableGeneratorConfig,
    WindPowerCurvePointConfig,
    load_config,
)
from energy_system_simulator.exceptions import ConfigurationError, DataValidationError
from energy_system_simulator.simulation.assets import RenewableAsset


def test_detailed_solar_profile_reports_hand_calculated_derating() -> None:
    asset = RenewableAsset.from_config(
        RenewableGeneratorConfig(
            id="solar",
            kind="solar",
            bus_id="bus",
            capacity_mw=80.0,
            availability_model="detailed",
            time_series_key="poa_w_m2",
            ambient_temperature_key="ambient_c",
            dc_capacity_mw=100.0,
            inverter_ac_capacity_mw=80.0,
            inverter_efficiency=0.98,
            degradation_factor=0.98,
            availability_factor=0.9,
            performance_ratio=0.95,
            reference_irradiance_w_m2=1000.0,
            temperature_coefficient_per_c=-0.004,
            nominal_operating_cell_temperature_c=45.0,
            soiling_loss_fraction=0.1,
            snow_loss_fraction=0.2,
        )
    )
    data = pd.DataFrame({"poa_w_m2": [1000.0], "ambient_c": [25.0]})

    profile = asset.availability_profile(data)

    dc_potential = 95.0
    cell_temperature = 25.0 + ((45.0 - 20.0) / 800.0) * 1000.0
    temperature_adjusted = dc_potential * (1.0 - 0.004 * (cell_temperature - 25.0))
    ac_before_clip = temperature_adjusted * 0.98
    ac_potential = 80.0
    derating_factor = 0.98 * 0.9 * 0.8 * 0.9
    assert profile.variables["dc_potential_mw"][0][0] == pytest.approx(dc_potential)
    assert profile.variables["temperature_loss_mw"][0][0] == pytest.approx(
        dc_potential - temperature_adjusted
    )
    assert profile.variables["clipping_loss_mw"][0][0] == pytest.approx(ac_before_clip - 80.0)
    assert profile.available_mw[0] == pytest.approx(ac_potential * derating_factor)
    assert profile.available_mw[0] <= 80.0


def test_wind_power_curve_profile_adjusts_height_density_and_losses() -> None:
    asset = RenewableAsset.from_config(
        RenewableGeneratorConfig(
            id="wind",
            kind="wind",
            bus_id="bus",
            capacity_mw=6.0,
            availability_model="power_curve",
            time_series_key="wind_10m",
            cut_in_speed_m_s=3.0,
            rated_speed_m_s=12.0,
            cut_out_speed_m_s=25.0,
            measurement_height_m=10.0,
            hub_height_m=80.0,
            wind_speed_adjustment="power_law",
            wind_shear_exponent=0.25,
            air_density_correction=True,
            air_temperature_key="air_temp_c",
            air_pressure_key="pressure_pa",
            turbine_count=2,
            turbine_rated_capacity_mw=3.0,
            power_curve=(
                WindPowerCurvePointConfig(wind_speed_m_s=0.0, power_mw=0.0),
                WindPowerCurvePointConfig(wind_speed_m_s=5.0, power_mw=0.0),
                WindPowerCurvePointConfig(wind_speed_m_s=10.0, power_mw=2.0),
                WindPowerCurvePointConfig(wind_speed_m_s=15.0, power_mw=3.0),
            ),
            wake_loss_fraction=0.1,
            electrical_loss_fraction=0.05,
            availability_factor=0.8,
        )
    )
    data = pd.DataFrame(
        {
            "wind_10m": [5.0],
            "air_temp_c": [15.0],
            "pressure_pa": [101325.0],
        }
    )

    profile = asset.availability_profile(data)

    hub_speed = 5.0 * (80.0 / 10.0) ** 0.25
    per_turbine = np.interp(hub_speed, [0.0, 5.0, 10.0, 15.0], [0.0, 0.0, 2.0, 3.0])
    density_factor = 101325.0 / (287.05 * (15.0 + 273.15)) / 1.225
    gross = per_turbine * 2.0
    expected = gross * density_factor * 0.9 * 0.95 * 0.8
    assert profile.variables["hub_height_wind_speed_m_s"][0][0] == pytest.approx(hub_speed)
    assert profile.variables["gross_potential_mw"][0][0] == pytest.approx(gross)
    assert profile.available_mw[0] == pytest.approx(expected)


def test_wind_power_curve_rejects_out_of_range_input() -> None:
    asset = RenewableAsset.from_config(
        RenewableGeneratorConfig(
            id="wind",
            kind="wind",
            bus_id="bus",
            capacity_mw=3.0,
            availability_model="power_curve",
            time_series_key="wind_speed_m_s",
            cut_in_speed_m_s=3.0,
            rated_speed_m_s=12.0,
            cut_out_speed_m_s=25.0,
            power_curve=(
                WindPowerCurvePointConfig(wind_speed_m_s=0.0, power_mw=0.0),
                WindPowerCurvePointConfig(wind_speed_m_s=10.0, power_mw=3.0),
            ),
        )
    )

    with pytest.raises(DataValidationError, match="outside the configured wind power curve range"):
        asset.availability_profile(pd.DataFrame({"wind_speed_m_s": [12.0]}))


def test_wind_power_curve_validation_requires_strictly_increasing_speeds(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "portfolio_two_thermal.yaml").read_text())
    raw["paths"]["output_directory"] = str(tmp_path / "outputs")
    raw["renewable_generators"][1].update(
        {
            "availability_model": "power_curve",
            "power_curve": [
                {"wind_speed_m_s": 0.0, "power_mw": 0.0},
                {"wind_speed_m_s": 5.0, "power_mw": 1.0},
                {"wind_speed_m_s": 5.0, "power_mw": 2.0},
            ],
        }
    )
    config_path = tmp_path / "bad-curve.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="strictly increasing"):
        load_config(config_path)
