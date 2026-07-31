from __future__ import annotations

import numpy as np

from energy_system_simulator.config import SolarConfig, WindConfig
from energy_system_simulator.generation import SolarPlant, WindFarm


def test_solar_output_is_zero_at_night_and_clipped_to_capacity() -> None:
    plant = SolarPlant(
        SolarConfig(
            capacity_mw=100.0,
            performance_ratio=0.9,
            reference_irradiance_w_m2=1000.0,
            temperature_coefficient_per_c=-0.004,
            nominal_operating_cell_temperature_c=45.0,
        )
    )
    output = plant.output_mw([0.0, 1000.0, 2000.0], [15.0, 25.0, 5.0])
    assert output[0] == 0.0
    assert np.all(output >= 0.0)
    assert np.all(output <= 100.0)


def test_wind_power_curve_regions() -> None:
    farm = WindFarm(
        WindConfig(
            capacity_mw=50.0,
            cut_in_speed_m_s=3.0,
            rated_speed_m_s=12.0,
            cut_out_speed_m_s=25.0,
        )
    )
    output = farm.output_mw([2.0, 3.0, 8.0, 12.0, 20.0, 25.0])
    assert output[0] == 0.0
    assert output[1] == 0.0
    assert 0.0 < output[2] < 50.0
    assert output[3] == 50.0
    assert output[4] == 50.0
    assert output[5] == 0.0
