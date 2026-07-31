from __future__ import annotations

import numpy as np
import numpy.typing as npt

from energy_system_simulator.config import SolarConfig

FloatArray = npt.NDArray[np.float64]


class SolarPlant:
    """Photovoltaic generation model based on irradiance and cell temperature."""

    def __init__(self, config: SolarConfig) -> None:
        self.config = config

    def output_mw(
        self,
        irradiance_w_m2: npt.ArrayLike,
        ambient_temperature_c: npt.ArrayLike,
    ) -> FloatArray:
        """Return available photovoltaic power in MW for each interval."""
        irradiance = np.asarray(irradiance_w_m2, dtype=np.float64)
        ambient = np.asarray(ambient_temperature_c, dtype=np.float64)
        if irradiance.shape != ambient.shape:
            raise ValueError("Irradiance and temperature arrays must have equal shape")
        if np.any(~np.isfinite(irradiance)) or np.any(~np.isfinite(ambient)):
            raise ValueError("Solar inputs must be finite")
        if np.any(irradiance < 0.0):
            raise ValueError("Irradiance must be non-negative")

        cell_temperature = (
            ambient
            + ((self.config.nominal_operating_cell_temperature_c - 20.0) / 800.0) * irradiance
        )
        temperature_factor = 1.0 + self.config.temperature_coefficient_per_c * (
            cell_temperature - 25.0
        )
        raw_output = (
            self.config.capacity_mw
            * self.config.performance_ratio
            * irradiance
            / self.config.reference_irradiance_w_m2
            * temperature_factor
        )
        return np.clip(raw_output, 0.0, self.config.capacity_mw).astype(np.float64)
