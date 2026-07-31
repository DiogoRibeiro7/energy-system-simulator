from __future__ import annotations

import numpy as np
import numpy.typing as npt

from energy_system_simulator.config import WindConfig

FloatArray = npt.NDArray[np.float64]


class WindFarm:
    """Piecewise wind-power model with a cubic partial-load region."""

    def __init__(self, config: WindConfig) -> None:
        self.config = config

    def output_mw(self, wind_speed_m_s: npt.ArrayLike) -> FloatArray:
        """Return available wind power in MW for each interval."""
        speed = np.asarray(wind_speed_m_s, dtype=np.float64)
        if np.any(~np.isfinite(speed)):
            raise ValueError("Wind speeds must be finite")
        if np.any(speed < 0.0):
            raise ValueError("Wind speeds must be non-negative")

        result = np.zeros_like(speed, dtype=np.float64)
        partial = (speed >= self.config.cut_in_speed_m_s) & (
            speed < self.config.rated_speed_m_s
        )
        rated = (speed >= self.config.rated_speed_m_s) & (
            speed < self.config.cut_out_speed_m_s
        )

        numerator = speed[partial] ** 3 - self.config.cut_in_speed_m_s**3
        denominator = (
            self.config.rated_speed_m_s**3 - self.config.cut_in_speed_m_s**3
        )
        result[partial] = self.config.capacity_mw * numerator / denominator
        result[rated] = self.config.capacity_mw
        return result
