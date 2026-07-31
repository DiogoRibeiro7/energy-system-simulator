from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy_system_simulator.data import load_input_data
from energy_system_simulator.exceptions import DataValidationError


def test_rejects_irregular_timestamps(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"],
            "demand_mw": [10.0, 12.0],
            "irradiance_w_m2": [0.0, 0.0],
            "ambient_temperature_c": [10.0, 10.0],
            "wind_speed_m_s": [5.0, 5.0],
        }
    )
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        load_input_data(path, time_step_hours=1.0)
