from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.exceptions import DataValidationError

REQUIRED_COLUMNS = (
    "timestamp",
    "demand_mw",
    "irradiance_w_m2",
    "ambient_temperature_c",
    "wind_speed_m_s",
)


def load_input_data(path: str | Path, time_step_hours: float) -> pd.DataFrame:
    """Load and validate the simulator input table."""
    input_path = Path(path)
    if not input_path.exists():
        raise DataValidationError(f"Input CSV does not exist: {input_path}")

    frame = pd.read_csv(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if frame.empty:
        raise DataValidationError("Input data must contain at least one row")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise DataValidationError("One or more timestamps cannot be parsed")
    if result["timestamp"].duplicated().any():
        raise DataValidationError("Timestamps must be unique")
    if not result["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamps must be strictly increasing")

    numeric_columns = REQUIRED_COLUMNS[1:]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataValidationError(f"Column {column!r} contains non-finite values")

    for column in ("demand_mw", "irradiance_w_m2", "wind_speed_m_s"):
        if (result[column] < 0.0).any():
            raise DataValidationError(f"Column {column!r} must be non-negative")

    if len(result) > 1:
        timestamps = pd.DatetimeIndex(result["timestamp"])
        timestamp_values = timestamps.to_numpy(dtype="datetime64[ns]")
        differences = np.diff(timestamp_values) / np.timedelta64(1, "s")
        expected_seconds = time_step_hours * 3600.0
        if not np.allclose(
            differences,
            expected_seconds,
            rtol=0.0,
            atol=DEFAULT_NUMERICAL_POLICY.time_axis_seconds,
        ):
            raise DataValidationError(
                "Timestamps are not equally spaced according to simulation.time_step_hours"
            )

    return result
