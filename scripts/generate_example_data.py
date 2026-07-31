from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_example_data(periods: int = 336, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic two-week hourly demand and weather dataset."""
    if periods <= 0:
        raise ValueError("periods must be positive")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-05", periods=periods, freq="h", tz="UTC")
    hour = timestamps.hour.to_numpy(dtype=float)
    day = np.arange(periods, dtype=float) / 24.0

    morning = 35.0 * np.exp(-0.5 * ((hour - 8.0) / 2.2) ** 2)
    evening = 70.0 * np.exp(-0.5 * ((hour - 19.0) / 3.0) ** 2)
    weekday = np.where(timestamps.dayofweek.to_numpy() < 5, 22.0, -8.0)
    demand = 180.0 + morning + evening + weekday + 10.0 * np.sin(2.0 * np.pi * day / 7.0)
    demand += rng.normal(0.0, 5.0, periods)

    solar_shape = np.maximum(0.0, np.sin(np.pi * (hour - 7.0) / 10.0))
    cloud_factor = np.clip(0.78 + 0.18 * np.sin(2.0 * np.pi * day / 4.0), 0.35, 1.0)
    irradiance = 850.0 * solar_shape * cloud_factor

    temperature = 10.0 + 5.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    temperature += 1.5 * np.sin(2.0 * np.pi * day / 7.0)

    wind = 7.5 + 2.6 * np.sin(2.0 * np.pi * day / 3.5)
    wind += 1.4 * np.sin(2.0 * np.pi * hour / 24.0) + rng.normal(0.0, 0.8, periods)
    wind = np.clip(wind, 0.0, 22.0)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": np.round(np.clip(demand, 0.0, None), 3),
            "irradiance_w_m2": np.round(irradiance, 3),
            "ambient_temperature_c": np.round(temperature, 3),
            "wind_speed_m_s": np.round(wind, 3),
        }
    )


def main() -> None:
    """Write the example dataset to the repository data directory."""
    output = Path(__file__).resolve().parents[1] / "data" / "example_hourly.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_example_data().to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
