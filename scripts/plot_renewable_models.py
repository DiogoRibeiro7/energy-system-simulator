from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_system_simulator.config import RenewableGeneratorConfig, WindPowerCurvePointConfig
from energy_system_simulator.simulation.assets import RenewableAsset


def main() -> None:
    output_dir = Path("docs") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_solar_derating(output_dir / "solar_derating_decomposition.png")
    _plot_wind_curve(output_dir / "wind_power_curve.png")


def _plot_solar_derating(path: Path) -> None:
    asset = RenewableAsset.from_config(
        RenewableGeneratorConfig(
            id="example-solar",
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
    profile = asset.availability_profile(pd.DataFrame({"poa_w_m2": [1000.0], "ambient_c": [25.0]}))
    values = {
        "DC potential": profile.variables["dc_potential_mw"][0][0],
        "Temperature loss": -profile.variables["temperature_loss_mw"][0][0],
        "Clipping loss": -profile.variables["clipping_loss_mw"][0][0],
        "Other derating": -profile.variables["other_derating_loss_mw"][0][0],
        "Final AC availability": profile.available_mw[0],
    }

    figure, axis = plt.subplots(figsize=(9, 4.5))
    colors = ["#2f6f9f", "#c44e52", "#dd8452", "#8172b2", "#55a868"]
    axis.bar(list(values.keys()), list(values.values()), color=colors)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_ylabel("MW")
    axis.set_title("Solar derating decomposition")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_wind_curve(path: Path) -> None:
    curve = (
        WindPowerCurvePointConfig(wind_speed_m_s=0.0, power_mw=0.0),
        WindPowerCurvePointConfig(wind_speed_m_s=3.0, power_mw=0.0),
        WindPowerCurvePointConfig(wind_speed_m_s=8.0, power_mw=1.4),
        WindPowerCurvePointConfig(wind_speed_m_s=12.0, power_mw=3.0),
        WindPowerCurvePointConfig(wind_speed_m_s=25.0, power_mw=3.0),
    )
    speeds = np.linspace(0.0, 25.0, 100)
    output = np.interp(
        speeds,
        [point.wind_speed_m_s for point in curve],
        [point.power_mw for point in curve],
    )

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(speeds, output, color="#2f6f9f", linewidth=2.0)
    axis.scatter(
        [point.wind_speed_m_s for point in curve],
        [point.power_mw for point in curve],
        color="#c44e52",
        zorder=3,
    )
    axis.set_xlabel("Hub-height wind speed (m/s)")
    axis.set_ylabel("Per-turbine power (MW)")
    axis.set_title("Tabulated wind power curve interpolation")
    axis.grid(True, color="#dddddd", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
