from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_system_simulator.simulation.engine import SimulationResult


def write_outputs(
    result: SimulationResult,
    output_directory: str | Path,
    *,
    create_plots: bool = True,
) -> None:
    """Write simulation time series, summary metrics, and optional plots."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result.timeseries.to_csv(output / "timeseries.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if create_plots:
        _plot_dispatch(result.timeseries, output / "dispatch.png")
        _plot_battery(result.timeseries, output / "battery_soc.png")


def _plot_dispatch(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 6))
    timestamp = pd.to_datetime(frame["timestamp"])
    axis.plot(timestamp, frame["end_user_demand_mw"], label="End-user demand", linewidth=1.8)
    axis.plot(timestamp, frame["renewable_used_mw"], label="Renewable used")
    axis.plot(timestamp, frame["thermal_output_mw"], label="Thermal output")
    axis.plot(timestamp, frame["imports_mw"], label="Imports")
    axis.set_xlabel("Time")
    axis.set_ylabel("Power (MW)")
    axis.set_title("Energy-system dispatch")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_battery(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 4.5))
    timestamp = pd.to_datetime(frame["timestamp"])
    axis.plot(timestamp, frame["battery_soc_mwh"], label="State of charge")
    axis.set_xlabel("Time")
    axis.set_ylabel("Energy (MWh)")
    axis.set_title("Battery state of charge")
    axis.grid(True, alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
