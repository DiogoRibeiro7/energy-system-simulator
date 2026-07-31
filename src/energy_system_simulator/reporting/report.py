from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_system_simulator.config import ModelConfig
from energy_system_simulator.simulation.engine import SimulationResult


def write_outputs(
    result: SimulationResult,
    output_directory: str | Path,
    *,
    config: ModelConfig | None = None,
    config_path: str | Path | None = None,
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
    if config is not None:
        _write_manifest(result, output / "manifest.json", config, config_path)
    if create_plots:
        _plot_dispatch(result.timeseries, output / "dispatch.png")
        _plot_battery(result.timeseries, output / "battery_soc.png")


def _write_manifest(
    result: SimulationResult,
    path: Path,
    config: ModelConfig,
    config_path: str | Path | None,
) -> None:
    manifest = {
        "package_version": _package_version(),
        "python_version": platform.python_version(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "input_file": str(config.paths.input_csv),
        "input_file_sha256": _sha256(config.paths.input_csv),
        "configuration_file": str(Path(config_path).resolve()) if config_path else None,
        "configuration_sha256": _sha256(Path(config_path)) if config_path else None,
        "solver": {
            "name": "scipy.optimize.milp",
            "time_limit_seconds": config.simulation.solver_time_limit_seconds,
            "mip_relative_gap": config.simulation.mip_relative_gap,
            "termination_message": result.solver_message,
            "reported_mip_gap": result.mip_gap,
        },
        "formulation": asdict(result.formulation_statistics),
        "resolved_configuration": _json_ready(asdict(config)),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _package_version() -> str:
    try:
        return version("energy-system-simulator")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


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
