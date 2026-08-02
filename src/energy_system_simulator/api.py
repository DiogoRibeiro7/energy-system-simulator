from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from energy_system_simulator.config import ModelConfig, load_config, validate_config
from energy_system_simulator.data import load_input_data, validate_input_frame
from energy_system_simulator.dispatch import FormulationProblem
from energy_system_simulator.exceptions import ConfigurationError
from energy_system_simulator.reporting import write_outputs
from energy_system_simulator.scenarios import apply_overrides
from energy_system_simulator.simulation import SimulationEngine, SimulationResult


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a file-based model configuration."""
    return load_config(path)


def validate_model_config(config: ModelConfig) -> ModelConfig:
    """Validate a model configuration and return it unchanged."""
    validate_config(config)
    return config


def load_data(config: ModelConfig) -> pd.DataFrame:
    """Load and validate the input CSV referenced by a model configuration."""
    validate_config(config)
    return load_input_data(config.paths.input_csv, config.simulation.time_step_hours)


def validate_data(data: pd.DataFrame, *, time_step_hours: float) -> pd.DataFrame:
    """Validate an in-memory simulator input table and return a normalized copy."""
    return validate_input_frame(data, time_step_hours)


def build_model(config: ModelConfig, data: pd.DataFrame | None = None) -> FormulationProblem:
    """Build a dispatch formulation without solving it."""
    validate_config(config)
    return SimulationEngine(config).build_model(data)


def solve(config: ModelConfig, data: pd.DataFrame | None = None) -> SimulationResult:
    """Solve a model configuration, optionally using an in-memory input table."""
    validate_config(config)
    engine = SimulationEngine(config)
    if data is None:
        return engine.run()
    frame = validate_input_frame(data, config.simulation.time_step_hours)
    if config.rolling_horizon.enabled:
        return engine._run_rolling_horizon(frame)
    return engine._run_full_horizon_data(frame)


def run_simulation(
    config: ModelConfig | str | Path,
    *,
    data: pd.DataFrame | None = None,
    output_directory: str | Path | None = None,
    create_plots: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Solve a simulation and write standard outputs when an output directory is available."""
    if isinstance(config, str | Path):
        config_path = Path(config)
        resolved = load_config(config_path)
    else:
        config_path = None
        resolved = config
    if overrides:
        resolved = apply_overrides(resolved, dict(overrides))
    if output_directory is not None:
        resolved = replace(
            resolved, paths=replace(resolved.paths, output_directory=Path(output_directory))
        )
    validate_config(resolved)

    destination = (
        Path(output_directory) if output_directory is not None else resolved.paths.output_directory
    )
    ensure_writable_output_directory(destination, overwrite=overwrite, resume=resume)
    result = solve(resolved, data)
    write_outputs(
        result,
        destination,
        config=resolved,
        config_path=config_path,
        create_plots=create_plots,
        command_line_overrides=overrides,
    )
    return result


def ensure_writable_output_directory(
    path: str | Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    """Validate output overwrite intent and create the directory when allowed."""
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()) and not (overwrite or resume):
        raise ConfigurationError(
            f"Output directory already exists and is not empty: {destination}. "
            "Use --overwrite or --resume to continue."
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination
