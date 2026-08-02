from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import energy_system_simulator as ess
from energy_system_simulator.config import load_config
from energy_system_simulator.exceptions import ConfigurationError


def test_public_api_exports_supported_lifecycle_functions() -> None:
    assert callable(ess.load_model_config)
    assert callable(ess.validate_model_config)
    assert callable(ess.validate_data)
    assert callable(ess.build_model)
    assert callable(ess.solve)
    assert callable(ess.run_simulation)
    assert issubclass(ess.ConfigurationError, ess.EnergySystemError)
    assert issubclass(ess.DataValidationError, ess.EnergySystemError)
    assert issubclass(ess.OptimisationError, ess.EnergySystemError)


def test_python_api_builds_model_without_solving(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    config = replace(config, paths=replace(config.paths, output_directory=tmp_path / "outputs"))

    problem = ess.build_model(config)

    assert problem.statistics.continuous_variables > 0
    assert problem.statistics.linear_constraints > 0


def test_python_api_solves_with_in_memory_dataframe(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    config = replace(config, paths=replace(config.paths, output_directory=tmp_path / "outputs"))
    data = pd.read_csv(root / "data" / "example_hourly.csv").head(24)

    result = ess.solve(config, data=data)

    assert result.solver_status == "optimal"
    assert len(result.timeseries) == 24


def test_python_api_run_simulation_enforces_overwrite_protection(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    config = replace(config, paths=replace(config.paths, output_directory=tmp_path / "outputs"))
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="--overwrite or --resume"):
        ess.run_simulation(config, create_plots=False)
