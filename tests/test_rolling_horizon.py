from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from energy_system_simulator.config import ModelConfig, RollingHorizonConfig, load_config
from energy_system_simulator.simulation import SimulationEngine


def _sliced_config(
    tmp_path: Path,
    config_name: str,
    data_name: str,
    periods: int,
    rolling: RollingHorizonConfig,
) -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / data_name).head(periods)
    input_path = tmp_path / f"{config_name}.csv"
    data.to_csv(input_path, index=False)
    config = load_config(root / "configs" / config_name)
    return replace(
        config,
        paths=replace(config.paths, input_csv=input_path, output_directory=tmp_path / "outputs"),
        rolling_horizon=rolling,
    )


def test_rolling_horizon_config_loads_from_yaml(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "portfolio_two_thermal.yaml").read_text())
    raw["paths"]["output_directory"] = str(tmp_path / "outputs")
    raw["rolling_horizon"] = {
        "enabled": True,
        "optimisation_window_periods": 6,
        "implementation_periods": 2,
        "lookahead_periods": 4,
        "terminal_treatment": "relaxed",
        "forecast_mode": "perfect_foresight",
        "checkpoint_directory": str(tmp_path / "checkpoints"),
        "resume_from_checkpoint": False,
        "compare_full_horizon": True,
    }
    config_path = tmp_path / "rolling.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.rolling_horizon.enabled is True
    assert config.rolling_horizon.optimisation_window_periods == 6
    assert config.rolling_horizon.terminal_treatment == "relaxed"
    assert config.rolling_horizon.checkpoint_directory == tmp_path / "checkpoints"


def test_rolling_window_covering_horizon_matches_full_horizon(tmp_path: Path) -> None:
    full_config = _sliced_config(
        tmp_path,
        "example.yaml",
        "example_hourly.csv",
        6,
        RollingHorizonConfig(),
    )
    rolling_config = replace(
        full_config,
        rolling_horizon=RollingHorizonConfig(
            enabled=True,
            optimisation_window_periods=6,
            implementation_periods=6,
            checkpoint_directory=tmp_path / "checkpoints",
        ),
    )

    full = SimulationEngine(full_config).run()
    rolling = SimulationEngine(rolling_config).run()

    columns = [
        "renewable_used_mw",
        "thermal_output_mw",
        "battery_soc_mwh",
        "served_demand_mw",
    ]
    pd.testing.assert_frame_equal(full.timeseries[columns], rolling.timeseries[columns])
    assert rolling.objective_eur == pytest.approx(full.objective_eur)
    assert rolling.summary["rolling_horizon"]["windows"][0]["implemented_periods"] == 6


def test_rolling_horizon_preserves_coverage_and_storage_state(tmp_path: Path) -> None:
    config = _sliced_config(
        tmp_path,
        "example.yaml",
        "example_hourly.csv",
        8,
        RollingHorizonConfig(
            enabled=True,
            optimisation_window_periods=4,
            implementation_periods=2,
            lookahead_periods=2,
            terminal_treatment="relaxed",
            checkpoint_directory=tmp_path / "checkpoints",
        ),
    )
    hydro_units = tuple(
        replace(
            unit,
            minimum_final_reservoir_mwh=0.0,
            terminal_reservoir_mode="free",
        )
        for unit in config.portfolio.hydro_units
    )
    config = replace(config, portfolio=replace(config.portfolio, hydro_units=hydro_units))

    result = SimulationEngine(config).run()
    windows = result.summary["rolling_horizon"]["windows"]

    assert len(result.timeseries) == 8
    assert not result.timeseries["timestamp"].duplicated().any()
    assert [window["implementation_end_period"] for window in windows] == [2, 4, 6, 8]
    assert windows[0]["transferred_state"]["storage"]["battery_1"][
        "initial_soc_mwh"
    ] == pytest.approx(result.timeseries["storage_soc_mwh__battery_1"].iloc[1])
    assert windows[0]["transferred_state"]["thermal"]["thermal_1"]["initial_on"] == bool(
        round(result.timeseries["thermal_on__thermal_1"].iloc[1])
    )


def test_rolling_resume_from_checkpoint_reproduces_run(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "checkpoints"
    config = _sliced_config(
        tmp_path,
        "example.yaml",
        "example_hourly.csv",
        8,
        RollingHorizonConfig(
            enabled=True,
            optimisation_window_periods=4,
            implementation_periods=2,
            lookahead_periods=2,
            terminal_treatment="relaxed",
            checkpoint_directory=checkpoint_directory,
        ),
    )
    uninterrupted = SimulationEngine(config).run()
    resume_config = replace(
        config,
        rolling_horizon=replace(config.rolling_horizon, resume_from_checkpoint=True),
    )

    resumed = SimulationEngine(resume_config).run()

    pd.testing.assert_frame_equal(uninterrupted.timeseries, resumed.timeseries)
    pd.testing.assert_frame_equal(uninterrupted.asset_timeseries, resumed.asset_timeseries)
    assert resumed.objective_eur == pytest.approx(uninterrupted.objective_eur)


def test_rolling_horizon_transfers_hydro_state(tmp_path: Path) -> None:
    config = _sliced_config(
        tmp_path,
        "portfolio_hydro.yaml",
        "hydro_seasonal_hourly.csv",
        8,
        RollingHorizonConfig(
            enabled=True,
            optimisation_window_periods=4,
            implementation_periods=2,
            lookahead_periods=2,
            terminal_treatment="relaxed",
            checkpoint_directory=tmp_path / "checkpoints",
        ),
    )

    engine = SimulationEngine(config)
    state = engine._initial_rolling_state()
    window_config = engine._rolling_window_config(state, window_start=0, is_final_window=False)
    frame = pd.DataFrame(
        {
            "thermal_on__gas-ccgt": [1.0, 1.0],
            "thermal_output_mw__gas-ccgt": [80.0, 90.0],
            "storage_soc_mwh__battery": [45.0, 42.0],
            "hydro_reservoir_mwh__alpine-reservoir": [520.0, 515.0],
            "hydro_reservoir_mwh__river-run": [0.0, 0.0],
        }
    )
    transferred = engine._rolling_state_after_segment(
        state,
        window_config,
        frame,
        window_start=0,
    )

    assert transferred["hydro"]["alpine-reservoir"]["initial_reservoir_mwh"] == pytest.approx(515.0)


def test_rolling_horizon_transfers_flexible_demand_state(tmp_path: Path) -> None:
    config = _sliced_config(
        tmp_path,
        "portfolio_demand_response.yaml",
        "demand_response_hourly.csv",
        8,
        RollingHorizonConfig(
            enabled=True,
            optimisation_window_periods=4,
            implementation_periods=2,
            lookahead_periods=2,
            terminal_treatment="relaxed",
            checkpoint_directory=tmp_path / "checkpoints",
        ),
    )

    result = SimulationEngine(config).run()
    first_state = result.summary["rolling_horizon"]["windows"][0]["transferred_state"]
    charged = (
        result.timeseries["demand_task_charge_mw__ev-fleet"].iloc[:2].sum()
        * config.simulation.time_step_hours
    )

    assert first_state["demand"]["ev-fleet"]["remaining_task_energy_mwh"] == pytest.approx(
        45.0 - charged
    )
