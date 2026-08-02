from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from energy_system_simulator.cli import main
from energy_system_simulator.config import load_config
from energy_system_simulator.exceptions import OptimisationError
from energy_system_simulator.scenarios import (
    ScenarioExperimentError,
    apply_overrides,
    resolve_scenarios,
    run_experiment_file,
    stable_scenario_id,
)


def _scenario_config(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "demand_mw": [10.0, 10.0],
            "irradiance_w_m2": [0.0, 0.0],
            "ambient_temperature_c": [15.0, 15.0],
            "wind_speed_m_s": [0.0, 0.0],
        }
    ).to_csv(input_path, index=False)
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        f"""
schema_version: 1

simulation:
  time_step_hours: 1.0
  solver_time_limit_seconds: 20.0
  mip_relative_gap: 0.001

solar:
  capacity_mw: 1.0
  performance_ratio: 0.86
  reference_irradiance_w_m2: 1000.0
  temperature_coefficient_per_c: -0.004
  nominal_operating_cell_temperature_c: 45.0

wind:
  capacity_mw: 1.0
  cut_in_speed_m_s: 3.0
  rated_speed_m_s: 12.0
  cut_out_speed_m_s: 25.0

thermal:
  name: Scenario unit
  minimum_output_mw: 0.0
  maximum_output_mw: 20.0
  ramp_up_mw_per_hour: 100.0
  ramp_down_mw_per_hour: 100.0
  startup_ramp_mw: 20.0
  shutdown_ramp_mw: 20.0
  variable_cost_eur_per_mwh: 10.0
  no_load_cost_eur_per_hour: 0.0
  startup_cost_eur: 0.0
  shutdown_cost_eur: 0.0
  emission_factor_tonnes_per_mwh: 0.5
  minimum_up_hours: 1.0
  minimum_down_hours: 1.0
  initial_on: true
  initial_output_mw: 0.0
  initial_up_time_hours: 2.0
  initial_down_time_hours: 0.0
  terminal_commitment_mode: carry_residual_obligations

battery:
  energy_capacity_mwh: 0.0
  power_capacity_mw: 0.0
  minimum_soc_mwh: 0.0
  maximum_soc_mwh: 0.0
  initial_soc_mwh: 0.0
  charge_efficiency: 1.0
  discharge_efficiency: 1.0
  throughput_cost_eur_per_mwh: 0.0
  minimum_final_soc_mwh: 0.0
  terminal_soc_mode: free

network:
  loss_fraction: 0.0
  transfer_capacity_mw: 100.0

imports:
  maximum_power_mw: 0.0
  price_eur_per_mwh: 100.0
  emission_factor_tonnes_per_mwh: 0.0

penalties:
  renewable_curtailment_eur_per_mwh: 1.0
  lost_load_eur_per_mwh: 1000.0
  carbon_price_eur_per_tonne: 0.0

paths:
  input_csv: {input_path.as_posix()}
  output_directory: {(tmp_path / "base-output").as_posix()}
""",
        encoding="utf-8",
    )
    return config_path


def _experiment_file(tmp_path: Path, *, output_name: str = "experiment") -> Path:
    config_path = _scenario_config(tmp_path)
    experiment_path = tmp_path / "experiment.yaml"
    payload = {
        "base_config": str(config_path),
        "output_directory": str(tmp_path / output_name),
        "workers": 1,
        "scenarios": [
            {
                "id": "base",
                "overrides": {"penalties.carbon_price_eur_per_tonne": 0.0},
            }
        ],
        "sweeps": [
            {
                "parameter": "thermal.variable_cost_eur_per_mwh",
                "values": [10.0, 20.0],
            }
        ],
    }
    experiment_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return experiment_path


def test_scenario_expansion_and_stable_ids_are_canonical() -> None:
    payload = {
        "scenarios": [{"id": "base", "overrides": {"b": 2, "a": 1}}],
        "sweeps": [{"parameter": "x", "values": [1, 2]}],
        "grid": {"parameters": {"y": [3, 4], "z": [5, 6]}},
    }

    scenarios = resolve_scenarios(payload)

    assert len(scenarios) == 7
    assert scenarios[0].id == stable_scenario_id({"a": 1, "b": 2})
    assert [scenario.order for scenario in scenarios] == list(range(7))


def test_overrides_validate_paths_without_mutating_base_config(tmp_path: Path) -> None:
    config = load_config(_scenario_config(tmp_path))

    changed = apply_overrides(config, {"thermal.variable_cost_eur_per_mwh": 25.0})

    assert config.thermal.variable_cost_eur_per_mwh == pytest.approx(10.0)
    assert changed.thermal.variable_cost_eur_per_mwh == pytest.approx(25.0)
    with pytest.raises(ScenarioExperimentError, match="Unknown override path"):
        apply_overrides(config, {"thermal.not_a_field": 1.0})


def test_run_experiment_writes_manifests_and_aggregate(tmp_path: Path) -> None:
    experiment_path = _experiment_file(tmp_path)

    aggregate = run_experiment_file(experiment_path, create_plots=False)

    output = tmp_path / "experiment"
    assert len(aggregate) == 3
    assert aggregate["ok"].tolist() == [True, True, True]
    assert (output / "summary.csv").is_file()
    assert (output / "experiment_manifest.json").is_file()
    scenario_id = aggregate["scenario_id"].iloc[0]
    manifest = json.loads(
        (output / scenario_id / "scenario_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scenario_id"] == scenario_id
    assert "param_penalties.carbon_price_eur_per_tonne" in aggregate.columns
    assert "cost_thermal_variable_cost_eur" in aggregate.columns
    assert "total_emissions_tonnes" in aggregate.columns


def test_resume_skips_verified_completed_scenarios(tmp_path: Path) -> None:
    experiment_path = _experiment_file(tmp_path)
    first = run_experiment_file(experiment_path, create_plots=False)

    with pytest.raises(ScenarioExperimentError, match="already exists"):
        run_experiment_file(experiment_path, create_plots=False)

    second = run_experiment_file(experiment_path, resume=True, create_plots=False)

    assert first["scenario_id"].tolist() == second["scenario_id"].tolist()
    assert second["resumed"].tolist() == [True, True, True]


def test_parallel_results_are_deterministic(tmp_path: Path) -> None:
    sequential_path = _experiment_file(tmp_path / "sequential", output_name="out")
    parallel_path = _experiment_file(tmp_path / "parallel", output_name="out")

    sequential = run_experiment_file(sequential_path, workers=1, create_plots=False)
    parallel = run_experiment_file(parallel_path, workers=2, create_plots=False)

    assert sequential["scenario_id"].tolist() == parallel["scenario_id"].tolist()
    assert sequential["objective_eur"].tolist() == pytest.approx(parallel["objective_eur"].tolist())


def test_failed_scenarios_are_aggregated_and_raise(tmp_path: Path) -> None:
    config_path = _scenario_config(tmp_path)
    experiment_path = tmp_path / "bad.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(config_path),
                "output_directory": str(tmp_path / "bad-output"),
                "scenarios": [{"id": "bad", "overrides": {"thermal.missing": 0.0}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(OptimisationError, match="scenarios failed"):
        run_experiment_file(experiment_path, create_plots=False)

    aggregate = pd.read_csv(tmp_path / "bad-output" / "summary.csv")
    assert aggregate["ok"].tolist() == [False]
    assert aggregate["error_type"].iloc[0] == "ScenarioExperimentError"


def test_cli_run_scenarios_writes_aggregate(tmp_path: Path, capsys) -> None:
    config_path = _scenario_config(tmp_path)
    experiment_path = tmp_path / "cli-experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(config_path),
                "output_directory": str(tmp_path / "cli-output"),
                "scenarios": [{"id": "base", "overrides": {}}],
            }
        ),
        encoding="utf-8",
    )

    main(["run-scenarios", "--experiment", str(experiment_path), "--no-plots"])

    captured = capsys.readouterr()
    assert "Scenario experiment complete: 1 scenarios" in captured.out
    assert (tmp_path / "cli-output" / "summary.csv").is_file()
