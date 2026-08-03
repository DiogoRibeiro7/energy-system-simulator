from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from energy_system_simulator.cli import main
from energy_system_simulator.experiments import (
    ExperimentError,
    run_research_experiment,
    verify_experiment_manifest,
)


def _write_study(tmp_path: Path, *, include_comparisons: bool = True) -> Path:
    study = tmp_path / "study"
    (study / "configs").mkdir(parents=True)
    (study / "data").mkdir()
    (study / "scenarios").mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "demand_mw": [10.0, 12.0],
            "irradiance_w_m2": [0.0, 0.0],
            "ambient_temperature_c": [15.0, 15.0],
            "wind_speed_m_s": [0.0, 0.0],
        }
    ).to_csv(study / "data" / "input.csv", index=False)
    (study / "configs" / "base.yaml").write_text(
        """
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
  name: Tiny unit
  minimum_output_mw: 0.0
  maximum_output_mw: 50.0
  ramp_up_mw_per_hour: 50.0
  ramp_down_mw_per_hour: 50.0
  startup_ramp_mw: 50.0
  shutdown_ramp_mw: 50.0
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
  input_csv: ../data/input.csv
  output_directory: ../outputs/base
""",
        encoding="utf-8",
    )
    (study / "scenarios" / "experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "base_config": "../configs/base.yaml",
                "output_directory": "../outputs",
                "scenarios": [
                    {"id": "base", "overrides": {}},
                    {
                        "id": "higher-cost",
                        "overrides": {"penalties.carbon_price_eur_per_tonne": 20.0},
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload: dict[str, object] = {
        "id": "tiny",
        "title": "Tiny Study",
        "research_question": "What changes when thermal cost doubles?",
        "model_version": "test",
        "data_version": "test",
        "seed": 7,
        "base_config": "configs/base.yaml",
        "scenario_file": "scenarios/experiment.yaml",
        "output_directory": "outputs",
        "hypotheses": ["Higher thermal cost increases objective value."],
        "metrics": [
            {
                "name": "objective_eur",
                "display_name": "Total system cost",
                "source_column": "objective_eur",
                "unit": "EUR",
            }
        ],
        "comparisons": [
            {
                "id": "cost_change",
                "baseline": "base",
                "scenario": "higher-cost",
                "metric": "objective_eur",
                "paired_seed": True,
            }
        ]
        if include_comparisons
        else [],
        "uncertainty_intervals": [
            {
                "method": "monte_carlo_95_percent_interval",
                "applies_when": "replication rows are present",
            }
        ],
        "sensitivity_ranges": [
            {
                "parameter": "penalties.carbon_price_eur_per_tonne",
                "metric": "objective_eur",
                "scenarios": ["base", "higher-cost"],
            }
        ],
        "figures": [
            {
                "id": "objective",
                "metric": "objective_eur",
                "filename": "objective.png",
                "caption": "Objective values generated from summary.csv.",
            }
        ],
        "model_can_identify": ["Within-model dispatch cost differences."],
        "model_cannot_identify": ["Historical causal effects."],
        "limitations": ["Synthetic two-hour fixture."],
    }
    (study / "study.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return study


def test_research_experiment_generates_manifest_tables_and_report(tmp_path: Path) -> None:
    study = _write_study(tmp_path)

    result = run_research_experiment(study, create_plots=False)

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["seed"] == 7
    assert manifest["paired_seed_groups"][0]["seed"] == 7
    assert manifest["solver"]["backend"] == "scipy.optimize.milp"
    assert len(manifest["files"]) == 5
    assert (
        (study / "tables" / "metrics.md")
        .read_text(encoding="utf-8")
        .startswith("| Scenario | Total system cost [EUR] |")
    )
    assert "\\begin{tabular}" in (study / "tables" / "metrics.tex").read_text(encoding="utf-8")
    figure_metadata = json.loads((study / "figures" / "figure_metadata.json").read_text())
    assert figure_metadata[0]["caption"] == "Objective values generated from summary.csv."
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "## What the Model Cannot Identify" in report
    assert "Historical causal effects" in report


def test_research_manifest_detects_changed_inputs(tmp_path: Path) -> None:
    study = _write_study(tmp_path)
    result = run_research_experiment(study, create_plots=False)
    manifest_path = Path(result["manifest"])

    verify_experiment_manifest(manifest_path)
    (study / "configs" / "base.yaml").write_text(
        (study / "configs" / "base.yaml").read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentError, match="checksum mismatch"):
        verify_experiment_manifest(manifest_path)


def test_research_experiment_requires_pre_specified_comparisons(tmp_path: Path) -> None:
    study = _write_study(tmp_path, include_comparisons=False)

    with pytest.raises(ExperimentError, match="pre-specify at least one comparison"):
        run_research_experiment(study, create_plots=False)


def test_cli_run_and_reproduce_research_experiment(tmp_path: Path, capsys) -> None:
    study = _write_study(tmp_path)

    main(["run-experiment", "--study", str(study), "--no-plots"])
    captured = capsys.readouterr()
    assert "Research experiment complete" in captured.out
    main(
        [
            "reproduce-experiment",
            "--manifest",
            str(study / "outputs" / "research_manifest.json"),
            "--overwrite",
            "--no-plots",
        ]
    )

    captured = capsys.readouterr()
    assert "Research experiment reproduced" in captured.out
