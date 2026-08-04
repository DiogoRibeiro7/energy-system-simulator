from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from energy_system_simulator.cli import main
from energy_system_simulator.exceptions import ConfigurationError
from energy_system_simulator.heat import (
    CHPUnitConfig,
    CHPVertex,
    HeatSystemProblem,
    load_heat_problem,
    run_heat_study,
)


def test_heat_study_balances_and_reports_coupling() -> None:
    problem = load_heat_problem("configs/chp_heat_system.yaml")

    result = run_heat_study(problem)

    assert result.summary["unmet_heat_mwh_th"] == pytest.approx(0.0)
    assert result.summary["electricity_shortage_mwh"] == pytest.approx(0.0)
    assert result.summary["heat_balance_max_abs_residual_mw"] <= 1e-9
    assert result.summary["electricity_balance_max_abs_residual_mw"] <= 1e-9
    assert result.summary["storage_balance_max_abs_residual_mwh"] <= 1e-9
    assert result.summary["electricity_from_chp_mwh"] > 0.0
    assert result.summary["heat_by_source_mwh_th"]["chp_mwh_th"] > 0.0
    assert result.summary["total_fuel_mwh"] == pytest.approx(
        sum(result.summary["fuel_use_mwh"].values())
    )
    assert result.summary["total_emissions_tonnes"] == pytest.approx(
        (result.timeseries["total_fuel_mwh"] * 0.20).sum()
    )


def test_chp_polytope_prevents_independent_max_heat_and_power() -> None:
    problem = HeatSystemProblem(
        periods=("peak",),
        heat_demand_mw_th=[10.0],
        electricity_demand_mw=[10.0],
        electricity_purchase_price_eur_per_mwh=[2000.0],
        heat_shortage_penalty_eur_per_mwh=1000.0,
        electricity_shortage_penalty_eur_per_mwh=1000.0,
        chp_units=(
            CHPUnitConfig(
                id="tradeoff",
                vertices=(
                    CHPVertex(
                        id="off",
                        electric_output_mw=0.0,
                        heat_output_mw_th=0.0,
                        fuel_input_mwh_per_hour=0.0,
                    ),
                    CHPVertex(
                        id="power",
                        electric_output_mw=10.0,
                        heat_output_mw_th=0.0,
                        fuel_input_mwh_per_hour=20.0,
                    ),
                    CHPVertex(
                        id="heat",
                        electric_output_mw=0.0,
                        heat_output_mw_th=10.0,
                        fuel_input_mwh_per_hour=20.0,
                    ),
                ),
            ),
        ),
    )

    result = run_heat_study(problem)
    produced = result.timeseries.loc[0, "chp_power_mw"] + result.timeseries.loc[0, "chp_heat_mw_th"]

    assert produced <= 10.0 + 1e-9
    assert result.summary["unmet_heat_mwh_th"] + result.summary["electricity_shortage_mwh"] > 0.0


def test_chp_vertex_efficiency_above_one_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/chp_heat_system.yaml").read_text(encoding="utf-8"))
    raw["chp_units"][0]["vertices"][1]["fuel_input_mwh_per_hour"] = 3.0
    problem_path = tmp_path / "bad_heat.yaml"
    problem_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"total efficiency above 1\.0"):
        load_heat_problem(problem_path)


def test_heat_cli_writes_outputs(tmp_path: Path) -> None:
    output = tmp_path / "heat"

    main(
        [
            "heat-study",
            "--problem",
            "configs/chp_heat_system.yaml",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "heat_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["heat_balance_max_abs_residual_mw"] <= 1e-9
    assert (output / "heat_timeseries.csv").exists()
