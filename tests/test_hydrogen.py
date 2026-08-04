from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from energy_system_simulator.cli import main
from energy_system_simulator.exceptions import ConfigurationError
from energy_system_simulator.hydrogen import (
    ElectrolyserConfig,
    HydrogenDemandConfig,
    HydrogenReconverterConfig,
    HydrogenStorageConfig,
    HydrogenSystemProblem,
    hydrogen_kg_to_mwh_lhv,
    hydrogen_mwh_lhv_to_kg,
    load_hydrogen_problem,
    run_hydrogen_study,
)


def test_hydrogen_surplus_can_be_stored_and_reconverted() -> None:
    problem = HydrogenSystemProblem(
        periods=("surplus_1", "surplus_2", "deficit_1", "deficit_2"),
        renewable_surplus_mw=[10.0, 10.0, 0.0, 0.0],
        electricity_deficit_mw=[0.0, 0.0, 3.0, 3.0],
        electrolyser=ElectrolyserConfig(
            id="electrolyser",
            power_capacity_mw=10.0,
            efficiency_mwh_h2_per_mwh_electric=0.7,
            variable_cost_eur_per_mwh_electric=1.0,
        ),
        storage=HydrogenStorageConfig(
            id="cavern",
            energy_capacity_mwh_lhv=20.0,
            charge_capacity_mwh_per_hour=20.0,
            discharge_capacity_mwh_per_hour=20.0,
        ),
        demand=HydrogenDemandConfig(
            id="industry",
            demand_mwh_lhv=[0.0, 2.0, 0.0, 0.0],
        ),
        reconverter=HydrogenReconverterConfig(
            id="fuel-cell",
            power_capacity_mw=3.0,
            efficiency_mwh_electric_per_mwh_h2=0.5,
        ),
    )

    result = run_hydrogen_study(problem)

    assert result.summary["hydrogen_produced_mwh_lhv"] == pytest.approx(14.0)
    assert result.summary["hydrogen_delivered_mwh_lhv"] == pytest.approx(2.0)
    assert result.summary["reconverted_electricity_mwh"] == pytest.approx(6.0)
    assert result.summary["unserved_electric_deficit_mwh"] == pytest.approx(0.0)
    assert result.summary["ending_inventory_mwh_lhv"] == pytest.approx(0.0)
    assert result.summary["round_trip_efficiency"] == pytest.approx(0.3)
    assert result.summary["hydrogen_balance_max_abs_residual_mwh"] <= 1e-9
    assert result.timeseries["hydrogen_carrier_balance_residual_mwh_lhv"].abs().max() <= 1e-9


def test_hydrogen_shortage_reports_when_storage_is_insufficient() -> None:
    problem = HydrogenSystemProblem(
        periods=("only_period",),
        renewable_surplus_mw=[4.0],
        electricity_deficit_mw=[0.0],
        electrolyser=ElectrolyserConfig(
            id="electrolyser",
            power_capacity_mw=4.0,
            efficiency_mwh_h2_per_mwh_electric=0.5,
        ),
        demand=HydrogenDemandConfig(id="industry", demand_mwh_lhv=[5.0]),
    )

    result = run_hydrogen_study(problem)

    assert result.summary["hydrogen_delivered_mwh_lhv"] == pytest.approx(2.0)
    assert result.summary["hydrogen_shortage_mwh_lhv"] == pytest.approx(3.0)


def test_hydrogen_efficiency_above_one_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/hydrogen_system.yaml").read_text(encoding="utf-8"))
    raw["electrolyser"]["efficiency_mwh_h2_per_mwh_electric"] = 1.1
    problem_path = tmp_path / "bad_hydrogen.yaml"
    problem_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="efficiency cannot exceed"):
        load_hydrogen_problem(problem_path)


def test_hydrogen_lhv_conversion_helpers_are_reversible() -> None:
    assert hydrogen_kg_to_mwh_lhv(1000.0) == pytest.approx(33.33)
    assert hydrogen_mwh_lhv_to_kg(33.33) == pytest.approx(1000.0)
    with pytest.raises(ConfigurationError, match="non-negative"):
        hydrogen_kg_to_mwh_lhv(-1.0)


def test_hydrogen_cli_writes_outputs(tmp_path: Path) -> None:
    output = tmp_path / "hydrogen"

    main(
        [
            "hydrogen-study",
            "--problem",
            "configs/hydrogen_system.yaml",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "hydrogen_summary.json").read_text(encoding="utf-8"))
    assert summary["canonical_hydrogen_unit"] == "MWh_LHV"
    assert summary["reconverted_electricity_mwh"] == pytest.approx(6.0)
    assert (output / "hydrogen_timeseries.csv").exists()
