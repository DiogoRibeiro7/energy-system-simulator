from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_system_simulator.config import load_config
from energy_system_simulator.reporting import (
    compare_output_directories,
    run_diagnostics,
    versioned_output_tables,
    write_outputs,
)
from energy_system_simulator.simulation import SimulationEngine


def test_write_outputs_includes_machine_readable_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "example.yaml"
    config = load_config(config_path)
    result = SimulationEngine(config).run()

    write_outputs(result, tmp_path, config=config, config_path=config_path, create_plots=False)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert (tmp_path / "asset_timeseries.csv").is_file()
    assert manifest["package_version"] == "1.0.0"
    assert len(manifest["input_file_sha256"]) == 64
    assert len(manifest["configuration_sha256"]) == 64
    assert manifest["solver"]["name"] == "scipy.optimize.milp"
    assert manifest["solver"]["status"] == "optimal"
    assert manifest["solver"]["backend_status"] == "optimal"
    assert manifest["solver"]["backend_status_code"] == 0
    assert manifest["formulation"]["integer_variables"] == 1680
    assert "integrality_max_deviation" in manifest["numerical_diagnostics"]
    assert (
        manifest["terminal_commitment"]["terminal_commitment_mode"]
        == "forbid_incomplete_transitions"
    )
    assert manifest["resolved_configuration"]["paths"]["input_csv"].endswith(
        "data\\example_hourly.csv"
    ) or manifest["resolved_configuration"]["paths"]["input_csv"].endswith(
        "data/example_hourly.csv"
    )


def test_write_outputs_creates_versioned_tables_dictionary_diagnostics_and_report(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "example.yaml"
    config = load_config(config_path)
    result = SimulationEngine(config).run()

    write_outputs(result, tmp_path, config=config, config_path=config_path, create_plots=False)

    expected = {
        "system_timeseries_v1.csv",
        "asset_timeseries_v1.csv",
        "bus_timeseries_v1.csv",
        "line_timeseries_v1.csv",
        "cost_components_v1.csv",
        "emissions_v1.csv",
        "reliability_events_v1.csv",
        "solver_diagnostics_v1.csv",
        "summary_metrics_v1.csv",
        "data_dictionary.csv",
        "diagnostics.json",
        "dashboard",
        "dashboard.html",
        "report.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["schema_version"] == 1
    assert diagnostics["status"] == "ok"
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Solver status: `optimal`" in report
    assert "dashboard/index.html" in report
    assert "dashboard.html" in report
    dashboard = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "Energy System Dashboard" in dashboard
    assert "system_timeseries_v1.csv" in dashboard
    assert (tmp_path / "dashboard" / "index.html").is_file()
    assert (tmp_path / "dashboard" / "styles.css").is_file()
    assert (tmp_path / "dashboard" / "app.js").is_file()
    assert (tmp_path / "dashboard" / "data.json").is_file()
    assert (tmp_path / "dashboard" / "data.js").is_file()


def test_versioned_tables_reproduce_core_summary_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    result = SimulationEngine(load_config(root / "configs" / "example.yaml")).run()

    tables = versioned_output_tables(result)
    costs = tables["cost_components_v1"]
    emissions = tables["emissions_v1"]

    assert costs["value_eur"].sum() == result.summary["objective_eur"]
    assert emissions["total_emissions_tonnes"].sum() == pytest.approx(
        result.summary["total_emissions_tonnes"]
    )
    assert set(tables["bus_timeseries_v1"].columns) == {
        "schema_version",
        "timestamp",
        "bus_id",
        "voltage_angle_rad",
        "balance_residual_mw",
    }
    assert set(tables["line_timeseries_v1"].columns) == {
        "schema_version",
        "timestamp",
        "line_id",
        "flow_mw",
        "abs_utilisation",
        "overload_residual_mw",
    }


def test_reporting_diagnostics_flag_modified_result() -> None:
    root = Path(__file__).resolve().parents[1]
    result = SimulationEngine(load_config(root / "configs" / "example.yaml")).run()
    result.timeseries.loc[0, "source_balance_residual_mw"] = 1.0
    result.summary["energy_reconciliation"]["max_abs_source_balance_residual_mw"] = 1.0
    result.timeseries.loc[0, "battery_charge_mw"] = 1.0
    result.timeseries.loc[0, "battery_discharge_mw"] = 1.0

    findings = run_diagnostics(result)

    assert {"balance_residuals", "simultaneous_incompatible_modes"} <= {
        finding.check for finding in findings
    }


def test_compare_output_directories_writes_markdown_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "example.yaml"
    config = load_config(config_path)
    result = SimulationEngine(config).run()
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_outputs(result, first, config=config, config_path=config_path, create_plots=False)
    write_outputs(result, second, config=config, config_path=config_path, create_plots=False)
    output = tmp_path / "comparison.md"

    table = compare_output_directories((first, second), output)

    assert not table.empty
    assert "Scenario Output Comparison" in output.read_text(encoding="utf-8")


def test_write_outputs_creates_expanded_plot_set(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "example.yaml"
    config = load_config(config_path)
    result = SimulationEngine(config).run()

    write_outputs(result, tmp_path, config=config, config_path=config_path, create_plots=True)

    assert {
        "dispatch.png",
        "battery_soc.png",
        "thermal_commitment.png",
        "storage_dispatch.png",
        "hydro_reservoir.png",
        "network.png",
        "reserves.png",
        "unserved_energy.png",
        "cost_breakdown.png",
        "emissions.png",
        "duration_curves.png",
    } <= {path.name for path in tmp_path.iterdir()}
