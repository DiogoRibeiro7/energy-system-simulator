from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_system_simulator.config import ModelConfig, resolved_config_to_dict
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.metadata import get_package_version
from energy_system_simulator.reporting.dashboard import write_dashboard
from energy_system_simulator.simulation.engine import SimulationResult

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DiagnosticFinding:
    """One automatic reporting diagnostic finding."""

    check: str
    severity: str
    message: str
    metric: str
    value: float | str
    threshold: float | str


def write_outputs(
    result: SimulationResult,
    output_directory: str | Path,
    *,
    config: ModelConfig | None = None,
    config_path: str | Path | None = None,
    create_plots: bool = True,
    command_line_overrides: Mapping[str, Any] | None = None,
) -> None:
    """Write simulation time series, summary metrics, and optional plots."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result.timeseries.to_csv(output / "timeseries.csv", index=False)
    result.asset_timeseries.to_csv(output / "asset_timeseries.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tables = versioned_output_tables(result)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    data_dictionary(tables).to_csv(output / "data_dictionary.csv", index=False)
    diagnostics = run_diagnostics(result)
    (output / "diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "findings": [asdict(finding) for finding in diagnostics],
                "status": "ok"
                if all(finding.severity != "error" for finding in diagnostics)
                else "error",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if config is not None:
        _write_manifest(
            result,
            output / "manifest.json",
            config,
            config_path,
            command_line_overrides=command_line_overrides,
        )
    if create_plots:
        _write_plots(result, output)
    write_dashboard(output)
    _write_markdown_report(result, output, diagnostics, create_plots=create_plots)


def versioned_output_tables(result: SimulationResult) -> dict[str, pd.DataFrame]:
    """Return stable, versioned output tables derived from a simulation result."""
    frame = result.timeseries.copy()
    asset = result.asset_timeseries.copy()
    return {
        "system_timeseries_v1": frame,
        "asset_timeseries_v1": asset,
        "bus_timeseries_v1": _bus_timeseries(frame),
        "line_timeseries_v1": _line_timeseries(frame),
        "cost_components_v1": _cost_components(result.summary),
        "emissions_v1": _emissions_table(frame),
        "reliability_events_v1": _reliability_events(frame),
        "solver_diagnostics_v1": _solver_diagnostics(result),
        "summary_metrics_v1": _summary_metrics(result.summary),
    }


def run_diagnostics(result: SimulationResult) -> list[DiagnosticFinding]:
    """Run automatic output diagnostics over a completed result."""
    findings: list[DiagnosticFinding] = []
    frame = result.timeseries
    policy = DEFAULT_NUMERICAL_POLICY
    reconciliation = result.summary.get("energy_reconciliation", {})
    if isinstance(reconciliation, Mapping):
        for metric in (
            "max_abs_source_balance_residual_mw",
            "max_abs_delivered_demand_residual_mw",
            "max_abs_battery_energy_residual_mwh",
            "max_abs_curtailment_residual_mw",
        ):
            value = float(reconciliation.get(metric, 0.0))
            threshold = (
                policy.energy_reconciliation_mwh
                if metric.endswith("_mwh")
                else policy.primal_feasibility_mw
            )
            if value > threshold:
                findings.append(
                    DiagnosticFinding(
                        "balance_residuals",
                        "error",
                        f"{metric} exceeds the numerical policy",
                        metric,
                        value,
                        threshold,
                    )
                )
    lower_bound_violation = _minimum_nonnegative_value(frame)
    if lower_bound_violation < -policy.nonnegative_cleanup:
        findings.append(
            DiagnosticFinding(
                "bound_violations",
                "error",
                "A non-negative reported column contains a negative value",
                "minimum_nonnegative_column_value",
                lower_bound_violation,
                -policy.nonnegative_cleanup,
            )
        )
    if {"battery_charge_mw", "battery_discharge_mw"} <= set(frame.columns):
        simultaneous = float((frame["battery_charge_mw"] * frame["battery_discharge_mw"]).max())
        if simultaneous > policy.primal_feasibility_mw:
            findings.append(
                DiagnosticFinding(
                    "simultaneous_incompatible_modes",
                    "error",
                    "Battery charge and discharge are both active in at least one period",
                    "max_charge_discharge_product",
                    simultaneous,
                    policy.primal_feasibility_mw,
                )
            )
    unserved = float(result.summary.get("unserved_energy_mwh", 0.0))
    if unserved > policy.energy_reconciliation_mwh:
        findings.append(
            DiagnosticFinding(
                "suspicious_load_shedding",
                "warning",
                "Simulation contains unserved energy",
                "unserved_energy_mwh",
                unserved,
                policy.energy_reconciliation_mwh,
            )
        )
    terminal = result.summary.get("terminal_commitment", {})
    if isinstance(terminal, Mapping) and terminal.get("terminal_commitment_mode") == (
        "forbid_incomplete_transitions"
    ):
        residual = max(
            float(terminal.get("residual_minimum_up_hours", 0.0)),
            float(terminal.get("residual_minimum_down_hours", 0.0)),
        )
        if residual > policy.report_rounding:
            findings.append(
                DiagnosticFinding(
                    "terminal_state",
                    "error",
                    "Strict terminal mode ended with residual commitment obligations",
                    "terminal_residual_hours",
                    residual,
                    policy.report_rounding,
                )
            )
    if result.solver_status != "optimal":
        findings.append(
            DiagnosticFinding(
                "solver_status",
                "error",
                "Solver did not report an optimal solution",
                "solver_status",
                result.solver_status,
                "optimal",
            )
        )
    missing_periods = _missing_period_count(
        frame, float(result.summary.get("time_step_hours", 1.0))
    )
    if missing_periods:
        findings.append(
            DiagnosticFinding(
                "missing_time_periods",
                "error",
                "Timestamp spacing is not contiguous",
                "missing_period_count",
                float(missing_periods),
                0.0,
            )
        )
    if {"renewable_used_mw", "renewable_available_mw"} <= set(frame.columns):
        maximum_overuse = float(
            (frame["renewable_used_mw"] - frame["renewable_available_mw"]).max()
        )
        if maximum_overuse > policy.primal_feasibility_mw:
            findings.append(
                DiagnosticFinding(
                    "asset_availability",
                    "error",
                    "Renewable use exceeds reported renewable availability",
                    "max_renewable_overuse_mw",
                    maximum_overuse,
                    policy.primal_feasibility_mw,
                )
            )
    return findings


def compare_output_directories(
    output_directories: Sequence[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """Write a Markdown comparison report for two or more simulation output directories."""
    if len(output_directories) < 2:
        raise ValueError("At least two output directories are required for comparison")
    rows = [_summary_for_comparison(Path(path)) for path in output_directories]
    baseline = rows[0]
    records: list[dict[str, float | str]] = []
    metric_names = sorted(set().union(*(set(row["metrics"]) for row in rows)))
    for metric in metric_names:
        baseline_value = baseline["metrics"].get(metric)
        if baseline_value is None:
            continue
        for row in rows[1:]:
            value = row["metrics"].get(metric)
            if value is None:
                continue
            absolute = value - baseline_value
            percent = absolute / baseline_value * 100.0 if baseline_value != 0.0 else np.nan
            records.append(
                {
                    "baseline": baseline["label"],
                    "comparison": row["label"],
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "comparison_value": value,
                    "absolute_difference": absolute,
                    "percent_difference": percent,
                }
            )
    table = pd.DataFrame.from_records(records)
    markdown = _comparison_markdown(table)
    Path(output_path).write_text(markdown, encoding="utf-8")
    return table


def _write_manifest(
    result: SimulationResult,
    path: Path,
    config: ModelConfig,
    config_path: str | Path | None,
    *,
    command_line_overrides: Mapping[str, Any] | None = None,
) -> None:
    manifest = {
        "package_version": get_package_version(),
        "python_version": platform.python_version(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "input_file": str(config.paths.input_csv),
        "input_file_sha256": _sha256(config.paths.input_csv),
        "configuration_file": str(Path(config_path).resolve()) if config_path else None,
        "configuration_sha256": _sha256(Path(config_path)) if config_path else None,
        "command_line_overrides": dict(command_line_overrides or {}),
        "solver": {
            "name": "scipy.optimize.milp",
            "time_limit_seconds": config.simulation.solver_time_limit_seconds,
            "mip_relative_gap": config.simulation.mip_relative_gap,
            "allow_non_optimal_solution": config.simulation.allow_non_optimal_solution,
            "status": result.solver_status,
            "backend_status": result.backend_solver_status,
            "backend_status_code": result.backend_solver_status_code,
            "termination_message": result.solver_message,
            "reported_mip_gap": result.mip_gap,
            "objective_bound_eur": result.objective_bound_eur,
            "absolute_gap_eur": result.absolute_gap_eur,
            "relative_gap": result.relative_gap,
            "runtime_seconds": result.solver_runtime_seconds,
            "node_count": result.solver_node_count,
        },
        "formulation": asdict(result.formulation_statistics),
        "terminal_commitment": asdict(result.terminal_commitment_state),
        "terminal_commitment_by_unit": {
            unit_id: asdict(state) for unit_id, state in result.terminal_commitment_by_unit.items()
        },
        "numerical_diagnostics": result.numerical_diagnostics,
        "resolved_configuration": resolved_config_to_dict(config),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


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


def _bus_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    residual_series = (
        frame["bus_balance_residual_mw"]
        if "bus_balance_residual_mw" in frame
        else pd.Series(np.zeros(len(frame), dtype=np.float64))
    )
    for column in frame.columns:
        if not column.startswith("bus_voltage_angle_rad__"):
            continue
        bus_id = column.removeprefix("bus_voltage_angle_rad__")
        for period, value in enumerate(frame[column]):
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "timestamp": frame["timestamp"].iloc[period],
                    "bus_id": bus_id,
                    "voltage_angle_rad": float(value),
                    "balance_residual_mw": float(residual_series.iloc[period]),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "schema_version",
            "timestamp",
            "bus_id",
            "voltage_angle_rad",
            "balance_residual_mw",
        ],
    )


def _line_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for column in frame.columns:
        if not column.startswith("line_flow_mw__"):
            continue
        line_id = column.removeprefix("line_flow_mw__")
        utilisation_column = f"line_abs_utilisation__{line_id}"
        overload_column = f"line_overload_residual_mw__{line_id}"
        for period, value in enumerate(frame[column]):
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "timestamp": frame["timestamp"].iloc[period],
                    "line_id": line_id,
                    "flow_mw": float(value),
                    "abs_utilisation": float(frame[utilisation_column].iloc[period])
                    if utilisation_column in frame
                    else 0.0,
                    "overload_residual_mw": float(frame[overload_column].iloc[period])
                    if overload_column in frame
                    else 0.0,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "schema_version",
            "timestamp",
            "line_id",
            "flow_mw",
            "abs_utilisation",
            "overload_residual_mw",
        ],
    )


def _cost_components(summary: Mapping[str, Any]) -> pd.DataFrame:
    costs = summary.get("cost_components_eur", {})
    if not isinstance(costs, Mapping):
        costs = {}
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "component": str(component),
                "value_eur": float(value),
                "aggregation": "sum",
            }
            for component, value in sorted(costs.items())
        ],
        columns=["schema_version", "component", "value_eur", "aggregation"],
    )


def _emissions_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        thermal = float(row.get("thermal_emissions_tonnes", 0.0))
        imports = float(row.get("import_emissions_tonnes", 0.0))
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp": row["timestamp"],
                "thermal_emissions_tonnes": thermal,
                "import_emissions_tonnes": imports,
                "total_emissions_tonnes": thermal + imports,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "schema_version",
            "timestamp",
            "thermal_emissions_tonnes",
            "import_emissions_tonnes",
            "total_emissions_tonnes",
        ],
    )


def _reliability_events(frame: pd.DataFrame) -> pd.DataFrame:
    if "total_load_shed_mw" not in frame:
        events = pd.DataFrame()
    else:
        event_rows = frame[
            frame["total_load_shed_mw"] > DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw
        ]
        events = pd.DataFrame(
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp": event_rows["timestamp"],
                "event_type": "unserved_energy",
                "unserved_load_mw": event_rows["total_load_shed_mw"],
            }
        )
    return events.reindex(columns=["schema_version", "timestamp", "event_type", "unserved_load_mw"])


def _solver_diagnostics(result: SimulationResult) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "solver_status": result.solver_status,
                "backend_status": result.backend_solver_status,
                "backend_status_code": result.backend_solver_status_code,
                "mip_gap": result.mip_gap,
                "objective_eur": result.objective_eur,
                "objective_bound_eur": result.objective_bound_eur,
                "absolute_gap_eur": result.absolute_gap_eur,
                "relative_gap": result.relative_gap,
                "runtime_seconds": result.solver_runtime_seconds,
                "node_count": result.solver_node_count,
                "message": result.solver_message,
            }
        ]
    )


def _summary_metrics(summary: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for metric, value in _flatten_scalars(summary).items():
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "metric": metric,
                "value": value,
                "unit": _unit_for_name(metric),
                "aggregation": _aggregation_for_name(metric),
            }
        )
    return pd.DataFrame.from_records(
        sorted(records, key=lambda row: str(row["metric"])),
        columns=["schema_version", "metric", "value", "unit", "aggregation"],
    )


def data_dictionary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a data dictionary row for every generated output column."""
    rows: list[dict[str, str | int]] = []
    for table_name, table in sorted(tables.items()):
        for column in table.columns:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "table": table_name,
                    "column": column,
                    "unit": _unit_for_name(column),
                    "sign_convention": _sign_convention_for_name(column),
                    "aggregation_rule": _aggregation_for_name(column),
                    "description": _description_for_name(column),
                }
            )
    return pd.DataFrame.from_records(rows)


def _write_plots(result: SimulationResult, output: Path) -> None:
    frame = result.timeseries
    _plot_dispatch(frame, output / "dispatch.png")
    _plot_battery(frame, output / "battery_soc.png")
    _plot_thermal(frame, output / "thermal_commitment.png")
    _plot_storage(frame, output / "storage_dispatch.png")
    _plot_hydro(frame, output / "hydro_reservoir.png")
    _plot_network(frame, output / "network.png")
    _plot_reserves(frame, output / "reserves.png")
    _plot_unserved_energy(frame, output / "unserved_energy.png")
    _plot_cost_breakdown(result.summary, output / "cost_breakdown.png")
    _plot_emissions(frame, output / "emissions.png")
    _plot_duration_curves(frame, output / "duration_curves.png")


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


def _plot_thermal(frame: pd.DataFrame, path: Path) -> None:
    output_columns = _limited_columns(frame, "thermal_output_mw__")
    startup_columns = _limited_columns(frame, "thermal_on__")
    _plot_multi_axis(
        frame,
        path,
        output_columns,
        "Thermal output by unit",
        "MW",
        secondary_columns=startup_columns,
        secondary_label="Commitment",
    )


def _plot_storage(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        column
        for column in ("battery_charge_mw", "battery_discharge_mw", "battery_soc_mwh")
        if column in frame
    ]
    _plot_multi_axis(frame, path, columns, "Storage dispatch and state", "MW / MWh")


def _plot_hydro(frame: pd.DataFrame, path: Path) -> None:
    columns = _limited_columns(frame, "hydro_reservoir_mwh__") + _limited_columns(
        frame, "hydro_spill_mw__"
    )
    _plot_multi_axis(frame, path, columns, "Hydro reservoir and spill", "MWh / MW")


def _plot_network(frame: pd.DataFrame, path: Path) -> None:
    columns = _limited_columns(frame, "line_flow_mw__") + _limited_columns(
        frame, "line_abs_utilisation__"
    )
    _plot_multi_axis(frame, path, columns, "Bus and line diagnostics", "MW / pu")


def _plot_reserves(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        column
        for column in (
            "reserve_upward_requirement_mw",
            "reserve_upward_procured_mw",
            "reserve_upward_shortfall_mw",
            "reserve_downward_requirement_mw",
            "reserve_downward_procured_mw",
            "reserve_downward_shortfall_mw",
        )
        if column in frame
    ]
    _plot_multi_axis(frame, path, columns, "Reserve requirement and provision", "MW")


def _plot_unserved_energy(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        column
        for column in ("total_load_shed_mw", "network_capacity_shed_mw", "dispatch_load_shed_mw")
        if column in frame
    ]
    _plot_multi_axis(frame, path, columns, "Unserved energy events", "MW")


def _plot_cost_breakdown(summary: Mapping[str, Any], path: Path) -> None:
    costs = summary.get("cost_components_eur", {})
    if not isinstance(costs, Mapping) or not costs:
        _plot_empty(path, "Cost decomposition")
        return
    series = pd.Series({str(key): float(value) for key, value in costs.items() if value != 0.0})
    if series.empty:
        _plot_empty(path, "Cost decomposition")
        return
    series = _top_series(series, limit=10)
    figure, axis = plt.subplots(figsize=(11, 5))
    series.sort_values().plot(kind="barh", ax=axis)
    axis.set_xlabel("EUR")
    axis.set_title("Cost decomposition")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_emissions(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        column
        for column in ("thermal_emissions_tonnes", "import_emissions_tonnes")
        if column in frame
    ]
    _plot_multi_axis(frame, path, columns, "Emissions", "tonnes")


def _plot_duration_curves(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        column
        for column in ("end_user_demand_mw", "renewable_used_mw", "thermal_output_mw", "imports_mw")
        if column in frame
    ]
    if not columns:
        _plot_empty(path, "Duration curves")
        return
    figure, axis = plt.subplots(figsize=(11, 5))
    for column in columns:
        sorted_values = np.sort(frame[column].to_numpy(dtype=np.float64))[::-1]
        axis.plot(np.arange(1, len(sorted_values) + 1), sorted_values, label=column)
    axis.set_xlabel("Ranked period")
    axis.set_ylabel("MW")
    axis.set_title("Duration curves")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_multi_axis(
    frame: pd.DataFrame,
    path: Path,
    columns: Sequence[str],
    title: str,
    ylabel: str,
    *,
    secondary_columns: Sequence[str] = (),
    secondary_label: str = "",
) -> None:
    if not columns and not secondary_columns:
        _plot_empty(path, title)
        return
    figure, axis = plt.subplots(figsize=(13, 5))
    timestamp = pd.to_datetime(frame["timestamp"])
    for column in columns:
        axis.plot(timestamp, frame[column], label=_short_label(column), linewidth=1.2)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if secondary_columns:
        secondary = axis.twinx()
        for column in secondary_columns:
            secondary.step(
                timestamp,
                frame[column],
                label=_short_label(column),
                linewidth=1.0,
                alpha=0.45,
            )
        secondary.set_ylabel(secondary_label)
        handles, labels = axis.get_legend_handles_labels()
        handles_2, labels_2 = secondary.get_legend_handles_labels()
        axis.legend(handles + handles_2, labels + labels_2, fontsize="small", ncol=2)
    else:
        axis.legend(fontsize="small", ncol=2)
    axis.grid(True, alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_empty(path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 3))
    axis.set_title(title)
    axis.text(0.5, 0.5, "No data", ha="center", va="center")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_markdown_report(
    result: SimulationResult,
    output: Path,
    diagnostics: Sequence[DiagnosticFinding],
    *,
    create_plots: bool,
) -> None:
    plot_names = [
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
    ]
    reconciliation = result.summary.get("energy_reconciliation", {})
    balance_residual = (
        float(reconciliation.get("max_abs_source_balance_residual_mw", 0.0))
        if isinstance(reconciliation, Mapping)
        else 0.0
    )
    lines = [
        "# Energy System Simulation Report",
        "",
        f"- Solver status: `{result.solver_status}`",
        f"- Backend status: `{result.backend_solver_status}`",
        f"- Objective: EUR {result.objective_eur:,.2f}",
        f"- Unserved energy: {float(result.summary.get('unserved_energy_mwh', 0.0)):,.3f} MWh",
        f"- Balance max residual: {balance_residual:g} MW",
        "",
        "## Tables",
        "",
    ]
    for table_name in versioned_output_tables(result):
        lines.append(f"- `{table_name}.csv`")
    lines.extend(["- `data_dictionary.csv`", "- `diagnostics.json`", "", "## Diagnostics", ""])
    if diagnostics:
        for finding in diagnostics:
            lines.append(
                f"- `{finding.severity}` `{finding.check}`: {finding.message} "
                f"({finding.metric}={finding.value}, threshold={finding.threshold})"
            )
    else:
        lines.append("- No diagnostic findings.")
    lines.extend(["", "## Dashboard", "", "- [`dashboard.html`](dashboard.html)"])
    if create_plots:
        lines.extend(["", "## Plots", ""])
        for name in plot_names:
            lines.append(f"![{name}]({name})")
    output.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flatten_scalars(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float | int | str | bool]:
    result: dict[str, float | int | str | bool] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten_scalars(value, prefix=name))
        elif isinstance(value, str | int | float | bool) or value is None:
            result[name] = "" if value is None else value
    return result


def _minimum_nonnegative_value(frame: pd.DataFrame) -> float:
    columns = [
        column
        for column in frame.columns
        if column.endswith(("_mw", "_mwh", "_tonnes", "_kg"))
        and "residual" not in column
        and "angle" not in column
        and "change" not in column
        and "delta" not in column
    ]
    if not columns:
        return 0.0
    return float(frame[columns].min(numeric_only=True).min())


def _missing_period_count(frame: pd.DataFrame, time_step_hours: float) -> int:
    if "timestamp" not in frame or len(frame) < 2:
        return 0
    timestamps = pd.to_datetime(frame["timestamp"])
    deltas = timestamps.diff().dropna()
    expected = pd.Timedelta(hours=time_step_hours)
    return int((deltas != expected).sum())


def _summary_for_comparison(path: Path) -> dict[str, Any]:
    summary_path = path / "summary_metrics_v1.csv"
    if summary_path.exists():
        table = pd.read_csv(summary_path)
        metrics = {
            str(row["metric"]): float(row["value"])
            for _, row in table.iterrows()
            if pd.notna(row["value"]) and _is_number(row["value"])
        }
    else:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        metrics = {
            key: float(value)
            for key, value in _flatten_scalars(summary).items()
            if isinstance(value, int | float)
        }
    return {"label": path.name, "metrics": metrics}


def _comparison_markdown(table: pd.DataFrame) -> str:
    lines = ["# Scenario Output Comparison", ""]
    if table.empty:
        return "# Scenario Output Comparison\n\nNo comparable scalar metrics.\n"
    columns = list(table.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in table.iterrows():
        lines.append(
            "| " + " | ".join(_format_markdown_value(row[column]) for column in columns) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _limited_columns(frame: pd.DataFrame, prefix: str, *, limit: int = 8) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    return columns[:limit]


def _top_series(series: pd.Series, *, limit: int) -> pd.Series:
    if len(series) <= limit:
        return series
    ordered = series.abs().sort_values(ascending=False)
    keep = list(ordered.iloc[: limit - 1].index)
    other = float(series.drop(index=keep).sum())
    result: pd.Series = pd.concat([series.loc[keep], pd.Series({"other": other})])
    return result


def _short_label(column: str) -> str:
    return column.replace("__", " ").replace("_", " ")


def _unit_for_name(name: str) -> str:
    lower = name.lower()
    if lower == "schema_version":
        return "version"
    if lower == "timestamp":
        return "datetime"
    if "eur_per_mwh" in lower:
        return "EUR/MWh"
    if lower.endswith("_eur") or lower.endswith("value_eur") or "objective" in lower:
        return "EUR"
    if lower.endswith("_mw") or lower.endswith("_load_mw"):
        return "MW"
    if lower.endswith("_mwh") or "energy" in lower:
        return "MWh"
    if "tonnes" in lower:
        return "tonnes"
    if lower.endswith("_kg"):
        return "kg"
    if "seconds" in lower:
        return "s"
    if "gap" in lower or "share" in lower or "probability" in lower or "utilisation" in lower:
        return "fraction"
    return "dimensionless"


def _aggregation_for_name(name: str) -> str:
    lower = name.lower()
    if lower == "timestamp":
        return "time index"
    if lower.endswith("_mw") or "price" in lower or "soc" in lower or "angle" in lower:
        return "time series"
    if lower.endswith("_mwh") or lower.endswith("_eur") or "emissions" in lower:
        return "sum"
    if "max" in lower or "peak" in lower:
        return "maximum"
    if "average" in lower or "mean" in lower:
        return "mean"
    return "reported"


def _sign_convention_for_name(name: str) -> str:
    lower = name.lower()
    if "flow" in lower:
        return "positive in configured from-to direction"
    if "charge" in lower:
        return "positive increases served load or stored energy"
    if "discharge" in lower or "generation" in lower or "imports" in lower:
        return "positive supplies energy"
    if "cost" in lower or lower.endswith("_eur"):
        return "positive is cost, negative is credit"
    if "residual" in lower:
        return "positive residual follows equation definition"
    return "positive magnitude"


def _description_for_name(name: str) -> str:
    return name.replace("_", " ")


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
