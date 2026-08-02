from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from energy_system_simulator.config import load_config
from energy_system_simulator.data import load_input_data

CASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = CASE_DIR / "configs" / "iberia_baseline.yaml"
INPUT_PATH = CASE_DIR / "data" / "iberia_2024_week_hourly.csv"
TARGET_PATH = CASE_DIR / "provenance" / "ember_2024_targets.csv"
EXPERIMENT_PATH = CASE_DIR / "scenarios" / "iberia_sensitivity.yaml"
REPORT_PATH = CASE_DIR / "reports" / "case_study_report.md"
OUTPUT_DIR = CASE_DIR / "outputs"
HOURS_PER_YEAR = 8760.0


def main() -> int:
    config = load_config(CONFIG_PATH)
    data = load_input_data(INPUT_PATH, config.simulation.time_step_hours)
    targets = pd.read_csv(TARGET_PATH)
    _validate_hourly_snapshot(data, targets)
    _validate_experiment_file()
    _validate_provenance_and_report()
    _validate_outputs_if_present()
    print("iberia case study validation ok")
    return 0


def _validate_hourly_snapshot(data: pd.DataFrame, targets: pd.DataFrame) -> None:
    if len(data) != 168:
        raise AssertionError("Iberia snapshot must contain one 168-hour week")
    required = {
        "pt_demand_mw",
        "es_demand_mw",
        "pt_irradiance_w_m2",
        "es_irradiance_w_m2",
        "pt_wind_speed_m_s",
        "es_wind_speed_m_s",
        "pt_hydro_inflow_mw_water",
        "es_hydro_inflow_mw_water",
    }
    missing = required - set(data.columns)
    if missing:
        raise AssertionError(f"Iberia snapshot is missing columns: {sorted(missing)}")
    for area, column in (("Portugal", "pt_demand_mw"), ("Spain", "es_demand_mw")):
        target = float(
            targets[
                targets["area"].eq(area)
                & targets["variable"].eq("Demand")
                & targets["unit"].eq("TWh")
            ]["value"].iloc[0]
        )
        annualized = float(
            data[column].sum() * config_time_step_hours(data) * HOURS_PER_YEAR / len(data) / 1e6
        )
        if abs(annualized - target) > 0.01:
            raise AssertionError(f"{area} annualized demand does not match Ember target")


def _validate_experiment_file() -> None:
    payload = yaml.safe_load(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    if len(scenarios) < 6:
        raise AssertionError("Iberia sensitivity file must define at least six scenarios")
    labels = {scenario["id"] for scenario in scenarios}
    expected = {
        "solar-plus-25",
        "interconnection-plus-50",
        "gas-retirement-30",
        "hydro-drought-proxy",
        "battery-plus-50",
        "carbon-price-120",
    }
    if labels != expected:
        raise AssertionError("Iberia sensitivity scenario IDs changed unexpectedly")


def _validate_provenance_and_report() -> None:
    provenance = (CASE_DIR / "provenance.md").read_text(encoding="utf-8")
    report = REPORT_PATH.read_text(encoding="utf-8")
    for phrase in (
        "CC BY 4.0",
        "2026-08-02",
        "not historical measurements",
    ):
        if phrase not in provenance:
            raise AssertionError(f"Missing provenance phrase: {phrase}")
    for phrase in (
        "not for operational replay",
        "positive `line_flow_mw__pt-es` is Portugal to Spain",
        "not a full nodal representation",
    ):
        if phrase not in report:
            raise AssertionError(f"Missing report caveat: {phrase}")


def _validate_outputs_if_present() -> None:
    summary_path = OUTPUT_DIR / "baseline" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("solver_status", "optimal") != "optimal":
            raise AssertionError("Baseline solver status is not optimal")
        diagnostics = json.loads(
            (OUTPUT_DIR / "baseline" / "diagnostics.json").read_text(encoding="utf-8")
        )
        if diagnostics["status"] != "ok":
            raise AssertionError("Baseline diagnostics are not ok")
    aggregate_path = OUTPUT_DIR / "scenarios" / "summary.csv"
    if aggregate_path.is_file():
        aggregate = pd.read_csv(aggregate_path)
        if len(aggregate) != 6 or not aggregate["ok"].all():
            raise AssertionError("Sensitivity outputs are incomplete or failed")


def config_time_step_hours(data: pd.DataFrame) -> float:
    timestamps = pd.to_datetime(data["timestamp"])
    if len(timestamps) < 2:
        return 1.0
    return float((timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 3600.0)


if __name__ == "__main__":
    raise SystemExit(main())
