from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from energy_system_simulator.api import run_simulation
from energy_system_simulator.scenarios import run_experiment_file

CASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CASE_DIR / "data"
CONFIG_DIR = CASE_DIR / "configs"
SCENARIO_DIR = CASE_DIR / "scenarios"
OUTPUT_DIR = CASE_DIR / "outputs"
REPORT_DIR = CASE_DIR / "reports"
TARGET_PATH = CASE_DIR / "provenance" / "ember_2024_targets.csv"
INPUT_PATH = DATA_DIR / "iberia_2024_week_hourly.csv"
CONFIG_PATH = CONFIG_DIR / "iberia_baseline.yaml"
EXPERIMENT_PATH = SCENARIO_DIR / "iberia_sensitivity.yaml"
REPORT_PATH = REPORT_DIR / "case_study_report.md"

STUDY_START = "2024-06-24T00:00:00Z"
PERIODS = 168
HOURS_PER_YEAR = 8760.0

EMBER_TARGETS = [
    ("Portugal", "Electricity demand", "Demand", "Demand", "TWh", 57.74),
    ("Portugal", "Electricity generation", "Fuel", "Solar", "TWh", 7.09),
    ("Portugal", "Electricity generation", "Fuel", "Wind", "TWh", 14.34),
    ("Portugal", "Electricity generation", "Fuel", "Hydro", "TWh", 14.89),
    ("Portugal", "Electricity generation", "Fuel", "Gas", "TWh", 5.63),
    ("Portugal", "Electricity generation", "Fuel", "Nuclear", "TWh", 0.0),
    ("Portugal", "Electricity generation", "Total", "Total Generation", "TWh", 47.27),
    ("Portugal", "Electricity imports", "Electricity imports", "Net Imports", "TWh", 10.47),
    ("Portugal", "Power sector emissions", "Total", "Total emissions", "mtCO2", 5.23),
    ("Portugal", "Capacity", "Fuel", "Solar", "GW", 5.81),
    ("Portugal", "Capacity", "Fuel", "Wind", "GW", 5.60),
    ("Portugal", "Capacity", "Fuel", "Hydro", "GW", 8.36),
    ("Portugal", "Capacity", "Fuel", "Gas", "GW", 4.19),
    ("Portugal", "Capacity", "Fuel", "Nuclear", "GW", 0.0),
    ("Spain", "Electricity demand", "Demand", "Demand", "TWh", 270.71),
    ("Spain", "Electricity generation", "Fuel", "Solar", "TWh", 58.29),
    ("Spain", "Electricity generation", "Fuel", "Wind", "TWh", 62.20),
    ("Spain", "Electricity generation", "Fuel", "Hydro", "TWh", 34.41),
    ("Spain", "Electricity generation", "Fuel", "Gas", "TWh", 52.34),
    ("Spain", "Electricity generation", "Fuel", "Nuclear", "TWh", 54.53),
    ("Spain", "Electricity generation", "Total", "Total Generation", "TWh", 280.94),
    ("Spain", "Electricity imports", "Electricity imports", "Net Imports", "TWh", -10.23),
    ("Spain", "Power sector emissions", "Total", "Total emissions", "mtCO2", 41.08),
    ("Spain", "Capacity", "Fuel", "Solar", "GW", 38.16),
    ("Spain", "Capacity", "Fuel", "Wind", "GW", 32.18),
    ("Spain", "Capacity", "Fuel", "Hydro", "GW", 16.81),
    ("Spain", "Capacity", "Fuel", "Gas", "GW", 27.40),
    ("Spain", "Capacity", "Fuel", "Nuclear", "GW", 7.12),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and optionally run the Iberia case study.")
    parser.add_argument("--run", action="store_true", help="Run baseline and sensitivity cases")
    parser.add_argument("--overwrite", action="store_true", help="Refresh generated outputs")
    args = parser.parse_args()

    build_case_inputs()
    if args.run:
        if args.overwrite:
            _remove_case_outputs()
        run_simulation(CONFIG_PATH, create_plots=True, overwrite=True)
        run_experiment_file(EXPERIMENT_PATH, create_plots=False)
    results_available = (OUTPUT_DIR / "baseline" / "summary.json").is_file()
    if args.run or results_available or not REPORT_PATH.exists():
        write_report(results_available=results_available)
    print(f"Iberia case study files written under {CASE_DIR}")
    print(f"Report written: {REPORT_PATH}")
    return 0


def build_case_inputs() -> None:
    for directory in (DATA_DIR, CONFIG_DIR, SCENARIO_DIR, TARGET_PATH.parent, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    _write_targets()
    _write_hourly_snapshot()
    _write_baseline_config()
    _write_experiment()


def _write_targets() -> None:
    with TARGET_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["area", "category", "subcategory", "variable", "unit", "value"])
        writer.writerows(EMBER_TARGETS)


def _write_hourly_snapshot() -> None:
    timestamps = pd.date_range(STUDY_START, periods=PERIODS, freq="h")
    pt_shape = _normalised_demand_shape(PERIODS, morning=8.0, evening=21.0, weekend_discount=0.08)
    es_shape = _normalised_demand_shape(PERIODS, morning=9.0, evening=20.0, weekend_discount=0.06)
    pt_average_mw = _annual_twh_to_average_mw(_target("Portugal", "Demand", "TWh"))
    es_average_mw = _annual_twh_to_average_mw(_target("Spain", "Demand", "TWh"))
    pt_demand = [pt_average_mw * value for value in pt_shape]
    es_demand = [es_average_mw * value for value in es_shape]

    rows: list[dict[str, float | str]] = []
    for index, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day = index // 24
        solar_shape = max(0.0, math.sin(math.pi * (hour - 6) / 14.0))
        wind_cycle = math.sin(2.0 * math.pi * (index + 5) / 48.0)
        rows.append(
            {
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "demand_mw": pt_demand[index] + es_demand[index],
                "pt_demand_mw": pt_demand[index],
                "es_demand_mw": es_demand[index],
                "irradiance_w_m2": 920.0 * solar_shape,
                "ambient_temperature_c": 24.0 + 5.0 * solar_shape,
                "wind_speed_m_s": 7.3 + 1.4 * wind_cycle,
                "pt_irradiance_w_m2": 880.0 * solar_shape,
                "es_irradiance_w_m2": 960.0 * solar_shape,
                "pt_ambient_temperature_c": 22.0 + 4.0 * solar_shape,
                "es_ambient_temperature_c": 25.0 + 6.0 * solar_shape,
                "pt_wind_speed_m_s": 7.6 + 1.6 * wind_cycle,
                "es_wind_speed_m_s": 7.0 + 1.2 * math.sin(2.0 * math.pi * (index + 18) / 72.0),
                "pt_hydro_inflow_mw_water": 1550.0 + 220.0 * math.sin(2.0 * math.pi * day / 7.0),
                "es_hydro_inflow_mw_water": 2650.0
                + 360.0 * math.sin(2.0 * math.pi * (day + 1) / 7.0),
                "import_price_eur_per_mwh": 82.0
                + 18.0 * solar_shape
                + 9.0 * (hour in (19, 20, 21)),
            }
        )
    pd.DataFrame(rows).round(6).to_csv(INPUT_PATH, index=False)


def _write_baseline_config() -> None:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "scenario": {"id": "iberia-2024-week-baseline"},
        "simulation": {"time_step_hours": 1.0},
        "solver": {
            "solver_time_limit_seconds": 120.0,
            "mip_relative_gap": 0.001,
            "allow_non_optimal_solution": False,
        },
        "fuels": [
            {
                "id": "gas",
                "price_eur_per_mwh_thermal": 42.0,
                "co2_factor_tonnes_per_mwh_thermal": 0.20,
            },
            {
                "id": "nuclear",
                "price_eur_per_mwh_thermal": 5.0,
                "co2_factor_tonnes_per_mwh_thermal": 0.0,
            },
        ],
        "zones": [{"id": "pt"}, {"id": "es"}],
        "buses": [{"id": "pt", "zone_id": "pt"}, {"id": "es", "zone_id": "es"}],
        "lines": [
            {
                "id": "pt-es",
                "from_bus_id": "pt",
                "to_bus_id": "es",
                "susceptance": 8.0,
                "capacity_mw": 3500.0,
            }
        ],
        "aggregate_network": {
            "loss_fraction": 0.015,
            "transfer_capacity_mw": 90000.0,
            "network_mode": "nodal",
            "slack_bus_id": "es",
        },
        "renewable_generators": [
            _solar("pt-solar", "pt", 5810.0, "pt_irradiance_w_m2", "pt_ambient_temperature_c"),
            _wind("pt-wind", "pt", 5600.0, "pt_wind_speed_m_s"),
            _solar("es-solar", "es", 38160.0, "es_irradiance_w_m2", "es_ambient_temperature_c"),
            _wind("es-wind", "es", 32180.0, "es_wind_speed_m_s"),
        ],
        "thermal_generators": [
            _thermal("pt-gas", "pt", "gas", 4190.0, 0.39, 86.0, 1200.0),
            _thermal("es-gas", "es", "gas", 27400.0, 0.38, 82.0, 3500.0),
            _thermal("es-nuclear", "es", "nuclear", 7120.0, 0.0, 12.0, 10000.0, minimum=4550.0),
        ],
        "storage_units": [
            _storage("pt-battery", "pt", power=900.0, energy=3600.0),
            _storage("es-battery", "es", power=2600.0, energy=10400.0),
        ],
        "hydro_units": [
            _hydro("pt-hydro", "pt", "pt_hydro_inflow_mw_water", turbine=5200.0, reservoir=52000.0),
            _hydro("es-hydro", "es", "es_hydro_inflow_mw_water", turbine=9800.0, reservoir=95000.0),
        ],
        "imports": [
            {
                "id": "rest-of-europe",
                "bus_id": "es",
                "maximum_power_mw": 6000.0,
                "price_eur_per_mwh": 96.0,
                "emission_factor_tonnes_per_mwh": 0.18,
            }
        ],
        "demand": [
            {"id": "pt-load", "bus_id": "pt", "time_series_key": "pt_demand_mw"},
            {"id": "es-load", "bus_id": "es", "time_series_key": "es_demand_mw"},
        ],
        "penalties": {
            "renewable_curtailment_eur_per_mwh": 1.0,
            "lost_load_eur_per_mwh": 15000.0,
            "carbon_price_eur_per_tonne": 65.0,
        },
        "paths": {
            "input_csv": "../data/iberia_2024_week_hourly.csv",
            "output_directory": "../outputs/baseline",
        },
    }
    CONFIG_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_experiment() -> None:
    payload = {
        "base_config": "../configs/iberia_baseline.yaml",
        "output_directory": "../outputs/scenarios",
        "workers": 1,
        "resume": False,
        "scenarios": [
            {
                "id": "solar-plus-25",
                "overrides": {
                    "portfolio.renewable_generators[0].capacity_mw": 7262.5,
                    "portfolio.renewable_generators[2].capacity_mw": 47700.0,
                },
            },
            {
                "id": "interconnection-plus-50",
                "overrides": {"portfolio.lines[0].capacity_mw": 5250.0},
            },
            {
                "id": "gas-retirement-30",
                "overrides": {
                    "portfolio.thermal_generators[0].config.maximum_output_mw": 2933.0,
                    "portfolio.thermal_generators[1].config.maximum_output_mw": 19180.0,
                },
            },
            {
                "id": "hydro-drought-proxy",
                "overrides": {
                    "portfolio.hydro_units[0].turbine_efficiency": 0.55,
                    "portfolio.hydro_units[1].turbine_efficiency": 0.55,
                },
            },
            {
                "id": "battery-plus-50",
                "overrides": {
                    "portfolio.storage_units[0].config.power_capacity_mw": 1350.0,
                    "portfolio.storage_units[0].config.charge_power_capacity_mw": 1350.0,
                    "portfolio.storage_units[0].config.discharge_power_capacity_mw": 1350.0,
                    "portfolio.storage_units[0].config.energy_capacity_mwh": 5400.0,
                    "portfolio.storage_units[0].config.maximum_soc_mwh": 5400.0,
                    "portfolio.storage_units[1].config.power_capacity_mw": 3900.0,
                    "portfolio.storage_units[1].config.charge_power_capacity_mw": 3900.0,
                    "portfolio.storage_units[1].config.discharge_power_capacity_mw": 3900.0,
                    "portfolio.storage_units[1].config.energy_capacity_mwh": 15600.0,
                    "portfolio.storage_units[1].config.maximum_soc_mwh": 15600.0,
                },
            },
            {
                "id": "carbon-price-120",
                "overrides": {"penalties.carbon_price_eur_per_tonne": 120.0},
            },
        ],
    }
    EXPERIMENT_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_report(*, results_available: bool) -> None:
    target_frame = pd.read_csv(TARGET_PATH)
    demand_rows = _demand_validation_rows(target_frame)
    lines = [
        "# Iberian Approximation Case Study",
        "",
        "This case study is a two-zone Portugal-Spain approximation for one summer week.",
        "It is designed for reproducible model exercises, not for operational replay.",
        "",
        "## Study Design",
        "",
        f"- Period: {STUDY_START} for {PERIODS} hourly periods.",
        "- Topology: two buses, `pt` and `es`, connected by line `pt-es`.",
        "- Flow sign convention: positive `line_flow_mw__pt-es` is Portugal to Spain.",
        "- Demand and installed capacities are calibrated to Ember 2024 country aggregates.",
        "- Hourly demand, weather, hydro inflow, and prices are deterministic approximations.",
        "",
        "## Aggregate Validation",
        "",
        "| Area | Ember demand TWh | Annualized snapshot TWh | Difference % |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['area']} | {row['target_twh']:.2f} | {row['snapshot_twh']:.2f} | "
        f"{row['difference_pct']:.3f} |"
        for row in demand_rows
    )
    lines.extend(
        [
            "",
            "The demand snapshot is normalized to reproduce annual demand if the modeled week is",
            "repeated across a year. Generation and flow totals are model outcomes and",
            "are compared against Ember annual totals only as a reasonableness check.",
            "",
        ]
    )
    if results_available:
        lines.extend(_result_report_lines(target_frame))
    else:
        lines.extend(
            [
                "## Results",
                "",
                "Run `poetry run python case_studies/iberia/scripts/build_case_study.py "
                "--run --overwrite`",
                "to refresh baseline and sensitivity outputs.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- This is not a full nodal representation of the Portuguese or Spanish grids.",
            "- Thermal fleets are grouped by technology and represented as a few aggregate units.",
            "- Hydro inflows are stylized weekly profiles calibrated only qualitatively.",
            "- External trade is represented as a single import resource at Spain plus a "
            "PT-ES line.",
            "- Weather and price profiles are deterministic approximations, not historical "
            "reanalysis.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _result_report_lines(target_frame: pd.DataFrame) -> list[str]:
    summary = json.loads((OUTPUT_DIR / "baseline" / "summary.json").read_text(encoding="utf-8"))
    timeseries = pd.read_csv(OUTPUT_DIR / "baseline" / "timeseries.csv")
    dt = float(summary["time_step_hours"])
    annual_factor = HOURS_PER_YEAR / (len(timeseries) * dt)
    model_rows = [
        ("Demand", float(timeseries["end_user_demand_mw"].sum() * dt * annual_factor / 1e6)),
        ("Solar", _annualized_sum(timeseries, "solar", dt, annual_factor)),
        ("Wind", _annualized_sum(timeseries, "wind", dt, annual_factor)),
        ("Hydro", float(timeseries["hydro_generation_mw"].sum() * dt * annual_factor / 1e6)),
        ("Gas", _annualized_sum(timeseries, "gas", dt, annual_factor)),
        ("Nuclear", _annualized_sum(timeseries, "nuclear", dt, annual_factor)),
    ]
    target_totals = (
        target_frame[
            target_frame["unit"].eq("TWh")
            & target_frame["variable"].isin({"Demand", "Solar", "Wind", "Hydro", "Gas", "Nuclear"})
        ]
        .groupby("variable")["value"]
        .sum()
        .to_dict()
    )
    lines = [
        "## Baseline Results",
        "",
        f"- Solver status: `{summary.get('solver_status', 'optimal')}`",
        f"- Objective: EUR {float(summary['objective_eur']):,.0f}",
        f"- Unserved energy: {float(summary['unserved_energy_mwh']):,.3f} MWh",
        f"- Renewable share: {float(summary['renewable_share_of_primary_generation']):.2%}",
        f"- CO2 emissions: {float(summary['total_emissions_tonnes']):,.0f} tonnes",
        "",
        "| Quantity | Annualized model TWh | Ember 2024 TWh | Difference % |",
        "|---|---:|---:|---:|",
    ]
    for variable, model_twh in model_rows:
        target = target_totals.get(variable)
        difference = (model_twh - target) / target * 100.0 if target else 0.0
        lines.append(f"| {variable} | {model_twh:.2f} | {target or 0.0:.2f} | {difference:.1f} |")
    lines.extend(
        [
            "",
            "The summer-week weather approximation intentionally overstates annualized solar",
            "generation when repeated across a full year; this is a limitation of using one",
            "compact representative week rather than a weather-year sample.",
        ]
    )
    if "line_flow_mw__pt-es" in timeseries:
        lines.extend(
            [
                "",
                "## Cross-Border Flow",
                "",
                "- Maximum Portugal-to-Spain flow: "
                f"{timeseries['line_flow_mw__pt-es'].max():,.1f} MW",
                "- Maximum Spain-to-Portugal flow: "
                f"{-timeseries['line_flow_mw__pt-es'].min():,.1f} MW",
                "",
            ]
        )
    aggregate = OUTPUT_DIR / "scenarios" / "summary.csv"
    if aggregate.is_file():
        scenario_frame = pd.read_csv(aggregate)
        lines.extend(
            [
                "## Sensitivities",
                "",
                "| Scenario | Objective EUR | Emissions tonnes | Unserved MWh | Renewable share |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in scenario_frame.iterrows():
            lines.append(
                f"| {row['scenario_label']} | {float(row['objective_eur']):,.0f} | "
                f"{float(row['total_emissions_tonnes']):,.0f} | "
                f"{float(row['unserved_energy_mwh']):,.3f} | {float(row['renewable_share']):.2%} |"
            )
        lines.append("")
    return lines


def _annualized_sum(
    timeseries: pd.DataFrame,
    token: str,
    dt: float,
    annual_factor: float,
) -> float:
    columns = [
        column
        for column in timeseries.columns
        if column.startswith("thermal_output_mw__") and token in column
    ]
    if token in {"solar", "wind"}:
        columns = [
            column
            for column in timeseries.columns
            if column.startswith("renewable_used_mw__") and token in column
        ]
    if not columns:
        return 0.0
    return float(timeseries[columns].sum(axis=1).sum() * dt * annual_factor / 1e6)


def _demand_validation_rows(target_frame: pd.DataFrame) -> list[dict[str, float | str]]:
    data = pd.read_csv(INPUT_PATH)
    rows: list[dict[str, float | str]] = []
    for area, column in (("Portugal", "pt_demand_mw"), ("Spain", "es_demand_mw")):
        target = float(
            target_frame[
                target_frame["area"].eq(area)
                & target_frame["variable"].eq("Demand")
                & target_frame["unit"].eq("TWh")
            ]["value"].iloc[0]
        )
        snapshot = float(data[column].sum() * HOURS_PER_YEAR / PERIODS / 1e6)
        rows.append(
            {
                "area": area,
                "target_twh": target,
                "snapshot_twh": snapshot,
                "difference_pct": (snapshot - target) / target * 100.0,
            }
        )
    return rows


def _solar(
    asset_id: str,
    bus_id: str,
    capacity: float,
    key: str,
    temperature_key: str,
) -> dict[str, Any]:
    return {
        "id": asset_id,
        "kind": "solar",
        "bus_id": bus_id,
        "capacity_mw": capacity,
        "performance_ratio": 0.86,
        "reference_irradiance_w_m2": 1000.0,
        "temperature_coefficient_per_c": -0.004,
        "nominal_operating_cell_temperature_c": 45.0,
        "time_series_key": key,
        "ambient_temperature_key": temperature_key,
    }


def _wind(asset_id: str, bus_id: str, capacity: float, key: str) -> dict[str, Any]:
    return {
        "id": asset_id,
        "kind": "wind",
        "bus_id": bus_id,
        "capacity_mw": capacity,
        "cut_in_speed_m_s": 3.0,
        "rated_speed_m_s": 12.0,
        "cut_out_speed_m_s": 25.0,
        "time_series_key": key,
    }


def _thermal(
    asset_id: str,
    bus_id: str,
    fuel_id: str,
    capacity: float,
    emissions: float,
    variable_cost: float,
    startup_cost: float,
    *,
    minimum: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": asset_id,
        "bus_id": bus_id,
        "fuel_id": fuel_id,
        "name": asset_id.replace("-", " ").title(),
        "minimum_output_mw": minimum,
        "maximum_output_mw": capacity,
        "ramp_up_mw_per_hour": capacity,
        "ramp_down_mw_per_hour": capacity,
        "startup_ramp_mw": capacity,
        "shutdown_ramp_mw": capacity,
        "variable_cost_eur_per_mwh": variable_cost,
        "no_load_cost_eur_per_hour": 0.0,
        "startup_cost_eur": startup_cost,
        "shutdown_cost_eur": startup_cost * 0.1,
        "emission_factor_tonnes_per_mwh": emissions,
        "minimum_up_hours": 3.0,
        "minimum_down_hours": 3.0,
        "initial_on": minimum > 0.0,
        "initial_output_mw": minimum,
        "initial_up_time_hours": 24.0 if minimum > 0.0 else 0.0,
        "initial_down_time_hours": 0.0 if minimum > 0.0 else 24.0,
        "terminal_commitment_mode": "carry_residual_obligations",
    }


def _storage(asset_id: str, bus_id: str, *, power: float, energy: float) -> dict[str, Any]:
    return {
        "id": asset_id,
        "bus_id": bus_id,
        "technology": "battery",
        "energy_capacity_mwh": energy,
        "power_capacity_mw": power,
        "charge_power_capacity_mw": power,
        "discharge_power_capacity_mw": power,
        "minimum_soc_mwh": energy * 0.05,
        "maximum_soc_mwh": energy,
        "initial_soc_mwh": energy * 0.45,
        "charge_efficiency": 0.93,
        "discharge_efficiency": 0.93,
        "self_discharge_rate_per_hour": 0.0001,
        "throughput_cost_eur_per_mwh": 2.0,
        "minimum_final_soc_mwh": energy * 0.30,
        "terminal_soc_mode": "minimum",
    }


def _hydro(
    asset_id: str,
    bus_id: str,
    inflow_key: str,
    *,
    turbine: float,
    reservoir: float,
) -> dict[str, Any]:
    return {
        "id": asset_id,
        "kind": "reservoir",
        "bus_id": bus_id,
        "inflow_time_series_key": inflow_key,
        "minimum_reservoir_mwh": reservoir * 0.10,
        "maximum_reservoir_mwh": reservoir,
        "initial_reservoir_mwh": reservoir * 0.55,
        "minimum_final_reservoir_mwh": reservoir * 0.35,
        "terminal_reservoir_mode": "minimum",
        "turbine_capacity_mw": turbine,
        "turbine_efficiency": 0.90,
        "spill_capacity_mw": turbine * 1.3,
        "minimum_release_mw": turbine * 0.02,
        "evaporation_rate_per_hour": 0.00002,
        "water_value_eur_per_mwh": 6.0,
    }


def _normalised_demand_shape(
    periods: int,
    *,
    morning: float,
    evening: float,
    weekend_discount: float,
) -> list[float]:
    values: list[float] = []
    for index in range(periods):
        hour = index % 24
        day = index // 24
        weekend = 1.0 - weekend_discount if day in {5, 6} else 1.0
        daily = (
            0.90
            + 0.10 * math.exp(-(((hour - morning) / 3.5) ** 2))
            + 0.16 * math.exp(-(((hour - evening) / 3.2) ** 2))
        )
        values.append(daily * weekend)
    mean = sum(values) / len(values)
    return [value / mean for value in values]


def _annual_twh_to_average_mw(value: float) -> float:
    return value * 1_000_000.0 / HOURS_PER_YEAR


def _target(area: str, variable: str, unit: str) -> float:
    for item_area, _, _, item_variable, item_unit, value in EMBER_TARGETS:
        if item_area == area and item_variable == variable and item_unit == unit:
            return value
    raise KeyError((area, variable, unit))


def _remove_case_outputs() -> None:
    if OUTPUT_DIR.exists():
        if CASE_DIR not in OUTPUT_DIR.resolve().parents:
            raise RuntimeError(f"Refusing to remove output outside case study: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
