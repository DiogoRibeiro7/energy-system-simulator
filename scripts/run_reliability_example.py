from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.errors import PerformanceWarning

from energy_system_simulator.config import load_config
from energy_system_simulator.simulation import (
    OutageModel,
    ReliabilityResult,
    ReliabilityStudy,
    ReliabilityStudyConfig,
)

ROOT = Path(__file__).resolve().parents[1]
WORKDIR = ROOT / ".tmp" / "reliability-example"


def main() -> None:
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    base_config = _scenario_config(extra_assets=False)
    augmented_config = _scenario_config(extra_assets=True)

    base = _run_study(base_config)
    augmented = _run_study(augmented_config)

    print("scenario,eue_mwh,lole_hours_per_year,edns_mw")
    for name, result in (("base", base), ("with_extra_peaker_battery", augmented)):
        print(
            f"{name},"
            f"{result.metrics['expected_unserved_energy_mwh']:.3f},"
            f"{result.metrics['loss_of_load_expectation_hours_per_year']:.3f},"
            f"{result.metrics['expected_demand_not_served_mw']:.3f}"
        )


def _scenario_config(extra_assets: bool) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "portfolio_two_thermal.yaml").read_text())
    if not isinstance(raw, dict):
        raise TypeError("Example configuration must load as a mapping")
    input_path = WORKDIR / ("augmented.csv" if extra_assets else "base.csv")
    source = pd.read_csv(ROOT / "data" / "example_hourly.csv").head(48)
    source.to_csv(input_path, index=False)
    raw["paths"]["input_csv"] = str(input_path)
    output_name = "augmented-output" if extra_assets else "base-output"
    raw["paths"]["output_directory"] = str(WORKDIR / output_name)
    if extra_assets:
        raw["thermal_generators"].append(_extra_peaker())
        raw["storage_units"].append(_extra_battery())
    config_path = WORKDIR / ("augmented.yaml" if extra_assets else "base.yaml")
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def _run_study(config_path: Path) -> ReliabilityResult:
    config = load_config(config_path)
    outage_models = [
        OutageModel("north-ccgt", "thermal", 0.06, 12.0),
        OutageModel("south-peaker", "thermal", 0.10, 6.0),
        OutageModel("south-battery", "storage", 0.04, 8.0),
        OutageModel("north-south", "line", 0.02, 4.0),
        OutageModel("market-imports", "import", 0.03, 5.0),
    ]
    if "augmented" in config_path.name:
        outage_models.extend(
            [
                OutageModel("reserve-peaker", "thermal", 0.12, 5.0),
                OutageModel("reserve-battery", "storage", 0.05, 8.0),
            ]
        )
    return ReliabilityStudy(
        config,
        ReliabilityStudyConfig(
            replications=10,
            seed=20260802,
            outage_models=tuple(outage_models),
            parallel_workers=2,
        ),
    ).run()


def _extra_peaker() -> dict[str, Any]:
    return {
        "id": "reserve-peaker",
        "bus_id": "south-hub",
        "fuel_id": "gas",
        "name": "Reserve peaking turbine",
        "minimum_output_mw": 0.0,
        "maximum_output_mw": 60.0,
        "ramp_up_mw_per_hour": 60.0,
        "ramp_down_mw_per_hour": 60.0,
        "startup_ramp_mw": 60.0,
        "shutdown_ramp_mw": 60.0,
        "variable_cost_eur_per_mwh": 130.0,
        "no_load_cost_eur_per_hour": 150.0,
        "startup_cost_eur": 1200.0,
        "shutdown_cost_eur": 300.0,
        "emission_factor_tonnes_per_mwh": 0.55,
        "minimum_up_hours": 1.0,
        "minimum_down_hours": 1.0,
        "initial_on": False,
        "initial_output_mw": 0.0,
        "initial_up_time_hours": 0.0,
        "initial_down_time_hours": 2.0,
        "terminal_commitment_mode": "carry_residual_obligations",
    }


def _extra_battery() -> dict[str, Any]:
    return {
        "id": "reserve-battery",
        "bus_id": "south-hub",
        "technology": "battery",
        "energy_capacity_mwh": 120.0,
        "power_capacity_mw": 40.0,
        "charge_power_capacity_mw": 40.0,
        "discharge_power_capacity_mw": 40.0,
        "minimum_soc_mwh": 12.0,
        "maximum_soc_mwh": 120.0,
        "initial_soc_mwh": 60.0,
        "charge_efficiency": 0.94,
        "discharge_efficiency": 0.94,
        "self_discharge_rate_per_hour": 0.0,
        "minimum_charge_mw": 0.0,
        "minimum_discharge_mw": 0.0,
        "throughput_cost_eur_per_mwh": 2.5,
        "minimum_final_soc_mwh": 40.0,
        "terminal_soc_mode": "minimum",
    }


if __name__ == "__main__":
    main()
