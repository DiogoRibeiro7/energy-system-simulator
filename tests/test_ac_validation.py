from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

from energy_system_simulator.ac_validation import (
    ACValidationOptions,
    select_ac_validation_periods,
    validate_ac_power_flow,
)
from energy_system_simulator.config import (
    BusConfig,
    DemandConfig,
    ImportConfig,
    ImportResourceConfig,
    ModelConfig,
    NetworkConfig,
    PortfolioConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
    load_config,
)


@dataclass(frozen=True)
class _SolvedDispatch:
    timeseries: pd.DataFrame
    objective_eur: float = 0.0


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal() -> ThermalConfig:
    base = _example_config().thermal
    return replace(base, minimum_output_mw=0.0, maximum_output_mw=100.0)


def _two_bus_config(
    *,
    reactive_demand_mvar_per_mw: float = 0.0,
    voltage_min_pu: float = 0.95,
    ac_rating_mva: float = 100.0,
    reactive_power_max_mvar: float = 100.0,
    ac_reactance_pu: float = 0.1,
) -> ModelConfig:
    base = _example_config()
    imports = ImportConfig(0.0, 0.0, 0.0)
    thermal = ThermalGeneratorConfig(
        id="gen",
        bus_id="source",
        fuel_id="gas",
        config=_thermal(),
        reactive_power_min_mvar=-100.0,
        reactive_power_max_mvar=reactive_power_max_mvar,
    )
    demand = DemandConfig(
        id="load",
        bus_id="sink",
        time_series_key="load_mw",
        reactive_demand_mvar_per_mw=reactive_demand_mvar_per_mw,
    )
    portfolio = PortfolioConfig(
        scenario=base.portfolio.scenario,
        fuels=base.portfolio.fuels,
        zones=base.portfolio.zones,
        buses=(
            BusConfig(id="source", zone_id="system"),
            BusConfig(id="sink", zone_id="system", voltage_min_pu=voltage_min_pu),
        ),
        lines=(
            TransmissionLineConfig(
                id="source_sink",
                from_bus_id="source",
                to_bus_id="sink",
                susceptance=10.0,
                capacity_mw=100.0,
                ac_reactance_pu=ac_reactance_pu,
                ac_rating_mva=ac_rating_mva,
            ),
        ),
        renewable_generators=(),
        thermal_generators=(thermal,),
        storage_units=(),
        hydro_units=(),
        imports=(ImportResourceConfig(id="imports", bus_id="source", config=imports),),
        demand=(demand,),
    )
    return replace(
        base,
        portfolio=portfolio,
        imports=imports,
        thermal=thermal.config,
        network=NetworkConfig(
            loss_fraction=0.0,
            transfer_capacity_mw=100.0,
            network_mode="nodal",
            slack_bus_id="source",
            ac_base_mva=100.0,
        ),
    )


def _dispatch_frame(
    *, active_mw: float = 10.0, reactive_policy_columns: bool = True
) -> pd.DataFrame:
    data = {
        "timestamp": ["2026-01-01T00:00:00", "2026-01-01T01:00:00", "2026-01-01T02:00:00"],
        "bus_net_injection_mw__source": [active_mw, active_mw * 2.0, active_mw * 3.0],
        "bus_net_injection_mw__sink": [-active_mw, -active_mw * 2.0, -active_mw * 3.0],
        "line_flow_mw__source_sink": [active_mw, active_mw * 2.0, active_mw * 3.0],
        "thermal_on__gen": [1.0, 1.0, 1.0],
        "demand_served_mw__load": [active_mw, active_mw * 2.0, active_mw * 3.0],
        "demand_adjusted_mw": [active_mw, active_mw * 3.0, active_mw * 2.0],
        "renewable_available_mw": [0.0, 50.0, 10.0],
        "line_max_abs_utilisation": [0.1, 0.2, 0.9],
    }
    if reactive_policy_columns:
        data["bus_voltage_angle_rad__source"] = [0.0, 0.0, 0.0]
        data["bus_voltage_angle_rad__sink"] = [-0.1, -0.2, -0.3]
    return pd.DataFrame(data)


def test_ac_validation_converges_on_two_bus_case() -> None:
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "ac_two_bus_reference.json").read_text(
            encoding="utf-8"
        )
    )
    config = _two_bus_config()
    result = _SolvedDispatch(_dispatch_frame(active_mw=10.0))

    validation = validate_ac_power_flow(
        config,
        result,
        ACValidationOptions(policies=(), periods=(0,)),
    )

    row = validation.records.iloc[0]
    assert validation.valid
    assert row["converged"]
    assert row["max_power_mismatch_pu"] < 1e-8
    assert row["min_voltage_pu"] == pytest.approx(reference["expected_sink_voltage_pu"])
    assert row["active_losses_mw"] == pytest.approx(0.0, abs=1e-8)
    assert row["max_dc_active_flow_mismatch_mw"] < 0.01


def test_ac_validation_reports_voltage_violation() -> None:
    config = _two_bus_config(
        reactive_demand_mvar_per_mw=2.0,
        voltage_min_pu=0.99,
        ac_reactance_pu=0.25,
    )
    result = _SolvedDispatch(_dispatch_frame(active_mw=20.0))

    validation = validate_ac_power_flow(
        config,
        result,
        ACValidationOptions(policies=(), periods=(0,)),
    )

    assert not validation.valid
    assert validation.records["max_voltage_violation_pu"].iloc[0] > 0.0


def test_ac_validation_reports_reactive_limit_violation() -> None:
    config = _two_bus_config(
        reactive_demand_mvar_per_mw=1.0,
        reactive_power_max_mvar=5.0,
    )
    result = _SolvedDispatch(_dispatch_frame(active_mw=20.0))

    validation = validate_ac_power_flow(
        config,
        result,
        ACValidationOptions(policies=(), periods=(0,)),
    )

    assert not validation.valid
    assert validation.records["max_reactive_limit_violation_mvar"].iloc[0] > 0.0


def test_ac_validation_reports_branch_overload() -> None:
    config = _two_bus_config(ac_rating_mva=5.0)
    result = _SolvedDispatch(_dispatch_frame(active_mw=10.0))

    validation = validate_ac_power_flow(
        config,
        result,
        ACValidationOptions(policies=(), periods=(0,)),
    )

    assert not validation.valid
    assert validation.records["binding_branch_id"].iloc[0] == "source_sink"
    assert validation.records["max_branch_overload_mva"].iloc[0] > 0.0


def test_ac_period_selection_combines_policies_and_explicit_periods() -> None:
    config = _two_bus_config()
    result = _SolvedDispatch(_dispatch_frame(active_mw=10.0))

    periods = select_ac_validation_periods(
        config,
        result,
        ACValidationOptions(
            policies=("peak_demand", "peak_renewable", "congestion"),
            periods=(0,),
            timestamps=("2026-01-01T01:00:00",),
        ),
    )

    assert periods == (0, 1, 2)
