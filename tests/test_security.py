from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_system_simulator.config import (
    BusConfig,
    DemandConfig,
    ImportConfig,
    ImportResourceConfig,
    ModelConfig,
    NetworkConfig,
    ReserveConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.security import (
    Contingency,
    SecurityOptions,
    evaluate_security,
    explicit_line_outage_flows,
    lodf_line_outage_flows,
)


@dataclass(frozen=True)
class _SolvedDispatch:
    timeseries: pd.DataFrame
    objective_eur: float


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal(
    *,
    maximum_output_mw: float,
    variable_cost_eur_per_mwh: float = 1.0,
    no_load_cost_eur_per_hour: float = 0.0,
) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        minimum_output_mw=0.0,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=1_000.0,
        ramp_down_mw_per_hour=1_000.0,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=variable_cost_eur_per_mwh,
        no_load_cost_eur_per_hour=no_load_cost_eur_per_hour,
        startup_cost_eur=0.0,
        shutdown_cost_eur=0.0,
        emission_factor_tonnes_per_mwh=0.0,
        minimum_up_hours=1.0,
        minimum_down_hours=1.0,
        initial_on=False,
        initial_output_mw=0.0,
        initial_up_time_hours=0.0,
        initial_down_time_hours=10.0,
        terminal_commitment_mode="forbid_incomplete_transitions",
        terminal_on=None,
    )


def _unit(
    unit_id: str,
    bus_id: str,
    thermal: ThermalConfig,
) -> ThermalGeneratorConfig:
    return ThermalGeneratorConfig(id=unit_id, bus_id=bus_id, fuel_id="gas", config=thermal)


def _config(
    *,
    buses: tuple[str, ...],
    lines: tuple[TransmissionLineConfig, ...],
    thermal: tuple[ThermalGeneratorConfig, ...],
    demand: tuple[DemandConfig, ...],
    reserves: ReserveConfig | None = None,
) -> ModelConfig:
    base = _example_config()
    imports = ImportConfig(
        maximum_power_mw=0.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    return replace(
        base,
        thermal=thermal[0].config,
        battery=replace(
            base.battery,
            energy_capacity_mwh=0.0,
            power_capacity_mw=0.0,
            minimum_soc_mwh=0.0,
            maximum_soc_mwh=0.0,
            initial_soc_mwh=0.0,
            minimum_final_soc_mwh=0.0,
            terminal_soc_mode="free",
        ),
        imports=imports,
        reserves=reserves or ReserveConfig(),
        network=NetworkConfig(
            loss_fraction=0.0,
            transfer_capacity_mw=1_000.0,
            network_mode="nodal",
            slack_bus_id=buses[0],
        ),
        penalties=replace(
            base.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
        portfolio=replace(
            base.portfolio,
            buses=tuple(BusConfig(id=bus_id, zone_id="zone") for bus_id in buses),
            lines=lines,
            renewable_generators=(),
            thermal_generators=thermal,
            storage_units=(),
            hydro_units=(),
            imports=(ImportResourceConfig(id="imports", bus_id=buses[0], config=imports),),
            demand=demand,
        ),
    )


def _solve(
    config: ModelConfig,
    *,
    demand_profiles_mw: dict[str, np.ndarray],
) -> _SolvedDispatch:
    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(1, dtype=float),
        gross_demand_mw=sum(demand_profiles_mw.values()),
        demand_profiles_mw=demand_profiles_mw,
        renewable_availability_by_asset_mw={},
    )
    return _SolvedDispatch(timeseries=result.frame, objective_eur=result.objective_eur)


def _triangle_config(*, line_capacity_mw: float = 100.0) -> ModelConfig:
    thermal = (_unit("thermal", "a", _thermal(maximum_output_mw=100.0)),)
    return _config(
        buses=("a", "b", "c"),
        lines=(
            TransmissionLineConfig(
                id="ab",
                from_bus_id="a",
                to_bus_id="b",
                susceptance=10.0,
                capacity_mw=line_capacity_mw,
            ),
            TransmissionLineConfig(
                id="bc",
                from_bus_id="b",
                to_bus_id="c",
                susceptance=10.0,
                capacity_mw=line_capacity_mw,
            ),
            TransmissionLineConfig(
                id="ac",
                from_bus_id="a",
                to_bus_id="c",
                susceptance=10.0,
                capacity_mw=line_capacity_mw,
            ),
        ),
        thermal=thermal,
        demand=(
            DemandConfig(id="load_b", bus_id="b", time_series_key="load_b_mw"),
            DemandConfig(id="load_c", bus_id="c", time_series_key="load_c_mw"),
        ),
    )


def test_three_bus_line_outage_uses_alternative_path() -> None:
    config = _triangle_config()
    result = _solve(
        config,
        demand_profiles_mw={
            "load_b": np.array([45.0], dtype=float),
            "load_c": np.array([45.0], dtype=float),
        },
    )

    evaluation = evaluate_security(
        config,
        result,
        contingencies=(Contingency(id="line:ab", kind="line", asset_id="ab"),),
        options=SecurityOptions(allow_emergency_actions=False),
    )

    assert evaluation.secure
    record = evaluation.records.iloc[0]
    assert record["emergency_load_shed_mw"] == pytest.approx(0.0)
    assert record["emergency_overload_mw"] == pytest.approx(0.0)


def test_generator_outage_uses_procured_upward_reserve() -> None:
    cheap = _unit(
        "cheap",
        "north",
        _thermal(maximum_output_mw=50.0, variable_cost_eur_per_mwh=1.0),
    )
    peaker = _unit(
        "peaker",
        "north",
        _thermal(
            maximum_output_mw=50.0,
            variable_cost_eur_per_mwh=100.0,
            no_load_cost_eur_per_hour=10.0,
        ),
    )
    config = _config(
        buses=("north", "south"),
        lines=(
            TransmissionLineConfig(
                id="north_south",
                from_bus_id="north",
                to_bus_id="south",
                susceptance=10.0,
                capacity_mw=100.0,
            ),
        ),
        thermal=(cheap, peaker),
        demand=(DemandConfig(id="load", bus_id="south", time_series_key="load_mw"),),
        reserves=ReserveConfig(upward_fixed_mw=50.0),
    )
    result = _solve(config, demand_profiles_mw={"load": np.array([50.0], dtype=float)})

    evaluation = evaluate_security(
        config,
        result,
        contingencies=(Contingency(id="generator:cheap", kind="generator", asset_id="cheap"),),
        options=SecurityOptions(allow_emergency_actions=False),
    )

    assert evaluation.secure
    assert result.timeseries["thermal_upward_reserve_mw__peaker"].iloc[0] == pytest.approx(50.0)
    assert evaluation.records["redispatch_up_mw"].iloc[0] == pytest.approx(50.0)


def test_hard_security_fails_without_committed_reserve() -> None:
    cheap = _unit(
        "cheap",
        "north",
        _thermal(maximum_output_mw=50.0, variable_cost_eur_per_mwh=1.0),
    )
    peaker = _unit(
        "peaker",
        "north",
        _thermal(
            maximum_output_mw=50.0,
            variable_cost_eur_per_mwh=100.0,
            no_load_cost_eur_per_hour=10.0,
        ),
    )
    config = _config(
        buses=("north", "south"),
        lines=(
            TransmissionLineConfig(
                id="north_south",
                from_bus_id="north",
                to_bus_id="south",
                susceptance=10.0,
                capacity_mw=100.0,
            ),
        ),
        thermal=(cheap, peaker),
        demand=(DemandConfig(id="load", bus_id="south", time_series_key="load_mw"),),
    )
    result = _solve(config, demand_profiles_mw={"load": np.array([50.0], dtype=float)})

    evaluation = evaluate_security(
        config,
        result,
        contingencies=(Contingency(id="generator:cheap", kind="generator", asset_id="cheap"),),
        options=SecurityOptions(allow_emergency_actions=False),
    )

    assert not evaluation.secure
    assert evaluation.records["solver_status"].iloc[0] == "infeasible"


def test_lodf_line_outage_matches_explicit_dc_solve() -> None:
    config = _triangle_config()
    result = _solve(
        config,
        demand_profiles_mw={
            "load_b": np.array([45.0], dtype=float),
            "load_c": np.array([45.0], dtype=float),
        },
    )
    row = result.timeseries.iloc[0]
    base_flows = {
        line.id: float(row[f"line_flow_mw__{line.id}"]) for line in config.portfolio.lines
    }
    bus_injections = {
        bus.id: float(row[f"bus_net_injection_mw__{bus.id}"]) for bus in config.portfolio.buses
    }

    lodf = lodf_line_outage_flows(config, base_flows, failed_line_id="ab")
    explicit = explicit_line_outage_flows(config, bus_injections, failed_line_id="ab")

    assert lodf["ab"] == pytest.approx(0.0)
    for line_id in ("bc", "ac"):
        assert lodf[line_id] == pytest.approx(explicit[line_id], abs=1e-6)
