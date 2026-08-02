from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from energy_system_simulator.config import (
    BatteryConfig,
    BusConfig,
    DemandConfig,
    HydroUnitConfig,
    ImportConfig,
    ImportResourceConfig,
    ModelConfig,
    NetworkConfig,
    RenewableGeneratorConfig,
    ReserveConfig,
    StorageUnitConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
    load_config,
)
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.simulation import ReliabilityReplication, ReliabilityStudy

ROOT = Path(__file__).resolve().parents[2]
ABS_TOL = 1e-6


@dataclass(frozen=True)
class EnumeratedUnit:
    """Independent thermal unit data used by brute-force checks."""

    unit_id: str
    minimum_output_mw: float
    maximum_output_mw: float
    variable_cost_eur_per_mwh: float
    no_load_cost_eur_per_hour: float
    startup_cost_eur: float


@dataclass(frozen=True)
class EnumeratedSolution:
    """Best solution found by independent binary commitment enumeration."""

    objective_eur: float
    commitment: tuple[tuple[int, ...], ...]
    dispatch_mw: tuple[tuple[float, ...], ...]


def _example_config() -> ModelConfig:
    return load_config(ROOT / "configs" / "example.yaml")


def _thermal(
    *,
    name: str = "thermal",
    minimum_output_mw: float = 0.0,
    maximum_output_mw: float,
    variable_cost_eur_per_mwh: float = 0.0,
    no_load_cost_eur_per_hour: float = 0.0,
    startup_cost_eur: float = 0.0,
    initial_on: bool = False,
    terminal_commitment_mode: Literal[
        "forbid_incomplete_transitions",
        "carry_residual_obligations",
        "fixed_terminal_commitment",
    ] = "forbid_incomplete_transitions",
) -> ThermalConfig:
    base = _example_config().thermal
    return replace(
        base,
        name=name,
        minimum_output_mw=minimum_output_mw,
        maximum_output_mw=maximum_output_mw,
        ramp_up_mw_per_hour=1_000.0,
        ramp_down_mw_per_hour=1_000.0,
        startup_ramp_mw=maximum_output_mw,
        shutdown_ramp_mw=maximum_output_mw,
        variable_cost_eur_per_mwh=variable_cost_eur_per_mwh,
        no_load_cost_eur_per_hour=no_load_cost_eur_per_hour,
        startup_cost_eur=startup_cost_eur,
        shutdown_cost_eur=0.0,
        emission_factor_tonnes_per_mwh=0.0,
        minimum_up_hours=1.0,
        minimum_down_hours=1.0,
        initial_on=initial_on,
        initial_output_mw=minimum_output_mw if initial_on else 0.0,
        initial_up_time_hours=10.0 if initial_on else 0.0,
        initial_down_time_hours=0.0 if initial_on else 10.0,
        terminal_commitment_mode=terminal_commitment_mode,
        terminal_on=None,
        minimum_fuel_input_mwh_per_hour=0.0,
        heat_rate_segments=(),
        startup_categories=(),
    )


def _battery(
    *,
    capacity_mwh: float = 0.0,
    power_mw: float = 0.0,
    initial_soc_mwh: float = 0.0,
    throughput_cost_eur_per_mwh: float = 0.0,
    terminal_soc_mode: Literal["minimum", "exact", "cyclic", "free"] = "free",
) -> BatteryConfig:
    base = _example_config().battery
    return replace(
        base,
        energy_capacity_mwh=capacity_mwh,
        power_capacity_mw=power_mw,
        charge_power_capacity_mw=None,
        discharge_power_capacity_mw=None,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=capacity_mwh,
        initial_soc_mwh=initial_soc_mwh,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        throughput_cost_eur_per_mwh=throughput_cost_eur_per_mwh,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode=terminal_soc_mode,
        self_discharge_rate_per_hour=0.0,
        minimum_charge_mw=0.0,
        minimum_discharge_mw=0.0,
        charge_ramp_mw_per_hour=None,
        discharge_ramp_mw_per_hour=None,
        degradation_bands=(),
    )


def _unit(unit_id: str, config: ThermalConfig, bus_id: str = "system") -> ThermalGeneratorConfig:
    return ThermalGeneratorConfig(id=unit_id, bus_id=bus_id, fuel_id="gas", config=config)


def _aggregate_config(
    *,
    thermal_units: tuple[ThermalGeneratorConfig, ...],
    storage_units: tuple[StorageUnitConfig, ...] = (),
    hydro_units: tuple[HydroUnitConfig, ...] = (),
    imports: ImportConfig | None = None,
    reserves: ReserveConfig | None = None,
    demand: tuple[DemandConfig, ...] | None = None,
) -> ModelConfig:
    base = _example_config()
    reserve_config = reserves or ReserveConfig()
    import_config = imports or ImportConfig(
        maximum_power_mw=0.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    demand_units = demand or (DemandConfig(id="load", bus_id="system", time_series_key="load_mw"),)
    return replace(
        base,
        thermal=thermal_units[0].config if thermal_units else _thermal(maximum_output_mw=0.0),
        battery=storage_units[0].config if storage_units else _battery(),
        imports=import_config,
        reserves=reserve_config,
        network=replace(base.network, loss_fraction=0.0, transfer_capacity_mw=1_000.0),
        penalties=replace(
            base.penalties,
            renewable_curtailment_eur_per_mwh=0.0,
            lost_load_eur_per_mwh=10_000.0,
            carbon_price_eur_per_tonne=0.0,
        ),
        portfolio=replace(
            base.portfolio,
            buses=(BusConfig(id="system", zone_id="zone"),),
            lines=(),
            renewable_generators=(),
            thermal_generators=thermal_units,
            storage_units=storage_units,
            hydro_units=hydro_units,
            imports=(ImportResourceConfig(id="imports", bus_id="system", config=import_config),),
            demand=demand_units,
        ),
    )


def _nodal_config(
    *,
    buses: tuple[str, ...],
    lines: tuple[TransmissionLineConfig, ...],
    renewable_bus: str,
    demand_units: tuple[DemandConfig, ...],
) -> ModelConfig:
    base = _aggregate_config(
        thermal_units=(_unit("offline", _thermal(maximum_output_mw=0.0)),),
    )
    import_config = ImportConfig(
        maximum_power_mw=0.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    return replace(
        base,
        network=NetworkConfig(
            loss_fraction=0.0,
            transfer_capacity_mw=1_000.0,
            network_mode=cast(Literal["aggregate", "nodal"], "nodal"),
            slack_bus_id=buses[0],
        ),
        imports=import_config,
        portfolio=replace(
            base.portfolio,
            buses=tuple(BusConfig(id=bus_id, zone_id="zone") for bus_id in buses),
            lines=lines,
            renewable_generators=(
                RenewableGeneratorConfig(
                    id="solar",
                    kind="solar",
                    bus_id=renewable_bus,
                    capacity_mw=1_000.0,
                    time_series_key="solar_mw",
                ),
            ),
            imports=(ImportResourceConfig(id="imports", bus_id=buses[0], config=import_config),),
            demand=demand_units,
        ),
    )


def _dispatch_for_commitment(
    units: tuple[EnumeratedUnit, ...],
    online: tuple[int, ...],
    demand_mw: float,
) -> tuple[float, ...] | None:
    generation = [0.0] * len(units)
    required = demand_mw
    for index, unit in enumerate(units):
        if online[index] == 0:
            continue
        generation[index] = unit.minimum_output_mw
        required -= unit.minimum_output_mw
    if required < -ABS_TOL:
        return None
    if required <= ABS_TOL:
        return tuple(generation)

    merit_order = sorted(
        (unit.variable_cost_eur_per_mwh, index) for index, unit in enumerate(units)
    )
    for _, index in merit_order:
        if online[index] == 0:
            continue
        headroom = units[index].maximum_output_mw - generation[index]
        addition = min(headroom, required)
        generation[index] += addition
        required -= addition
        if required <= ABS_TOL:
            return tuple(generation)
    return None


def _enumerate_commitment(
    units: tuple[EnumeratedUnit, ...],
    demand_mw: tuple[float, ...],
) -> EnumeratedSolution:
    best: EnumeratedSolution | None = None
    for flat in product((0, 1), repeat=len(units) * len(demand_mw)):
        commitment = tuple(
            tuple(flat[period * len(units) + unit] for unit in range(len(units)))
            for period in range(len(demand_mw))
        )
        total = 0.0
        dispatch: list[tuple[float, ...]] = []
        previous = (0,) * len(units)
        feasible = True
        for period, demand in enumerate(demand_mw):
            generation = _dispatch_for_commitment(units, commitment[period], demand)
            if generation is None:
                feasible = False
                break
            dispatch.append(generation)
            for unit_index, unit in enumerate(units):
                total += generation[unit_index] * unit.variable_cost_eur_per_mwh
                total += commitment[period][unit_index] * unit.no_load_cost_eur_per_hour
                if commitment[period][unit_index] and not previous[unit_index]:
                    total += unit.startup_cost_eur
            previous = commitment[period]
        if feasible and (best is None or total < best.objective_eur):
            best = EnumeratedSolution(total, commitment, tuple(dispatch))
    if best is None:
        raise AssertionError("No feasible enumerated commitment")
    return best


def test_single_period_economic_dispatch_matches_hand_merit_order() -> None:
    cheap = _unit(
        "cheap",
        _thermal(name="cheap", maximum_output_mw=50.0, variable_cost_eur_per_mwh=5.0),
    )
    peaker = _unit(
        "peaker",
        _thermal(name="peaker", maximum_output_mw=100.0, variable_cost_eur_per_mwh=50.0),
    )

    result = UnitCommitment(_aggregate_config(thermal_units=(cheap, peaker))).solve(
        np.zeros(1),
        np.array([80.0]),
    )

    assert result.frame["thermal_output_mw__cheap"].iloc[0] == pytest.approx(50.0)
    assert result.frame["thermal_output_mw__peaker"].iloc[0] == pytest.approx(30.0)
    assert result.objective_eur == pytest.approx(1_750.0)


def test_single_unit_commitment_matches_independent_enumeration() -> None:
    thermal = _thermal(
        maximum_output_mw=80.0,
        variable_cost_eur_per_mwh=20.0,
        no_load_cost_eur_per_hour=15.0,
        startup_cost_eur=100.0,
    )
    demand = (0.0, 50.0, 50.0)
    expected = _enumerate_commitment(
        (
            EnumeratedUnit(
                "unit",
                thermal.minimum_output_mw,
                thermal.maximum_output_mw,
                thermal.variable_cost_eur_per_mwh,
                thermal.no_load_cost_eur_per_hour,
                thermal.startup_cost_eur,
            ),
        ),
        demand,
    )

    result = UnitCommitment(_aggregate_config(thermal_units=(_unit("unit", thermal),))).solve(
        np.zeros(3),
        np.array(demand),
    )

    assert result.objective_eur == pytest.approx(expected.objective_eur)
    assert result.frame["thermal_on__unit"].tolist() == [row[0] for row in expected.commitment]


def test_two_unit_startup_cost_case_matches_independent_enumeration() -> None:
    cheap = _thermal(
        name="cheap",
        maximum_output_mw=50.0,
        variable_cost_eur_per_mwh=1.0,
        startup_cost_eur=1_000.0,
    )
    peaker = _thermal(
        name="peaker",
        maximum_output_mw=50.0,
        variable_cost_eur_per_mwh=50.0,
        startup_cost_eur=0.0,
    )
    demand = (10.0, 10.0)
    expected = _enumerate_commitment(
        (
            EnumeratedUnit("cheap", 0.0, 50.0, 1.0, 0.0, 1_000.0),
            EnumeratedUnit("peaker", 0.0, 50.0, 50.0, 0.0, 0.0),
        ),
        demand,
    )

    result = UnitCommitment(
        _aggregate_config(thermal_units=(_unit("cheap", cheap), _unit("peaker", peaker)))
    ).solve(np.zeros(2), np.array(demand))

    assert result.objective_eur == pytest.approx(expected.objective_eur)
    assert result.frame["thermal_on__cheap"].tolist() == [row[0] for row in expected.commitment]
    assert result.frame["thermal_on__peaker"].tolist() == [row[1] for row in expected.commitment]


def test_battery_arbitrage_matches_known_price_solution() -> None:
    imports = ImportConfig(
        maximum_power_mw=20.0,
        price_eur_per_mwh=0.0,
        emission_factor_tonnes_per_mwh=0.0,
    )
    battery = _battery(capacity_mwh=10.0, power_mw=10.0)
    config = _aggregate_config(
        thermal_units=(_unit("offline", _thermal(maximum_output_mw=0.0)),),
        storage_units=(StorageUnitConfig(id="battery", bus_id="system", config=battery),),
        imports=imports,
    )

    result = UnitCommitment(config).solve(
        np.zeros(2),
        np.array([10.0, 10.0]),
        import_price_series=np.array([10.0, 100.0]),
    )

    assert result.frame["imports_mw"].tolist() == pytest.approx([20.0, 0.0])
    assert result.frame["battery_charge_mw"].tolist() == pytest.approx([10.0, 0.0])
    assert result.frame["battery_discharge_mw"].tolist() == pytest.approx([0.0, 10.0])
    assert result.objective_eur == pytest.approx(200.0)


def test_reservoir_allocation_matches_two_period_water_value_solution() -> None:
    hydro = HydroUnitConfig(
        id="reservoir",
        bus_id="system",
        kind="reservoir",
        inflow_time_series_key="hydro_inflow_mw",
        turbine_capacity_mw=10.0,
        turbine_efficiency=1.0,
        minimum_reservoir_mwh=0.0,
        maximum_reservoir_mwh=10.0,
        initial_reservoir_mwh=10.0,
        minimum_final_reservoir_mwh=0.0,
        terminal_reservoir_mode="free",
        water_value_eur_per_mwh=75.0,
    )
    thermal = _thermal(maximum_output_mw=20.0, variable_cost_eur_per_mwh=50.0)
    result = UnitCommitment(
        _aggregate_config(thermal_units=(_unit("thermal", thermal),), hydro_units=(hydro,))
    ).solve(
        np.zeros(2),
        np.array([10.0, 10.0]),
        hydro_inflows_mw={"reservoir": np.zeros(2)},
    )

    assert result.frame["hydro_generation_mw__reservoir"].sum() == pytest.approx(0.0)
    assert result.frame["hydro_reservoir_mwh__reservoir"].iloc[-1] == pytest.approx(10.0)
    assert result.objective_eur == pytest.approx(250.0)


def test_two_bus_dc_opf_matches_known_congestion_result() -> None:
    line = TransmissionLineConfig(
        id="north_south",
        from_bus_id="north",
        to_bus_id="south",
        susceptance=10.0,
        capacity_mw=40.0,
    )
    demand = DemandConfig(id="load", bus_id="south", time_series_key="load_mw")
    config = _nodal_config(
        buses=("north", "south"),
        lines=(line,),
        renewable_bus="north",
        demand_units=(demand,),
    )

    result = UnitCommitment(config).solve(
        np.array([100.0]),
        np.array([100.0]),
        demand_profiles_mw={"load": np.array([100.0])},
        renewable_availability_by_asset_mw={"solar": np.array([100.0])},
    )

    assert result.frame["line_flow_mw__north_south"].iloc[0] == pytest.approx(40.0)
    assert result.frame["renewable_used_mw"].iloc[0] == pytest.approx(40.0)
    assert result.frame["source_load_shed_mw"].iloc[0] == pytest.approx(60.0)


def test_three_bus_loop_flow_matches_independent_matrix_calculation() -> None:
    config = _nodal_config(
        buses=("a", "b", "c"),
        lines=(
            TransmissionLineConfig(
                id="ab",
                from_bus_id="a",
                to_bus_id="b",
                susceptance=10.0,
                capacity_mw=100.0,
            ),
            TransmissionLineConfig(
                id="bc",
                from_bus_id="b",
                to_bus_id="c",
                susceptance=10.0,
                capacity_mw=100.0,
            ),
            TransmissionLineConfig(
                id="ac",
                from_bus_id="a",
                to_bus_id="c",
                susceptance=10.0,
                capacity_mw=100.0,
            ),
        ),
        renewable_bus="a",
        demand_units=(
            DemandConfig(id="load_b", bus_id="b", time_series_key="load_b_mw"),
            DemandConfig(id="load_c", bus_id="c", time_series_key="load_c_mw"),
        ),
    )

    result = UnitCommitment(config).solve(
        np.array([90.0]),
        np.array([90.0]),
        demand_profiles_mw={"load_b": np.array([45.0]), "load_c": np.array([45.0])},
        renewable_availability_by_asset_mw={"solar": np.array([90.0])},
    )

    assert result.frame["line_flow_mw__ab"].iloc[0] == pytest.approx(45.0, abs=ABS_TOL)
    assert result.frame["line_flow_mw__ac"].iloc[0] == pytest.approx(45.0, abs=ABS_TOL)
    assert result.frame["line_flow_mw__bc"].iloc[0] == pytest.approx(0.0, abs=ABS_TOL)


def test_reserve_procurement_matches_headroom_limit() -> None:
    thermal = _thermal(maximum_output_mw=100.0, variable_cost_eur_per_mwh=1.0)
    result = UnitCommitment(
        _aggregate_config(
            thermal_units=(_unit("thermal", thermal),),
            reserves=ReserveConfig(
                upward_fixed_mw=30.0,
                thermal_upward_cost_eur_per_mw_hour=2.0,
            ),
        )
    ).solve(np.zeros(1), np.array([60.0]))

    assert result.frame["thermal_output_mw__thermal"].iloc[0] == pytest.approx(60.0)
    assert result.frame["thermal_upward_reserve_mw__thermal"].iloc[0] == pytest.approx(30.0)
    assert result.frame["reserve_upward_shortfall_mw"].iloc[0] == pytest.approx(0.0)
    assert result.objective_eur == pytest.approx(120.0)


def test_demand_shifting_preserves_exact_energy() -> None:
    demand = DemandConfig(
        id="industrial",
        bus_id="system",
        time_series_key="industrial_mw",
        kind="shiftable",
        shift_up_capacity_mw=10.0,
        shift_down_capacity_mw=10.0,
    )
    config = _aggregate_config(
        thermal_units=(_unit("offline", _thermal(maximum_output_mw=0.0)),),
        demand=(demand,),
    )

    result = UnitCommitment(config).solve(
        np.array([10.0, 0.0]),
        np.array([0.0, 10.0]),
        demand_profiles_mw={"industrial": np.array([0.0, 10.0])},
    )

    shifted_up = float(result.frame["demand_shift_up_mw__industrial"].sum())
    shifted_down = float(result.frame["demand_shift_down_mw__industrial"].sum())
    assert shifted_up == pytest.approx(10.0)
    assert shifted_down == pytest.approx(10.0)
    assert shifted_up - shifted_down == pytest.approx(0.0)


def test_reliability_metrics_match_small_enumerated_outage_model() -> None:
    demand_mw = 80.0
    forced_outage_rate = 0.25
    expected_eue = demand_mw * forced_outage_rate
    replications = [
        ReliabilityReplication(
            index=0,
            seed=1,
            metrics={
                "expected_unserved_energy_mwh": 0.0,
                "loss_of_load_probability": 0.0,
                "expected_demand_not_served_mw": 0.0,
            },
            outage_hours_by_asset={"thermal": 0.0},
            outage_unserved_energy_mwh_by_type={"thermal": 0.0},
            failed=False,
            error="",
        ),
        ReliabilityReplication(
            index=1,
            seed=2,
            metrics={
                "expected_unserved_energy_mwh": demand_mw,
                "loss_of_load_probability": 1.0,
                "expected_demand_not_served_mw": demand_mw,
            },
            outage_hours_by_asset={"thermal": 1.0},
            outage_unserved_energy_mwh_by_type={"thermal": demand_mw},
            failed=False,
            error="",
        ),
        ReliabilityReplication(
            index=2,
            seed=3,
            metrics={
                "expected_unserved_energy_mwh": 0.0,
                "loss_of_load_probability": 0.0,
                "expected_demand_not_served_mw": 0.0,
            },
            outage_hours_by_asset={"thermal": 0.0},
            outage_unserved_energy_mwh_by_type={"thermal": 0.0},
            failed=False,
            error="",
        ),
        ReliabilityReplication(
            index=3,
            seed=4,
            metrics={
                "expected_unserved_energy_mwh": 0.0,
                "loss_of_load_probability": 0.0,
                "expected_demand_not_served_mw": 0.0,
            },
            outage_hours_by_asset={"thermal": 0.0},
            outage_unserved_energy_mwh_by_type={"thermal": 0.0},
            failed=False,
            error="",
        ),
    ]
    study = ReliabilityStudy.__new__(ReliabilityStudy)

    metrics = study._aggregate_metrics(replications, attempted=len(replications))
    attribution = study._aggregate_outage_attribution(replications)

    assert metrics["expected_unserved_energy_mwh"] == pytest.approx(expected_eue)
    assert metrics["loss_of_load_probability"] == pytest.approx(forced_outage_rate)
    assert metrics["expected_demand_not_served_mw"] == pytest.approx(expected_eue)
    assert attribution["thermal"] == pytest.approx(expected_eue)
