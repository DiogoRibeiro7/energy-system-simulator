from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from energy_system_simulator.exceptions import ConfigurationError, OptimisationError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HeatBoilerConfig:
    id: str
    heat_capacity_mw: float
    efficiency_mwh_heat_per_mwh_fuel: float
    fuel_cost_eur_per_mwh: float = 0.0
    variable_cost_eur_per_mwh_heat: float = 0.0
    emission_tonnes_per_mwh_fuel: float = 0.0


@dataclass(frozen=True)
class ElectricBoilerConfig:
    id: str
    heat_capacity_mw: float
    efficiency_mwh_heat_per_mwh_electric: float
    variable_cost_eur_per_mwh_heat: float = 0.0


@dataclass(frozen=True)
class HeatPumpConfig:
    id: str
    heat_capacity_mw: float
    cop: npt.ArrayLike
    variable_cost_eur_per_mwh_heat: float = 0.0


@dataclass(frozen=True)
class ThermalStorageConfig:
    id: str
    energy_capacity_mwh_th: float
    charge_capacity_mw_th: float
    discharge_capacity_mw_th: float
    initial_inventory_mwh_th: float = 0.0
    minimum_final_inventory_mwh_th: float = 0.0
    standing_loss_fraction_per_hour: float = 0.0


@dataclass(frozen=True)
class CHPVertex:
    id: str
    electric_output_mw: float
    heat_output_mw_th: float
    fuel_input_mwh_per_hour: float


@dataclass(frozen=True)
class CHPUnitConfig:
    id: str
    vertices: tuple[CHPVertex, ...]
    fuel_cost_eur_per_mwh: float = 0.0
    variable_cost_eur_per_mwh_fuel: float = 0.0
    emission_tonnes_per_mwh_fuel: float = 0.0


@dataclass(frozen=True)
class HeatSystemProblem:
    periods: tuple[str, ...]
    heat_demand_mw_th: npt.ArrayLike
    electricity_demand_mw: npt.ArrayLike = 0.0
    time_step_hours: float = 1.0
    network_delivery_efficiency: float = 1.0
    electricity_purchase_price_eur_per_mwh: npt.ArrayLike = 0.0
    electricity_export_value_eur_per_mwh: npt.ArrayLike = 0.0
    heat_shortage_penalty_eur_per_mwh: float = 10_000.0
    electricity_shortage_penalty_eur_per_mwh: float = 10_000.0
    heat_dumping_cost_eur_per_mwh: float = 0.0
    carbon_price_eur_per_tonne: float = 0.0
    boilers: tuple[HeatBoilerConfig, ...] = ()
    electric_boilers: tuple[ElectricBoilerConfig, ...] = ()
    heat_pumps: tuple[HeatPumpConfig, ...] = ()
    chp_units: tuple[CHPUnitConfig, ...] = ()
    storage: ThermalStorageConfig = field(
        default_factory=lambda: ThermalStorageConfig(
            id="thermal-store",
            energy_capacity_mwh_th=0.0,
            charge_capacity_mw_th=0.0,
            discharge_capacity_mw_th=0.0,
        )
    )


@dataclass(frozen=True)
class HeatSystemResult:
    timeseries: pd.DataFrame
    summary: dict[str, Any]
    solver_message: str

    def write(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        self.timeseries.to_csv(output_directory / "heat_timeseries.csv", index=False)
        (output_directory / "heat_summary.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class HeatSystemModel:
    """Continuous district-heat and CHP co-optimisation model."""

    def solve(self, problem: HeatSystemProblem) -> HeatSystemResult:
        _validate_problem(problem)
        data = _PreparedHeat.from_problem(problem)
        registry = _Registry()
        registry.add("storage_charge_mw_th", data.periods)
        registry.add("storage_discharge_mw_th", data.periods)
        registry.add("storage_inventory_mwh_th", data.periods)
        registry.add("heat_dump_mw_th", data.periods)
        registry.add("unmet_heat_mw_th", data.periods)
        registry.add("electricity_purchase_mw", data.periods)
        registry.add("electricity_export_mw", data.periods)
        registry.add("electricity_shortage_mw", data.periods)
        for fuel_boiler in data.boilers:
            registry.add(_asset("boiler_heat_mw_th", fuel_boiler.id), data.periods)
            registry.add(_asset("boiler_fuel_mwh_per_hour", fuel_boiler.id), data.periods)
        for electric_boiler in data.electric_boilers:
            registry.add(_asset("electric_boiler_heat_mw_th", electric_boiler.id), data.periods)
            registry.add(_asset("electric_boiler_electric_mw", electric_boiler.id), data.periods)
        for pump in data.heat_pumps:
            registry.add(_asset("heat_pump_heat_mw_th", pump.id), data.periods)
            registry.add(_asset("heat_pump_electric_mw", pump.id), data.periods)
        for unit in data.chp_units:
            registry.add(_asset("chp_power_mw", unit.id), data.periods)
            registry.add(_asset("chp_heat_mw_th", unit.id), data.periods)
            registry.add(_asset("chp_fuel_mwh_per_hour", unit.id), data.periods)
            for vertex in unit.vertices:
                registry.add(_asset(f"chp_weight__{unit.id}", vertex.id), data.periods)

        objective = _objective(data, registry)
        lower, upper = _bounds(data, registry)
        equality, equality_rhs, inequality, inequality_rhs = _constraints(data, registry)
        result = linprog(
            c=objective,
            A_eq=equality,
            b_eq=equality_rhs,
            A_ub=inequality,
            b_ub=inequality_rhs,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        if result.status != 0 or result.x is None:
            raise OptimisationError(f"Heat system optimisation failed: {result.message}")
        solution = np.asarray(result.x, dtype=np.float64)
        return _result(data, registry, solution, float(result.fun), str(result.message))


def load_heat_problem(path: Path | str) -> HeatSystemProblem:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Heat study YAML must contain a mapping")
    if int(payload.get("schema_version", 1)) != 1:
        raise ConfigurationError("Heat study schema_version must be 1")
    periods = tuple(str(item) for item in payload.get("periods", ["0"]))
    if not periods:
        raise ConfigurationError("Heat study must define at least one period")
    problem = HeatSystemProblem(
        periods=periods,
        heat_demand_mw_th=_profile(payload.get("heat_demand_mw_th", 0.0), len(periods)),
        electricity_demand_mw=_profile(payload.get("electricity_demand_mw", 0.0), len(periods)),
        time_step_hours=_positive(payload, "time_step_hours", 1.0),
        network_delivery_efficiency=_fraction(payload, "network_delivery_efficiency", 1.0),
        electricity_purchase_price_eur_per_mwh=_finite_profile(
            payload.get("electricity_purchase_price_eur_per_mwh", 0.0),
            len(periods),
        ),
        electricity_export_value_eur_per_mwh=_finite_profile(
            payload.get("electricity_export_value_eur_per_mwh", 0.0),
            len(periods),
        ),
        heat_shortage_penalty_eur_per_mwh=float(
            payload.get("heat_shortage_penalty_eur_per_mwh", 10_000.0)
        ),
        electricity_shortage_penalty_eur_per_mwh=float(
            payload.get("electricity_shortage_penalty_eur_per_mwh", 10_000.0)
        ),
        heat_dumping_cost_eur_per_mwh=float(payload.get("heat_dumping_cost_eur_per_mwh", 0.0)),
        carbon_price_eur_per_tonne=float(payload.get("carbon_price_eur_per_tonne", 0.0)),
        boilers=tuple(_load_boiler(item) for item in payload.get("boilers", []) or []),
        electric_boilers=tuple(
            _load_electric_boiler(item) for item in payload.get("electric_boilers", []) or []
        ),
        heat_pumps=tuple(_load_heat_pump(item) for item in payload.get("heat_pumps", []) or []),
        chp_units=tuple(_load_chp_unit(item) for item in payload.get("chp_units", []) or []),
        storage=_load_storage(payload.get("storage") or {}),
    )
    _validate_problem(problem)
    return problem


def run_heat_study(problem: HeatSystemProblem) -> HeatSystemResult:
    _validate_problem(problem)
    return HeatSystemModel().solve(problem)


class _Registry:
    def __init__(self) -> None:
        self._blocks: dict[str, tuple[int, int]] = {}
        self.size = 0

    def add(self, name: str, size: int) -> None:
        self._blocks[name] = (self.size, size)
        self.size += size

    def at(self, name: str, period: int) -> int:
        offset, size = self._blocks[name]
        if period < 0 or period >= size:
            raise IndexError(period)
        return offset + period

    def values(self, solution: FloatArray, name: str) -> FloatArray:
        offset, size = self._blocks[name]
        return solution[offset : offset + size]


@dataclass(frozen=True)
class _PreparedHeat:
    periods: int
    labels: tuple[str, ...]
    heat_demand_mw_th: FloatArray
    electricity_demand_mw: FloatArray
    time_step_hours: float
    network_delivery_efficiency: float
    electricity_purchase_price_eur_per_mwh: FloatArray
    electricity_export_value_eur_per_mwh: FloatArray
    heat_shortage_penalty_eur_per_mwh: float
    electricity_shortage_penalty_eur_per_mwh: float
    heat_dumping_cost_eur_per_mwh: float
    carbon_price_eur_per_tonne: float
    boilers: tuple[HeatBoilerConfig, ...]
    electric_boilers: tuple[ElectricBoilerConfig, ...]
    heat_pumps: tuple[HeatPumpConfig, ...]
    heat_pump_cop: dict[str, FloatArray]
    chp_units: tuple[CHPUnitConfig, ...]
    storage: ThermalStorageConfig

    @classmethod
    def from_problem(cls, problem: HeatSystemProblem) -> _PreparedHeat:
        periods = len(problem.periods)
        return cls(
            periods=periods,
            labels=problem.periods,
            heat_demand_mw_th=_profile(problem.heat_demand_mw_th, periods),
            electricity_demand_mw=_profile(problem.electricity_demand_mw, periods),
            time_step_hours=problem.time_step_hours,
            network_delivery_efficiency=problem.network_delivery_efficiency,
            electricity_purchase_price_eur_per_mwh=_finite_profile(
                problem.electricity_purchase_price_eur_per_mwh,
                periods,
            ),
            electricity_export_value_eur_per_mwh=_finite_profile(
                problem.electricity_export_value_eur_per_mwh,
                periods,
            ),
            heat_shortage_penalty_eur_per_mwh=problem.heat_shortage_penalty_eur_per_mwh,
            electricity_shortage_penalty_eur_per_mwh=(
                problem.electricity_shortage_penalty_eur_per_mwh
            ),
            heat_dumping_cost_eur_per_mwh=problem.heat_dumping_cost_eur_per_mwh,
            carbon_price_eur_per_tonne=problem.carbon_price_eur_per_tonne,
            boilers=problem.boilers,
            electric_boilers=problem.electric_boilers,
            heat_pumps=problem.heat_pumps,
            heat_pump_cop={
                pump.id: _positive_profile(pump.cop, periods, f"heat_pumps.{pump.id}.cop")
                for pump in problem.heat_pumps
            },
            chp_units=problem.chp_units,
            storage=problem.storage,
        )


def _objective(data: _PreparedHeat, registry: _Registry) -> FloatArray:
    objective = np.zeros(registry.size, dtype=np.float64)
    dt = data.time_step_hours
    for t in range(data.periods):
        objective[registry.at("electricity_purchase_mw", t)] = (
            data.electricity_purchase_price_eur_per_mwh[t] * dt
        )
        objective[registry.at("electricity_export_mw", t)] = (
            -data.electricity_export_value_eur_per_mwh[t] * dt
        )
        objective[registry.at("electricity_shortage_mw", t)] = (
            data.electricity_shortage_penalty_eur_per_mwh * dt
        )
        objective[registry.at("unmet_heat_mw_th", t)] = data.heat_shortage_penalty_eur_per_mwh * dt
        objective[registry.at("heat_dump_mw_th", t)] = data.heat_dumping_cost_eur_per_mwh * dt
        for fuel_boiler in data.boilers:
            fuel_cost = fuel_boiler.fuel_cost_eur_per_mwh
            fuel_cost += fuel_boiler.emission_tonnes_per_mwh_fuel * data.carbon_price_eur_per_tonne
            objective[registry.at(_asset("boiler_fuel_mwh_per_hour", fuel_boiler.id), t)] = (
                fuel_cost * dt
            )
            objective[registry.at(_asset("boiler_heat_mw_th", fuel_boiler.id), t)] = (
                fuel_boiler.variable_cost_eur_per_mwh_heat * dt
            )
        for electric_boiler in data.electric_boilers:
            objective[registry.at(_asset("electric_boiler_heat_mw_th", electric_boiler.id), t)] = (
                electric_boiler.variable_cost_eur_per_mwh_heat * dt
            )
        for pump in data.heat_pumps:
            objective[registry.at(_asset("heat_pump_heat_mw_th", pump.id), t)] = (
                pump.variable_cost_eur_per_mwh_heat * dt
            )
        for unit in data.chp_units:
            fuel_cost = unit.fuel_cost_eur_per_mwh + unit.variable_cost_eur_per_mwh_fuel
            fuel_cost += unit.emission_tonnes_per_mwh_fuel * data.carbon_price_eur_per_tonne
            objective[registry.at(_asset("chp_fuel_mwh_per_hour", unit.id), t)] = fuel_cost * dt
    return objective


def _bounds(data: _PreparedHeat, registry: _Registry) -> tuple[FloatArray, FloatArray]:
    lower = np.zeros(registry.size, dtype=np.float64)
    upper = np.full(registry.size, np.inf, dtype=np.float64)
    for t in range(data.periods):
        upper[registry.at("storage_charge_mw_th", t)] = data.storage.charge_capacity_mw_th
        upper[registry.at("storage_discharge_mw_th", t)] = data.storage.discharge_capacity_mw_th
        upper[registry.at("storage_inventory_mwh_th", t)] = data.storage.energy_capacity_mwh_th
        upper[registry.at("unmet_heat_mw_th", t)] = data.heat_demand_mw_th[t]
        upper[registry.at("electricity_shortage_mw", t)] = data.electricity_demand_mw[t]
        for fuel_boiler in data.boilers:
            upper[registry.at(_asset("boiler_heat_mw_th", fuel_boiler.id), t)] = (
                fuel_boiler.heat_capacity_mw
            )
            upper[registry.at(_asset("boiler_fuel_mwh_per_hour", fuel_boiler.id), t)] = (
                fuel_boiler.heat_capacity_mw / fuel_boiler.efficiency_mwh_heat_per_mwh_fuel
            )
        for electric_boiler in data.electric_boilers:
            upper[registry.at(_asset("electric_boiler_heat_mw_th", electric_boiler.id), t)] = (
                electric_boiler.heat_capacity_mw
            )
            upper[registry.at(_asset("electric_boiler_electric_mw", electric_boiler.id), t)] = (
                electric_boiler.heat_capacity_mw
                / electric_boiler.efficiency_mwh_heat_per_mwh_electric
            )
        for pump in data.heat_pumps:
            upper[registry.at(_asset("heat_pump_heat_mw_th", pump.id), t)] = pump.heat_capacity_mw
            upper[registry.at(_asset("heat_pump_electric_mw", pump.id), t)] = (
                pump.heat_capacity_mw / data.heat_pump_cop[pump.id][t]
            )
        for unit in data.chp_units:
            for vertex in unit.vertices:
                upper[registry.at(_asset(f"chp_weight__{unit.id}", vertex.id), t)] = 1.0
    return lower, upper


def _constraints(
    data: _PreparedHeat,
    registry: _Registry,
) -> tuple[coo_matrix, FloatArray, coo_matrix, FloatArray]:
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    eq_rhs: list[float] = []
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    ub_rhs: list[float] = []

    def add_eq(terms: Mapping[int, float], rhs: float) -> None:
        row = len(eq_rhs)
        for col, value in terms.items():
            eq_rows.append(row)
            eq_cols.append(col)
            eq_vals.append(value)
        eq_rhs.append(rhs)

    def add_ub(terms: Mapping[int, float], rhs: float) -> None:
        row = len(ub_rhs)
        for col, value in terms.items():
            ub_rows.append(row)
            ub_cols.append(col)
            ub_vals.append(value)
        ub_rhs.append(rhs)

    dt = data.time_step_hours
    retention = (1.0 - data.storage.standing_loss_fraction_per_hour) ** dt
    eta_network = data.network_delivery_efficiency
    for t in range(data.periods):
        heat_terms: dict[int, float] = {
            registry.at("storage_discharge_mw_th", t): 1.0,
            registry.at("unmet_heat_mw_th", t): 1.0 / eta_network,
            registry.at("storage_charge_mw_th", t): -1.0,
            registry.at("heat_dump_mw_th", t): -1.0,
        }
        electricity_terms: dict[int, float] = {
            registry.at("electricity_purchase_mw", t): 1.0,
            registry.at("electricity_shortage_mw", t): 1.0,
            registry.at("electricity_export_mw", t): -1.0,
        }
        export_limit_terms: dict[int, float] = {
            registry.at("electricity_export_mw", t): 1.0,
        }
        for fuel_boiler in data.boilers:
            heat = registry.at(_asset("boiler_heat_mw_th", fuel_boiler.id), t)
            fuel = registry.at(_asset("boiler_fuel_mwh_per_hour", fuel_boiler.id), t)
            heat_terms[heat] = 1.0
            add_eq({heat: 1.0, fuel: -fuel_boiler.efficiency_mwh_heat_per_mwh_fuel}, 0.0)
        for electric_boiler in data.electric_boilers:
            heat = registry.at(_asset("electric_boiler_heat_mw_th", electric_boiler.id), t)
            power = registry.at(_asset("electric_boiler_electric_mw", electric_boiler.id), t)
            heat_terms[heat] = 1.0
            electricity_terms[power] = -1.0
            add_eq(
                {
                    heat: 1.0,
                    power: -electric_boiler.efficiency_mwh_heat_per_mwh_electric,
                },
                0.0,
            )
        for pump in data.heat_pumps:
            heat = registry.at(_asset("heat_pump_heat_mw_th", pump.id), t)
            power = registry.at(_asset("heat_pump_electric_mw", pump.id), t)
            heat_terms[heat] = 1.0
            electricity_terms[power] = -1.0
            add_eq({heat: 1.0, power: -data.heat_pump_cop[pump.id][t]}, 0.0)
        for unit in data.chp_units:
            power = registry.at(_asset("chp_power_mw", unit.id), t)
            heat = registry.at(_asset("chp_heat_mw_th", unit.id), t)
            fuel = registry.at(_asset("chp_fuel_mwh_per_hour", unit.id), t)
            heat_terms[heat] = 1.0
            electricity_terms[power] = 1.0
            export_limit_terms[power] = -1.0
            power_terms = {power: 1.0}
            heat_link_terms = {heat: 1.0}
            fuel_terms = {fuel: 1.0}
            weight_sum_terms: dict[int, float] = {}
            for vertex in unit.vertices:
                weight = registry.at(_asset(f"chp_weight__{unit.id}", vertex.id), t)
                power_terms[weight] = -vertex.electric_output_mw
                heat_link_terms[weight] = -vertex.heat_output_mw_th
                fuel_terms[weight] = -vertex.fuel_input_mwh_per_hour
                weight_sum_terms[weight] = 1.0
            add_eq(power_terms, 0.0)
            add_eq(heat_link_terms, 0.0)
            add_eq(fuel_terms, 0.0)
            add_ub(weight_sum_terms, 1.0)
        add_eq(heat_terms, data.heat_demand_mw_th[t] / eta_network)
        add_eq(electricity_terms, data.electricity_demand_mw[t])
        add_ub(export_limit_terms, 0.0)

        previous = (
            data.storage.initial_inventory_mwh_th
            if t == 0
            else registry.at("storage_inventory_mwh_th", t - 1)
        )
        inventory_terms = {
            registry.at("storage_inventory_mwh_th", t): 1.0,
            registry.at("storage_charge_mw_th", t): -dt,
            registry.at("storage_discharge_mw_th", t): dt,
        }
        if isinstance(previous, int):
            inventory_terms[previous] = -retention
            rhs = 0.0
        else:
            rhs = retention * previous
        add_eq(inventory_terms, rhs)
    add_ub(
        {registry.at("storage_inventory_mwh_th", data.periods - 1): -1.0},
        -data.storage.minimum_final_inventory_mwh_th,
    )
    return (
        coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(len(eq_rhs), registry.size)),
        np.asarray(eq_rhs, dtype=np.float64),
        coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(len(ub_rhs), registry.size)),
        np.asarray(ub_rhs, dtype=np.float64),
    )


def _result(
    data: _PreparedHeat,
    registry: _Registry,
    solution: FloatArray,
    objective_eur: float,
    solver_message: str,
) -> HeatSystemResult:
    dt = data.time_step_hours
    frame = pd.DataFrame(
        {
            "period": data.labels,
            "heat_demand_mw_th": data.heat_demand_mw_th,
            "heat_source_requirement_mw_th": (
                data.heat_demand_mw_th / data.network_delivery_efficiency
            ),
            "electricity_demand_mw": data.electricity_demand_mw,
            "storage_charge_mw_th": registry.values(solution, "storage_charge_mw_th"),
            "storage_discharge_mw_th": registry.values(solution, "storage_discharge_mw_th"),
            "storage_inventory_mwh_th": registry.values(solution, "storage_inventory_mwh_th"),
            "heat_dump_mw_th": registry.values(solution, "heat_dump_mw_th"),
            "unmet_heat_mw_th": registry.values(solution, "unmet_heat_mw_th"),
            "electricity_purchase_mw": registry.values(solution, "electricity_purchase_mw"),
            "electricity_export_mw": registry.values(solution, "electricity_export_mw"),
            "electricity_shortage_mw": registry.values(solution, "electricity_shortage_mw"),
        }
    )
    frame["boiler_heat_mw_th"] = 0.0
    frame["boiler_fuel_mwh_per_hour"] = 0.0
    frame["boiler_emissions_tonnes"] = 0.0
    for fuel_boiler in data.boilers:
        heat = registry.values(solution, _asset("boiler_heat_mw_th", fuel_boiler.id))
        fuel = registry.values(solution, _asset("boiler_fuel_mwh_per_hour", fuel_boiler.id))
        frame[f"boiler_heat_mw_th__{fuel_boiler.id}"] = heat
        frame[f"boiler_fuel_mwh_per_hour__{fuel_boiler.id}"] = fuel
        frame["boiler_heat_mw_th"] += heat
        frame["boiler_fuel_mwh_per_hour"] += fuel
        frame["boiler_emissions_tonnes"] += fuel * fuel_boiler.emission_tonnes_per_mwh_fuel * dt
    frame["electric_boiler_heat_mw_th"] = 0.0
    frame["electric_boiler_electric_mw"] = 0.0
    for electric_boiler in data.electric_boilers:
        heat = registry.values(
            solution,
            _asset("electric_boiler_heat_mw_th", electric_boiler.id),
        )
        power = registry.values(
            solution,
            _asset("electric_boiler_electric_mw", electric_boiler.id),
        )
        frame[f"electric_boiler_heat_mw_th__{electric_boiler.id}"] = heat
        frame[f"electric_boiler_electric_mw__{electric_boiler.id}"] = power
        frame["electric_boiler_heat_mw_th"] += heat
        frame["electric_boiler_electric_mw"] += power
    frame["heat_pump_heat_mw_th"] = 0.0
    frame["heat_pump_electric_mw"] = 0.0
    for pump in data.heat_pumps:
        heat = registry.values(solution, _asset("heat_pump_heat_mw_th", pump.id))
        power = registry.values(solution, _asset("heat_pump_electric_mw", pump.id))
        frame[f"heat_pump_heat_mw_th__{pump.id}"] = heat
        frame[f"heat_pump_electric_mw__{pump.id}"] = power
        frame["heat_pump_heat_mw_th"] += heat
        frame["heat_pump_electric_mw"] += power
    frame["chp_power_mw"] = 0.0
    frame["chp_heat_mw_th"] = 0.0
    frame["chp_fuel_mwh_per_hour"] = 0.0
    frame["chp_emissions_tonnes"] = 0.0
    for unit in data.chp_units:
        power = registry.values(solution, _asset("chp_power_mw", unit.id))
        heat = registry.values(solution, _asset("chp_heat_mw_th", unit.id))
        fuel = registry.values(solution, _asset("chp_fuel_mwh_per_hour", unit.id))
        frame[f"chp_power_mw__{unit.id}"] = power
        frame[f"chp_heat_mw_th__{unit.id}"] = heat
        frame[f"chp_fuel_mwh_per_hour__{unit.id}"] = fuel
        frame["chp_power_mw"] += power
        frame["chp_heat_mw_th"] += heat
        frame["chp_fuel_mwh_per_hour"] += fuel
        frame["chp_emissions_tonnes"] += fuel * unit.emission_tonnes_per_mwh_fuel * dt
    heat_supply = (
        frame["boiler_heat_mw_th"]
        + frame["electric_boiler_heat_mw_th"]
        + frame["heat_pump_heat_mw_th"]
        + frame["chp_heat_mw_th"]
        + frame["storage_discharge_mw_th"]
        + frame["unmet_heat_mw_th"] / data.network_delivery_efficiency
    )
    heat_uses = (
        frame["heat_source_requirement_mw_th"]
        + frame["storage_charge_mw_th"]
        + frame["heat_dump_mw_th"]
    )
    frame["heat_balance_residual_mw_th"] = heat_supply - heat_uses
    electricity_supply = (
        frame["chp_power_mw"] + frame["electricity_purchase_mw"] + frame["electricity_shortage_mw"]
    )
    electricity_uses = (
        frame["electricity_demand_mw"]
        + frame["electric_boiler_electric_mw"]
        + frame["heat_pump_electric_mw"]
        + frame["electricity_export_mw"]
    )
    frame["electricity_balance_residual_mw"] = electricity_supply - electricity_uses
    frame["network_heat_losses_mwh_th"] = (
        frame["heat_source_requirement_mw_th"] - frame["heat_demand_mw_th"]
    ) * dt
    frame["total_fuel_mwh"] = (
        frame["boiler_fuel_mwh_per_hour"] + frame["chp_fuel_mwh_per_hour"]
    ) * dt
    frame["total_emissions_tonnes"] = (
        frame["boiler_emissions_tonnes"] + frame["chp_emissions_tonnes"]
    )
    storage_residual = _storage_balance_residual(data, frame)
    heat_by_source = {
        "boiler_mwh_th": float(frame["boiler_heat_mw_th"].sum() * dt),
        "electric_boiler_mwh_th": float(frame["electric_boiler_heat_mw_th"].sum() * dt),
        "heat_pump_mwh_th": float(frame["heat_pump_heat_mw_th"].sum() * dt),
        "chp_mwh_th": float(frame["chp_heat_mw_th"].sum() * dt),
        "storage_discharge_mwh_th": float(frame["storage_discharge_mw_th"].sum() * dt),
    }
    fuel_by_source = {
        "boiler_mwh": float(frame["boiler_fuel_mwh_per_hour"].sum() * dt),
        "chp_mwh": float(frame["chp_fuel_mwh_per_hour"].sum() * dt),
    }
    summary = {
        "schema_version": 1,
        "objective_eur": objective_eur,
        "heat_by_source_mwh_th": heat_by_source,
        "electricity_from_chp_mwh": float(frame["chp_power_mw"].sum() * dt),
        "electricity_purchased_mwh": float(frame["electricity_purchase_mw"].sum() * dt),
        "electricity_exported_mwh": float(frame["electricity_export_mw"].sum() * dt),
        "electricity_shortage_mwh": float(frame["electricity_shortage_mw"].sum() * dt),
        "fuel_use_mwh": fuel_by_source,
        "total_fuel_mwh": float(frame["total_fuel_mwh"].sum()),
        "total_emissions_tonnes": float(frame["total_emissions_tonnes"].sum()),
        "heat_dumped_mwh_th": float(frame["heat_dump_mw_th"].sum() * dt),
        "unmet_heat_mwh_th": float(frame["unmet_heat_mw_th"].sum() * dt),
        "storage_charge_mwh_th": float(frame["storage_charge_mw_th"].sum() * dt),
        "storage_discharge_mwh_th": float(frame["storage_discharge_mw_th"].sum() * dt),
        "ending_storage_mwh_th": float(frame["storage_inventory_mwh_th"].iloc[-1]),
        "network_heat_losses_mwh_th": float(frame["network_heat_losses_mwh_th"].sum()),
        "heat_balance_max_abs_residual_mw": float(frame["heat_balance_residual_mw_th"].abs().max()),
        "electricity_balance_max_abs_residual_mw": float(
            frame["electricity_balance_residual_mw"].abs().max()
        ),
        "storage_balance_max_abs_residual_mwh": storage_residual,
        "coupling_statement": (
            "CHP fuel, costs, emissions, heat output, and electricity output are "
            "reported once through the configured operating-region vertices."
        ),
    }
    return HeatSystemResult(frame, summary, solver_message)


def _storage_balance_residual(data: _PreparedHeat, frame: pd.DataFrame) -> float:
    retention = (1.0 - data.storage.standing_loss_fraction_per_hour) ** data.time_step_hours
    previous = frame["storage_inventory_mwh_th"].shift(1)
    previous.iloc[0] = data.storage.initial_inventory_mwh_th
    expected = (
        retention * previous
        + frame["storage_charge_mw_th"] * data.time_step_hours
        - frame["storage_discharge_mw_th"] * data.time_step_hours
    )
    return float((frame["storage_inventory_mwh_th"] - expected).abs().max())


def _load_boiler(payload: Any) -> HeatBoilerConfig:
    raw = _as_mapping(payload, "boiler")
    return HeatBoilerConfig(
        id=str(raw.get("id", "boiler")),
        heat_capacity_mw=_nonnegative(raw, "heat_capacity_mw", 0.0),
        efficiency_mwh_heat_per_mwh_fuel=_positive(
            raw,
            "efficiency_mwh_heat_per_mwh_fuel",
            0.9,
        ),
        fuel_cost_eur_per_mwh=float(raw.get("fuel_cost_eur_per_mwh", 0.0)),
        variable_cost_eur_per_mwh_heat=float(raw.get("variable_cost_eur_per_mwh_heat", 0.0)),
        emission_tonnes_per_mwh_fuel=_nonnegative(
            raw,
            "emission_tonnes_per_mwh_fuel",
            0.0,
        ),
    )


def _load_electric_boiler(payload: Any) -> ElectricBoilerConfig:
    raw = _as_mapping(payload, "electric_boiler")
    return ElectricBoilerConfig(
        id=str(raw.get("id", "electric-boiler")),
        heat_capacity_mw=_nonnegative(raw, "heat_capacity_mw", 0.0),
        efficiency_mwh_heat_per_mwh_electric=_positive(
            raw,
            "efficiency_mwh_heat_per_mwh_electric",
            0.98,
        ),
        variable_cost_eur_per_mwh_heat=float(raw.get("variable_cost_eur_per_mwh_heat", 0.0)),
    )


def _load_heat_pump(payload: Any) -> HeatPumpConfig:
    raw = _as_mapping(payload, "heat_pump")
    return HeatPumpConfig(
        id=str(raw.get("id", "heat-pump")),
        heat_capacity_mw=_nonnegative(raw, "heat_capacity_mw", 0.0),
        cop=raw.get("cop", 3.0),
        variable_cost_eur_per_mwh_heat=float(raw.get("variable_cost_eur_per_mwh_heat", 0.0)),
    )


def _load_storage(payload: Any) -> ThermalStorageConfig:
    raw = _as_mapping(payload, "storage") if isinstance(payload, Mapping) else {}
    return ThermalStorageConfig(
        id=str(raw.get("id", "thermal-store")),
        energy_capacity_mwh_th=_nonnegative(raw, "energy_capacity_mwh_th", 0.0),
        charge_capacity_mw_th=_nonnegative(raw, "charge_capacity_mw_th", 0.0),
        discharge_capacity_mw_th=_nonnegative(raw, "discharge_capacity_mw_th", 0.0),
        initial_inventory_mwh_th=_nonnegative(raw, "initial_inventory_mwh_th", 0.0),
        minimum_final_inventory_mwh_th=_nonnegative(raw, "minimum_final_inventory_mwh_th", 0.0),
        standing_loss_fraction_per_hour=_fraction(raw, "standing_loss_fraction_per_hour", 0.0),
    )


def _load_chp_unit(payload: Any) -> CHPUnitConfig:
    raw = _as_mapping(payload, "chp_unit")
    vertices = tuple(_load_chp_vertex(item) for item in raw.get("vertices", []) or [])
    return CHPUnitConfig(
        id=str(raw.get("id", "chp")),
        vertices=vertices,
        fuel_cost_eur_per_mwh=float(raw.get("fuel_cost_eur_per_mwh", 0.0)),
        variable_cost_eur_per_mwh_fuel=float(raw.get("variable_cost_eur_per_mwh_fuel", 0.0)),
        emission_tonnes_per_mwh_fuel=_nonnegative(
            raw,
            "emission_tonnes_per_mwh_fuel",
            0.0,
        ),
    )


def _load_chp_vertex(payload: Any) -> CHPVertex:
    raw = _as_mapping(payload, "chp_vertex")
    return CHPVertex(
        id=str(raw.get("id", "vertex")),
        electric_output_mw=_nonnegative(raw, "electric_output_mw", 0.0),
        heat_output_mw_th=_nonnegative(raw, "heat_output_mw_th", 0.0),
        fuel_input_mwh_per_hour=_nonnegative(raw, "fuel_input_mwh_per_hour", 0.0),
    )


def _validate_problem(problem: HeatSystemProblem) -> None:
    periods = len(problem.periods)
    if periods == 0:
        raise ConfigurationError("Heat study must define at least one period")
    _profile(problem.heat_demand_mw_th, periods, "heat_demand_mw_th")
    _profile(problem.electricity_demand_mw, periods, "electricity_demand_mw")
    _finite_profile(
        problem.electricity_purchase_price_eur_per_mwh,
        periods,
        "electricity_purchase_price_eur_per_mwh",
    )
    _finite_profile(
        problem.electricity_export_value_eur_per_mwh,
        periods,
        "electricity_export_value_eur_per_mwh",
    )
    if problem.time_step_hours <= 0.0:
        raise ConfigurationError("time_step_hours must be positive")
    if not 0.0 < problem.network_delivery_efficiency <= 1.0:
        raise ConfigurationError("network_delivery_efficiency must be greater than 0 and at most 1")
    for name, value in (
        ("heat_shortage_penalty_eur_per_mwh", problem.heat_shortage_penalty_eur_per_mwh),
        (
            "electricity_shortage_penalty_eur_per_mwh",
            problem.electricity_shortage_penalty_eur_per_mwh,
        ),
        ("heat_dumping_cost_eur_per_mwh", problem.heat_dumping_cost_eur_per_mwh),
        ("carbon_price_eur_per_tonne", problem.carbon_price_eur_per_tonne),
    ):
        if value < 0.0:
            raise ConfigurationError(f"{name} must be non-negative")
    for fuel_boiler in problem.boilers:
        _validate_id(fuel_boiler.id, "boiler id")
        _validate_nonnegative(fuel_boiler.heat_capacity_mw, f"{fuel_boiler.id}.heat_capacity_mw")
        _validate_efficiency(
            fuel_boiler.efficiency_mwh_heat_per_mwh_fuel,
            f"{fuel_boiler.id}.efficiency_mwh_heat_per_mwh_fuel",
            upper=1.0,
        )
        _validate_nonnegative(
            fuel_boiler.emission_tonnes_per_mwh_fuel,
            f"{fuel_boiler.id}.emission_tonnes_per_mwh_fuel",
        )
    for electric_boiler in problem.electric_boilers:
        _validate_id(electric_boiler.id, "electric boiler id")
        _validate_nonnegative(
            electric_boiler.heat_capacity_mw,
            f"{electric_boiler.id}.heat_capacity_mw",
        )
        _validate_efficiency(
            electric_boiler.efficiency_mwh_heat_per_mwh_electric,
            f"{electric_boiler.id}.efficiency_mwh_heat_per_mwh_electric",
            upper=1.0,
        )
    for pump in problem.heat_pumps:
        _validate_id(pump.id, "heat pump id")
        _validate_nonnegative(pump.heat_capacity_mw, f"{pump.id}.heat_capacity_mw")
        _positive_profile(pump.cop, periods, f"{pump.id}.cop")
    _validate_storage(problem.storage)
    for unit in problem.chp_units:
        _validate_chp_unit(unit)
    if not (
        problem.boilers
        or problem.electric_boilers
        or problem.heat_pumps
        or problem.chp_units
        or problem.storage.initial_inventory_mwh_th > 0.0
    ):
        raise ConfigurationError("At least one heat source or initial thermal storage is required")


def _validate_storage(storage: ThermalStorageConfig) -> None:
    _validate_id(storage.id, "thermal storage id")
    for name, value in (
        ("energy_capacity_mwh_th", storage.energy_capacity_mwh_th),
        ("charge_capacity_mw_th", storage.charge_capacity_mw_th),
        ("discharge_capacity_mw_th", storage.discharge_capacity_mw_th),
        ("initial_inventory_mwh_th", storage.initial_inventory_mwh_th),
        ("minimum_final_inventory_mwh_th", storage.minimum_final_inventory_mwh_th),
    ):
        _validate_nonnegative(value, f"{storage.id}.{name}")
    if not 0.0 <= storage.standing_loss_fraction_per_hour <= 1.0:
        raise ConfigurationError(f"{storage.id}.standing_loss_fraction_per_hour must be 0 to 1")
    if storage.initial_inventory_mwh_th > storage.energy_capacity_mwh_th:
        raise ConfigurationError("Initial thermal storage inventory exceeds capacity")
    if storage.minimum_final_inventory_mwh_th > storage.energy_capacity_mwh_th:
        raise ConfigurationError("Final thermal storage inventory exceeds capacity")


def _validate_chp_unit(unit: CHPUnitConfig) -> None:
    _validate_id(unit.id, "CHP id")
    if not unit.vertices:
        raise ConfigurationError(f"CHP unit {unit.id} must define operating vertices")
    if unit.fuel_cost_eur_per_mwh < 0.0:
        raise ConfigurationError(f"{unit.id}.fuel_cost_eur_per_mwh must be non-negative")
    if unit.variable_cost_eur_per_mwh_fuel < 0.0:
        raise ConfigurationError(f"{unit.id}.variable_cost_eur_per_mwh_fuel must be non-negative")
    _validate_nonnegative(
        unit.emission_tonnes_per_mwh_fuel,
        f"{unit.id}.emission_tonnes_per_mwh_fuel",
    )
    has_zero_vertex = False
    for vertex in unit.vertices:
        _validate_id(vertex.id, f"{unit.id} vertex id")
        for name, value in (
            ("electric_output_mw", vertex.electric_output_mw),
            ("heat_output_mw_th", vertex.heat_output_mw_th),
            ("fuel_input_mwh_per_hour", vertex.fuel_input_mwh_per_hour),
        ):
            _validate_nonnegative(value, f"{unit.id}.{vertex.id}.{name}")
        if (
            vertex.electric_output_mw == 0.0
            and vertex.heat_output_mw_th == 0.0
            and vertex.fuel_input_mwh_per_hour == 0.0
        ):
            has_zero_vertex = True
        if (vertex.electric_output_mw > 0.0 or vertex.heat_output_mw_th > 0.0) and (
            vertex.fuel_input_mwh_per_hour <= 0.0
        ):
            raise ConfigurationError(f"CHP vertex {unit.id}.{vertex.id} has output without fuel")
        if vertex.fuel_input_mwh_per_hour > 0.0:
            total_efficiency = (
                vertex.electric_output_mw + vertex.heat_output_mw_th
            ) / vertex.fuel_input_mwh_per_hour
            if total_efficiency > 1.0 + 1e-9:
                raise ConfigurationError(
                    f"CHP vertex {unit.id}.{vertex.id} has total efficiency above 1.0"
                )
    if not has_zero_vertex:
        raise ConfigurationError(f"CHP unit {unit.id} must include a zero-output vertex")


def _validate_id(value: str, label: str) -> None:
    if not value:
        raise ConfigurationError(f"{label} must not be empty")
    if "__" in value:
        raise ConfigurationError(f"{label} must not contain '__'")


def _validate_nonnegative(value: float, path: str) -> None:
    if value < 0.0:
        raise ConfigurationError(f"{path} must be non-negative")


def _validate_efficiency(value: float, path: str, *, upper: float) -> None:
    if value <= 0.0 or value > upper:
        raise ConfigurationError(f"{path} must be greater than 0 and at most {upper}")


def _asset(prefix: str, asset_id: str) -> str:
    return f"{prefix}__{asset_id}"


def _as_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    return payload


def _positive(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = float(payload.get(key, default))
    if value <= 0.0:
        raise ConfigurationError(f"{key} must be positive")
    return value


def _nonnegative(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = float(payload.get(key, default))
    if value < 0.0:
        raise ConfigurationError(f"{key} must be non-negative")
    return value


def _fraction(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = float(payload.get(key, default))
    if not 0.0 < value <= 1.0:
        raise ConfigurationError(f"{key} must be greater than 0 and at most 1")
    return value


def _profile(value: npt.ArrayLike, periods: int, path: str = "profile") -> FloatArray:
    array = _finite_profile(value, periods, path)
    if np.any(array < 0.0):
        raise ConfigurationError(f"{path} must contain finite non-negative numbers")
    return array


def _positive_profile(value: npt.ArrayLike, periods: int, path: str = "profile") -> FloatArray:
    array = _finite_profile(value, periods, path)
    if np.any(array <= 0.0):
        raise ConfigurationError(f"{path} must contain finite positive numbers")
    return array


def _finite_profile(value: npt.ArrayLike, periods: int, path: str = "profile") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(periods, float(array), dtype=np.float64)
    if array.ndim != 1 or len(array) != periods:
        raise ConfigurationError(f"{path} must be a scalar or a list with {periods} values")
    if np.any(~np.isfinite(array)):
        raise ConfigurationError(f"{path} must contain finite numbers")
    return array.astype(np.float64)
