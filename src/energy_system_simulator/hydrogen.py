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
MWH_LHV_PER_KG_H2 = 0.03333


def hydrogen_kg_to_mwh_lhv(value_kg: float) -> float:
    if value_kg < 0.0:
        raise ConfigurationError("Hydrogen mass must be non-negative")
    return value_kg * MWH_LHV_PER_KG_H2


def hydrogen_mwh_lhv_to_kg(value_mwh_lhv: float) -> float:
    if value_mwh_lhv < 0.0:
        raise ConfigurationError("Hydrogen energy must be non-negative")
    return value_mwh_lhv / MWH_LHV_PER_KG_H2


@dataclass(frozen=True)
class ElectrolyserConfig:
    id: str
    power_capacity_mw: float
    efficiency_mwh_h2_per_mwh_electric: float
    minimum_load_mw: float = 0.0
    ramp_mw_per_hour: float | None = None
    variable_cost_eur_per_mwh_electric: float = 0.0


@dataclass(frozen=True)
class HydrogenStorageConfig:
    id: str
    energy_capacity_mwh_lhv: float
    charge_capacity_mwh_per_hour: float
    discharge_capacity_mwh_per_hour: float
    initial_inventory_mwh_lhv: float = 0.0
    minimum_final_inventory_mwh_lhv: float = 0.0
    standing_loss_fraction_per_hour: float = 0.0


@dataclass(frozen=True)
class HydrogenDemandConfig:
    id: str
    demand_mwh_lhv: npt.ArrayLike
    shortage_penalty_eur_per_mwh: float = 10_000.0


@dataclass(frozen=True)
class HydrogenReconverterConfig:
    id: str
    power_capacity_mw: float
    efficiency_mwh_electric_per_mwh_h2: float
    variable_cost_eur_per_mwh_electric: float = 0.0
    emission_tonnes_per_mwh_h2: float = 0.0


@dataclass(frozen=True)
class HydrogenSystemProblem:
    periods: tuple[str, ...]
    renewable_surplus_mw: npt.ArrayLike
    electricity_deficit_mw: npt.ArrayLike
    time_step_hours: float = 1.0
    electricity_purchase_price_eur_per_mwh: npt.ArrayLike = 0.0
    reconversion_value_eur_per_mwh: float = 1_000.0
    electrolyser: ElectrolyserConfig = field(
        default_factory=lambda: ElectrolyserConfig(
            id="electrolyser",
            power_capacity_mw=0.0,
            efficiency_mwh_h2_per_mwh_electric=0.7,
        )
    )
    storage: HydrogenStorageConfig = field(
        default_factory=lambda: HydrogenStorageConfig(
            id="hydrogen-store",
            energy_capacity_mwh_lhv=0.0,
            charge_capacity_mwh_per_hour=0.0,
            discharge_capacity_mwh_per_hour=0.0,
        )
    )
    demand: HydrogenDemandConfig = field(
        default_factory=lambda: HydrogenDemandConfig(id="hydrogen-demand", demand_mwh_lhv=0.0)
    )
    reconverter: HydrogenReconverterConfig = field(
        default_factory=lambda: HydrogenReconverterConfig(
            id="fuel-cell",
            power_capacity_mw=0.0,
            efficiency_mwh_electric_per_mwh_h2=0.5,
        )
    )


@dataclass(frozen=True)
class HydrogenSystemResult:
    timeseries: pd.DataFrame
    summary: dict[str, Any]
    solver_message: str

    def write(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        self.timeseries.to_csv(output_directory / "hydrogen_timeseries.csv", index=False)
        (output_directory / "hydrogen_summary.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class HydrogenSystemModel:
    """Continuous hydrogen production, storage, demand, and reconversion model."""

    def solve(self, problem: HydrogenSystemProblem) -> HydrogenSystemResult:
        _validate_problem(problem)
        data = _PreparedHydrogen.from_problem(problem)
        registry = _Registry()
        for block in (
            "electrolyser_power_mw",
            "hydrogen_produced_mwh_lhv",
            "hydrogen_to_storage_mwh_lhv",
            "hydrogen_storage_discharge_mwh_lhv",
            "hydrogen_inventory_mwh_lhv",
            "hydrogen_delivered_mwh_lhv",
            "hydrogen_shortage_mwh_lhv",
            "hydrogen_curtailed_mwh_lhv",
            "reconversion_hydrogen_mwh_lhv",
            "reconversion_power_mw",
            "unserved_electric_deficit_mw",
        ):
            registry.add(block, data.periods)
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
            raise OptimisationError(f"Hydrogen system optimisation failed: {result.message}")
        solution = np.asarray(result.x, dtype=np.float64)
        return _result(data, registry, solution, float(result.fun), str(result.message))


def load_hydrogen_problem(path: Path | str) -> HydrogenSystemProblem:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Hydrogen study YAML must contain a mapping")
    if int(payload.get("schema_version", 1)) != 1:
        raise ConfigurationError("Hydrogen study schema_version must be 1")
    periods = tuple(str(item) for item in payload.get("periods", ["0"]))
    if not periods:
        raise ConfigurationError("Hydrogen study must define at least one period")
    electrolyser_raw = _mapping(payload, "electrolyser")
    storage_raw = _mapping(payload, "storage")
    demand_raw = _mapping(payload, "demand")
    reconverter_raw = _mapping(payload, "reconverter")
    problem = HydrogenSystemProblem(
        periods=periods,
        renewable_surplus_mw=_profile(payload.get("renewable_surplus_mw", 0.0), len(periods)),
        electricity_deficit_mw=_profile(payload.get("electricity_deficit_mw", 0.0), len(periods)),
        time_step_hours=_positive(payload, "time_step_hours", 1.0),
        electricity_purchase_price_eur_per_mwh=_finite_profile(
            payload.get("electricity_purchase_price_eur_per_mwh", 0.0),
            len(periods),
        ),
        reconversion_value_eur_per_mwh=float(
            payload.get("reconversion_value_eur_per_mwh", 1_000.0)
        ),
        electrolyser=ElectrolyserConfig(
            id=str(electrolyser_raw.get("id", "electrolyser")),
            power_capacity_mw=_nonnegative(electrolyser_raw, "power_capacity_mw", 0.0),
            efficiency_mwh_h2_per_mwh_electric=_positive(
                electrolyser_raw,
                "efficiency_mwh_h2_per_mwh_electric",
                0.7,
            ),
            minimum_load_mw=_nonnegative(electrolyser_raw, "minimum_load_mw", 0.0),
            ramp_mw_per_hour=(
                _nonnegative(electrolyser_raw, "ramp_mw_per_hour", 0.0)
                if "ramp_mw_per_hour" in electrolyser_raw
                else None
            ),
            variable_cost_eur_per_mwh_electric=float(
                electrolyser_raw.get("variable_cost_eur_per_mwh_electric", 0.0)
            ),
        ),
        storage=HydrogenStorageConfig(
            id=str(storage_raw.get("id", "hydrogen-store")),
            energy_capacity_mwh_lhv=_nonnegative(
                storage_raw,
                "energy_capacity_mwh_lhv",
                0.0,
            ),
            charge_capacity_mwh_per_hour=_nonnegative(
                storage_raw,
                "charge_capacity_mwh_per_hour",
                0.0,
            ),
            discharge_capacity_mwh_per_hour=_nonnegative(
                storage_raw,
                "discharge_capacity_mwh_per_hour",
                0.0,
            ),
            initial_inventory_mwh_lhv=_nonnegative(
                storage_raw,
                "initial_inventory_mwh_lhv",
                0.0,
            ),
            minimum_final_inventory_mwh_lhv=_nonnegative(
                storage_raw,
                "minimum_final_inventory_mwh_lhv",
                0.0,
            ),
            standing_loss_fraction_per_hour=_fraction(
                storage_raw,
                "standing_loss_fraction_per_hour",
                0.0,
            ),
        ),
        demand=HydrogenDemandConfig(
            id=str(demand_raw.get("id", "hydrogen-demand")),
            demand_mwh_lhv=_profile(demand_raw.get("demand_mwh_lhv", 0.0), len(periods)),
            shortage_penalty_eur_per_mwh=float(
                demand_raw.get("shortage_penalty_eur_per_mwh", 10_000.0)
            ),
        ),
        reconverter=HydrogenReconverterConfig(
            id=str(reconverter_raw.get("id", "fuel-cell")),
            power_capacity_mw=_nonnegative(reconverter_raw, "power_capacity_mw", 0.0),
            efficiency_mwh_electric_per_mwh_h2=_positive(
                reconverter_raw,
                "efficiency_mwh_electric_per_mwh_h2",
                0.5,
            ),
            variable_cost_eur_per_mwh_electric=float(
                reconverter_raw.get("variable_cost_eur_per_mwh_electric", 0.0)
            ),
            emission_tonnes_per_mwh_h2=_nonnegative(
                reconverter_raw,
                "emission_tonnes_per_mwh_h2",
                0.0,
            ),
        ),
    )
    _validate_problem(problem)
    return problem


def run_hydrogen_study(problem: HydrogenSystemProblem) -> HydrogenSystemResult:
    _validate_problem(problem)
    return HydrogenSystemModel().solve(problem)


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
class _PreparedHydrogen:
    periods: int
    labels: tuple[str, ...]
    renewable_surplus_mw: FloatArray
    electricity_deficit_mw: FloatArray
    electricity_purchase_price_eur_per_mwh: FloatArray
    time_step_hours: float
    reconversion_value_eur_per_mwh: float
    electrolyser: ElectrolyserConfig
    storage: HydrogenStorageConfig
    demand: HydrogenDemandConfig
    demand_mwh_lhv: FloatArray
    reconverter: HydrogenReconverterConfig

    @classmethod
    def from_problem(cls, problem: HydrogenSystemProblem) -> _PreparedHydrogen:
        periods = len(problem.periods)
        return cls(
            periods=periods,
            labels=problem.periods,
            renewable_surplus_mw=_profile(problem.renewable_surplus_mw, periods),
            electricity_deficit_mw=_profile(problem.electricity_deficit_mw, periods),
            electricity_purchase_price_eur_per_mwh=_finite_profile(
                problem.electricity_purchase_price_eur_per_mwh,
                periods,
            ),
            time_step_hours=problem.time_step_hours,
            reconversion_value_eur_per_mwh=problem.reconversion_value_eur_per_mwh,
            electrolyser=problem.electrolyser,
            storage=problem.storage,
            demand=problem.demand,
            demand_mwh_lhv=_profile(problem.demand.demand_mwh_lhv, periods),
            reconverter=problem.reconverter,
        )


def _objective(data: _PreparedHydrogen, registry: _Registry) -> FloatArray:
    objective = np.zeros(registry.size, dtype=np.float64)
    dt = data.time_step_hours
    for t in range(data.periods):
        objective[registry.at("electrolyser_power_mw", t)] = (
            data.electricity_purchase_price_eur_per_mwh[t]
            + data.electrolyser.variable_cost_eur_per_mwh_electric
        ) * dt
        objective[registry.at("hydrogen_shortage_mwh_lhv", t)] = (
            data.demand.shortage_penalty_eur_per_mwh
        )
        objective[registry.at("reconversion_power_mw", t)] = (
            data.reconverter.variable_cost_eur_per_mwh_electric
            - data.reconversion_value_eur_per_mwh
        ) * dt
        objective[registry.at("unserved_electric_deficit_mw", t)] = (
            data.reconversion_value_eur_per_mwh * dt
        )
    return objective


def _bounds(data: _PreparedHydrogen, registry: _Registry) -> tuple[FloatArray, FloatArray]:
    lower = np.zeros(registry.size, dtype=np.float64)
    upper = np.full(registry.size, np.inf, dtype=np.float64)
    for t in range(data.periods):
        upper[registry.at("electrolyser_power_mw", t)] = data.electrolyser.power_capacity_mw
        upper[registry.at("hydrogen_to_storage_mwh_lhv", t)] = (
            data.storage.charge_capacity_mwh_per_hour * data.time_step_hours
        )
        upper[registry.at("hydrogen_storage_discharge_mwh_lhv", t)] = (
            data.storage.discharge_capacity_mwh_per_hour * data.time_step_hours
        )
        upper[registry.at("hydrogen_inventory_mwh_lhv", t)] = data.storage.energy_capacity_mwh_lhv
        upper[registry.at("hydrogen_delivered_mwh_lhv", t)] = data.demand_mwh_lhv[t]
        upper[registry.at("hydrogen_shortage_mwh_lhv", t)] = data.demand_mwh_lhv[t]
        upper[registry.at("reconversion_power_mw", t)] = min(
            data.reconverter.power_capacity_mw,
            data.electricity_deficit_mw[t],
        )
        upper[registry.at("unserved_electric_deficit_mw", t)] = data.electricity_deficit_mw[t]
    return lower, upper


def _constraints(
    data: _PreparedHydrogen,
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
    for t in range(data.periods):
        add_eq(
            {
                registry.at("hydrogen_produced_mwh_lhv", t): 1.0,
                registry.at("electrolyser_power_mw", t): (
                    -data.electrolyser.efficiency_mwh_h2_per_mwh_electric * dt
                ),
            },
            0.0,
        )
        add_eq(
            {
                registry.at("hydrogen_produced_mwh_lhv", t): 1.0,
                registry.at("hydrogen_storage_discharge_mwh_lhv", t): 1.0,
                registry.at("hydrogen_to_storage_mwh_lhv", t): -1.0,
                registry.at("hydrogen_delivered_mwh_lhv", t): -1.0,
                registry.at("reconversion_hydrogen_mwh_lhv", t): -1.0,
                registry.at("hydrogen_curtailed_mwh_lhv", t): -1.0,
            },
            0.0,
        )
        previous = (
            data.storage.initial_inventory_mwh_lhv
            if t == 0
            else registry.at("hydrogen_inventory_mwh_lhv", t - 1)
        )
        inventory_terms = {
            registry.at("hydrogen_inventory_mwh_lhv", t): 1.0,
            registry.at("hydrogen_to_storage_mwh_lhv", t): -1.0,
            registry.at("hydrogen_storage_discharge_mwh_lhv", t): 1.0,
        }
        if isinstance(previous, int):
            inventory_terms[previous] = -retention
            rhs = 0.0
        else:
            rhs = retention * previous
        add_eq(inventory_terms, rhs)
        add_eq(
            {
                registry.at("hydrogen_delivered_mwh_lhv", t): 1.0,
                registry.at("hydrogen_shortage_mwh_lhv", t): 1.0,
            },
            data.demand_mwh_lhv[t],
        )
        add_eq(
            {
                registry.at("reconversion_power_mw", t): 1.0,
                registry.at("reconversion_hydrogen_mwh_lhv", t): (
                    -data.reconverter.efficiency_mwh_electric_per_mwh_h2 / dt
                ),
            },
            0.0,
        )
        add_eq(
            {
                registry.at("reconversion_power_mw", t): 1.0,
                registry.at("unserved_electric_deficit_mw", t): 1.0,
            },
            data.electricity_deficit_mw[t],
        )
        add_ub(
            {registry.at("electrolyser_power_mw", t): 1.0},
            data.renewable_surplus_mw[t],
        )
        if data.electrolyser.minimum_load_mw > 0.0:
            add_ub(
                {registry.at("electrolyser_power_mw", t): -1.0},
                -data.electrolyser.minimum_load_mw,
            )
        if data.electrolyser.ramp_mw_per_hour is not None and t > 0:
            ramp = data.electrolyser.ramp_mw_per_hour * dt
            add_ub(
                {
                    registry.at("electrolyser_power_mw", t): 1.0,
                    registry.at("electrolyser_power_mw", t - 1): -1.0,
                },
                ramp,
            )
            add_ub(
                {
                    registry.at("electrolyser_power_mw", t): -1.0,
                    registry.at("electrolyser_power_mw", t - 1): 1.0,
                },
                ramp,
            )
    add_ub(
        {registry.at("hydrogen_inventory_mwh_lhv", data.periods - 1): -1.0},
        -data.storage.minimum_final_inventory_mwh_lhv,
    )
    return (
        coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(len(eq_rhs), registry.size)),
        np.asarray(eq_rhs, dtype=np.float64),
        coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(len(ub_rhs), registry.size)),
        np.asarray(ub_rhs, dtype=np.float64),
    )


def _result(
    data: _PreparedHydrogen,
    registry: _Registry,
    solution: FloatArray,
    objective_eur: float,
    solver_message: str,
) -> HydrogenSystemResult:
    dt = data.time_step_hours
    frame = pd.DataFrame(
        {
            "period": data.labels,
            "renewable_surplus_mw": data.renewable_surplus_mw,
            "electricity_deficit_mw": data.electricity_deficit_mw,
            "electrolyser_power_mw": registry.values(solution, "electrolyser_power_mw"),
            "hydrogen_produced_mwh_lhv": registry.values(
                solution,
                "hydrogen_produced_mwh_lhv",
            ),
            "hydrogen_to_storage_mwh_lhv": registry.values(
                solution,
                "hydrogen_to_storage_mwh_lhv",
            ),
            "hydrogen_storage_discharge_mwh_lhv": registry.values(
                solution,
                "hydrogen_storage_discharge_mwh_lhv",
            ),
            "hydrogen_inventory_mwh_lhv": registry.values(
                solution,
                "hydrogen_inventory_mwh_lhv",
            ),
            "hydrogen_delivered_mwh_lhv": registry.values(
                solution,
                "hydrogen_delivered_mwh_lhv",
            ),
            "hydrogen_shortage_mwh_lhv": registry.values(
                solution,
                "hydrogen_shortage_mwh_lhv",
            ),
            "hydrogen_curtailed_mwh_lhv": registry.values(
                solution,
                "hydrogen_curtailed_mwh_lhv",
            ),
            "reconversion_hydrogen_mwh_lhv": registry.values(
                solution,
                "reconversion_hydrogen_mwh_lhv",
            ),
            "reconversion_power_mw": registry.values(solution, "reconversion_power_mw"),
            "unserved_electric_deficit_mw": registry.values(
                solution,
                "unserved_electric_deficit_mw",
            ),
        }
    )
    frame["conversion_loss_mwh_lhv"] = (
        frame["electrolyser_power_mw"] * dt - frame["hydrogen_produced_mwh_lhv"]
    )
    frame["reconversion_loss_mwh"] = (
        frame["reconversion_hydrogen_mwh_lhv"] - frame["reconversion_power_mw"] * dt
    )
    frame["reconversion_emissions_tonnes"] = (
        frame["reconversion_hydrogen_mwh_lhv"] * data.reconverter.emission_tonnes_per_mwh_h2
    )
    frame["hydrogen_carrier_balance_residual_mwh_lhv"] = (
        frame["hydrogen_produced_mwh_lhv"]
        + frame["hydrogen_storage_discharge_mwh_lhv"]
        - frame["hydrogen_to_storage_mwh_lhv"]
        - frame["hydrogen_delivered_mwh_lhv"]
        - frame["reconversion_hydrogen_mwh_lhv"]
        - frame["hydrogen_curtailed_mwh_lhv"]
    )
    produced = float(frame["hydrogen_produced_mwh_lhv"].sum())
    reconverted = float(frame["reconversion_power_mw"].sum() * dt)
    electricity_to_h2 = float(frame["electrolyser_power_mw"].sum() * dt)
    balance_residual = max(
        _hydrogen_balance_residual(data, frame),
        float(frame["hydrogen_carrier_balance_residual_mwh_lhv"].abs().max()),
    )
    summary = {
        "schema_version": 1,
        "canonical_hydrogen_unit": "MWh_LHV",
        "objective_eur": objective_eur,
        "hydrogen_balance_max_abs_residual_mwh": balance_residual,
        "electricity_consumed_mwh": electricity_to_h2,
        "hydrogen_produced_mwh_lhv": produced,
        "hydrogen_delivered_mwh_lhv": float(frame["hydrogen_delivered_mwh_lhv"].sum()),
        "hydrogen_shortage_mwh_lhv": float(frame["hydrogen_shortage_mwh_lhv"].sum()),
        "hydrogen_curtailed_mwh_lhv": float(frame["hydrogen_curtailed_mwh_lhv"].sum()),
        "ending_inventory_mwh_lhv": float(frame["hydrogen_inventory_mwh_lhv"].iloc[-1]),
        "reconverted_electricity_mwh": reconverted,
        "unserved_electric_deficit_mwh": float(frame["unserved_electric_deficit_mw"].sum() * dt),
        "round_trip_efficiency": (
            reconverted / electricity_to_h2 if electricity_to_h2 > 0.0 else 0.0
        ),
        "conversion_losses_mwh": float(frame["conversion_loss_mwh_lhv"].sum()),
        "reconversion_losses_mwh": float(frame["reconversion_loss_mwh"].sum()),
        "reconversion_emissions_tonnes": float(frame["reconversion_emissions_tonnes"].sum()),
        "marginal_system_value_proxy_eur_per_mwh": data.reconversion_value_eur_per_mwh,
        "emissions_statement": (
            "Hydrogen emissions are only zero when configured process emissions and "
            "the electricity source assumptions are zero."
        ),
    }
    return HydrogenSystemResult(frame, summary, solver_message)


def _hydrogen_balance_residual(data: _PreparedHydrogen, frame: pd.DataFrame) -> float:
    retention = (1.0 - data.storage.standing_loss_fraction_per_hour) ** data.time_step_hours
    previous = frame["hydrogen_inventory_mwh_lhv"].shift(1)
    previous.iloc[0] = data.storage.initial_inventory_mwh_lhv
    expected = (
        retention * previous
        + frame["hydrogen_to_storage_mwh_lhv"]
        - frame["hydrogen_storage_discharge_mwh_lhv"]
    )
    return float((frame["hydrogen_inventory_mwh_lhv"] - expected).abs().max())


def _validate_problem(problem: HydrogenSystemProblem) -> None:
    periods = len(problem.periods)
    for name, values in (
        ("renewable_surplus_mw", problem.renewable_surplus_mw),
        ("electricity_deficit_mw", problem.electricity_deficit_mw),
        ("demand.demand_mwh_lhv", problem.demand.demand_mwh_lhv),
    ):
        _profile(values, periods, name)
    _finite_profile(
        problem.electricity_purchase_price_eur_per_mwh,
        periods,
        "electricity_purchase_price_eur_per_mwh",
    )
    if problem.time_step_hours <= 0.0:
        raise ConfigurationError("time_step_hours must be positive")
    if problem.reconversion_value_eur_per_mwh < 0.0:
        raise ConfigurationError("Reconversion value must be non-negative")
    if problem.electrolyser.power_capacity_mw < 0.0:
        raise ConfigurationError("Electrolyser capacity must be non-negative")
    if problem.electrolyser.minimum_load_mw < 0.0:
        raise ConfigurationError("Electrolyser minimum load must be non-negative")
    if (
        problem.electrolyser.ramp_mw_per_hour is not None
        and problem.electrolyser.ramp_mw_per_hour < 0.0
    ):
        raise ConfigurationError("Electrolyser ramp limit must be non-negative")
    if problem.electrolyser.efficiency_mwh_h2_per_mwh_electric <= 0.0:
        raise ConfigurationError("Electrolyser efficiency must be positive")
    if problem.electrolyser.efficiency_mwh_h2_per_mwh_electric > 1.0:
        raise ConfigurationError("Electrolyser efficiency cannot exceed 1.0")
    if problem.storage.energy_capacity_mwh_lhv < 0.0:
        raise ConfigurationError("Hydrogen storage capacity must be non-negative")
    if problem.storage.charge_capacity_mwh_per_hour < 0.0:
        raise ConfigurationError("Hydrogen storage charge capacity must be non-negative")
    if problem.storage.discharge_capacity_mwh_per_hour < 0.0:
        raise ConfigurationError("Hydrogen storage discharge capacity must be non-negative")
    if problem.storage.initial_inventory_mwh_lhv < 0.0:
        raise ConfigurationError("Initial hydrogen inventory must be non-negative")
    if problem.storage.minimum_final_inventory_mwh_lhv < 0.0:
        raise ConfigurationError("Final hydrogen inventory must be non-negative")
    if not 0.0 <= problem.storage.standing_loss_fraction_per_hour <= 1.0:
        raise ConfigurationError("Hydrogen storage standing loss must be between 0 and 1")
    if problem.reconverter.power_capacity_mw < 0.0:
        raise ConfigurationError("Reconverter capacity must be non-negative")
    if problem.reconverter.efficiency_mwh_electric_per_mwh_h2 <= 0.0:
        raise ConfigurationError("Reconverter efficiency must be positive")
    if problem.reconverter.efficiency_mwh_electric_per_mwh_h2 > 1.0:
        raise ConfigurationError("Reconverter efficiency cannot exceed 1.0")
    if problem.reconverter.emission_tonnes_per_mwh_h2 < 0.0:
        raise ConfigurationError("Reconverter emissions must be non-negative")
    if problem.electrolyser.minimum_load_mw > problem.electrolyser.power_capacity_mw:
        raise ConfigurationError("Electrolyser minimum load exceeds capacity")
    if problem.storage.initial_inventory_mwh_lhv > problem.storage.energy_capacity_mwh_lhv:
        raise ConfigurationError("Initial hydrogen inventory exceeds storage capacity")
    if problem.storage.minimum_final_inventory_mwh_lhv > problem.storage.energy_capacity_mwh_lhv:
        raise ConfigurationError("Final hydrogen inventory exceeds storage capacity")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


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
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{key} must be between 0 and 1")
    return value


def _profile(value: npt.ArrayLike, periods: int, path: str = "profile") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(periods, float(array), dtype=np.float64)
    if array.ndim != 1 or len(array) != periods:
        raise ConfigurationError(f"{path} must be a scalar or a list with {periods} values")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ConfigurationError(f"{path} must contain finite non-negative numbers")
    return array.astype(np.float64)


def _finite_profile(value: npt.ArrayLike, periods: int, path: str = "profile") -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(periods, float(array), dtype=np.float64)
    if array.ndim != 1 or len(array) != periods:
        raise ConfigurationError(f"{path} must be a scalar or a list with {periods} values")
    if np.any(~np.isfinite(array)):
        raise ConfigurationError(f"{path} must contain finite numbers")
    return array.astype(np.float64)
