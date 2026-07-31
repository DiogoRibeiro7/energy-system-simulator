from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from energy_system_simulator.config import BatteryConfig, FuelConfig, ModelConfig, ThermalConfig
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.dispatch.solver import (
    absolute_gap,
    interpret_solver_result,
    objective_bound_with_constant,
    relative_gap,
    solve_milp,
)
from energy_system_simulator.dispatch.variables import VariableRegistry
from energy_system_simulator.exceptions import OptimisationError

FloatArray = npt.NDArray[np.float64]

SYSTEM_BLOCKS = (
    "renewable_used_mw",
    "imports_mw",
    "source_load_shed_mw",
)
STORAGE_BLOCKS = (
    "storage_charge_mw",
    "storage_discharge_mw",
    "storage_charge_mode",
    "storage_discharge_mode",
    "storage_soc_mwh",
)
STORAGE_DEGRADATION_BLOCK = "storage_degradation_throughput_mwh"
THERMAL_BLOCKS = (
    "thermal_output_mw",
    "thermal_on",
    "thermal_startup",
    "thermal_shutdown",
)
THERMAL_SEGMENT_OUTPUT_BLOCK = "thermal_segment_output_mw"
THERMAL_STARTUP_CATEGORY_BLOCK = "thermal_startup_category"


def _segment_asset_id(unit_id: str, segment_id: str) -> str:
    return f"{unit_id}::{segment_id}"


def _startup_category_asset_id(unit_id: str, category_id: str) -> str:
    return f"{unit_id}::{category_id}"


def _storage_degradation_asset_id(unit_id: str, band_id: str) -> str:
    return f"{unit_id}::{band_id}"


def _storage_charge_capacity(config: BatteryConfig) -> float:
    return (
        config.charge_power_capacity_mw
        if config.charge_power_capacity_mw is not None
        else config.power_capacity_mw
    )


def _storage_discharge_capacity(config: BatteryConfig) -> float:
    return (
        config.discharge_power_capacity_mw
        if config.discharge_power_capacity_mw is not None
        else config.power_capacity_mw
    )


@dataclass(frozen=True)
class FormulationStatistics:
    """Size metrics for the mixed-integer dispatch formulation."""

    continuous_variables: int
    integer_variables: int
    binary_variables: int
    linear_constraints: int
    matrix_nonzeros: int
    variable_counts_by_block: dict[str, int] = field(default_factory=dict)
    constraint_counts_by_component: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ThermalUnit:
    """Resolved thermal generator used by the indexed formulation."""

    id: str
    fuel_id: str
    config: ThermalConfig
    must_run: bool = False
    availability_factor: float = 1.0
    availability_factor_key: str | None = None


@dataclass(frozen=True)
class StorageUnit:
    """Resolved storage asset used by the indexed formulation."""

    id: str
    config: BatteryConfig


@dataclass(frozen=True)
class FormulationProblem:
    """Complete linear mixed-integer problem and source input arrays."""

    objective: FloatArray
    integrality: npt.NDArray[np.int_]
    bounds: Bounds
    constraints: LinearConstraint
    renewable_available_mw: FloatArray
    gross_demand_mw: FloatArray
    thermal_units: tuple[ThermalUnit, ...]
    storage_units: tuple[StorageUnit, ...]
    thermal_capacity_available_mw: dict[str, FloatArray]
    storage_availability_factor: dict[str, FloatArray]
    fuel_prices_eur_per_mwh_thermal: dict[str, FloatArray]
    registry: VariableRegistry
    statistics: FormulationStatistics


@dataclass(frozen=True)
class TerminalCommitmentState:
    """Commitment state and residual obligations at the final model period."""

    thermal_on: bool
    thermal_output_mw: float
    consecutive_on_hours: float
    consecutive_off_hours: float
    residual_minimum_up_hours: float
    residual_minimum_down_hours: float
    terminal_commitment_mode: str


@dataclass(frozen=True)
class DispatchResult:
    """Optimised dispatch table and solver diagnostics."""

    frame: pd.DataFrame
    objective_eur: float
    solver_message: str
    solver_status: str
    backend_solver_status: str
    backend_solver_status_code: int | None
    mip_gap: float | None
    primal_objective_eur: float | None
    objective_bound_eur: float | None
    absolute_gap_eur: float | None
    relative_gap: float | None
    solver_runtime_seconds: float
    solver_node_count: int | None
    formulation_statistics: FormulationStatistics
    cost_components_eur: dict[str, float]
    terminal_commitment_state: TerminalCommitmentState
    terminal_commitment_by_unit: dict[str, TerminalCommitmentState]
    numerical_diagnostics: dict[str, float]


class _ConstraintBuilder:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = variable_count
        self.row_indices: list[int] = []
        self.column_indices: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.component_counts: dict[str, int] = {}

    def add(
        self,
        coefficients: dict[int, float],
        lower: float,
        upper: float,
        *,
        component: str,
    ) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if value != 0.0:
                self.row_indices.append(row)
                self.column_indices.append(column)
                self.values.append(value)
        self.lower.append(lower)
        self.upper.append(upper)
        self.component_counts[component] = self.component_counts.get(component, 0) + 1

    def build(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.row_indices, self.column_indices)),
            shape=(len(self.lower), self.variable_count),
            dtype=np.float64,
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=np.float64),
            np.asarray(self.upper, dtype=np.float64),
        )


class UnitCommitment:
    """Generator-indexed mixed-integer unit-commitment dispatch model."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def build_formulation(
        self,
        renewable_available_mw: npt.ArrayLike,
        gross_demand_mw: npt.ArrayLike,
        thermal_availability_factors: Mapping[str, npt.ArrayLike] | None = None,
        fuel_price_series: Mapping[str, npt.ArrayLike] | None = None,
        storage_availability_factors: Mapping[str, npt.ArrayLike] | None = None,
    ) -> FormulationProblem:
        """Build the MILP formulation without solving it."""
        renewable = np.asarray(renewable_available_mw, dtype=np.float64)
        demand = np.asarray(gross_demand_mw, dtype=np.float64)
        if renewable.ndim != 1 or demand.ndim != 1 or renewable.shape != demand.shape:
            raise ValueError(
                "Renewable availability and demand must be equal one-dimensional arrays"
            )
        if renewable.size == 0:
            raise ValueError("The dispatch horizon cannot be empty")
        if np.any(~np.isfinite(renewable)) or np.any(~np.isfinite(demand)):
            raise ValueError("Dispatch inputs must be finite")
        if np.any(renewable < 0.0) or np.any(demand < 0.0):
            raise ValueError("Dispatch inputs must be non-negative")

        periods = renewable.size
        thermal_units = self._thermal_units()
        storage_units = self._storage_units()
        thermal_capacity = self._thermal_capacity_available(
            thermal_units,
            periods,
            thermal_availability_factors or {},
        )
        storage_availability = self._storage_availability_factors(
            storage_units,
            periods,
            storage_availability_factors or {},
        )
        fuel_prices = self._fuel_prices(periods, fuel_price_series or {})
        registry = self._variable_registry(periods, thermal_units, storage_units)
        objective = self._objective(registry, renewable, thermal_units, storage_units, fuel_prices)
        bounds, integrality = self._bounds(
            registry,
            renewable,
            demand,
            thermal_units,
            storage_units,
            storage_availability,
        )
        constraints, component_counts = self._constraints(
            registry,
            demand,
            thermal_units,
            storage_units,
            thermal_capacity,
        )

        integer_variables = int(np.count_nonzero(integrality))
        statistics = FormulationStatistics(
            continuous_variables=registry.size - integer_variables,
            integer_variables=integer_variables,
            binary_variables=integer_variables,
            linear_constraints=constraints.A.shape[0],
            matrix_nonzeros=constraints.A.nnz,
            variable_counts_by_block=registry.variable_counts_by_block(),
            constraint_counts_by_component=component_counts,
        )
        return FormulationProblem(
            objective=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            renewable_available_mw=renewable,
            gross_demand_mw=demand,
            thermal_units=thermal_units,
            storage_units=storage_units,
            thermal_capacity_available_mw=thermal_capacity,
            storage_availability_factor=storage_availability,
            fuel_prices_eur_per_mwh_thermal=fuel_prices,
            registry=registry,
            statistics=statistics,
        )

    def solve(
        self,
        renewable_available_mw: npt.ArrayLike,
        gross_demand_mw: npt.ArrayLike,
        thermal_availability_factors: Mapping[str, npt.ArrayLike] | None = None,
        fuel_price_series: Mapping[str, npt.ArrayLike] | None = None,
        storage_availability_factors: Mapping[str, npt.ArrayLike] | None = None,
    ) -> DispatchResult:
        """Solve unit commitment over the full input horizon."""
        problem = self.build_formulation(
            renewable_available_mw,
            gross_demand_mw,
            thermal_availability_factors,
            fuel_price_series,
            storage_availability_factors,
        )
        return self.solve_formulation(problem)

    def solve_formulation(self, problem: FormulationProblem) -> DispatchResult:
        """Solve a previously built formulation."""
        solve_started = perf_counter()
        backend_result = solve_milp(
            objective=problem.objective,
            integrality=problem.integrality,
            bounds=problem.bounds,
            constraints=problem.constraints,
            time_limit_seconds=self.config.simulation.solver_time_limit_seconds,
            mip_relative_gap=self.config.simulation.mip_relative_gap,
        )
        solver = interpret_solver_result(
            backend_result,
            allow_non_optimal_solution=self.config.simulation.allow_non_optimal_solution,
        )
        solver_runtime_seconds = perf_counter() - solve_started
        if not solver.accepted or solver.solution is None:
            raise OptimisationError(
                f"Unit commitment failed with status {solver.status}: {solver.message}"
            )

        frame = self._solution_frame(problem, solver.solution)
        integrality_max_deviation = self._coerce_binary_columns(
            frame,
            problem.thermal_units,
            problem.storage_units,
        )
        frame["renewable_available_mw"] = problem.renewable_available_mw
        frame["renewable_curtailed_mw"] = (
            problem.renewable_available_mw - frame["renewable_used_mw"]
        )
        frame["gross_demand_mw"] = problem.gross_demand_mw
        self._add_thermal_accounting_columns(frame, problem.thermal_units, problem)
        self._add_storage_accounting_columns(frame, problem.storage_units)
        nonnegative_cleanup_max_abs = self._clip_nonnegative_solver_noise(
            frame,
            problem.thermal_units,
            problem.storage_units,
        )

        constant_curtailment_cost = (
            self.config.penalties.renewable_curtailment_eur_per_mwh
            * problem.renewable_available_mw.sum()
            * self.config.simulation.time_step_hours
        )
        primal_objective_eur = (
            float(solver.objective_value + constant_curtailment_cost)
            if solver.objective_value is not None
            else None
        )
        cost_components = self._cost_components(frame, problem.thermal_units, problem.storage_units)
        objective_eur = float(sum(cost_components.values()))
        if (
            primal_objective_eur is not None
            and abs(primal_objective_eur - objective_eur)
            > DEFAULT_NUMERICAL_POLICY.objective_reconciliation_eur
        ):
            raise OptimisationError(
                "Reported dispatch cost components do not reconcile with solver objective"
            )
        objective_bound_eur = objective_bound_with_constant(
            solver.objective_bound,
            constant_curtailment_cost,
        )
        terminal_by_unit = {
            unit.id: self._terminal_commitment_state_for_unit(frame, unit)
            for unit in problem.thermal_units
        }
        primary_unit_id = problem.thermal_units[0].id
        return DispatchResult(
            frame=frame,
            objective_eur=objective_eur,
            solver_message=solver.message,
            solver_status=solver.status,
            backend_solver_status=solver.backend_status,
            backend_solver_status_code=solver.status_code,
            mip_gap=solver.backend_relative_gap,
            primal_objective_eur=primal_objective_eur,
            objective_bound_eur=objective_bound_eur,
            absolute_gap_eur=absolute_gap(objective_eur, objective_bound_eur),
            relative_gap=relative_gap(objective_eur, objective_bound_eur),
            solver_runtime_seconds=solver_runtime_seconds,
            solver_node_count=solver.node_count,
            formulation_statistics=problem.statistics,
            cost_components_eur=cost_components,
            terminal_commitment_state=terminal_by_unit[primary_unit_id],
            terminal_commitment_by_unit=terminal_by_unit,
            numerical_diagnostics={
                "integrality_max_deviation": integrality_max_deviation,
                "nonnegative_cleanup_max_abs": nonnegative_cleanup_max_abs,
            },
        )

    def _thermal_units(self) -> tuple[ThermalUnit, ...]:
        configured = self.config.portfolio.thermal_generators
        if len(configured) == 1:
            unit = configured[0]
            return (
                ThermalUnit(
                    id=unit.id,
                    fuel_id=unit.fuel_id,
                    config=self.config.thermal,
                    must_run=unit.must_run,
                    availability_factor=unit.availability_factor,
                    availability_factor_key=unit.availability_factor_key,
                ),
            )
        return tuple(
            ThermalUnit(
                id=unit.id,
                fuel_id=unit.fuel_id,
                config=unit.config,
                must_run=unit.must_run,
                availability_factor=unit.availability_factor,
                availability_factor_key=unit.availability_factor_key,
            )
            for unit in configured
        )

    def _storage_units(self) -> tuple[StorageUnit, ...]:
        configured = self.config.portfolio.storage_units
        if len(configured) == 1:
            unit = configured[0]
            return (StorageUnit(id=unit.id, config=self.config.battery),)
        return tuple(StorageUnit(id=unit.id, config=unit.config) for unit in configured)

    def _fuels_by_id(self) -> dict[str, FuelConfig]:
        fuels = {fuel.id: fuel for fuel in self.config.portfolio.fuels}
        for generator in self.config.portfolio.thermal_generators:
            fuels.setdefault(
                generator.fuel_id,
                FuelConfig(
                    id=generator.fuel_id,
                    price_eur_per_mwh_thermal=0.0,
                    co2_factor_tonnes_per_mwh_thermal=0.0,
                ),
            )
        return fuels

    def _fuel_prices(
        self,
        periods: int,
        fuel_price_series: Mapping[str, npt.ArrayLike],
    ) -> dict[str, FloatArray]:
        prices: dict[str, FloatArray] = {}
        for fuel in self._fuels_by_id().values():
            values = np.full(periods, fuel.price_eur_per_mwh_thermal, dtype=np.float64)
            if fuel.id in fuel_price_series:
                values = np.asarray(fuel_price_series[fuel.id], dtype=np.float64)
                if values.shape != (periods,):
                    raise ValueError(f"Fuel price series for {fuel.id} has wrong shape")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"Fuel prices for {fuel.id} must be finite and non-negative")
            prices[fuel.id] = values
        return prices

    def _storage_availability_factors(
        self,
        units: tuple[StorageUnit, ...],
        periods: int,
        storage_availability_factors: Mapping[str, npt.ArrayLike],
    ) -> dict[str, FloatArray]:
        result: dict[str, FloatArray] = {}
        for unit in units:
            factor = np.full(periods, unit.config.availability_factor, dtype=np.float64)
            if unit.id in storage_availability_factors:
                series_factor = np.asarray(storage_availability_factors[unit.id], dtype=np.float64)
                if series_factor.shape != (periods,):
                    raise ValueError(f"Storage availability factor for {unit.id} has wrong shape")
                factor = factor * series_factor
            if np.any(~np.isfinite(factor)) or np.any((factor < 0.0) | (factor > 1.0)):
                raise ValueError(
                    f"Storage availability factors for {unit.id} must be finite in [0, 1]"
                )
            result[unit.id] = factor
        return result

    def _thermal_capacity_available(
        self,
        units: tuple[ThermalUnit, ...],
        periods: int,
        thermal_availability_factors: Mapping[str, npt.ArrayLike],
    ) -> dict[str, FloatArray]:
        result: dict[str, FloatArray] = {}
        for unit in units:
            factor = np.full(periods, unit.availability_factor, dtype=np.float64)
            if unit.id in thermal_availability_factors:
                series_factor = np.asarray(thermal_availability_factors[unit.id], dtype=np.float64)
                if series_factor.shape != (periods,):
                    raise ValueError(f"Availability factor for {unit.id} has wrong shape")
                factor = factor * series_factor
            if np.any(~np.isfinite(factor)) or np.any((factor < 0.0) | (factor > 1.0)):
                raise ValueError(f"Availability factors for {unit.id} must be finite in [0, 1]")
            result[unit.id] = factor * unit.config.maximum_output_mw
        return result

    def _variable_registry(
        self,
        periods: int,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
    ) -> VariableRegistry:
        registry = VariableRegistry()
        for block in SYSTEM_BLOCKS:
            registry.add(block, periods, binary=block == "battery_charge_mode")
        for storage in storage_units:
            for block in STORAGE_BLOCKS:
                registry.add(
                    block,
                    periods,
                    asset_id=storage.id,
                    binary=block in {"storage_charge_mode", "storage_discharge_mode"},
                )
            for band in storage.config.degradation_bands:
                registry.add(
                    STORAGE_DEGRADATION_BLOCK,
                    periods,
                    asset_id=_storage_degradation_asset_id(storage.id, band.id),
                )
        for thermal_unit in thermal_units:
            for block in THERMAL_BLOCKS:
                registry.add(
                    block,
                    periods,
                    asset_id=thermal_unit.id,
                    binary=block in {"thermal_on", "thermal_startup", "thermal_shutdown"},
                )
            for segment in thermal_unit.config.heat_rate_segments:
                registry.add(
                    THERMAL_SEGMENT_OUTPUT_BLOCK,
                    periods,
                    asset_id=_segment_asset_id(thermal_unit.id, segment.id),
                )
            for category in thermal_unit.config.startup_categories:
                registry.add(
                    THERMAL_STARTUP_CATEGORY_BLOCK,
                    periods,
                    asset_id=_startup_category_asset_id(thermal_unit.id, category.id),
                    binary=True,
                )
        return registry

    def _objective(
        self,
        registry: VariableRegistry,
        renewable: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        fuel_prices: dict[str, FloatArray],
    ) -> FloatArray:
        periods = renewable.size
        dt = self.config.simulation.time_step_hours
        imports = self.config.imports
        penalties = self.config.penalties
        fuels = self._fuels_by_id()
        coefficients = np.zeros(registry.size, dtype=np.float64)

        for t in range(periods):
            coefficients[registry.at("renewable_used_mw", t)] = (
                -penalties.renewable_curtailment_eur_per_mwh * dt
            )
            for storage in storage_units:
                battery = storage.config
                coefficients[registry.at("storage_charge_mw", t, asset_id=storage.id)] = (
                    battery.throughput_cost_eur_per_mwh * dt
                )
                coefficients[registry.at("storage_discharge_mw", t, asset_id=storage.id)] = (
                    battery.throughput_cost_eur_per_mwh * dt
                )
                for band in battery.degradation_bands:
                    coefficients[
                        registry.at(
                            STORAGE_DEGRADATION_BLOCK,
                            t,
                            asset_id=_storage_degradation_asset_id(storage.id, band.id),
                        )
                    ] = band.cost_eur_per_mwh
            coefficients[registry.at("imports_mw", t)] = dt * (
                imports.price_eur_per_mwh
                + penalties.carbon_price_eur_per_tonne * imports.emission_factor_tonnes_per_mwh
            )
            coefficients[registry.at("source_load_shed_mw", t)] = (
                penalties.lost_load_eur_per_mwh * (1.0 - self.config.network.loss_fraction) * dt
            )
            for unit in thermal_units:
                thermal = unit.config
                fuel = fuels[unit.fuel_id]
                fuel_price = fuel_prices[unit.fuel_id][t]
                carbon_price = penalties.carbon_price_eur_per_tonne
                if thermal.heat_rate_segments:
                    coefficients[registry.at("thermal_output_mw", t, asset_id=unit.id)] = 0.0
                    coefficients[registry.at("thermal_on", t, asset_id=unit.id)] = dt * (
                        thermal.no_load_cost_eur_per_hour
                        + thermal.minimum_fuel_input_mwh_per_hour
                        * (fuel_price + carbon_price * fuel.co2_factor_tonnes_per_mwh_thermal)
                    )
                    for segment in thermal.heat_rate_segments:
                        segment_input_cost = segment.heat_rate_mwh_thermal_per_mwh * (
                            fuel_price + carbon_price * fuel.co2_factor_tonnes_per_mwh_thermal
                        )
                        coefficients[
                            registry.at(
                                THERMAL_SEGMENT_OUTPUT_BLOCK,
                                t,
                                asset_id=_segment_asset_id(unit.id, segment.id),
                            )
                        ] = dt * segment_input_cost
                else:
                    coefficients[registry.at("thermal_output_mw", t, asset_id=unit.id)] = dt * (
                        thermal.variable_cost_eur_per_mwh
                        + carbon_price * thermal.emission_factor_tonnes_per_mwh
                    )
                    coefficients[registry.at("thermal_on", t, asset_id=unit.id)] = (
                        thermal.no_load_cost_eur_per_hour * dt
                    )
                if thermal.startup_categories:
                    coefficients[registry.at("thermal_startup", t, asset_id=unit.id)] = 0.0
                    for category in thermal.startup_categories:
                        startup_fuel_cost = category.startup_fuel_input_mwh_thermal * (
                            fuel_price + carbon_price * fuel.co2_factor_tonnes_per_mwh_thermal
                        )
                        coefficients[
                            registry.at(
                                THERMAL_STARTUP_CATEGORY_BLOCK,
                                t,
                                asset_id=_startup_category_asset_id(unit.id, category.id),
                            )
                        ] = category.startup_cost_eur + startup_fuel_cost
                else:
                    coefficients[registry.at("thermal_startup", t, asset_id=unit.id)] = (
                        thermal.startup_cost_eur
                    )
                coefficients[registry.at("thermal_shutdown", t, asset_id=unit.id)] = (
                    thermal.shutdown_cost_eur
                )
        return coefficients

    def _bounds(
        self,
        registry: VariableRegistry,
        renewable: FloatArray,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        storage_availability: dict[str, FloatArray],
    ) -> tuple[Bounds, npt.NDArray[np.int_]]:
        periods = renewable.size
        lower = np.zeros(registry.size, dtype=np.float64)
        upper = np.full(registry.size, np.inf, dtype=np.float64)
        integrality = registry.integrality()

        for t in range(periods):
            upper[registry.at("renewable_used_mw", t)] = renewable[t]
            upper[registry.at("imports_mw", t)] = self.config.imports.maximum_power_mw
            upper[registry.at("source_load_shed_mw", t)] = demand[t]
            for storage in storage_units:
                battery = storage.config
                availability = storage_availability[storage.id][t]
                upper[registry.at("storage_charge_mw", t, asset_id=storage.id)] = (
                    _storage_charge_capacity(battery) * availability
                )
                upper[registry.at("storage_discharge_mw", t, asset_id=storage.id)] = (
                    _storage_discharge_capacity(battery) * availability
                )
                lower[registry.at("storage_soc_mwh", t, asset_id=storage.id)] = (
                    battery.minimum_soc_mwh
                )
                upper[registry.at("storage_soc_mwh", t, asset_id=storage.id)] = (
                    battery.maximum_soc_mwh
                )
                upper[registry.at("storage_charge_mode", t, asset_id=storage.id)] = 1.0
                upper[registry.at("storage_discharge_mode", t, asset_id=storage.id)] = 1.0
                for band in battery.degradation_bands:
                    upper[
                        registry.at(
                            STORAGE_DEGRADATION_BLOCK,
                            t,
                            asset_id=_storage_degradation_asset_id(storage.id, band.id),
                        )
                    ] = band.capacity_mwh
            for unit in thermal_units:
                upper[registry.at("thermal_output_mw", t, asset_id=unit.id)] = (
                    unit.config.maximum_output_mw
                )
                for block in ("thermal_on", "thermal_startup", "thermal_shutdown"):
                    upper[registry.at(block, t, asset_id=unit.id)] = 1.0
                for segment in unit.config.heat_rate_segments:
                    upper[
                        registry.at(
                            THERMAL_SEGMENT_OUTPUT_BLOCK,
                            t,
                            asset_id=_segment_asset_id(unit.id, segment.id),
                        )
                    ] = segment.capacity_mw
                for category in unit.config.startup_categories:
                    upper[
                        registry.at(
                            THERMAL_STARTUP_CATEGORY_BLOCK,
                            t,
                            asset_id=_startup_category_asset_id(unit.id, category.id),
                        )
                    ] = 1.0
        return Bounds(lower, upper), integrality

    def _constraints(
        self,
        registry: VariableRegistry,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        thermal_capacity_available: dict[str, FloatArray],
    ) -> tuple[LinearConstraint, dict[str, int]]:
        builder = _ConstraintBuilder(registry.size)
        self._add_balance_constraints(builder, registry, demand, thermal_units, storage_units)
        self._add_thermal_constraints(
            builder,
            registry,
            demand.size,
            thermal_units,
            thermal_capacity_available,
        )
        self._add_storage_constraints(builder, registry, demand.size, storage_units)
        self._add_terminal_soc_constraints(builder, registry, demand.size, storage_units)
        return builder.build(), builder.component_counts

    def _add_balance_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
    ) -> None:
        for t, value in enumerate(demand):
            coefficients = {
                registry.at("renewable_used_mw", t): 1.0,
                registry.at("imports_mw", t): 1.0,
                registry.at("source_load_shed_mw", t): 1.0,
            }
            for unit in thermal_units:
                coefficients[registry.at("thermal_output_mw", t, asset_id=unit.id)] = 1.0
            for storage in storage_units:
                coefficients[registry.at("storage_discharge_mw", t, asset_id=storage.id)] = 1.0
                coefficients[registry.at("storage_charge_mw", t, asset_id=storage.id)] = -1.0
            builder.add(coefficients, value, value, component="balance")

    def _add_thermal_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
        units: tuple[ThermalUnit, ...],
        capacity_available: dict[str, FloatArray],
    ) -> None:
        dt = self.config.simulation.time_step_hours
        for unit in units:
            thermal = unit.config
            for t in range(periods):
                self._add_thermal_bounds(builder, registry, unit, t, capacity_available[unit.id][t])
                self._add_heat_rate_segment_constraints(builder, registry, unit, t)
                self._add_thermal_state(builder, registry, unit, t)
                self._add_startup_category_constraints(builder, registry, unit, t)
                self._add_thermal_ramps(builder, registry, unit, t, dt)
                if unit.must_run:
                    builder.add(
                        {registry.at("thermal_on", t, asset_id=unit.id): 1.0},
                        1.0,
                        1.0,
                        component="thermal_must_run",
                    )
            self._add_minimum_duration_constraints(builder, registry, unit, periods)
            self._add_initial_duration_obligations(builder, registry, unit, periods)
            self._add_terminal_commitment_constraints(builder, registry, unit, periods)
            if thermal.terminal_commitment_mode == "fixed_terminal_commitment":
                terminal_on = 1.0 if thermal.terminal_on else 0.0
                builder.add(
                    {registry.at("thermal_on", periods - 1, asset_id=unit.id): 1.0},
                    terminal_on,
                    terminal_on,
                    component="thermal_terminal_fixed",
                )

    def _add_thermal_bounds(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        period: int,
        capacity_available_mw: float,
    ) -> None:
        thermal = unit.config
        output = registry.at("thermal_output_mw", period, asset_id=unit.id)
        online = registry.at("thermal_on", period, asset_id=unit.id)
        builder.add(
            {output: 1.0, online: -capacity_available_mw},
            -np.inf,
            0.0,
            component="thermal_capacity",
        )
        builder.add(
            {output: 1.0, online: -thermal.minimum_output_mw},
            0.0,
            np.inf,
            component="thermal_minimum_output",
        )

    def _add_heat_rate_segment_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        period: int,
    ) -> None:
        thermal = unit.config
        if not thermal.heat_rate_segments:
            return
        output = registry.at("thermal_output_mw", period, asset_id=unit.id)
        online = registry.at("thermal_on", period, asset_id=unit.id)
        coefficients = {output: 1.0, online: -thermal.minimum_output_mw}
        for segment in thermal.heat_rate_segments:
            segment_output = registry.at(
                THERMAL_SEGMENT_OUTPUT_BLOCK,
                period,
                asset_id=_segment_asset_id(unit.id, segment.id),
            )
            coefficients[segment_output] = -1.0
            builder.add(
                {segment_output: 1.0, online: -segment.capacity_mw},
                -np.inf,
                0.0,
                component="thermal_segment_capacity",
            )
        builder.add(
            coefficients,
            0.0,
            0.0,
            component="thermal_segment_output_sum",
        )

    def _add_startup_category_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        period: int,
    ) -> None:
        categories = unit.config.startup_categories
        if not categories:
            return
        startup = registry.at("thermal_startup", period, asset_id=unit.id)
        category_columns = {
            registry.at(
                THERMAL_STARTUP_CATEGORY_BLOCK,
                period,
                asset_id=_startup_category_asset_id(unit.id, category.id),
            ): 1.0
            for category in categories
        }
        builder.add(
            {**category_columns, startup: -1.0},
            0.0,
            0.0,
            component="thermal_startup_category_sum",
        )
        dt = self.config.simulation.time_step_hours
        for index, category in enumerate(categories):
            category_column = registry.at(
                THERMAL_STARTUP_CATEGORY_BLOCK,
                period,
                asset_id=_startup_category_asset_id(unit.id, category.id),
            )
            self._add_startup_category_minimum_downtime(
                builder,
                registry,
                unit,
                period,
                category_column,
                category.minimum_down_time_hours,
                dt,
            )
            if index + 1 < len(categories):
                self._add_startup_category_maximum_downtime(
                    builder,
                    registry,
                    unit,
                    period,
                    category_column,
                    categories[index + 1].minimum_down_time_hours,
                    dt,
                )

    def _add_startup_category_minimum_downtime(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        period: int,
        category_column: int,
        minimum_down_time_hours: float,
        dt: float,
    ) -> None:
        if minimum_down_time_hours == 0.0:
            return
        if not unit.config.initial_on:
            elapsed_initial_down = unit.config.initial_down_time_hours + period * dt
            if elapsed_initial_down < minimum_down_time_hours:
                builder.add(
                    {category_column: 1.0},
                    0.0,
                    0.0,
                    component="thermal_startup_category_min_down",
                )
        periods = self._duration_periods(minimum_down_time_hours)
        for offset in range(1, periods + 1):
            prior = period - offset
            if prior >= 0:
                builder.add(
                    {
                        category_column: 1.0,
                        registry.at("thermal_on", prior, asset_id=unit.id): 1.0,
                    },
                    -np.inf,
                    1.0,
                    component="thermal_startup_category_min_down",
                )

    def _add_startup_category_maximum_downtime(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        period: int,
        category_column: int,
        next_minimum_down_time_hours: float,
        dt: float,
    ) -> None:
        periods = self._duration_periods(next_minimum_down_time_hours)
        coefficients = {category_column: 1.0}
        recent_initial_online = self._initial_online_within_lookback(
            unit,
            period,
            next_minimum_down_time_hours,
            dt,
        )
        for offset in range(1, periods + 1):
            prior = period - offset
            if prior >= 0:
                coefficients[registry.at("thermal_on", prior, asset_id=unit.id)] = -1.0
        upper = recent_initial_online
        builder.add(
            coefficients,
            -np.inf,
            upper,
            component="thermal_startup_category_max_down",
        )

    @staticmethod
    def _initial_online_within_lookback(
        unit: ThermalUnit,
        period: int,
        lookback_hours: float,
        dt: float,
    ) -> float:
        if unit.config.initial_on:
            return 1.0
        elapsed_down_at_period_start = unit.config.initial_down_time_hours + period * dt
        return 1.0 if elapsed_down_at_period_start < lookback_hours else 0.0

    def _add_thermal_state(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        period: int,
    ) -> None:
        coefficients = {
            registry.at("thermal_on", period, asset_id=unit.id): 1.0,
            registry.at("thermal_startup", period, asset_id=unit.id): -1.0,
            registry.at("thermal_shutdown", period, asset_id=unit.id): 1.0,
        }
        rhs = float(unit.config.initial_on) if period == 0 else 0.0
        if period > 0:
            coefficients[registry.at("thermal_on", period - 1, asset_id=unit.id)] = -1.0
        builder.add(coefficients, rhs, rhs, component="thermal_state")
        builder.add(
            {
                registry.at("thermal_startup", period, asset_id=unit.id): 1.0,
                registry.at("thermal_shutdown", period, asset_id=unit.id): 1.0,
            },
            -np.inf,
            1.0,
            component="thermal_transition_exclusivity",
        )

    def _add_thermal_ramps(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        period: int,
        dt: float,
    ) -> None:
        thermal = unit.config
        ramp_up_coefficients = {
            registry.at("thermal_output_mw", period, asset_id=unit.id): 1.0,
            registry.at("thermal_startup", period, asset_id=unit.id): -thermal.startup_ramp_mw,
        }
        ramp_down_coefficients = {
            registry.at("thermal_output_mw", period, asset_id=unit.id): -1.0,
            registry.at("thermal_shutdown", period, asset_id=unit.id): -thermal.shutdown_ramp_mw,
        }
        if period > 0:
            previous_output = registry.at("thermal_output_mw", period - 1, asset_id=unit.id)
            ramp_up_coefficients[previous_output] = -1.0
            ramp_up_coefficients[registry.at("thermal_on", period - 1, asset_id=unit.id)] = (
                -thermal.ramp_up_mw_per_hour * dt
            )
            ramp_down_coefficients[
                registry.at("thermal_output_mw", period - 1, asset_id=unit.id)
            ] = 1.0
            ramp_down_coefficients[registry.at("thermal_on", period, asset_id=unit.id)] = (
                -thermal.ramp_down_mw_per_hour * dt
            )
            ramp_up_upper = 0.0
            ramp_down_upper = 0.0
        else:
            ramp_up_upper = thermal.initial_output_mw + thermal.ramp_up_mw_per_hour * dt * float(
                thermal.initial_on
            )
            ramp_down_coefficients[registry.at("thermal_on", period, asset_id=unit.id)] = (
                -thermal.ramp_down_mw_per_hour * dt
            )
            ramp_down_upper = -thermal.initial_output_mw
        builder.add(ramp_up_coefficients, -np.inf, ramp_up_upper, component="thermal_ramp_up")
        builder.add(
            ramp_down_coefficients,
            -np.inf,
            ramp_down_upper,
            component="thermal_ramp_down",
        )

    def _add_minimum_duration_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        periods: int,
    ) -> None:
        up = self._duration_periods(unit.config.minimum_up_hours)
        down = self._duration_periods(unit.config.minimum_down_hours)
        for t in range(periods):
            recent_startups = {
                registry.at("thermal_startup", k, asset_id=unit.id): 1.0
                for k in range(max(0, t - up + 1), t + 1)
            }
            recent_startups[registry.at("thermal_on", t, asset_id=unit.id)] = -1.0
            builder.add(recent_startups, -np.inf, 0.0, component="thermal_minimum_up")
            recent_shutdowns = {
                registry.at("thermal_shutdown", k, asset_id=unit.id): 1.0
                for k in range(max(0, t - down + 1), t + 1)
            }
            recent_shutdowns[registry.at("thermal_on", t, asset_id=unit.id)] = 1.0
            builder.add(recent_shutdowns, -np.inf, 1.0, component="thermal_minimum_down")

    def _add_initial_duration_obligations(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        periods: int,
    ) -> None:
        thermal = unit.config
        dt = self.config.simulation.time_step_hours
        if thermal.initial_on:
            remaining_hours = max(0.0, thermal.minimum_up_hours - thermal.initial_up_time_hours)
            forced_periods = min(periods, ceil(remaining_hours / dt))
            for t in range(forced_periods):
                builder.add(
                    {registry.at("thermal_on", t, asset_id=unit.id): 1.0},
                    1.0,
                    1.0,
                    component="thermal_initial_up",
                )
            return

        remaining_hours = max(0.0, thermal.minimum_down_hours - thermal.initial_down_time_hours)
        forced_periods = min(periods, ceil(remaining_hours / dt))
        for t in range(forced_periods):
            builder.add(
                {registry.at("thermal_on", t, asset_id=unit.id): 1.0},
                0.0,
                0.0,
                component="thermal_initial_down",
            )

    def _add_terminal_commitment_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: ThermalUnit,
        periods: int,
    ) -> None:
        thermal = unit.config
        if thermal.terminal_commitment_mode not in {
            "forbid_incomplete_transitions",
            "fixed_terminal_commitment",
        }:
            return
        up_periods = self._duration_periods(thermal.minimum_up_hours)
        down_periods = self._duration_periods(thermal.minimum_down_hours)
        for t in range(max(0, periods - up_periods + 1), periods):
            builder.add(
                {registry.at("thermal_startup", t, asset_id=unit.id): 1.0},
                0.0,
                0.0,
                component="thermal_terminal_up",
            )
        for t in range(max(0, periods - down_periods + 1), periods):
            builder.add(
                {registry.at("thermal_shutdown", t, asset_id=unit.id): 1.0},
                0.0,
                0.0,
                component="thermal_terminal_down",
            )

    def _add_storage_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
        storage_units: tuple[StorageUnit, ...],
    ) -> None:
        dt = self.config.simulation.time_step_hours
        for storage in storage_units:
            battery = storage.config
            charge_capacity = _storage_charge_capacity(battery)
            discharge_capacity = _storage_discharge_capacity(battery)
            retention = (1.0 - battery.self_discharge_rate_per_hour) ** dt
            for t in range(periods):
                charge = registry.at("storage_charge_mw", t, asset_id=storage.id)
                discharge = registry.at("storage_discharge_mw", t, asset_id=storage.id)
                charge_mode = registry.at("storage_charge_mode", t, asset_id=storage.id)
                discharge_mode = registry.at("storage_discharge_mode", t, asset_id=storage.id)
                builder.add(
                    {charge: 1.0, charge_mode: -charge_capacity},
                    -np.inf,
                    0.0,
                    component="storage_charge_mode",
                )
                builder.add(
                    {discharge: 1.0, discharge_mode: -discharge_capacity},
                    -np.inf,
                    0.0,
                    component="storage_discharge_mode",
                )
                if battery.minimum_charge_mw > 0.0:
                    builder.add(
                        {charge: 1.0, charge_mode: -battery.minimum_charge_mw},
                        0.0,
                        np.inf,
                        component="storage_minimum_charge",
                    )
                if battery.minimum_discharge_mw > 0.0:
                    builder.add(
                        {discharge: 1.0, discharge_mode: -battery.minimum_discharge_mw},
                        0.0,
                        np.inf,
                        component="storage_minimum_discharge",
                    )
                builder.add(
                    {charge_mode: 1.0, discharge_mode: 1.0},
                    -np.inf,
                    1.0,
                    component="storage_mode_exclusivity",
                )
                soc_coefficients = {
                    registry.at("storage_soc_mwh", t, asset_id=storage.id): 1.0,
                    charge: -battery.charge_efficiency * dt,
                    discharge: dt / battery.discharge_efficiency,
                }
                soc_rhs = retention * battery.initial_soc_mwh if t == 0 else 0.0
                if t > 0:
                    soc_coefficients[
                        registry.at("storage_soc_mwh", t - 1, asset_id=storage.id)
                    ] = -retention
                builder.add(soc_coefficients, soc_rhs, soc_rhs, component="storage_soc")
                self._add_storage_ramp_constraints(builder, registry, storage, t, dt)
                self._add_storage_degradation_constraints(builder, registry, storage, t, dt)

    def _add_storage_ramp_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        storage: StorageUnit,
        period: int,
        dt: float,
    ) -> None:
        battery = storage.config
        for block, limit, component in (
            ("storage_charge_mw", battery.charge_ramp_mw_per_hour, "storage_charge_ramp"),
            (
                "storage_discharge_mw",
                battery.discharge_ramp_mw_per_hour,
                "storage_discharge_ramp",
            ),
        ):
            if limit is None:
                continue
            current = registry.at(block, period, asset_id=storage.id)
            previous_value = 0.0
            coefficients = {current: 1.0}
            if period > 0:
                previous = registry.at(block, period - 1, asset_id=storage.id)
                coefficients[previous] = -1.0
            builder.add(coefficients, -np.inf, previous_value + limit * dt, component=component)
            coefficients_down = {current: -1.0}
            if period > 0:
                coefficients_down[registry.at(block, period - 1, asset_id=storage.id)] = 1.0
            builder.add(
                coefficients_down,
                -np.inf,
                previous_value + limit * dt,
                component=component,
            )

    def _add_storage_degradation_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        storage: StorageUnit,
        period: int,
        dt: float,
    ) -> None:
        if not storage.config.degradation_bands:
            return
        coefficients = {
            registry.at("storage_charge_mw", period, asset_id=storage.id): -dt,
            registry.at("storage_discharge_mw", period, asset_id=storage.id): -dt,
        }
        for band in storage.config.degradation_bands:
            coefficients[
                registry.at(
                    STORAGE_DEGRADATION_BLOCK,
                    period,
                    asset_id=_storage_degradation_asset_id(storage.id, band.id),
                )
            ] = 1.0
        builder.add(coefficients, 0.0, 0.0, component="storage_degradation_throughput")

    def _add_terminal_soc_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
        storage_units: tuple[StorageUnit, ...],
    ) -> None:
        for storage in storage_units:
            battery = storage.config
            terminal = {registry.at("storage_soc_mwh", periods - 1, asset_id=storage.id): 1.0}
            if battery.terminal_soc_mode == "minimum":
                builder.add(
                    terminal,
                    battery.minimum_final_soc_mwh,
                    np.inf,
                    component="storage_terminal",
                )
            elif battery.terminal_soc_mode == "exact":
                builder.add(
                    terminal,
                    battery.minimum_final_soc_mwh,
                    battery.minimum_final_soc_mwh,
                    component="storage_terminal",
                )
            elif battery.terminal_soc_mode == "cyclic":
                builder.add(
                    terminal,
                    battery.initial_soc_mwh,
                    battery.initial_soc_mwh,
                    component="storage_terminal",
                )

    def _duration_periods(self, duration_hours: float) -> int:
        return max(1, ceil(duration_hours / self.config.simulation.time_step_hours))

    def _solution_frame(self, problem: FormulationProblem, solution: FloatArray) -> pd.DataFrame:
        registry = problem.registry
        data: dict[str, FloatArray] = {
            block: registry.values(solution, block) for block in SYSTEM_BLOCKS
        }
        for storage in problem.storage_units:
            for block in STORAGE_BLOCKS:
                data[f"{block}__{storage.id}"] = registry.values(
                    solution,
                    block,
                    asset_id=storage.id,
                )
            for band in storage.config.degradation_bands:
                data[f"storage_degradation_throughput_mwh__{storage.id}__{band.id}"] = (
                    registry.values(
                        solution,
                        STORAGE_DEGRADATION_BLOCK,
                        asset_id=_storage_degradation_asset_id(storage.id, band.id),
                    )
                )
        for unit in problem.thermal_units:
            for block in THERMAL_BLOCKS:
                data[f"{block}__{unit.id}"] = registry.values(
                    solution,
                    block,
                    asset_id=unit.id,
                )
            for segment in unit.config.heat_rate_segments:
                data[f"thermal_segment_output_mw__{unit.id}__{segment.id}"] = registry.values(
                    solution,
                    THERMAL_SEGMENT_OUTPUT_BLOCK,
                    asset_id=_segment_asset_id(unit.id, segment.id),
                )
            for category in unit.config.startup_categories:
                data[f"thermal_startup_category__{unit.id}__{category.id}"] = registry.values(
                    solution,
                    THERMAL_STARTUP_CATEGORY_BLOCK,
                    asset_id=_startup_category_asset_id(unit.id, category.id),
                )
        frame = pd.DataFrame(data)
        self._add_storage_aggregate_columns(frame, problem.storage_units)
        for block in THERMAL_BLOCKS:
            columns = [f"{block}__{unit.id}" for unit in problem.thermal_units]
            frame[block] = frame[columns].sum(axis=1)
        return frame

    @staticmethod
    def _add_storage_aggregate_columns(
        frame: pd.DataFrame,
        storage_units: tuple[StorageUnit, ...],
    ) -> None:
        mapping = {
            "battery_charge_mw": "storage_charge_mw",
            "battery_discharge_mw": "storage_discharge_mw",
            "battery_charge_mode": "storage_charge_mode",
            "battery_soc_mwh": "storage_soc_mwh",
        }
        for aggregate, block in mapping.items():
            columns = [f"{block}__{unit.id}" for unit in storage_units]
            frame[aggregate] = frame[columns].sum(axis=1)

    def _add_thermal_accounting_columns(
        self,
        frame: pd.DataFrame,
        units: tuple[ThermalUnit, ...],
        problem: FormulationProblem,
    ) -> None:
        dt = self.config.simulation.time_step_hours
        carbon_price = self.config.penalties.carbon_price_eur_per_tonne
        fuels = self._fuels_by_id()
        for unit in units:
            thermal = unit.config
            fuel = fuels[unit.fuel_id]
            fuel_price = problem.fuel_prices_eur_per_mwh_thermal[unit.fuel_id]
            output = frame[f"thermal_output_mw__{unit.id}"]
            online = frame[f"thermal_on__{unit.id}"]
            startup = frame[f"thermal_startup__{unit.id}"]
            shutdown = frame[f"thermal_shutdown__{unit.id}"]
            capacity = problem.thermal_capacity_available_mw[unit.id]
            frame[f"thermal_capacity_available_mw__{unit.id}"] = capacity
            frame[f"thermal_capacity_factor__{unit.id}"] = np.divide(
                output.to_numpy(dtype=np.float64),
                capacity,
                out=np.zeros_like(capacity, dtype=np.float64),
                where=capacity > 0.0,
            )
            running_fuel_input = np.zeros(len(frame), dtype=np.float64)
            startup_fuel_input = np.zeros(len(frame), dtype=np.float64)
            startup_fixed_cost = np.zeros(len(frame), dtype=np.float64)
            if thermal.heat_rate_segments:
                running_fuel_input += (
                    online.to_numpy(dtype=np.float64) * thermal.minimum_fuel_input_mwh_per_hour * dt
                )
                for segment in thermal.heat_rate_segments:
                    segment_output = frame[
                        f"thermal_segment_output_mw__{unit.id}__{segment.id}"
                    ].to_numpy(dtype=np.float64)
                    running_fuel_input += (
                        segment_output * segment.heat_rate_mwh_thermal_per_mwh * dt
                    )
                    frame[f"thermal_segment_marginal_cost_eur_per_mwh__{unit.id}__{segment.id}"] = (
                        fuel_price * segment.heat_rate_mwh_thermal_per_mwh
                    )
            if thermal.startup_categories:
                for category in thermal.startup_categories:
                    category_startups = frame[
                        f"thermal_startup_category__{unit.id}__{category.id}"
                    ].to_numpy(dtype=np.float64)
                    startup_fuel_input += (
                        category_startups * category.startup_fuel_input_mwh_thermal
                    )
                    startup_fixed_cost += category_startups * category.startup_cost_eur
            else:
                startup_fixed_cost += startup.to_numpy(dtype=np.float64) * thermal.startup_cost_eur

            if thermal.heat_rate_segments:
                running_fuel_cost = running_fuel_input * fuel_price
                frame[f"thermal_fuel_input_mwh_thermal__{unit.id}"] = (
                    running_fuel_input + startup_fuel_input
                )
                frame[f"thermal_running_fuel_input_mwh_thermal__{unit.id}"] = running_fuel_input
                frame[f"thermal_startup_fuel_input_mwh_thermal__{unit.id}"] = startup_fuel_input
                frame[f"thermal_fuel_cost_eur__{unit.id}"] = (
                    running_fuel_cost + startup_fuel_input * fuel_price
                )
                frame[f"thermal_variable_cost_eur__{unit.id}"] = running_fuel_cost
                running_co2 = running_fuel_input * fuel.co2_factor_tonnes_per_mwh_thermal
                startup_co2 = startup_fuel_input * fuel.co2_factor_tonnes_per_mwh_thermal
                frame[f"thermal_direct_co2_emissions_tonnes__{unit.id}"] = running_co2 + startup_co2
                frame[f"thermal_emissions_tonnes__{unit.id}"] = running_co2 + startup_co2
                frame[f"thermal_methane_emissions_tonnes__{unit.id}"] = (
                    running_fuel_input + startup_fuel_input
                ) * fuel.methane_factor_tonnes_per_mwh_thermal
                frame[f"thermal_nox_emissions_kg__{unit.id}"] = (
                    running_fuel_input + startup_fuel_input
                ) * fuel.nox_factor_kg_per_mwh_thermal
                frame[f"thermal_sox_emissions_kg__{unit.id}"] = (
                    running_fuel_input + startup_fuel_input
                ) * fuel.sox_factor_kg_per_mwh_thermal
                output_mwh = output.to_numpy(dtype=np.float64) * dt
                total_fuel_input = running_fuel_input + startup_fuel_input
                frame[f"thermal_efficiency__{unit.id}"] = np.divide(
                    output_mwh,
                    total_fuel_input,
                    out=np.zeros_like(total_fuel_input, dtype=np.float64),
                    where=total_fuel_input > 0.0,
                )
            else:
                legacy_running_co2 = output * dt * thermal.emission_factor_tonnes_per_mwh
                startup_co2 = startup_fuel_input * fuel.co2_factor_tonnes_per_mwh_thermal
                frame[f"thermal_fuel_input_mwh_thermal__{unit.id}"] = startup_fuel_input
                frame[f"thermal_running_fuel_input_mwh_thermal__{unit.id}"] = 0.0
                frame[f"thermal_startup_fuel_input_mwh_thermal__{unit.id}"] = startup_fuel_input
                frame[f"thermal_fuel_cost_eur__{unit.id}"] = startup_fuel_input * fuel_price
                frame[f"thermal_variable_cost_eur__{unit.id}"] = (
                    output * dt * thermal.variable_cost_eur_per_mwh
                )
                frame[f"thermal_direct_co2_emissions_tonnes__{unit.id}"] = (
                    legacy_running_co2 + startup_co2
                )
                frame[f"thermal_emissions_tonnes__{unit.id}"] = legacy_running_co2 + startup_co2
                frame[f"thermal_methane_emissions_tonnes__{unit.id}"] = (
                    startup_fuel_input * fuel.methane_factor_tonnes_per_mwh_thermal
                )
                frame[f"thermal_nox_emissions_kg__{unit.id}"] = (
                    startup_fuel_input * fuel.nox_factor_kg_per_mwh_thermal
                )
                frame[f"thermal_sox_emissions_kg__{unit.id}"] = (
                    startup_fuel_input * fuel.sox_factor_kg_per_mwh_thermal
                )
                frame[f"thermal_efficiency__{unit.id}"] = 0.0
            frame[f"thermal_no_load_cost_eur__{unit.id}"] = (
                online * dt * thermal.no_load_cost_eur_per_hour
            )
            frame[f"thermal_startup_cost_eur__{unit.id}"] = (
                startup_fixed_cost + startup_fuel_input * fuel_price
            )
            frame[f"thermal_shutdown_cost_eur__{unit.id}"] = shutdown * thermal.shutdown_cost_eur
            frame[f"thermal_carbon_cost_eur__{unit.id}"] = (
                frame[f"thermal_emissions_tonnes__{unit.id}"] * carbon_price
            )
        frame["online_thermal_capacity_mw"] = sum(
            frame[f"thermal_on__{unit.id}"] * unit.config.maximum_output_mw for unit in units
        )
        frame["available_thermal_capacity_mw"] = sum(
            frame[f"thermal_capacity_available_mw__{unit.id}"] for unit in units
        )
        frame["unused_committed_capacity_mw"] = sum(
            frame[f"thermal_on__{unit.id}"] * problem.thermal_capacity_available_mw[unit.id]
            - frame[f"thermal_output_mw__{unit.id}"]
            for unit in units
        )

    def _add_storage_accounting_columns(
        self,
        frame: pd.DataFrame,
        units: tuple[StorageUnit, ...],
    ) -> None:
        dt = self.config.simulation.time_step_hours
        for unit in units:
            battery = unit.config
            charge = frame[f"storage_charge_mw__{unit.id}"]
            discharge = frame[f"storage_discharge_mw__{unit.id}"]
            soc = frame[f"storage_soc_mwh__{unit.id}"]
            throughput_mwh = (charge + discharge) * dt
            frame[f"storage_throughput_mwh__{unit.id}"] = throughput_mwh
            frame[f"storage_throughput_cost_eur__{unit.id}"] = (
                throughput_mwh * battery.throughput_cost_eur_per_mwh
            )
            degradation_cost = np.zeros(len(frame), dtype=np.float64)
            for band in battery.degradation_bands:
                column = f"storage_degradation_throughput_mwh__{unit.id}__{band.id}"
                degradation_cost += frame[column].to_numpy(dtype=np.float64) * band.cost_eur_per_mwh
            frame[f"storage_degradation_cost_eur__{unit.id}"] = degradation_cost
            charged = charge * dt
            discharged = discharge * dt
            frame[f"storage_charged_mwh__{unit.id}"] = charged
            frame[f"storage_discharged_mwh__{unit.id}"] = discharged
            previous_soc = soc.shift(1)
            previous_soc.iloc[0] = battery.initial_soc_mwh
            retention = (1.0 - battery.self_discharge_rate_per_hour) ** dt
            expected_soc = (
                retention * previous_soc
                + battery.charge_efficiency * charged
                - discharged / battery.discharge_efficiency
            )
            frame[f"storage_energy_residual_mwh__{unit.id}"] = soc - expected_soc
            frame[f"storage_round_trip_losses_mwh__{unit.id}"] = (
                (1.0 - battery.charge_efficiency) * charged
                + (1.0 / battery.discharge_efficiency - 1.0) * discharged
                + (1.0 - retention) * previous_soc
            )
            usable_energy = battery.maximum_soc_mwh - battery.minimum_soc_mwh
            frame[f"storage_depth_of_discharge__{unit.id}"] = (
                (battery.maximum_soc_mwh - soc) / usable_energy if usable_energy > 0.0 else 0.0
            )
            frame[f"storage_at_min_soc__{unit.id}"] = (
                soc <= battery.minimum_soc_mwh + DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw
            ).astype(int)
            frame[f"storage_at_max_soc__{unit.id}"] = (
                soc >= battery.maximum_soc_mwh - DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw
            ).astype(int)
            frame[f"storage_equivalent_full_cycles__{unit.id}"] = (
                throughput_mwh / (2.0 * usable_energy) if usable_energy > 0.0 else 0.0
            )
        frame["storage_throughput_cost_eur"] = sum(
            frame[f"storage_throughput_cost_eur__{unit.id}"] for unit in units
        )
        frame["storage_degradation_cost_eur"] = sum(
            frame[f"storage_degradation_cost_eur__{unit.id}"] for unit in units
        )
        frame["battery_throughput_cost_eur"] = frame["storage_throughput_cost_eur"]

    def _coerce_binary_columns(
        self,
        frame: pd.DataFrame,
        thermal_units: tuple[ThermalUnit, ...] | None = None,
        storage_units: tuple[StorageUnit, ...] | None = None,
    ) -> float:
        columns: list[str] = []
        if storage_units is None:
            columns.append("battery_charge_mode")
        else:
            for storage in storage_units:
                columns.extend(
                    [
                        f"storage_charge_mode__{storage.id}",
                        f"storage_discharge_mode__{storage.id}",
                    ]
                )
        if thermal_units is None:
            columns.extend(["thermal_on", "thermal_startup", "thermal_shutdown"])
        else:
            for unit in thermal_units:
                columns.extend(
                    [
                        f"thermal_on__{unit.id}",
                        f"thermal_startup__{unit.id}",
                        f"thermal_shutdown__{unit.id}",
                    ]
                )
                columns.extend(
                    f"thermal_startup_category__{unit.id}__{category.id}"
                    for category in unit.config.startup_categories
                )
        max_deviation = 0.0
        for column in columns:
            raw = frame[column].to_numpy(dtype=np.float64)
            rounded = np.rint(raw)
            deviations = np.abs(raw - rounded)
            column_max_deviation = float(deviations.max()) if deviations.size else 0.0
            max_deviation = max(max_deviation, column_max_deviation)
            if column_max_deviation > DEFAULT_NUMERICAL_POLICY.integrality:
                period = int(deviations.argmax())
                raise OptimisationError(
                    f"Integrality residual exceeds tolerance for {column} at period {period}: "
                    f"{column_max_deviation}"
                )
            frame[column] = rounded.astype(int)
        if thermal_units is not None:
            for block in ("thermal_on", "thermal_startup", "thermal_shutdown"):
                columns_for_block = [f"{block}__{unit.id}" for unit in thermal_units]
                frame[block] = frame[columns_for_block].sum(axis=1)
        if storage_units is not None:
            self._add_storage_aggregate_columns(frame, storage_units)
        return max_deviation

    def _clip_nonnegative_solver_noise(
        self,
        frame: pd.DataFrame,
        thermal_units: tuple[ThermalUnit, ...] | None = None,
        storage_units: tuple[StorageUnit, ...] | None = None,
    ) -> float:
        columns = [
            "renewable_used_mw",
            "imports_mw",
            "source_load_shed_mw",
            "renewable_curtailed_mw",
        ]
        if storage_units is None:
            columns.extend(["battery_charge_mw", "battery_discharge_mw", "battery_soc_mwh"])
        else:
            for storage in storage_units:
                columns.extend(
                    [
                        f"storage_charge_mw__{storage.id}",
                        f"storage_discharge_mw__{storage.id}",
                        f"storage_soc_mwh__{storage.id}",
                    ]
                )
                columns.extend(
                    f"storage_degradation_throughput_mwh__{storage.id}__{band.id}"
                    for band in storage.config.degradation_bands
                )
        if thermal_units is None:
            columns.append("thermal_output_mw")
        else:
            for unit in thermal_units:
                columns.append(f"thermal_output_mw__{unit.id}")
                columns.extend(
                    f"thermal_segment_output_mw__{unit.id}__{segment.id}"
                    for segment in unit.config.heat_rate_segments
                )
        max_clipped = 0.0
        for column in columns:
            raw = frame[column].to_numpy(dtype=np.float64)
            negative = raw < 0.0
            if not negative.any():
                continue
            min_value = float(raw[negative].min())
            if min_value < -DEFAULT_NUMERICAL_POLICY.nonnegative_cleanup:
                period = int(raw.argmin())
                raise OptimisationError(
                    f"Negative solver value exceeds cleanup tolerance for {column} at "
                    f"period {period}: {min_value}"
                )
            max_clipped = max(max_clipped, abs(min_value))
            frame.loc[negative, column] = 0.0
        if thermal_units:
            frame["thermal_output_mw"] = sum(
                frame[f"thermal_output_mw__{unit.id}"] for unit in thermal_units
            )
        if storage_units:
            self._add_storage_aggregate_columns(frame, storage_units)
        return max_clipped

    def _cost_components(
        self,
        frame: pd.DataFrame,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
    ) -> dict[str, float]:
        dt = self.config.simulation.time_step_hours
        imports = self.config.imports
        penalties = self.config.penalties
        network_efficiency = 1.0 - self.config.network.loss_fraction
        return {
            "thermal_variable_cost_eur": float(
                sum(frame[f"thermal_variable_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "thermal_no_load_cost_eur": float(
                sum(frame[f"thermal_no_load_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "startup_cost_eur": float(
                sum(frame[f"thermal_startup_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "shutdown_cost_eur": float(
                sum(frame[f"thermal_shutdown_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "import_energy_cost_eur": float(
                frame["imports_mw"].sum() * dt * imports.price_eur_per_mwh
            ),
            "battery_throughput_cost_eur": float(
                sum(
                    frame[f"storage_throughput_cost_eur__{unit.id}"].sum() for unit in storage_units
                )
            ),
            "storage_degradation_cost_eur": float(
                sum(
                    frame[f"storage_degradation_cost_eur__{unit.id}"].sum()
                    for unit in storage_units
                )
            ),
            "thermal_carbon_cost_eur": float(
                sum(frame[f"thermal_carbon_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "import_carbon_cost_eur": float(
                frame["imports_mw"].sum()
                * dt
                * imports.emission_factor_tonnes_per_mwh
                * penalties.carbon_price_eur_per_tonne
            ),
            "renewable_curtailment_cost_eur": float(
                frame["renewable_curtailed_mw"].sum()
                * dt
                * penalties.renewable_curtailment_eur_per_mwh
            ),
            "dispatch_load_shedding_cost_eur": float(
                frame["source_load_shed_mw"].sum()
                * network_efficiency
                * dt
                * penalties.lost_load_eur_per_mwh
            ),
        }

    def _terminal_commitment_state_for_unit(
        self,
        frame: pd.DataFrame,
        unit: ThermalUnit,
    ) -> TerminalCommitmentState:
        thermal = unit.config
        dt = self.config.simulation.time_step_hours
        on_column = f"thermal_on__{unit.id}"
        output_column = f"thermal_output_mw__{unit.id}"
        terminal_on = bool(int(frame[on_column].iloc[-1]))
        terminal_output_mw = float(frame[output_column].iloc[-1])

        matching_periods = 0
        for value in reversed(frame[on_column].tolist()):
            if bool(int(value)) != terminal_on:
                break
            matching_periods += 1

        consecutive_hours = matching_periods * dt
        if matching_periods == len(frame):
            if terminal_on and thermal.initial_on:
                consecutive_hours += thermal.initial_up_time_hours
            elif not terminal_on and not thermal.initial_on:
                consecutive_hours += thermal.initial_down_time_hours

        consecutive_on_hours = consecutive_hours if terminal_on else 0.0
        consecutive_off_hours = 0.0 if terminal_on else consecutive_hours
        residual_up = (
            max(0.0, thermal.minimum_up_hours - consecutive_on_hours) if terminal_on else 0.0
        )
        residual_down = (
            0.0 if terminal_on else max(0.0, thermal.minimum_down_hours - consecutive_off_hours)
        )
        return TerminalCommitmentState(
            thermal_on=terminal_on,
            thermal_output_mw=terminal_output_mw,
            consecutive_on_hours=float(consecutive_on_hours),
            consecutive_off_hours=float(consecutive_off_hours),
            residual_minimum_up_hours=float(residual_up),
            residual_minimum_down_hours=float(residual_down),
            terminal_commitment_mode=thermal.terminal_commitment_mode,
        )
