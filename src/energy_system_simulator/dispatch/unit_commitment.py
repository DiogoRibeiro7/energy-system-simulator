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

from energy_system_simulator.config import ModelConfig, ThermalConfig
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
    "battery_charge_mw",
    "battery_discharge_mw",
    "battery_charge_mode",
    "battery_soc_mwh",
    "imports_mw",
    "source_load_shed_mw",
)
THERMAL_BLOCKS = (
    "thermal_output_mw",
    "thermal_on",
    "thermal_startup",
    "thermal_shutdown",
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
    config: ThermalConfig
    must_run: bool = False
    availability_factor: float = 1.0
    availability_factor_key: str | None = None


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
    thermal_capacity_available_mw: dict[str, FloatArray]
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
        thermal_capacity = self._thermal_capacity_available(
            thermal_units,
            periods,
            thermal_availability_factors or {},
        )
        registry = self._variable_registry(periods, thermal_units)
        objective = self._objective(registry, renewable, thermal_units)
        bounds, integrality = self._bounds(registry, renewable, demand, thermal_units)
        constraints, component_counts = self._constraints(
            registry,
            demand,
            thermal_units,
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
            thermal_capacity_available_mw=thermal_capacity,
            registry=registry,
            statistics=statistics,
        )

    def solve(
        self,
        renewable_available_mw: npt.ArrayLike,
        gross_demand_mw: npt.ArrayLike,
        thermal_availability_factors: Mapping[str, npt.ArrayLike] | None = None,
    ) -> DispatchResult:
        """Solve unit commitment over the full input horizon."""
        problem = self.build_formulation(
            renewable_available_mw,
            gross_demand_mw,
            thermal_availability_factors,
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
        integrality_max_deviation = self._coerce_binary_columns(frame, problem.thermal_units)
        frame["renewable_available_mw"] = problem.renewable_available_mw
        frame["renewable_curtailed_mw"] = (
            problem.renewable_available_mw - frame["renewable_used_mw"]
        )
        frame["gross_demand_mw"] = problem.gross_demand_mw
        self._add_thermal_accounting_columns(frame, problem.thermal_units, problem)
        nonnegative_cleanup_max_abs = self._clip_nonnegative_solver_noise(
            frame,
            problem.thermal_units,
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
        cost_components = self._cost_components(frame, problem.thermal_units)
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
                    config=self.config.thermal,
                    must_run=unit.must_run,
                    availability_factor=unit.availability_factor,
                    availability_factor_key=unit.availability_factor_key,
                ),
            )
        return tuple(
            ThermalUnit(
                id=unit.id,
                config=unit.config,
                must_run=unit.must_run,
                availability_factor=unit.availability_factor,
                availability_factor_key=unit.availability_factor_key,
            )
            for unit in configured
        )

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
    ) -> VariableRegistry:
        registry = VariableRegistry()
        for block in SYSTEM_BLOCKS:
            registry.add(block, periods, binary=block == "battery_charge_mode")
        for unit in thermal_units:
            for block in THERMAL_BLOCKS:
                registry.add(
                    block,
                    periods,
                    asset_id=unit.id,
                    binary=block in {"thermal_on", "thermal_startup", "thermal_shutdown"},
                )
        return registry

    def _objective(
        self,
        registry: VariableRegistry,
        renewable: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
    ) -> FloatArray:
        periods = renewable.size
        dt = self.config.simulation.time_step_hours
        imports = self.config.imports
        battery = self.config.battery
        penalties = self.config.penalties
        coefficients = np.zeros(registry.size, dtype=np.float64)

        for t in range(periods):
            coefficients[registry.at("renewable_used_mw", t)] = (
                -penalties.renewable_curtailment_eur_per_mwh * dt
            )
            coefficients[registry.at("battery_charge_mw", t)] = (
                battery.throughput_cost_eur_per_mwh * dt
            )
            coefficients[registry.at("battery_discharge_mw", t)] = (
                battery.throughput_cost_eur_per_mwh * dt
            )
            coefficients[registry.at("imports_mw", t)] = dt * (
                imports.price_eur_per_mwh
                + penalties.carbon_price_eur_per_tonne * imports.emission_factor_tonnes_per_mwh
            )
            coefficients[registry.at("source_load_shed_mw", t)] = (
                penalties.lost_load_eur_per_mwh * (1.0 - self.config.network.loss_fraction) * dt
            )
            for unit in thermal_units:
                thermal = unit.config
                coefficients[registry.at("thermal_output_mw", t, asset_id=unit.id)] = dt * (
                    thermal.variable_cost_eur_per_mwh
                    + penalties.carbon_price_eur_per_tonne * thermal.emission_factor_tonnes_per_mwh
                )
                coefficients[registry.at("thermal_on", t, asset_id=unit.id)] = (
                    thermal.no_load_cost_eur_per_hour * dt
                )
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
    ) -> tuple[Bounds, npt.NDArray[np.int_]]:
        periods = renewable.size
        lower = np.zeros(registry.size, dtype=np.float64)
        upper = np.full(registry.size, np.inf, dtype=np.float64)
        integrality = registry.integrality()

        for t in range(periods):
            upper[registry.at("renewable_used_mw", t)] = renewable[t]
            upper[registry.at("battery_charge_mw", t)] = self.config.battery.power_capacity_mw
            upper[registry.at("battery_discharge_mw", t)] = self.config.battery.power_capacity_mw
            lower[registry.at("battery_soc_mwh", t)] = self.config.battery.minimum_soc_mwh
            upper[registry.at("battery_soc_mwh", t)] = self.config.battery.maximum_soc_mwh
            upper[registry.at("battery_charge_mode", t)] = 1.0
            upper[registry.at("imports_mw", t)] = self.config.imports.maximum_power_mw
            upper[registry.at("source_load_shed_mw", t)] = demand[t]
            for unit in thermal_units:
                upper[registry.at("thermal_output_mw", t, asset_id=unit.id)] = (
                    unit.config.maximum_output_mw
                )
                for block in ("thermal_on", "thermal_startup", "thermal_shutdown"):
                    upper[registry.at(block, t, asset_id=unit.id)] = 1.0
        return Bounds(lower, upper), integrality

    def _constraints(
        self,
        registry: VariableRegistry,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        thermal_capacity_available: dict[str, FloatArray],
    ) -> tuple[LinearConstraint, dict[str, int]]:
        builder = _ConstraintBuilder(registry.size)
        self._add_balance_constraints(builder, registry, demand, thermal_units)
        self._add_thermal_constraints(
            builder,
            registry,
            demand.size,
            thermal_units,
            thermal_capacity_available,
        )
        self._add_storage_constraints(builder, registry, demand.size)
        self._add_terminal_soc_constraint(builder, registry, demand.size)
        return builder.build(), builder.component_counts

    def _add_balance_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
    ) -> None:
        for t, value in enumerate(demand):
            coefficients = {
                registry.at("renewable_used_mw", t): 1.0,
                registry.at("battery_discharge_mw", t): 1.0,
                registry.at("imports_mw", t): 1.0,
                registry.at("source_load_shed_mw", t): 1.0,
                registry.at("battery_charge_mw", t): -1.0,
            }
            for unit in thermal_units:
                coefficients[registry.at("thermal_output_mw", t, asset_id=unit.id)] = 1.0
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
                self._add_thermal_state(builder, registry, unit, t)
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
    ) -> None:
        dt = self.config.simulation.time_step_hours
        battery = self.config.battery
        for t in range(periods):
            builder.add(
                {
                    registry.at("battery_charge_mw", t): 1.0,
                    registry.at("battery_charge_mode", t): -battery.power_capacity_mw,
                },
                -np.inf,
                0.0,
                component="storage_charge_mode",
            )
            builder.add(
                {
                    registry.at("battery_discharge_mw", t): 1.0,
                    registry.at("battery_charge_mode", t): battery.power_capacity_mw,
                },
                -np.inf,
                battery.power_capacity_mw,
                component="storage_discharge_mode",
            )
            soc_coefficients = {
                registry.at("battery_soc_mwh", t): 1.0,
                registry.at("battery_charge_mw", t): -battery.charge_efficiency * dt,
                registry.at("battery_discharge_mw", t): dt / battery.discharge_efficiency,
            }
            soc_rhs = battery.initial_soc_mwh if t == 0 else 0.0
            if t > 0:
                soc_coefficients[registry.at("battery_soc_mwh", t - 1)] = -1.0
            builder.add(soc_coefficients, soc_rhs, soc_rhs, component="storage_soc")

    def _add_terminal_soc_constraint(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
    ) -> None:
        battery = self.config.battery
        terminal = {registry.at("battery_soc_mwh", periods - 1): 1.0}
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
        for unit in problem.thermal_units:
            for block in THERMAL_BLOCKS:
                data[f"{block}__{unit.id}"] = registry.values(
                    solution,
                    block,
                    asset_id=unit.id,
                )
        frame = pd.DataFrame(data)
        for block in THERMAL_BLOCKS:
            columns = [f"{block}__{unit.id}" for unit in problem.thermal_units]
            frame[block] = frame[columns].sum(axis=1)
        return frame

    def _add_thermal_accounting_columns(
        self,
        frame: pd.DataFrame,
        units: tuple[ThermalUnit, ...],
        problem: FormulationProblem,
    ) -> None:
        dt = self.config.simulation.time_step_hours
        carbon_price = self.config.penalties.carbon_price_eur_per_tonne
        for unit in units:
            thermal = unit.config
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
            frame[f"thermal_variable_cost_eur__{unit.id}"] = (
                output * dt * thermal.variable_cost_eur_per_mwh
            )
            frame[f"thermal_no_load_cost_eur__{unit.id}"] = (
                online * dt * thermal.no_load_cost_eur_per_hour
            )
            frame[f"thermal_startup_cost_eur__{unit.id}"] = startup * thermal.startup_cost_eur
            frame[f"thermal_shutdown_cost_eur__{unit.id}"] = shutdown * thermal.shutdown_cost_eur
            frame[f"thermal_emissions_tonnes__{unit.id}"] = (
                output * dt * thermal.emission_factor_tonnes_per_mwh
            )
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

    def _coerce_binary_columns(
        self,
        frame: pd.DataFrame,
        thermal_units: tuple[ThermalUnit, ...] | None = None,
    ) -> float:
        columns = ["battery_charge_mode"]
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
        return max_deviation

    def _clip_nonnegative_solver_noise(
        self,
        frame: pd.DataFrame,
        thermal_units: tuple[ThermalUnit, ...] | None = None,
    ) -> float:
        columns = [
            "renewable_used_mw",
            "battery_charge_mw",
            "battery_discharge_mw",
            "battery_soc_mwh",
            "imports_mw",
            "source_load_shed_mw",
            "renewable_curtailed_mw",
        ]
        if thermal_units is None:
            columns.append("thermal_output_mw")
        else:
            for unit in thermal_units:
                columns.append(f"thermal_output_mw__{unit.id}")
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
        return max_clipped

    def _cost_components(
        self,
        frame: pd.DataFrame,
        thermal_units: tuple[ThermalUnit, ...],
    ) -> dict[str, float]:
        dt = self.config.simulation.time_step_hours
        imports = self.config.imports
        battery = self.config.battery
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
                (frame["battery_charge_mw"].sum() + frame["battery_discharge_mw"].sum())
                * dt
                * battery.throughput_cost_eur_per_mwh
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
