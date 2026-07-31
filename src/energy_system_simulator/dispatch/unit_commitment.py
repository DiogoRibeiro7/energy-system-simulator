from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix

from energy_system_simulator.config import ModelConfig
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.dispatch.solver import (
    absolute_gap,
    interpret_solver_result,
    objective_bound_with_constant,
    relative_gap,
    solve_milp,
)
from energy_system_simulator.exceptions import OptimisationError

FloatArray = npt.NDArray[np.float64]

BLOCKS: Final[tuple[str, ...]] = (
    "renewable_used_mw",
    "thermal_output_mw",
    "thermal_on",
    "thermal_startup",
    "thermal_shutdown",
    "battery_charge_mw",
    "battery_discharge_mw",
    "battery_charge_mode",
    "battery_soc_mwh",
    "imports_mw",
    "source_load_shed_mw",
)


@dataclass(frozen=True)
class FormulationStatistics:
    """Size metrics for the mixed-integer dispatch formulation."""

    continuous_variables: int
    integer_variables: int
    binary_variables: int
    linear_constraints: int
    matrix_nonzeros: int


@dataclass(frozen=True)
class FormulationProblem:
    """Complete linear mixed-integer problem and source input arrays."""

    objective: FloatArray
    integrality: npt.NDArray[np.int_]
    bounds: Bounds
    constraints: LinearConstraint
    renewable_available_mw: FloatArray
    gross_demand_mw: FloatArray
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
    numerical_diagnostics: dict[str, float]


class _VariableIndex:
    def __init__(self, periods: int) -> None:
        self.periods = periods
        self.offsets = {name: block * periods for block, name in enumerate(BLOCKS)}
        self.size = len(BLOCKS) * periods

    def at(self, block: str, period: int) -> int:
        return self.offsets[block] + period

    def values(self, solution: FloatArray, block: str) -> FloatArray:
        start = self.offsets[block]
        return solution[start : start + self.periods]


class _ConstraintBuilder:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = variable_count
        self.row_indices: list[int] = []
        self.column_indices: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, coefficients: dict[int, float], lower: float, upper: float) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if value != 0.0:
                self.row_indices.append(row)
                self.column_indices.append(column)
                self.values.append(value)
        self.lower.append(lower)
        self.upper.append(upper)

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
    """Single-thermal-unit mixed-integer dispatch model."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def build_formulation(
        self,
        renewable_available_mw: npt.ArrayLike,
        gross_demand_mw: npt.ArrayLike,
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
        index = _VariableIndex(periods)
        objective = self._objective(index, renewable)
        bounds, integrality = self._bounds(index, renewable, demand)
        constraints = self._constraints(index, demand)

        integer_variables = int(np.count_nonzero(integrality))
        statistics = FormulationStatistics(
            continuous_variables=index.size - integer_variables,
            integer_variables=integer_variables,
            binary_variables=integer_variables,
            linear_constraints=constraints.A.shape[0],
            matrix_nonzeros=constraints.A.nnz,
        )
        return FormulationProblem(
            objective=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            renewable_available_mw=renewable,
            gross_demand_mw=demand,
            statistics=statistics,
        )

    def solve(
        self,
        renewable_available_mw: npt.ArrayLike,
        gross_demand_mw: npt.ArrayLike,
    ) -> DispatchResult:
        """Solve unit commitment over the full input horizon."""
        problem = self.build_formulation(renewable_available_mw, gross_demand_mw)
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

        solution = solver.solution
        index = _VariableIndex(problem.renewable_available_mw.size)
        frame = pd.DataFrame({name: index.values(solution, name) for name in BLOCKS})
        integrality_max_deviation = self._coerce_binary_columns(frame)
        frame["renewable_available_mw"] = problem.renewable_available_mw
        frame["renewable_curtailed_mw"] = (
            problem.renewable_available_mw - frame["renewable_used_mw"]
        )
        frame["gross_demand_mw"] = problem.gross_demand_mw
        nonnegative_cleanup_max_abs = self._clip_nonnegative_solver_noise(frame)

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
        cost_components = self._cost_components(frame)
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
        absolute_gap_eur = absolute_gap(objective_eur, objective_bound_eur)
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
            absolute_gap_eur=absolute_gap_eur,
            relative_gap=relative_gap(objective_eur, objective_bound_eur),
            solver_runtime_seconds=solver_runtime_seconds,
            solver_node_count=solver.node_count,
            formulation_statistics=problem.statistics,
            cost_components_eur=cost_components,
            terminal_commitment_state=self._terminal_commitment_state(frame),
            numerical_diagnostics={
                "integrality_max_deviation": integrality_max_deviation,
                "nonnegative_cleanup_max_abs": nonnegative_cleanup_max_abs,
            },
        )

    def _objective(self, index: _VariableIndex, renewable: FloatArray) -> FloatArray:
        periods = renewable.size
        dt = self.config.simulation.time_step_hours
        thermal = self.config.thermal
        imports = self.config.imports
        battery = self.config.battery
        penalties = self.config.penalties

        coefficients = np.zeros(index.size, dtype=np.float64)
        for t in range(periods):
            coefficients[index.at("renewable_used_mw", t)] = (
                -penalties.renewable_curtailment_eur_per_mwh * dt
            )
            coefficients[index.at("thermal_output_mw", t)] = dt * (
                thermal.variable_cost_eur_per_mwh
                + penalties.carbon_price_eur_per_tonne * thermal.emission_factor_tonnes_per_mwh
            )
            coefficients[index.at("thermal_on", t)] = thermal.no_load_cost_eur_per_hour * dt
            coefficients[index.at("thermal_startup", t)] = thermal.startup_cost_eur
            coefficients[index.at("thermal_shutdown", t)] = thermal.shutdown_cost_eur
            coefficients[index.at("battery_charge_mw", t)] = (
                battery.throughput_cost_eur_per_mwh * dt
            )
            coefficients[index.at("battery_discharge_mw", t)] = (
                battery.throughput_cost_eur_per_mwh * dt
            )
            coefficients[index.at("imports_mw", t)] = dt * (
                imports.price_eur_per_mwh
                + penalties.carbon_price_eur_per_tonne * imports.emission_factor_tonnes_per_mwh
            )
            coefficients[index.at("source_load_shed_mw", t)] = (
                penalties.lost_load_eur_per_mwh * (1.0 - self.config.network.loss_fraction) * dt
            )
        return coefficients

    def _bounds(
        self,
        index: _VariableIndex,
        renewable: FloatArray,
        demand: FloatArray,
    ) -> tuple[Bounds, npt.NDArray[np.int_]]:
        periods = renewable.size
        lower = np.zeros(index.size, dtype=np.float64)
        upper = np.full(index.size, np.inf, dtype=np.float64)
        integrality = np.zeros(index.size, dtype=int)

        for t in range(periods):
            upper[index.at("renewable_used_mw", t)] = renewable[t]
            upper[index.at("thermal_output_mw", t)] = self.config.thermal.maximum_output_mw
            for block in ("thermal_on", "thermal_startup", "thermal_shutdown"):
                upper[index.at(block, t)] = 1.0
                integrality[index.at(block, t)] = 1
            upper[index.at("battery_charge_mw", t)] = self.config.battery.power_capacity_mw
            upper[index.at("battery_discharge_mw", t)] = self.config.battery.power_capacity_mw
            lower[index.at("battery_soc_mwh", t)] = self.config.battery.minimum_soc_mwh
            upper[index.at("battery_soc_mwh", t)] = self.config.battery.maximum_soc_mwh
            upper[index.at("battery_charge_mode", t)] = 1.0
            integrality[index.at("battery_charge_mode", t)] = 1
            upper[index.at("imports_mw", t)] = self.config.imports.maximum_power_mw
            upper[index.at("source_load_shed_mw", t)] = demand[t]

        return Bounds(lower, upper), integrality

    def _constraints(self, index: _VariableIndex, demand: FloatArray) -> LinearConstraint:
        periods = demand.size
        dt = self.config.simulation.time_step_hours
        thermal = self.config.thermal
        battery = self.config.battery
        builder = _ConstraintBuilder(index.size)

        for t in range(periods):
            builder.add(
                {
                    index.at("renewable_used_mw", t): 1.0,
                    index.at("thermal_output_mw", t): 1.0,
                    index.at("battery_discharge_mw", t): 1.0,
                    index.at("imports_mw", t): 1.0,
                    index.at("source_load_shed_mw", t): 1.0,
                    index.at("battery_charge_mw", t): -1.0,
                },
                demand[t],
                demand[t],
            )

            builder.add(
                {
                    index.at("thermal_output_mw", t): 1.0,
                    index.at("thermal_on", t): -thermal.maximum_output_mw,
                },
                -np.inf,
                0.0,
            )
            builder.add(
                {
                    index.at("thermal_output_mw", t): 1.0,
                    index.at("thermal_on", t): -thermal.minimum_output_mw,
                },
                0.0,
                np.inf,
            )

            state_coefficients = {
                index.at("thermal_on", t): 1.0,
                index.at("thermal_startup", t): -1.0,
                index.at("thermal_shutdown", t): 1.0,
            }
            state_rhs = float(thermal.initial_on) if t == 0 else 0.0
            if t > 0:
                state_coefficients[index.at("thermal_on", t - 1)] = -1.0
            builder.add(state_coefficients, state_rhs, state_rhs)

            builder.add(
                {
                    index.at("thermal_startup", t): 1.0,
                    index.at("thermal_shutdown", t): 1.0,
                },
                -np.inf,
                1.0,
            )

            ramp_up_coefficients = {
                index.at("thermal_output_mw", t): 1.0,
                index.at("thermal_startup", t): -thermal.startup_ramp_mw,
            }
            ramp_down_coefficients = {
                index.at("thermal_output_mw", t): -1.0,
                index.at("thermal_shutdown", t): -thermal.shutdown_ramp_mw,
            }
            if t > 0:
                ramp_up_coefficients[index.at("thermal_output_mw", t - 1)] = -1.0
                ramp_up_coefficients[index.at("thermal_on", t - 1)] = (
                    -thermal.ramp_up_mw_per_hour * dt
                )
                ramp_down_coefficients[index.at("thermal_output_mw", t - 1)] = 1.0
                ramp_down_coefficients[index.at("thermal_on", t)] = (
                    -thermal.ramp_down_mw_per_hour * dt
                )
                ramp_up_upper = 0.0
                ramp_down_upper = 0.0
            else:
                ramp_up_upper = (
                    thermal.initial_output_mw
                    + thermal.ramp_up_mw_per_hour * dt * float(thermal.initial_on)
                )
                ramp_down_coefficients[index.at("thermal_on", t)] = (
                    -thermal.ramp_down_mw_per_hour * dt
                )
                ramp_down_upper = -thermal.initial_output_mw
            builder.add(
                ramp_up_coefficients,
                -np.inf,
                ramp_up_upper,
            )
            builder.add(
                ramp_down_coefficients,
                -np.inf,
                ramp_down_upper,
            )

            builder.add(
                {
                    index.at("battery_charge_mw", t): 1.0,
                    index.at("battery_charge_mode", t): -battery.power_capacity_mw,
                },
                -np.inf,
                0.0,
            )
            builder.add(
                {
                    index.at("battery_discharge_mw", t): 1.0,
                    index.at("battery_charge_mode", t): battery.power_capacity_mw,
                },
                -np.inf,
                battery.power_capacity_mw,
            )

            soc_coefficients = {
                index.at("battery_soc_mwh", t): 1.0,
                index.at("battery_charge_mw", t): -battery.charge_efficiency * dt,
                index.at("battery_discharge_mw", t): dt / battery.discharge_efficiency,
            }
            soc_rhs = battery.initial_soc_mwh if t == 0 else 0.0
            if t > 0:
                soc_coefficients[index.at("battery_soc_mwh", t - 1)] = -1.0
            builder.add(soc_coefficients, soc_rhs, soc_rhs)

        up = self._duration_periods(thermal.minimum_up_hours)
        down = self._duration_periods(thermal.minimum_down_hours)
        for t in range(periods):
            recent_startups = {
                index.at("thermal_startup", k): 1.0 for k in range(max(0, t - up + 1), t + 1)
            }
            recent_startups[index.at("thermal_on", t)] = -1.0
            builder.add(recent_startups, -np.inf, 0.0)

            recent_shutdowns = {
                index.at("thermal_shutdown", k): 1.0 for k in range(max(0, t - down + 1), t + 1)
            }
            recent_shutdowns[index.at("thermal_on", t)] = 1.0
            builder.add(recent_shutdowns, -np.inf, 1.0)

        self._add_initial_duration_obligations(builder, index, periods)
        self._add_terminal_commitment_constraints(builder, index, periods)
        self._add_terminal_soc_constraint(builder, index, periods)
        return builder.build()

    def _duration_periods(self, duration_hours: float) -> int:
        return max(1, ceil(duration_hours / self.config.simulation.time_step_hours))

    def _add_initial_duration_obligations(
        self,
        builder: _ConstraintBuilder,
        index: _VariableIndex,
        periods: int,
    ) -> None:
        thermal = self.config.thermal
        dt = self.config.simulation.time_step_hours
        if thermal.initial_on:
            remaining_hours = max(0.0, thermal.minimum_up_hours - thermal.initial_up_time_hours)
            forced_periods = min(periods, ceil(remaining_hours / dt))
            for t in range(forced_periods):
                builder.add({index.at("thermal_on", t): 1.0}, 1.0, 1.0)
            return

        remaining_hours = max(0.0, thermal.minimum_down_hours - thermal.initial_down_time_hours)
        forced_periods = min(periods, ceil(remaining_hours / dt))
        for t in range(forced_periods):
            builder.add({index.at("thermal_on", t): 1.0}, 0.0, 0.0)

    def _add_terminal_commitment_constraints(
        self,
        builder: _ConstraintBuilder,
        index: _VariableIndex,
        periods: int,
    ) -> None:
        thermal = self.config.thermal
        mode = thermal.terminal_commitment_mode
        if mode in {"forbid_incomplete_transitions", "fixed_terminal_commitment"}:
            up_periods = self._duration_periods(thermal.minimum_up_hours)
            down_periods = self._duration_periods(thermal.minimum_down_hours)
            for t in range(max(0, periods - up_periods + 1), periods):
                builder.add({index.at("thermal_startup", t): 1.0}, 0.0, 0.0)
            for t in range(max(0, periods - down_periods + 1), periods):
                builder.add({index.at("thermal_shutdown", t): 1.0}, 0.0, 0.0)

        if mode == "fixed_terminal_commitment":
            terminal_on = 1.0 if thermal.terminal_on else 0.0
            builder.add({index.at("thermal_on", periods - 1): 1.0}, terminal_on, terminal_on)

    def _add_terminal_soc_constraint(
        self,
        builder: _ConstraintBuilder,
        index: _VariableIndex,
        periods: int,
    ) -> None:
        battery = self.config.battery
        terminal = {index.at("battery_soc_mwh", periods - 1): 1.0}
        if battery.terminal_soc_mode == "minimum":
            builder.add(terminal, battery.minimum_final_soc_mwh, np.inf)
        elif battery.terminal_soc_mode == "exact":
            builder.add(terminal, battery.minimum_final_soc_mwh, battery.minimum_final_soc_mwh)
        elif battery.terminal_soc_mode == "cyclic":
            builder.add(terminal, battery.initial_soc_mwh, battery.initial_soc_mwh)

    def _terminal_commitment_state(self, frame: pd.DataFrame) -> TerminalCommitmentState:
        thermal = self.config.thermal
        dt = self.config.simulation.time_step_hours
        terminal_on = bool(int(frame["thermal_on"].iloc[-1]))
        terminal_output_mw = float(frame["thermal_output_mw"].iloc[-1])

        matching_periods = 0
        for value in reversed(frame["thermal_on"].tolist()):
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

    def _coerce_binary_columns(self, frame: pd.DataFrame) -> float:
        max_deviation = 0.0
        for column in ("thermal_on", "thermal_startup", "thermal_shutdown", "battery_charge_mode"):
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
        return max_deviation

    def _clip_nonnegative_solver_noise(self, frame: pd.DataFrame) -> float:
        columns = (
            "renewable_used_mw",
            "thermal_output_mw",
            "battery_charge_mw",
            "battery_discharge_mw",
            "battery_soc_mwh",
            "imports_mw",
            "source_load_shed_mw",
            "renewable_curtailed_mw",
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
        return max_clipped

    def _cost_components(self, frame: pd.DataFrame) -> dict[str, float]:
        dt = self.config.simulation.time_step_hours
        thermal = self.config.thermal
        imports = self.config.imports
        battery = self.config.battery
        penalties = self.config.penalties
        network_efficiency = 1.0 - self.config.network.loss_fraction

        return {
            "thermal_variable_cost_eur": float(
                frame["thermal_output_mw"].sum() * dt * thermal.variable_cost_eur_per_mwh
            ),
            "thermal_no_load_cost_eur": float(
                frame["thermal_on"].sum() * dt * thermal.no_load_cost_eur_per_hour
            ),
            "startup_cost_eur": float(frame["thermal_startup"].sum() * thermal.startup_cost_eur),
            "shutdown_cost_eur": float(frame["thermal_shutdown"].sum() * thermal.shutdown_cost_eur),
            "import_energy_cost_eur": float(
                frame["imports_mw"].sum() * dt * imports.price_eur_per_mwh
            ),
            "battery_throughput_cost_eur": float(
                (frame["battery_charge_mw"].sum() + frame["battery_discharge_mw"].sum())
                * dt
                * battery.throughput_cost_eur_per_mwh
            ),
            "thermal_carbon_cost_eur": float(
                frame["thermal_output_mw"].sum()
                * dt
                * thermal.emission_factor_tonnes_per_mwh
                * penalties.carbon_price_eur_per_tonne
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
