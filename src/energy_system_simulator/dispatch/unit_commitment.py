from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from energy_system_simulator.config import ModelConfig
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
class DispatchResult:
    """Optimised dispatch table and solver diagnostics."""

    frame: pd.DataFrame
    objective_eur: float
    solver_message: str
    mip_gap: float | None
    formulation_statistics: FormulationStatistics


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
        result = milp(
            c=problem.objective,
            integrality=problem.integrality,
            bounds=problem.bounds,
            constraints=problem.constraints,
            options={
                "time_limit": self.config.simulation.solver_time_limit_seconds,
                "mip_rel_gap": self.config.simulation.mip_relative_gap,
                "presolve": True,
            },
        )
        if result.x is None or not result.success:
            raise OptimisationError(
                f"Unit commitment failed with status {result.status}: {result.message}"
            )

        solution = np.asarray(result.x, dtype=np.float64)
        index = _VariableIndex(problem.renewable_available_mw.size)
        frame = pd.DataFrame({name: index.values(solution, name) for name in BLOCKS})
        for column in ("thermal_on", "thermal_startup", "thermal_shutdown"):
            frame[column] = np.rint(frame[column]).astype(int)
        frame["renewable_available_mw"] = problem.renewable_available_mw
        frame["renewable_curtailed_mw"] = (
            problem.renewable_available_mw - frame["renewable_used_mw"]
        )
        frame["gross_demand_mw"] = problem.gross_demand_mw

        constant_curtailment_cost = (
            self.config.penalties.renewable_curtailment_eur_per_mwh
            * problem.renewable_available_mw.sum()
            * self.config.simulation.time_step_hours
        )
        objective_eur = float(result.fun + constant_curtailment_cost)
        mip_gap_raw = getattr(result, "mip_gap", None)
        mip_gap = float(mip_gap_raw) if mip_gap_raw is not None else None
        return DispatchResult(
            frame=frame,
            objective_eur=objective_eur,
            solver_message=str(result.message),
            mip_gap=mip_gap,
            formulation_statistics=problem.statistics,
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

            previous_output = thermal.initial_output_mw if t == 0 else 0.0
            ramp_up_coefficients = {
                index.at("thermal_output_mw", t): 1.0,
                index.at("thermal_startup", t): -thermal.maximum_output_mw,
            }
            ramp_down_coefficients = {
                index.at("thermal_output_mw", t): -1.0,
                index.at("thermal_shutdown", t): -thermal.maximum_output_mw,
            }
            if t > 0:
                ramp_up_coefficients[index.at("thermal_output_mw", t - 1)] = -1.0
                ramp_down_coefficients[index.at("thermal_output_mw", t - 1)] = 1.0
            builder.add(
                ramp_up_coefficients,
                -np.inf,
                thermal.ramp_up_mw_per_hour * dt + previous_output,
            )
            builder.add(
                ramp_down_coefficients,
                -np.inf,
                thermal.ramp_down_mw_per_hour * dt - previous_output,
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

        up = thermal.minimum_up_hours
        down = thermal.minimum_down_hours
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

        builder.add(
            {index.at("battery_soc_mwh", periods - 1): 1.0},
            battery.minimum_final_soc_mwh,
            np.inf,
        )
        return builder.build()
