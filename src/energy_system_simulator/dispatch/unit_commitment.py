from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import ceil
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import coo_matrix

from energy_system_simulator.config import (
    BatteryConfig,
    DemandConfig,
    FuelConfig,
    HydroUnitConfig,
    ModelConfig,
    RenewableGeneratorConfig,
    ThermalConfig,
)
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.dispatch.solver import (
    LinearConstraintData,
    SolverProblem,
    VariableBounds,
    absolute_gap,
    interpret_solver_result,
    objective_bound_with_constant,
    relative_gap,
    solve_milp,
)
from energy_system_simulator.dispatch.variables import VariableMetadata, VariableRegistry
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
HYDRO_BLOCKS = (
    "hydro_generation_mw",
    "hydro_release_mw",
    "hydro_spill_mw",
    "hydro_reservoir_mwh",
)
DEMAND_BLOCKS = (
    "demand_involuntary_shed_mw",
    "demand_voluntary_curtailment_mw",
    "demand_shift_down_mw",
    "demand_shift_up_mw",
    "demand_task_charge_mw",
    "demand_task_unserved_mwh",
)
THERMAL_BLOCKS = (
    "thermal_output_mw",
    "thermal_on",
    "thermal_startup",
    "thermal_shutdown",
)
THERMAL_SEGMENT_OUTPUT_BLOCK = "thermal_segment_output_mw"
THERMAL_STARTUP_CATEGORY_BLOCK = "thermal_startup_category"
BUS_VOLTAGE_ANGLE_BLOCK = "bus_voltage_angle_rad"
LINE_FLOW_BLOCK = "line_flow_mw"
THERMAL_UPWARD_RESERVE_BLOCK = "thermal_upward_reserve_mw"
THERMAL_DOWNWARD_RESERVE_BLOCK = "thermal_downward_reserve_mw"
STORAGE_UPWARD_RESERVE_BLOCK = "storage_upward_reserve_mw"
STORAGE_DOWNWARD_RESERVE_BLOCK = "storage_downward_reserve_mw"
DEMAND_UPWARD_RESERVE_BLOCK = "demand_upward_reserve_mw"
IMPORT_UPWARD_RESERVE_BLOCK = "import_upward_reserve_mw"
IMPORT_DOWNWARD_RESERVE_BLOCK = "import_downward_reserve_mw"
RESERVE_UPWARD_SHORTFALL_BLOCK = "reserve_upward_shortfall_mw"
RESERVE_DOWNWARD_SHORTFALL_BLOCK = "reserve_downward_shortfall_mw"
RESERVE_LARGEST_CONTINGENCY_BLOCK = "reserve_largest_online_contingency_mw"


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
    build_profile_seconds: dict[str, float] = field(default_factory=dict)


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
class HydroUnit:
    """Resolved hydro unit used by the indexed formulation."""

    id: str
    config: HydroUnitConfig


@dataclass(frozen=True)
class DemandUnit:
    """Resolved demand entity used by the indexed formulation."""

    id: str
    config: DemandConfig


@dataclass(frozen=True)
class RenewableUnit:
    """Resolved renewable asset used by nodal dispatch."""

    id: str
    bus_id: str
    config: RenewableGeneratorConfig


@dataclass(frozen=True)
class NetworkLine:
    """Resolved transmission line used by nodal DC dispatch."""

    id: str
    from_bus_id: str
    to_bus_id: str
    susceptance: float
    capacity_available_mw: FloatArray


@dataclass(frozen=True)
class NodalNetwork:
    """Resolved nodal network metadata for the optimisation model."""

    enabled: bool
    bus_ids: tuple[str, ...] = ()
    slack_bus_id: str | None = None
    lines: tuple[NetworkLine, ...] = ()


@dataclass(frozen=True)
class ReserveModel:
    """Resolved reserve requirements for the optimisation model."""

    enabled: bool
    upward_requirement_mw: FloatArray
    downward_requirement_mw: FloatArray


@dataclass(frozen=True)
class FormulationProblem:
    """Complete linear mixed-integer problem and source input arrays."""

    objective: FloatArray
    integrality: npt.NDArray[np.int_]
    bounds: VariableBounds
    constraints: LinearConstraintData
    constraint_components: tuple[str, ...]
    variable_metadata: tuple[VariableMetadata, ...]
    renewable_available_mw: FloatArray
    gross_demand_mw: FloatArray
    thermal_units: tuple[ThermalUnit, ...]
    storage_units: tuple[StorageUnit, ...]
    hydro_units: tuple[HydroUnit, ...]
    demand_units: tuple[DemandUnit, ...]
    renewable_units: tuple[RenewableUnit, ...]
    renewable_available_by_asset_mw: dict[str, FloatArray]
    demand_profiles_mw: dict[str, FloatArray]
    thermal_capacity_available_mw: dict[str, FloatArray]
    storage_availability_factor: dict[str, FloatArray]
    hydro_inflow_mw: dict[str, FloatArray]
    fuel_prices_eur_per_mwh_thermal: dict[str, FloatArray]
    import_capacity_available_mw: FloatArray
    import_prices_eur_per_mwh: FloatArray
    network: NodalNetwork
    reserves: ReserveModel
    registry: VariableRegistry
    statistics: FormulationStatistics

    def solver_problem(self) -> SolverProblem:
        """Return the backend-neutral optimisation payload."""
        return SolverProblem(
            objective=self.objective,
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=self.constraints,
            variable_names=tuple(variable.name for variable in self.variable_metadata),
        )


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
        self.components: list[str] = []
        self.names: list[str] = []
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
        component_index = self.component_counts.get(component, 0)
        for column, value in coefficients.items():
            if value != 0.0:
                self.row_indices.append(row)
                self.column_indices.append(column)
                self.values.append(value)
        self.lower.append(lower)
        self.upper.append(upper)
        self.components.append(component)
        self.names.append(f"{component}[{component_index}]")
        self.component_counts[component] = component_index + 1

    def build(self) -> LinearConstraintData:
        matrix = coo_matrix(
            (self.values, (self.row_indices, self.column_indices)),
            shape=(len(self.lower), self.variable_count),
            dtype=np.float64,
        ).tocsr()
        return LinearConstraintData(
            matrix,
            np.asarray(self.lower, dtype=np.float64),
            np.asarray(self.upper, dtype=np.float64),
            tuple(self.names),
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
        hydro_inflows_mw: Mapping[str, npt.ArrayLike] | None = None,
        demand_profiles_mw: Mapping[str, npt.ArrayLike] | None = None,
        renewable_availability_by_asset_mw: Mapping[str, npt.ArrayLike] | None = None,
        line_availability_factors: Mapping[str, npt.ArrayLike] | None = None,
        import_availability_factors: npt.ArrayLike | None = None,
        import_price_series: npt.ArrayLike | None = None,
        fixed_thermal_commitment: Mapping[str, npt.ArrayLike] | None = None,
    ) -> FormulationProblem:
        """Build the MILP formulation without solving it."""
        build_started = perf_counter()
        profile: dict[str, float] = {}
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
        profile["input_validation"] = perf_counter() - build_started

        phase_started = perf_counter()
        periods = renewable.size
        thermal_units = self._thermal_units()
        storage_units = self._storage_units()
        hydro_units = self._hydro_units()
        demand_units = self._demand_units(demand_profiles_mw)
        profile["asset_resolution"] = perf_counter() - phase_started

        phase_started = perf_counter()
        demand_profiles = self._demand_profiles(demand_units, demand, demand_profiles_mw or {})
        network = self._nodal_network(periods, line_availability_factors or {})
        if network.enabled and not demand_units:
            raise ValueError("Nodal network mode requires demand profiles for each demand asset")
        renewable_units = self._renewable_units() if network.enabled else ()
        renewable_by_asset = self._renewable_availability_by_asset(
            renewable_units,
            renewable,
            renewable_availability_by_asset_mw or {},
        )
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
        hydro_inflow = self._hydro_inflows(hydro_units, periods, hydro_inflows_mw or {})
        fuel_prices = self._fuel_prices(periods, fuel_price_series or {})
        import_capacity = self._import_capacity_available(periods, import_availability_factors)
        import_prices = self._import_prices(periods, import_price_series)
        fixed_commitment = self._fixed_thermal_commitment(
            thermal_units,
            periods,
            fixed_thermal_commitment or {},
        )
        reserves = self._reserve_model(renewable, demand)
        profile["series_resolution"] = perf_counter() - phase_started

        phase_started = perf_counter()
        registry = self._variable_registry(
            periods,
            thermal_units,
            storage_units,
            hydro_units,
            demand_units,
            renewable_units,
            network,
            reserves,
        )
        profile["variable_registry"] = perf_counter() - phase_started

        phase_started = perf_counter()
        objective = self._objective(
            registry,
            renewable,
            thermal_units,
            storage_units,
            hydro_units,
            demand_units,
            fuel_prices,
            import_prices,
            reserves,
        )
        profile["objective"] = perf_counter() - phase_started

        phase_started = perf_counter()
        bounds, integrality = self._bounds(
            registry,
            renewable,
            demand,
            thermal_units,
            storage_units,
            hydro_units,
            demand_units,
            demand_profiles,
            storage_availability,
            renewable_by_asset,
            import_capacity,
            fixed_commitment,
            network,
            reserves,
        )
        profile["bounds"] = perf_counter() - phase_started

        phase_started = perf_counter()
        constraints, component_counts, constraint_components = self._constraints(
            registry,
            demand,
            thermal_units,
            storage_units,
            hydro_units,
            demand_units,
            demand_profiles,
            thermal_capacity,
            hydro_inflow,
            renewable_units,
            network,
            reserves,
            storage_availability,
            import_capacity,
        )
        profile["constraints"] = perf_counter() - phase_started

        integer_variables = int(np.count_nonzero(integrality))
        profile["total"] = perf_counter() - build_started
        statistics = FormulationStatistics(
            continuous_variables=registry.size - integer_variables,
            integer_variables=integer_variables,
            binary_variables=integer_variables,
            linear_constraints=constraints.A.shape[0],
            matrix_nonzeros=constraints.A.nnz,
            variable_counts_by_block=registry.variable_counts_by_block(),
            constraint_counts_by_component=component_counts,
            build_profile_seconds=profile,
        )
        return FormulationProblem(
            objective=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            constraint_components=constraint_components,
            variable_metadata=registry.variable_metadata(),
            renewable_available_mw=renewable,
            gross_demand_mw=demand,
            thermal_units=thermal_units,
            storage_units=storage_units,
            hydro_units=hydro_units,
            demand_units=demand_units,
            renewable_units=renewable_units,
            renewable_available_by_asset_mw=renewable_by_asset,
            demand_profiles_mw=demand_profiles,
            thermal_capacity_available_mw=thermal_capacity,
            storage_availability_factor=storage_availability,
            hydro_inflow_mw=hydro_inflow,
            fuel_prices_eur_per_mwh_thermal=fuel_prices,
            import_capacity_available_mw=import_capacity,
            import_prices_eur_per_mwh=import_prices,
            network=network,
            reserves=reserves,
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
        hydro_inflows_mw: Mapping[str, npt.ArrayLike] | None = None,
        demand_profiles_mw: Mapping[str, npt.ArrayLike] | None = None,
        renewable_availability_by_asset_mw: Mapping[str, npt.ArrayLike] | None = None,
        line_availability_factors: Mapping[str, npt.ArrayLike] | None = None,
        import_availability_factors: npt.ArrayLike | None = None,
        import_price_series: npt.ArrayLike | None = None,
        fixed_thermal_commitment: Mapping[str, npt.ArrayLike] | None = None,
    ) -> DispatchResult:
        """Solve unit commitment over the full input horizon."""
        problem = self.build_formulation(
            renewable_available_mw,
            gross_demand_mw,
            thermal_availability_factors,
            fuel_price_series,
            storage_availability_factors,
            hydro_inflows_mw,
            demand_profiles_mw,
            renewable_availability_by_asset_mw,
            line_availability_factors,
            import_availability_factors,
            import_price_series,
            fixed_thermal_commitment,
        )
        return self.solve_formulation(problem)

    def solve_formulation(self, problem: FormulationProblem) -> DispatchResult:
        """Solve a previously built formulation."""
        solve_started = perf_counter()
        backend_result = solve_milp(
            problem.solver_problem(),
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
        self._add_hydro_accounting_columns(frame, problem.hydro_units, problem)
        self._add_network_accounting_columns(frame, problem)
        self._add_demand_accounting_columns(frame, problem.demand_units)
        self._add_reserve_accounting_columns(frame, problem)
        nonnegative_cleanup_max_abs = self._clip_nonnegative_solver_noise(
            frame,
            problem.thermal_units,
            problem.storage_units,
            problem.hydro_units,
            problem.demand_units,
        )
        if problem.demand_units:
            self._add_demand_accounting_columns(frame, problem.demand_units)
        self._add_network_accounting_columns(frame, problem)
        self._add_reserve_accounting_columns(frame, problem)

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
        cost_components = self._cost_components(frame, problem)
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

    def _hydro_units(self) -> tuple[HydroUnit, ...]:
        return tuple(
            HydroUnit(id=unit.id, config=unit) for unit in self.config.portfolio.hydro_units
        )

    def _demand_units(
        self,
        demand_profiles_mw: Mapping[str, npt.ArrayLike] | None,
    ) -> tuple[DemandUnit, ...]:
        if demand_profiles_mw is None:
            return ()
        return tuple(DemandUnit(id=unit.id, config=unit) for unit in self.config.portfolio.demand)

    def _demand_profiles(
        self,
        units: tuple[DemandUnit, ...],
        aggregate_demand: FloatArray,
        demand_profiles_mw: Mapping[str, npt.ArrayLike],
    ) -> dict[str, FloatArray]:
        result: dict[str, FloatArray] = {}
        for unit in units:
            if unit.id not in demand_profiles_mw:
                raise ValueError(f"Demand profile series is required for {unit.id}")
            values = np.asarray(demand_profiles_mw[unit.id], dtype=np.float64)
            if values.shape != aggregate_demand.shape:
                raise ValueError(f"Demand profile series for {unit.id} has wrong shape")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"Demand profile for {unit.id} must be finite and non-negative")
            result[unit.id] = values
        if result:
            aggregate = np.asarray(
                np.sum(np.vstack(list(result.values())), axis=0, dtype=np.float64),
                dtype=np.float64,
            )
            if not np.allclose(
                aggregate,
                aggregate_demand,
                atol=DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw,
                rtol=0.0,
            ):
                raise ValueError("Demand profiles must sum to aggregate gross demand")
        return result

    def _renewable_units(self) -> tuple[RenewableUnit, ...]:
        return tuple(
            RenewableUnit(id=unit.id, bus_id=unit.bus_id, config=unit)
            for unit in self.config.portfolio.renewable_generators
        )

    def _renewable_availability_by_asset(
        self,
        units: tuple[RenewableUnit, ...],
        aggregate_renewable: FloatArray,
        availability_by_asset_mw: Mapping[str, npt.ArrayLike],
    ) -> dict[str, FloatArray]:
        if not units:
            return {}
        result: dict[str, FloatArray] = {}
        for unit in units:
            if unit.id not in availability_by_asset_mw:
                raise ValueError(f"Renewable availability series is required for {unit.id}")
            values = np.asarray(availability_by_asset_mw[unit.id], dtype=np.float64)
            if values.shape != aggregate_renewable.shape:
                raise ValueError(f"Renewable availability series for {unit.id} has wrong shape")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(
                    f"Renewable availability for {unit.id} must be finite and non-negative"
                )
            result[unit.id] = values
        aggregate = np.asarray(
            np.sum(np.vstack(list(result.values())), axis=0, dtype=np.float64),
            dtype=np.float64,
        )
        if not np.allclose(
            aggregate,
            aggregate_renewable,
            atol=DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw,
            rtol=0.0,
        ):
            raise ValueError("Renewable asset availability must sum to aggregate availability")
        return result

    def _nodal_network(
        self,
        periods: int,
        line_availability_factors: Mapping[str, npt.ArrayLike],
    ) -> NodalNetwork:
        if self.config.network.network_mode != "nodal":
            return NodalNetwork(enabled=False)
        bus_ids = tuple(bus.id for bus in self.config.portfolio.buses)
        slack_bus_id = self.config.network.slack_bus_id or bus_ids[0]
        if slack_bus_id not in bus_ids:
            raise ValueError("Slack bus references unknown bus")
        lines: list[NetworkLine] = []
        for config in self.config.portfolio.lines:
            availability = np.full(periods, config.availability_factor, dtype=np.float64)
            if config.id in line_availability_factors:
                series = np.asarray(line_availability_factors[config.id], dtype=np.float64)
                if series.shape != (periods,):
                    raise ValueError(f"Line availability factor for {config.id} has wrong shape")
                availability = availability * series
            if np.any(~np.isfinite(availability)) or np.any(
                (availability < 0.0) | (availability > 1.0)
            ):
                raise ValueError(
                    f"Line availability factors for {config.id} must be finite in [0, 1]"
                )
            lines.append(
                NetworkLine(
                    id=config.id,
                    from_bus_id=config.from_bus_id,
                    to_bus_id=config.to_bus_id,
                    susceptance=config.susceptance,
                    capacity_available_mw=availability * config.capacity_mw,
                )
            )
        return NodalNetwork(
            enabled=True,
            bus_ids=bus_ids,
            slack_bus_id=slack_bus_id,
            lines=tuple(lines),
        )

    def _reserve_model(self, renewable: FloatArray, demand: FloatArray) -> ReserveModel:
        config = self.config.reserves
        upward = (
            config.upward_fixed_mw
            + config.upward_demand_fraction * demand
            + config.upward_renewable_fraction * renewable
        )
        downward = (
            config.downward_fixed_mw
            + config.downward_demand_fraction * demand
            + config.downward_renewable_fraction * renewable
        )
        enabled = bool(
            np.any(upward > 0.0)
            or np.any(downward > 0.0)
            or config.largest_online_contingency_fraction > 0.0
        )
        return ReserveModel(
            enabled=enabled,
            upward_requirement_mw=np.asarray(upward, dtype=np.float64),
            downward_requirement_mw=np.asarray(downward, dtype=np.float64),
        )

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

    def _hydro_inflows(
        self,
        units: tuple[HydroUnit, ...],
        periods: int,
        hydro_inflows_mw: Mapping[str, npt.ArrayLike],
    ) -> dict[str, FloatArray]:
        result: dict[str, FloatArray] = {}
        for unit in units:
            if unit.id not in hydro_inflows_mw:
                raise ValueError(f"Hydro inflow series is required for {unit.id}")
            values = np.asarray(hydro_inflows_mw[unit.id], dtype=np.float64)
            if values.shape != (periods,):
                raise ValueError(f"Hydro inflow series for {unit.id} has wrong shape")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"Hydro inflows for {unit.id} must be finite and non-negative")
            result[unit.id] = values
        return result

    def _import_capacity_available(
        self,
        periods: int,
        import_availability_factors: npt.ArrayLike | None,
    ) -> FloatArray:
        factor = np.ones(periods, dtype=np.float64)
        if import_availability_factors is not None:
            factor = np.asarray(import_availability_factors, dtype=np.float64)
            if factor.shape != (periods,):
                raise ValueError("Import availability factors must match dispatch horizon")
            if np.any(~np.isfinite(factor)) or np.any((factor < 0.0) | (factor > 1.0)):
                raise ValueError("Import availability factors must be finite values in [0, 1]")
        return (self.config.imports.maximum_power_mw * factor).astype(np.float64)

    def _import_prices(
        self,
        periods: int,
        import_price_series: npt.ArrayLike | None,
    ) -> FloatArray:
        values = np.full(periods, self.config.imports.price_eur_per_mwh, dtype=np.float64)
        if import_price_series is not None:
            values = np.asarray(import_price_series, dtype=np.float64)
            if values.shape != (periods,):
                raise ValueError("Import price series must match dispatch horizon")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("Import price series must contain non-negative finite values")
        return values.astype(np.float64)

    def _fixed_thermal_commitment(
        self,
        units: tuple[ThermalUnit, ...],
        periods: int,
        fixed_thermal_commitment: Mapping[str, npt.ArrayLike],
    ) -> dict[str, FloatArray]:
        unit_ids = {unit.id for unit in units}
        unknown = set(fixed_thermal_commitment) - unit_ids
        if unknown:
            raise ValueError(
                f"Fixed thermal commitment references unknown units: {sorted(unknown)}"
            )
        result: dict[str, FloatArray] = {}
        for unit_id, raw_values in fixed_thermal_commitment.items():
            raw = np.asarray(raw_values, dtype=np.float64)
            if raw.ndim != 1 or raw.size > periods:
                raise ValueError(
                    "Fixed thermal commitment must be a one-dimensional horizon prefix"
                )
            values = np.full(periods, np.nan, dtype=np.float64)
            values[: raw.size] = raw
            fixed = values[~np.isnan(values)]
            if np.any(~np.isin(fixed, [0.0, 1.0])):
                raise ValueError("Fixed thermal commitment values must be 0, 1, or NaN")
            result[unit_id] = values
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
        hydro_units: tuple[HydroUnit, ...],
        demand_units: tuple[DemandUnit, ...],
        renewable_units: tuple[RenewableUnit, ...],
        network: NodalNetwork,
        reserves: ReserveModel,
    ) -> VariableRegistry:
        registry = VariableRegistry()
        for block in SYSTEM_BLOCKS:
            registry.add(block, periods, binary=block == "battery_charge_mode")
        if reserves.enabled:
            registry.add(RESERVE_UPWARD_SHORTFALL_BLOCK, periods)
            registry.add(RESERVE_DOWNWARD_SHORTFALL_BLOCK, periods)
            if self.config.reserves.largest_online_contingency_fraction > 0.0:
                registry.add(RESERVE_LARGEST_CONTINGENCY_BLOCK, periods)
        if network.enabled:
            for renewable in renewable_units:
                registry.add("renewable_used_mw", periods, asset_id=renewable.id)
            for bus_id in network.bus_ids:
                registry.add(BUS_VOLTAGE_ANGLE_BLOCK, periods, asset_id=bus_id)
            for line in network.lines:
                registry.add(LINE_FLOW_BLOCK, periods, asset_id=line.id)
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
            if reserves.enabled:
                registry.add(STORAGE_UPWARD_RESERVE_BLOCK, periods, asset_id=storage.id)
                registry.add(STORAGE_DOWNWARD_RESERVE_BLOCK, periods, asset_id=storage.id)
        for hydro in hydro_units:
            for block in HYDRO_BLOCKS:
                registry.add(block, periods, asset_id=hydro.id)
        for demand in demand_units:
            registry.add("demand_involuntary_shed_mw", periods, asset_id=demand.id)
            if demand.config.kind in {"curtailable", "shiftable"}:
                registry.add("demand_voluntary_curtailment_mw", periods, asset_id=demand.id)
            if demand.config.kind == "shiftable":
                registry.add("demand_shift_down_mw", periods, asset_id=demand.id)
                registry.add("demand_shift_up_mw", periods, asset_id=demand.id)
            if demand.config.kind in {"deferrable", "ev_charging"}:
                registry.add("demand_task_charge_mw", periods, asset_id=demand.id)
                registry.add("demand_task_unserved_mwh", periods, asset_id=demand.id)
            if reserves.enabled and demand.config.kind in {"curtailable", "shiftable"}:
                registry.add(DEMAND_UPWARD_RESERVE_BLOCK, periods, asset_id=demand.id)
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
            if reserves.enabled:
                registry.add(THERMAL_UPWARD_RESERVE_BLOCK, periods, asset_id=thermal_unit.id)
                registry.add(THERMAL_DOWNWARD_RESERVE_BLOCK, periods, asset_id=thermal_unit.id)
        if reserves.enabled and self.config.reserves.allow_import_reserves:
            registry.add(IMPORT_UPWARD_RESERVE_BLOCK, periods)
            registry.add(IMPORT_DOWNWARD_RESERVE_BLOCK, periods)
        return registry

    def _objective(
        self,
        registry: VariableRegistry,
        renewable: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        hydro_units: tuple[HydroUnit, ...],
        demand_units: tuple[DemandUnit, ...],
        fuel_prices: dict[str, FloatArray],
        import_prices: FloatArray,
        reserves: ReserveModel,
    ) -> FloatArray:
        periods = renewable.size
        dt = self.config.simulation.time_step_hours
        imports = self.config.imports
        penalties = self.config.penalties
        fuels = self._fuels_by_id()
        reserve_config = self.config.reserves
        coefficients = np.zeros(registry.size, dtype=np.float64)

        for t in range(periods):
            if reserves.enabled:
                coefficients[registry.at(RESERVE_UPWARD_SHORTFALL_BLOCK, t)] = (
                    reserve_config.upward_shortfall_penalty_eur_per_mw_hour * dt
                )
                coefficients[registry.at(RESERVE_DOWNWARD_SHORTFALL_BLOCK, t)] = (
                    reserve_config.downward_shortfall_penalty_eur_per_mw_hour * dt
                )
                if reserve_config.allow_import_reserves:
                    coefficients[registry.at(IMPORT_UPWARD_RESERVE_BLOCK, t)] = (
                        reserve_config.import_upward_cost_eur_per_mw_hour * dt
                    )
                    coefficients[registry.at(IMPORT_DOWNWARD_RESERVE_BLOCK, t)] = (
                        reserve_config.import_downward_cost_eur_per_mw_hour * dt
                    )
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
                if reserves.enabled:
                    coefficients[
                        registry.at(STORAGE_UPWARD_RESERVE_BLOCK, t, asset_id=storage.id)
                    ] = reserve_config.storage_upward_cost_eur_per_mw_hour * dt
                    coefficients[
                        registry.at(STORAGE_DOWNWARD_RESERVE_BLOCK, t, asset_id=storage.id)
                    ] = reserve_config.storage_downward_cost_eur_per_mw_hour * dt
                for band in battery.degradation_bands:
                    coefficients[
                        registry.at(
                            STORAGE_DEGRADATION_BLOCK,
                            t,
                            asset_id=_storage_degradation_asset_id(storage.id, band.id),
                        )
                    ] = band.cost_eur_per_mwh
            for hydro in hydro_units:
                if t == periods - 1:
                    coefficients[
                        registry.at("hydro_reservoir_mwh", t, asset_id=hydro.id)
                    ] = -hydro.config.water_value_eur_per_mwh
            coefficients[registry.at("imports_mw", t)] = dt * (
                import_prices[t]
                + penalties.carbon_price_eur_per_tonne * imports.emission_factor_tonnes_per_mwh
            )
            coefficients[registry.at("source_load_shed_mw", t)] = (
                penalties.lost_load_eur_per_mwh * (1.0 - self.config.network.loss_fraction) * dt
            )
            for demand in demand_units:
                demand_config = demand.config
                lost_load_cost = (
                    demand_config.value_of_lost_load_eur_per_mwh
                    if demand_config.value_of_lost_load_eur_per_mwh is not None
                    else penalties.lost_load_eur_per_mwh
                )
                coefficients[registry.at("demand_involuntary_shed_mw", t, asset_id=demand.id)] = (
                    lost_load_cost * (1.0 - self.config.network.loss_fraction) * dt
                )
                if demand_config.kind in {"curtailable", "shiftable"}:
                    coefficients[
                        registry.at("demand_voluntary_curtailment_mw", t, asset_id=demand.id)
                    ] = demand_config.voluntary_curtailment_cost_eur_per_mwh * dt
                    if reserves.enabled:
                        coefficients[
                            registry.at(DEMAND_UPWARD_RESERVE_BLOCK, t, asset_id=demand.id)
                        ] = reserve_config.demand_response_upward_cost_eur_per_mw_hour * dt
                if demand_config.kind == "shiftable":
                    shift_cost = demand_config.shift_cost_eur_per_mwh * dt
                    coefficients[registry.at("demand_shift_down_mw", t, asset_id=demand.id)] = (
                        shift_cost
                    )
                    coefficients[registry.at("demand_shift_up_mw", t, asset_id=demand.id)] = (
                        shift_cost
                    )
                if demand_config.kind in {"deferrable", "ev_charging"}:
                    coefficients[registry.at("demand_task_unserved_mwh", t, asset_id=demand.id)] = (
                        demand_config.task_unserved_penalty_eur_per_mwh
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
                if reserves.enabled:
                    coefficients[registry.at(THERMAL_UPWARD_RESERVE_BLOCK, t, asset_id=unit.id)] = (
                        reserve_config.thermal_upward_cost_eur_per_mw_hour * dt
                    )
                    coefficients[
                        registry.at(THERMAL_DOWNWARD_RESERVE_BLOCK, t, asset_id=unit.id)
                    ] = reserve_config.thermal_downward_cost_eur_per_mw_hour * dt
        return coefficients

    def _bounds(
        self,
        registry: VariableRegistry,
        renewable: FloatArray,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        hydro_units: tuple[HydroUnit, ...],
        demand_units: tuple[DemandUnit, ...],
        demand_profiles: dict[str, FloatArray],
        storage_availability: dict[str, FloatArray],
        renewable_available_by_asset: dict[str, FloatArray],
        import_capacity_available_mw: FloatArray,
        fixed_thermal_commitment: dict[str, FloatArray],
        network: NodalNetwork,
        reserves: ReserveModel,
    ) -> tuple[VariableBounds, npt.NDArray[np.int_]]:
        periods = renewable.size
        lower = np.zeros(registry.size, dtype=np.float64)
        upper = np.full(registry.size, np.inf, dtype=np.float64)
        integrality = registry.integrality()
        for t in range(periods):
            upper[registry.at("renewable_used_mw", t)] = renewable[t]
            upper[registry.at("imports_mw", t)] = import_capacity_available_mw[t]
            upper[registry.at("source_load_shed_mw", t)] = 0.0 if demand_units else demand[t]
            if network.enabled:
                for renewable_id, values in renewable_available_by_asset.items():
                    upper[registry.at("renewable_used_mw", t, asset_id=renewable_id)] = values[t]
                for bus_id in network.bus_ids:
                    column = registry.at(BUS_VOLTAGE_ANGLE_BLOCK, t, asset_id=bus_id)
                    lower[column] = -np.inf
                    upper[column] = np.inf
                if network.slack_bus_id is not None:
                    slack = registry.at(
                        BUS_VOLTAGE_ANGLE_BLOCK,
                        t,
                        asset_id=network.slack_bus_id,
                    )
                    lower[slack] = 0.0
                    upper[slack] = 0.0
                for line in network.lines:
                    column = registry.at(LINE_FLOW_BLOCK, t, asset_id=line.id)
                    lower[column] = -line.capacity_available_mw[t]
                    upper[column] = line.capacity_available_mw[t]
            for demand_unit in demand_units:
                demand_config = demand_unit.config
                baseline = demand_profiles[demand_unit.id][t]
                upper[registry.at("demand_involuntary_shed_mw", t, asset_id=demand_unit.id)] = (
                    baseline
                    + demand_config.shift_up_capacity_mw
                    + demand_config.task_power_capacity_mw
                )
                if demand_config.kind in {"curtailable", "shiftable"}:
                    curtailment_limit = baseline * demand_config.maximum_curtailment_fraction
                    if demand_config.maximum_curtailment_mw is not None:
                        curtailment_limit = min(
                            curtailment_limit,
                            demand_config.maximum_curtailment_mw,
                        )
                    upper[
                        registry.at(
                            "demand_voluntary_curtailment_mw",
                            t,
                            asset_id=demand_unit.id,
                        )
                    ] = curtailment_limit
                if demand_config.kind == "shiftable":
                    upper[registry.at("demand_shift_down_mw", t, asset_id=demand_unit.id)] = min(
                        baseline, demand_config.shift_down_capacity_mw
                    )
                    upper[registry.at("demand_shift_up_mw", t, asset_id=demand_unit.id)] = (
                        demand_config.shift_up_capacity_mw
                    )
                if demand_config.kind in {"deferrable", "ev_charging"}:
                    in_window = t >= demand_config.task_start_period and (
                        demand_config.task_end_period is None or t < demand_config.task_end_period
                    )
                    upper[registry.at("demand_task_charge_mw", t, asset_id=demand_unit.id)] = (
                        demand_config.task_power_capacity_mw if in_window else 0.0
                    )
                    upper[registry.at("demand_task_unserved_mwh", t, asset_id=demand_unit.id)] = (
                        demand_config.task_required_energy_mwh
                    )
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
            for hydro in hydro_units:
                config = hydro.config
                upper[registry.at("hydro_generation_mw", t, asset_id=hydro.id)] = (
                    config.turbine_capacity_mw
                )
                upper[registry.at("hydro_release_mw", t, asset_id=hydro.id)] = (
                    config.turbine_capacity_mw / config.turbine_efficiency
                )
                upper[registry.at("hydro_spill_mw", t, asset_id=hydro.id)] = (
                    config.spill_capacity_mw if config.spill_capacity_mw is not None else np.inf
                )
                lower[registry.at("hydro_reservoir_mwh", t, asset_id=hydro.id)] = (
                    config.minimum_reservoir_mwh
                )
                upper[registry.at("hydro_reservoir_mwh", t, asset_id=hydro.id)] = (
                    config.maximum_reservoir_mwh
                )
                if config.kind == "run_of_river":
                    upper[registry.at("hydro_reservoir_mwh", t, asset_id=hydro.id)] = 0.0
            for unit in thermal_units:
                upper[registry.at("thermal_output_mw", t, asset_id=unit.id)] = (
                    unit.config.maximum_output_mw
                )
                for block in ("thermal_on", "thermal_startup", "thermal_shutdown"):
                    upper[registry.at(block, t, asset_id=unit.id)] = 1.0
                if unit.id in fixed_thermal_commitment:
                    fixed_value = fixed_thermal_commitment[unit.id][t]
                    if not np.isnan(fixed_value):
                        column = registry.at("thermal_on", t, asset_id=unit.id)
                        lower[column] = fixed_value
                        upper[column] = fixed_value
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
        return VariableBounds(lower, upper), integrality

    def _constraints(
        self,
        registry: VariableRegistry,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        hydro_units: tuple[HydroUnit, ...],
        demand_units: tuple[DemandUnit, ...],
        demand_profiles: dict[str, FloatArray],
        thermal_capacity_available: dict[str, FloatArray],
        hydro_inflow: dict[str, FloatArray],
        renewable_units: tuple[RenewableUnit, ...],
        network: NodalNetwork,
        reserves: ReserveModel,
        storage_availability: dict[str, FloatArray],
        import_capacity_available_mw: FloatArray,
    ) -> tuple[LinearConstraintData, dict[str, int], tuple[str, ...]]:
        builder = _ConstraintBuilder(registry.size)
        if network.enabled:
            self._add_renewable_aggregate_constraints(
                builder,
                registry,
                demand.size,
                renewable_units,
            )
            self._add_nodal_network_constraints(
                builder,
                registry,
                demand.size,
                thermal_units,
                storage_units,
                hydro_units,
                demand_units,
                demand_profiles,
                renewable_units,
                network,
            )
        else:
            self._add_balance_constraints(
                builder,
                registry,
                demand,
                thermal_units,
                storage_units,
                hydro_units,
                demand_units,
                demand_profiles,
            )
        self._add_thermal_constraints(
            builder,
            registry,
            demand.size,
            thermal_units,
            thermal_capacity_available,
        )
        self._add_hydro_constraints(builder, registry, demand.size, hydro_units, hydro_inflow)
        self._add_storage_constraints(builder, registry, demand.size, storage_units)
        self._add_demand_constraints(builder, registry, demand_units, demand_profiles)
        if reserves.enabled:
            self._add_reserve_constraints(
                builder,
                registry,
                demand.size,
                thermal_units,
                storage_units,
                demand_units,
                demand_profiles,
                thermal_capacity_available,
                storage_availability,
                import_capacity_available_mw,
                reserves,
            )
        self._add_terminal_soc_constraints(builder, registry, demand.size, storage_units)
        return builder.build(), builder.component_counts, tuple(builder.components)

    def _add_renewable_aggregate_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
        renewable_units: tuple[RenewableUnit, ...],
    ) -> None:
        for t in range(periods):
            coefficients = {registry.at("renewable_used_mw", t): 1.0}
            for renewable in renewable_units:
                coefficients[registry.at("renewable_used_mw", t, asset_id=renewable.id)] = -1.0
            builder.add(coefficients, 0.0, 0.0, component="renewable_aggregate")

    def _add_nodal_network_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        hydro_units: tuple[HydroUnit, ...],
        demand_units: tuple[DemandUnit, ...],
        demand_profiles: dict[str, FloatArray],
        renewable_units: tuple[RenewableUnit, ...],
        network: NodalNetwork,
    ) -> None:
        renewable_bus = {unit.id: unit.bus_id for unit in renewable_units}
        thermal_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.thermal_generators}
        storage_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.storage_units}
        hydro_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.hydro_units}
        demand_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.demand}
        import_bus = self.config.portfolio.imports[0].bus_id

        for t in range(periods):
            for line in network.lines:
                builder.add(
                    {
                        registry.at(LINE_FLOW_BLOCK, t, asset_id=line.id): 1.0,
                        registry.at(
                            BUS_VOLTAGE_ANGLE_BLOCK,
                            t,
                            asset_id=line.from_bus_id,
                        ): -line.susceptance,
                        registry.at(
                            BUS_VOLTAGE_ANGLE_BLOCK,
                            t,
                            asset_id=line.to_bus_id,
                        ): line.susceptance,
                    },
                    0.0,
                    0.0,
                    component="dc_line_flow",
                )

            for bus_id in network.bus_ids:
                coefficients: dict[int, float] = {}
                rhs = 0.0
                for renewable in renewable_units:
                    if renewable_bus[renewable.id] == bus_id:
                        coefficients[registry.at("renewable_used_mw", t, asset_id=renewable.id)] = (
                            1.0
                        )
                for unit in thermal_units:
                    if thermal_bus[unit.id] == bus_id:
                        coefficients[registry.at("thermal_output_mw", t, asset_id=unit.id)] = 1.0
                for storage in storage_units:
                    if storage_bus[storage.id] == bus_id:
                        coefficients[
                            registry.at("storage_discharge_mw", t, asset_id=storage.id)
                        ] = 1.0
                        coefficients[
                            registry.at("storage_charge_mw", t, asset_id=storage.id)
                        ] = -1.0
                for hydro in hydro_units:
                    if hydro_bus[hydro.id] == bus_id:
                        coefficients[registry.at("hydro_generation_mw", t, asset_id=hydro.id)] = 1.0
                if import_bus == bus_id:
                    coefficients[registry.at("imports_mw", t)] = 1.0
                for demand_unit in demand_units:
                    if demand_bus[demand_unit.id] != bus_id:
                        continue
                    rhs += demand_profiles[demand_unit.id][t]
                    coefficients[
                        registry.at(
                            "demand_involuntary_shed_mw",
                            t,
                            asset_id=demand_unit.id,
                        )
                    ] = 1.0
                    if demand_unit.config.kind in {"curtailable", "shiftable"}:
                        coefficients[
                            registry.at(
                                "demand_voluntary_curtailment_mw",
                                t,
                                asset_id=demand_unit.id,
                            )
                        ] = 1.0
                    if demand_unit.config.kind == "shiftable":
                        coefficients[
                            registry.at("demand_shift_down_mw", t, asset_id=demand_unit.id)
                        ] = 1.0
                        coefficients[
                            registry.at("demand_shift_up_mw", t, asset_id=demand_unit.id)
                        ] = -1.0
                    if demand_unit.config.kind in {"deferrable", "ev_charging"}:
                        coefficients[
                            registry.at("demand_task_charge_mw", t, asset_id=demand_unit.id)
                        ] = -1.0
                for line in network.lines:
                    if line.from_bus_id == bus_id:
                        coefficients[registry.at(LINE_FLOW_BLOCK, t, asset_id=line.id)] = -1.0
                    elif line.to_bus_id == bus_id:
                        coefficients[registry.at(LINE_FLOW_BLOCK, t, asset_id=line.id)] = 1.0
                builder.add(coefficients, rhs, rhs, component="nodal_balance")

    def _add_reserve_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        demand_units: tuple[DemandUnit, ...],
        demand_profiles: dict[str, FloatArray],
        thermal_capacity_available: dict[str, FloatArray],
        storage_availability: dict[str, FloatArray],
        import_capacity_available_mw: FloatArray,
        reserves: ReserveModel,
    ) -> None:
        config = self.config.reserves
        duration = config.response_duration_hours
        for t in range(periods):
            upward_coefficients = {registry.at(RESERVE_UPWARD_SHORTFALL_BLOCK, t): 1.0}
            downward_coefficients = {registry.at(RESERVE_DOWNWARD_SHORTFALL_BLOCK, t): 1.0}
            if config.largest_online_contingency_fraction > 0.0:
                contingency = registry.at(RESERVE_LARGEST_CONTINGENCY_BLOCK, t)
                upward_coefficients[contingency] = -config.largest_online_contingency_fraction
                for unit in thermal_units:
                    builder.add(
                        {
                            contingency: 1.0,
                            registry.at("thermal_on", t, asset_id=unit.id): (
                                -thermal_capacity_available[unit.id][t]
                            ),
                        },
                        0.0,
                        np.inf,
                        component="reserve_largest_online_contingency",
                    )
            for unit in thermal_units:
                up = registry.at(THERMAL_UPWARD_RESERVE_BLOCK, t, asset_id=unit.id)
                down = registry.at(THERMAL_DOWNWARD_RESERVE_BLOCK, t, asset_id=unit.id)
                output = registry.at("thermal_output_mw", t, asset_id=unit.id)
                online = registry.at("thermal_on", t, asset_id=unit.id)
                capacity = thermal_capacity_available[unit.id][t]
                builder.add(
                    {output: 1.0, up: 1.0, online: -capacity},
                    -np.inf,
                    0.0,
                    component="reserve_thermal_up_headroom",
                )
                builder.add(
                    {up: 1.0, online: -unit.config.ramp_up_mw_per_hour * duration},
                    -np.inf,
                    0.0,
                    component="reserve_thermal_up_ramp",
                )
                builder.add(
                    {down: 1.0, output: -1.0, online: unit.config.minimum_output_mw},
                    -np.inf,
                    0.0,
                    component="reserve_thermal_down_headroom",
                )
                builder.add(
                    {down: 1.0, online: -unit.config.ramp_down_mw_per_hour * duration},
                    -np.inf,
                    0.0,
                    component="reserve_thermal_down_ramp",
                )
                upward_coefficients[up] = 1.0
                downward_coefficients[down] = 1.0
            for storage in storage_units:
                up = registry.at(STORAGE_UPWARD_RESERVE_BLOCK, t, asset_id=storage.id)
                down = registry.at(STORAGE_DOWNWARD_RESERVE_BLOCK, t, asset_id=storage.id)
                charge = registry.at("storage_charge_mw", t, asset_id=storage.id)
                discharge = registry.at("storage_discharge_mw", t, asset_id=storage.id)
                soc = registry.at("storage_soc_mwh", t, asset_id=storage.id)
                battery = storage.config
                availability = storage_availability[storage.id][t]
                builder.add(
                    {
                        discharge: 1.0,
                        up: 1.0,
                    },
                    -np.inf,
                    _storage_discharge_capacity(battery) * availability,
                    component="reserve_storage_up_headroom",
                )
                builder.add(
                    {up: duration / battery.discharge_efficiency, soc: -1.0},
                    -np.inf,
                    -battery.minimum_soc_mwh,
                    component="reserve_storage_up_energy",
                )
                builder.add(
                    {
                        charge: 1.0,
                        down: 1.0,
                    },
                    -np.inf,
                    _storage_charge_capacity(battery) * availability,
                    component="reserve_storage_down_headroom",
                )
                builder.add(
                    {down: duration * battery.charge_efficiency, soc: 1.0},
                    -np.inf,
                    battery.maximum_soc_mwh,
                    component="reserve_storage_down_energy",
                )
                upward_coefficients[up] = 1.0
                downward_coefficients[down] = 1.0
            for demand_unit in demand_units:
                if demand_unit.config.kind not in {"curtailable", "shiftable"}:
                    continue
                reserve = registry.at(DEMAND_UPWARD_RESERVE_BLOCK, t, asset_id=demand_unit.id)
                baseline = demand_profiles[demand_unit.id][t]
                demand_config = demand_unit.config
                reserve_limit = baseline * config.demand_response_upward_fraction
                curtailment_limit = baseline * demand_config.maximum_curtailment_fraction
                if demand_config.maximum_curtailment_mw is not None:
                    curtailment_limit = min(curtailment_limit, demand_config.maximum_curtailment_mw)
                limit = min(reserve_limit, curtailment_limit)
                coefficients = {reserve: 1.0}
                if demand_config.kind in {"curtailable", "shiftable"}:
                    coefficients[
                        registry.at(
                            "demand_voluntary_curtailment_mw",
                            t,
                            asset_id=demand_unit.id,
                        )
                    ] = 1.0
                if demand_config.kind == "shiftable":
                    coefficients[
                        registry.at("demand_shift_down_mw", t, asset_id=demand_unit.id)
                    ] = 1.0
                builder.add(
                    coefficients,
                    -np.inf,
                    limit,
                    component="reserve_demand_response_upward",
                )
                upward_coefficients[reserve] = 1.0
            if config.allow_import_reserves:
                import_up = registry.at(IMPORT_UPWARD_RESERVE_BLOCK, t)
                import_down = registry.at(IMPORT_DOWNWARD_RESERVE_BLOCK, t)
                imports = registry.at("imports_mw", t)
                builder.add(
                    {imports: 1.0, import_up: 1.0},
                    -np.inf,
                    import_capacity_available_mw[t],
                    component="reserve_import_up_headroom",
                )
                builder.add(
                    {import_down: 1.0, imports: -1.0},
                    -np.inf,
                    0.0,
                    component="reserve_import_down_headroom",
                )
                upward_coefficients[import_up] = 1.0
                downward_coefficients[import_down] = 1.0
            builder.add(
                upward_coefficients,
                reserves.upward_requirement_mw[t],
                np.inf,
                component="reserve_upward_requirement",
            )
            builder.add(
                downward_coefficients,
                reserves.downward_requirement_mw[t],
                np.inf,
                component="reserve_downward_requirement",
            )

    def _add_balance_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        demand: FloatArray,
        thermal_units: tuple[ThermalUnit, ...],
        storage_units: tuple[StorageUnit, ...],
        hydro_units: tuple[HydroUnit, ...],
        demand_units: tuple[DemandUnit, ...],
        demand_profiles: dict[str, FloatArray],
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
            for hydro in hydro_units:
                coefficients[registry.at("hydro_generation_mw", t, asset_id=hydro.id)] = 1.0
            if demand_units:
                value = 0.0
                for demand_unit in demand_units:
                    baseline = demand_profiles[demand_unit.id][t]
                    value += baseline
                    coefficients[
                        registry.at(
                            "demand_involuntary_shed_mw",
                            t,
                            asset_id=demand_unit.id,
                        )
                    ] = 1.0
                    if demand_unit.config.kind in {"curtailable", "shiftable"}:
                        coefficients[
                            registry.at(
                                "demand_voluntary_curtailment_mw",
                                t,
                                asset_id=demand_unit.id,
                            )
                        ] = 1.0
                    if demand_unit.config.kind == "shiftable":
                        coefficients[
                            registry.at("demand_shift_down_mw", t, asset_id=demand_unit.id)
                        ] = 1.0
                        coefficients[
                            registry.at("demand_shift_up_mw", t, asset_id=demand_unit.id)
                        ] = -1.0
                    if demand_unit.config.kind in {"deferrable", "ev_charging"}:
                        coefficients[
                            registry.at("demand_task_charge_mw", t, asset_id=demand_unit.id)
                        ] = -1.0
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

    def _add_demand_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        units: tuple[DemandUnit, ...],
        demand_profiles: dict[str, FloatArray],
    ) -> None:
        for unit in units:
            config = unit.config
            baseline = demand_profiles[unit.id]
            for t, baseline_mw in enumerate(baseline):
                coefficients = {registry.at("demand_involuntary_shed_mw", t, asset_id=unit.id): 1.0}
                if config.kind in {"curtailable", "shiftable"}:
                    coefficients[
                        registry.at("demand_voluntary_curtailment_mw", t, asset_id=unit.id)
                    ] = 1.0
                if config.kind == "shiftable":
                    coefficients[registry.at("demand_shift_down_mw", t, asset_id=unit.id)] = 1.0
                    coefficients[registry.at("demand_shift_up_mw", t, asset_id=unit.id)] = -1.0
                if config.kind in {"deferrable", "ev_charging"}:
                    coefficients[registry.at("demand_task_charge_mw", t, asset_id=unit.id)] = -1.0
                builder.add(
                    coefficients,
                    -np.inf,
                    baseline_mw,
                    component="demand_shed_limit",
                )
            if config.kind == "shiftable":
                self._add_shift_conservation_constraints(builder, registry, unit, baseline.size)
            if config.kind in {"deferrable", "ev_charging"}:
                self._add_task_completion_constraint(builder, registry, unit, baseline.size)

    def _add_shift_conservation_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: DemandUnit,
        periods: int,
    ) -> None:
        dt = self.config.simulation.time_step_hours
        config = unit.config
        window_periods = (
            periods
            if config.shift_window_hours <= 0.0
            else max(1, ceil(config.shift_window_hours / dt))
        )
        for start in range(0, periods, window_periods):
            end = min(periods, start + window_periods)
            coefficients: dict[int, float] = {}
            for t in range(start, end):
                coefficients[registry.at("demand_shift_up_mw", t, asset_id=unit.id)] = dt
                coefficients[registry.at("demand_shift_down_mw", t, asset_id=unit.id)] = (
                    -(1.0 + config.rebound_fraction) * dt
                )
            builder.add(coefficients, 0.0, 0.0, component="demand_shift_conservation")

    def _add_task_completion_constraint(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        unit: DemandUnit,
        periods: int,
    ) -> None:
        dt = self.config.simulation.time_step_hours
        config = unit.config
        coefficients: dict[int, float] = {}
        for t in range(periods):
            coefficients[registry.at("demand_task_charge_mw", t, asset_id=unit.id)] = dt
            coefficients[registry.at("demand_involuntary_shed_mw", t, asset_id=unit.id)] = -dt
            coefficients[registry.at("demand_task_unserved_mwh", t, asset_id=unit.id)] = 1.0
        builder.add(
            coefficients,
            config.task_required_energy_mwh,
            config.task_required_energy_mwh,
            component="demand_task_completion",
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

    def _add_hydro_constraints(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        periods: int,
        hydro_units: tuple[HydroUnit, ...],
        hydro_inflow: dict[str, FloatArray],
    ) -> None:
        dt = self.config.simulation.time_step_hours
        for hydro in hydro_units:
            config = hydro.config
            retention = (1.0 - config.evaporation_rate_per_hour) ** dt
            for t in range(periods):
                release = registry.at("hydro_release_mw", t, asset_id=hydro.id)
                generation = registry.at("hydro_generation_mw", t, asset_id=hydro.id)
                spill = registry.at("hydro_spill_mw", t, asset_id=hydro.id)
                reservoir = registry.at("hydro_reservoir_mwh", t, asset_id=hydro.id)
                builder.add(
                    {generation: 1.0, release: -config.turbine_efficiency},
                    0.0,
                    0.0,
                    component="hydro_turbine_conversion",
                )
                if config.minimum_release_mw > 0.0:
                    builder.add(
                        {release: 1.0, spill: 1.0},
                        config.minimum_release_mw,
                        np.inf,
                        component="hydro_environmental_release",
                    )
                coefficients = {
                    reservoir: 1.0,
                    release: dt,
                    spill: dt,
                }
                rhs = hydro_inflow[hydro.id][t] * dt
                if t == 0:
                    rhs += retention * config.initial_reservoir_mwh
                else:
                    coefficients[
                        registry.at("hydro_reservoir_mwh", t - 1, asset_id=hydro.id)
                    ] = -retention
                builder.add(coefficients, rhs, rhs, component="hydro_water_balance")
            self._add_hydro_terminal_constraint(builder, registry, hydro, periods)

    def _add_hydro_terminal_constraint(
        self,
        builder: _ConstraintBuilder,
        registry: VariableRegistry,
        hydro: HydroUnit,
        periods: int,
    ) -> None:
        config = hydro.config
        if config.kind == "run_of_river" or config.terminal_reservoir_mode == "free":
            return
        terminal = {registry.at("hydro_reservoir_mwh", periods - 1, asset_id=hydro.id): 1.0}
        if config.terminal_reservoir_mode == "minimum":
            builder.add(
                terminal,
                config.minimum_final_reservoir_mwh,
                np.inf,
                component="hydro_terminal",
            )
        elif config.terminal_reservoir_mode == "exact":
            builder.add(
                terminal,
                config.minimum_final_reservoir_mwh,
                config.minimum_final_reservoir_mwh,
                component="hydro_terminal",
            )
        elif config.terminal_reservoir_mode == "cyclic":
            builder.add(
                terminal,
                config.initial_reservoir_mwh,
                config.initial_reservoir_mwh,
                component="hydro_terminal",
            )

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
        if problem.reserves.enabled:
            data[RESERVE_UPWARD_SHORTFALL_BLOCK] = registry.values(
                solution,
                RESERVE_UPWARD_SHORTFALL_BLOCK,
            )
            data[RESERVE_DOWNWARD_SHORTFALL_BLOCK] = registry.values(
                solution,
                RESERVE_DOWNWARD_SHORTFALL_BLOCK,
            )
            if self.config.reserves.largest_online_contingency_fraction > 0.0:
                data[RESERVE_LARGEST_CONTINGENCY_BLOCK] = registry.values(
                    solution,
                    RESERVE_LARGEST_CONTINGENCY_BLOCK,
                )
            if self.config.reserves.allow_import_reserves:
                data[IMPORT_UPWARD_RESERVE_BLOCK] = registry.values(
                    solution,
                    IMPORT_UPWARD_RESERVE_BLOCK,
                )
                data[IMPORT_DOWNWARD_RESERVE_BLOCK] = registry.values(
                    solution,
                    IMPORT_DOWNWARD_RESERVE_BLOCK,
                )
        for storage in problem.storage_units:
            for block in STORAGE_BLOCKS:
                data[f"{block}__{storage.id}"] = registry.values(
                    solution,
                    block,
                    asset_id=storage.id,
                )
            if problem.reserves.enabled:
                data[f"{STORAGE_UPWARD_RESERVE_BLOCK}__{storage.id}"] = registry.values(
                    solution,
                    STORAGE_UPWARD_RESERVE_BLOCK,
                    asset_id=storage.id,
                )
                data[f"{STORAGE_DOWNWARD_RESERVE_BLOCK}__{storage.id}"] = registry.values(
                    solution,
                    STORAGE_DOWNWARD_RESERVE_BLOCK,
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
        for hydro in problem.hydro_units:
            for block in HYDRO_BLOCKS:
                data[f"{block}__{hydro.id}"] = registry.values(
                    solution,
                    block,
                    asset_id=hydro.id,
                )
        for renewable in problem.renewable_units:
            data[f"renewable_used_mw__{renewable.id}"] = registry.values(
                solution,
                "renewable_used_mw",
                asset_id=renewable.id,
            )
        if problem.network.enabled:
            for bus_id in problem.network.bus_ids:
                data[f"{BUS_VOLTAGE_ANGLE_BLOCK}__{bus_id}"] = registry.values(
                    solution,
                    BUS_VOLTAGE_ANGLE_BLOCK,
                    asset_id=bus_id,
                )
            for line in problem.network.lines:
                data[f"{LINE_FLOW_BLOCK}__{line.id}"] = registry.values(
                    solution,
                    LINE_FLOW_BLOCK,
                    asset_id=line.id,
                )
        for demand in problem.demand_units:
            data[f"demand_baseline_mw__{demand.id}"] = problem.demand_profiles_mw[demand.id]
            data[f"demand_involuntary_shed_mw__{demand.id}"] = registry.values(
                solution,
                "demand_involuntary_shed_mw",
                asset_id=demand.id,
            )
            if demand.config.kind in {"curtailable", "shiftable"}:
                data[f"demand_voluntary_curtailment_mw__{demand.id}"] = registry.values(
                    solution,
                    "demand_voluntary_curtailment_mw",
                    asset_id=demand.id,
                )
            if demand.config.kind == "shiftable":
                data[f"demand_shift_down_mw__{demand.id}"] = registry.values(
                    solution,
                    "demand_shift_down_mw",
                    asset_id=demand.id,
                )
                data[f"demand_shift_up_mw__{demand.id}"] = registry.values(
                    solution,
                    "demand_shift_up_mw",
                    asset_id=demand.id,
                )
            if demand.config.kind in {"deferrable", "ev_charging"}:
                data[f"demand_task_charge_mw__{demand.id}"] = registry.values(
                    solution,
                    "demand_task_charge_mw",
                    asset_id=demand.id,
                )
                data[f"demand_task_unserved_mwh__{demand.id}"] = registry.values(
                    solution,
                    "demand_task_unserved_mwh",
                    asset_id=demand.id,
                )
            if problem.reserves.enabled and demand.config.kind in {"curtailable", "shiftable"}:
                data[f"{DEMAND_UPWARD_RESERVE_BLOCK}__{demand.id}"] = registry.values(
                    solution,
                    DEMAND_UPWARD_RESERVE_BLOCK,
                    asset_id=demand.id,
                )
        for unit in problem.thermal_units:
            for block in THERMAL_BLOCKS:
                data[f"{block}__{unit.id}"] = registry.values(
                    solution,
                    block,
                    asset_id=unit.id,
                )
            if problem.reserves.enabled:
                data[f"{THERMAL_UPWARD_RESERVE_BLOCK}__{unit.id}"] = registry.values(
                    solution,
                    THERMAL_UPWARD_RESERVE_BLOCK,
                    asset_id=unit.id,
                )
                data[f"{THERMAL_DOWNWARD_RESERVE_BLOCK}__{unit.id}"] = registry.values(
                    solution,
                    THERMAL_DOWNWARD_RESERVE_BLOCK,
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
        self._add_hydro_aggregate_columns(frame, problem.hydro_units)
        self._add_demand_aggregate_columns(frame, problem.demand_units, problem.gross_demand_mw)
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

    @staticmethod
    def _add_hydro_aggregate_columns(
        frame: pd.DataFrame,
        hydro_units: tuple[HydroUnit, ...],
    ) -> None:
        mapping = {
            "hydro_generation_mw": "hydro_generation_mw",
            "hydro_release_mw": "hydro_release_mw",
            "hydro_spill_mw": "hydro_spill_mw",
            "hydro_reservoir_mwh": "hydro_reservoir_mwh",
        }
        for aggregate, block in mapping.items():
            if not hydro_units:
                frame[aggregate] = 0.0
                continue
            columns = [f"{block}__{unit.id}" for unit in hydro_units]
            frame[aggregate] = frame[columns].sum(axis=1)

    @staticmethod
    def _add_demand_aggregate_columns(
        frame: pd.DataFrame,
        demand_units: tuple[DemandUnit, ...],
        gross_demand_mw: FloatArray,
    ) -> None:
        frame["demand_baseline_mw"] = (
            sum(frame[f"demand_baseline_mw__{unit.id}"] for unit in demand_units)
            if demand_units
            else gross_demand_mw
        )
        mapping = {
            "demand_involuntary_shed_mw": "demand_involuntary_shed_mw",
            "demand_voluntary_curtailment_mw": "demand_voluntary_curtailment_mw",
            "demand_shift_down_mw": "demand_shift_down_mw",
            "demand_shift_up_mw": "demand_shift_up_mw",
            "demand_task_charge_mw": "demand_task_charge_mw",
            "demand_task_unserved_mwh": "demand_task_unserved_mwh",
        }
        for aggregate, block in mapping.items():
            columns = [
                f"{block}__{unit.id}" for unit in demand_units if f"{block}__{unit.id}" in frame
            ]
            frame[aggregate] = frame[columns].sum(axis=1) if columns else 0.0

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

    def _add_hydro_accounting_columns(
        self,
        frame: pd.DataFrame,
        units: tuple[HydroUnit, ...],
        problem: FormulationProblem,
    ) -> None:
        dt = self.config.simulation.time_step_hours
        for unit in units:
            config = unit.config
            inflow = problem.hydro_inflow_mw[unit.id]
            reservoir = frame[f"hydro_reservoir_mwh__{unit.id}"]
            release = frame[f"hydro_release_mw__{unit.id}"]
            spill = frame[f"hydro_spill_mw__{unit.id}"]
            generation = frame[f"hydro_generation_mw__{unit.id}"]
            previous = reservoir.shift(1)
            previous.iloc[0] = config.initial_reservoir_mwh
            retention = (1.0 - config.evaporation_rate_per_hour) ** dt
            expected = retention * previous + inflow * dt - release * dt - spill * dt
            frame[f"hydro_inflow_mw__{unit.id}"] = inflow
            frame[f"hydro_water_loss_mwh__{unit.id}"] = (1.0 - retention) * previous
            frame[f"hydro_water_balance_residual_mwh__{unit.id}"] = reservoir - expected
            frame[f"hydro_capacity_factor__{unit.id}"] = np.divide(
                generation.to_numpy(dtype=np.float64),
                config.turbine_capacity_mw,
                out=np.zeros(len(frame), dtype=np.float64),
                where=config.turbine_capacity_mw > 0.0,
            )
            frame[f"hydro_terminal_value_eur__{unit.id}"] = 0.0
            frame.loc[
                frame.index[-1],
                f"hydro_terminal_value_eur__{unit.id}",
            ] = float(reservoir.iloc[-1] * config.water_value_eur_per_mwh)
        frame["hydro_inflow_mw"] = (
            sum(frame[f"hydro_inflow_mw__{unit.id}"] for unit in units) if units else 0.0
        )
        frame["hydro_water_loss_mwh"] = (
            sum(frame[f"hydro_water_loss_mwh__{unit.id}"] for unit in units) if units else 0.0
        )
        frame["hydro_terminal_value_eur"] = (
            sum(frame[f"hydro_terminal_value_eur__{unit.id}"] for unit in units) if units else 0.0
        )

    def _add_network_accounting_columns(
        self,
        frame: pd.DataFrame,
        problem: FormulationProblem,
    ) -> None:
        network = problem.network
        if not network.enabled:
            return
        bus_residual_columns: list[str] = []
        max_abs_utilisation = np.zeros(len(frame), dtype=np.float64)
        congested_flags = np.zeros(len(frame), dtype=bool)
        for renewable in problem.renewable_units:
            used = frame[f"renewable_used_mw__{renewable.id}"]
            available = problem.renewable_available_by_asset_mw[renewable.id]
            frame[f"renewable_available_mw__{renewable.id}"] = available
            frame[f"renewable_curtailed_mw__{renewable.id}"] = available - used
        for line in network.lines:
            flow_column = f"{LINE_FLOW_BLOCK}__{line.id}"
            capacity_column = f"line_capacity_available_mw__{line.id}"
            signed_utilisation_column = f"line_signed_utilisation__{line.id}"
            abs_utilisation_column = f"line_abs_utilisation__{line.id}"
            overload_column = f"line_overload_residual_mw__{line.id}"
            flow = frame[flow_column].to_numpy(dtype=np.float64)
            capacity = line.capacity_available_mw
            abs_flow = np.abs(flow)
            utilisation = np.divide(
                abs_flow,
                capacity,
                out=np.zeros_like(abs_flow, dtype=np.float64),
                where=capacity > 0.0,
            )
            signed_utilisation = np.divide(
                flow,
                capacity,
                out=np.zeros_like(flow, dtype=np.float64),
                where=capacity > 0.0,
            )
            overload = np.maximum(0.0, abs_flow - capacity)
            frame[capacity_column] = capacity
            frame[signed_utilisation_column] = signed_utilisation
            frame[abs_utilisation_column] = utilisation
            frame[overload_column] = overload
            max_abs_utilisation = np.maximum(max_abs_utilisation, utilisation)
            congested_flags |= utilisation >= 1.0 - DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw
        for bus_id in network.bus_ids:
            generation = self._bus_generation(frame, problem, bus_id)
            load = self._bus_load(frame, problem, bus_id)
            net_line_out = np.zeros(len(frame), dtype=np.float64)
            for line in network.lines:
                flow = frame[f"{LINE_FLOW_BLOCK}__{line.id}"].to_numpy(dtype=np.float64)
                if line.from_bus_id == bus_id:
                    net_line_out += flow
                elif line.to_bus_id == bus_id:
                    net_line_out -= flow
            frame[f"bus_net_injection_mw__{bus_id}"] = net_line_out
            residual_column = f"bus_balance_residual_mw__{bus_id}"
            frame[residual_column] = generation - load - net_line_out
            bus_residual_columns.append(residual_column)
        frame["line_max_abs_utilisation"] = max_abs_utilisation
        frame["line_congested"] = congested_flags.astype(int)
        frame["line_overload_residual_mw"] = self._sum_prefixed_frame_columns(
            frame,
            "line_overload_residual_mw__",
        )
        frame["bus_balance_residual_mw"] = (
            frame[bus_residual_columns].abs().max(axis=1) if bus_residual_columns else 0.0
        )

    def _bus_generation(
        self,
        frame: pd.DataFrame,
        problem: FormulationProblem,
        bus_id: str,
    ) -> FloatArray:
        values = np.zeros(len(frame), dtype=np.float64)
        renewable_bus = {unit.id: unit.bus_id for unit in problem.renewable_units}
        thermal_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.thermal_generators}
        storage_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.storage_units}
        hydro_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.hydro_units}
        for unit in problem.renewable_units:
            if renewable_bus[unit.id] == bus_id:
                values += frame[f"renewable_used_mw__{unit.id}"].to_numpy(dtype=np.float64)
        for thermal in problem.thermal_units:
            if thermal_bus[thermal.id] == bus_id:
                values += frame[f"thermal_output_mw__{thermal.id}"].to_numpy(dtype=np.float64)
        for storage in problem.storage_units:
            if storage_bus[storage.id] == bus_id:
                values += frame[f"storage_discharge_mw__{storage.id}"].to_numpy(dtype=np.float64)
        for hydro in problem.hydro_units:
            if hydro_bus[hydro.id] == bus_id:
                values += frame[f"hydro_generation_mw__{hydro.id}"].to_numpy(dtype=np.float64)
        if self.config.portfolio.imports[0].bus_id == bus_id:
            values += frame["imports_mw"].to_numpy(dtype=np.float64)
        return values

    def _bus_load(
        self,
        frame: pd.DataFrame,
        problem: FormulationProblem,
        bus_id: str,
    ) -> FloatArray:
        values = np.zeros(len(frame), dtype=np.float64)
        storage_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.storage_units}
        demand_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.demand}
        for storage in problem.storage_units:
            if storage_bus[storage.id] == bus_id:
                values += frame[f"storage_charge_mw__{storage.id}"].to_numpy(dtype=np.float64)
        for demand in problem.demand_units:
            if demand_bus[demand.id] != bus_id:
                continue
            values += frame[f"demand_baseline_mw__{demand.id}"].to_numpy(dtype=np.float64)
            values -= frame[f"demand_involuntary_shed_mw__{demand.id}"].to_numpy(dtype=np.float64)
            if demand.config.kind in {"curtailable", "shiftable"}:
                values -= frame[f"demand_voluntary_curtailment_mw__{demand.id}"].to_numpy(
                    dtype=np.float64
                )
            if demand.config.kind == "shiftable":
                values -= frame[f"demand_shift_down_mw__{demand.id}"].to_numpy(dtype=np.float64)
                values += frame[f"demand_shift_up_mw__{demand.id}"].to_numpy(dtype=np.float64)
            if demand.config.kind in {"deferrable", "ev_charging"}:
                values += frame[f"demand_task_charge_mw__{demand.id}"].to_numpy(dtype=np.float64)
        return values

    def _add_demand_accounting_columns(
        self,
        frame: pd.DataFrame,
        units: tuple[DemandUnit, ...],
    ) -> None:
        dt = self.config.simulation.time_step_hours
        penalties = self.config.penalties
        for unit in units:
            config = unit.config
            baseline = frame[f"demand_baseline_mw__{unit.id}"]
            voluntary_column = f"demand_voluntary_curtailment_mw__{unit.id}"
            shift_down_column = f"demand_shift_down_mw__{unit.id}"
            shift_up_column = f"demand_shift_up_mw__{unit.id}"
            task_charge_column = f"demand_task_charge_mw__{unit.id}"
            shed = frame[f"demand_involuntary_shed_mw__{unit.id}"]
            voluntary = frame.get(voluntary_column, 0.0)
            shift_down = frame.get(shift_down_column, 0.0)
            shift_up = frame.get(shift_up_column, 0.0)
            task_charge = frame.get(task_charge_column, 0.0)
            adjusted = baseline - voluntary - shift_down + shift_up + task_charge
            frame[f"demand_adjusted_mw__{unit.id}"] = adjusted
            frame[f"demand_served_mw__{unit.id}"] = adjusted - shed
            lost_load_cost = (
                config.value_of_lost_load_eur_per_mwh
                if config.value_of_lost_load_eur_per_mwh is not None
                else penalties.lost_load_eur_per_mwh
            )
            frame[f"demand_involuntary_shed_cost_eur__{unit.id}"] = (
                shed * lost_load_cost * (1.0 - self.config.network.loss_fraction) * dt
            )
            if config.kind in {"curtailable", "shiftable"}:
                frame[f"demand_voluntary_curtailment_cost_eur__{unit.id}"] = (
                    frame[voluntary_column] * config.voluntary_curtailment_cost_eur_per_mwh * dt
                )
            if config.kind == "shiftable":
                frame[f"demand_shift_cost_eur__{unit.id}"] = (
                    (frame[shift_down_column] + frame[shift_up_column])
                    * config.shift_cost_eur_per_mwh
                    * dt
                )
            if config.kind in {"deferrable", "ev_charging"}:
                frame[f"demand_task_unserved_cost_eur__{unit.id}"] = (
                    frame[f"demand_task_unserved_mwh__{unit.id}"]
                    * config.task_unserved_penalty_eur_per_mwh
                )
        if units:
            self._add_demand_aggregate_columns(
                frame,
                units,
                frame["demand_baseline_mw"].to_numpy(dtype=np.float64),
            )
            frame["source_load_shed_mw"] = frame["demand_involuntary_shed_mw"]
        frame["demand_adjusted_mw"] = (
            frame["demand_baseline_mw"]
            - frame["demand_voluntary_curtailment_mw"]
            - frame["demand_shift_down_mw"]
            + frame["demand_shift_up_mw"]
            + frame["demand_task_charge_mw"]
        )
        frame["demand_response_delta_mw"] = (
            frame["demand_adjusted_mw"] - frame["demand_baseline_mw"]
        )
        frame["demand_served_before_network_mw"] = (
            frame["demand_adjusted_mw"] - frame["demand_involuntary_shed_mw"]
        )
        frame["demand_involuntary_shed_cost_eur"] = self._sum_prefixed_frame_columns(
            frame,
            "demand_involuntary_shed_cost_eur__",
        )
        frame["demand_voluntary_curtailment_cost_eur"] = self._sum_prefixed_frame_columns(
            frame,
            "demand_voluntary_curtailment_cost_eur__",
        )
        frame["demand_shift_cost_eur"] = self._sum_prefixed_frame_columns(
            frame,
            "demand_shift_cost_eur__",
        )
        frame["demand_task_unserved_cost_eur"] = self._sum_prefixed_frame_columns(
            frame,
            "demand_task_unserved_cost_eur__",
        )

    def _add_reserve_accounting_columns(
        self,
        frame: pd.DataFrame,
        problem: FormulationProblem,
    ) -> None:
        if not problem.reserves.enabled:
            return
        reserve_config = self.config.reserves
        thermal_up = self._sum_prefixed_frame_columns(frame, f"{THERMAL_UPWARD_RESERVE_BLOCK}__")
        thermal_down = self._sum_prefixed_frame_columns(
            frame, f"{THERMAL_DOWNWARD_RESERVE_BLOCK}__"
        )
        storage_up = self._sum_prefixed_frame_columns(frame, f"{STORAGE_UPWARD_RESERVE_BLOCK}__")
        storage_down = self._sum_prefixed_frame_columns(
            frame, f"{STORAGE_DOWNWARD_RESERVE_BLOCK}__"
        )
        demand_up = self._sum_prefixed_frame_columns(frame, f"{DEMAND_UPWARD_RESERVE_BLOCK}__")
        import_up = (
            frame[IMPORT_UPWARD_RESERVE_BLOCK]
            if IMPORT_UPWARD_RESERVE_BLOCK in frame
            else pd.Series(0.0, index=frame.index)
        )
        import_down = (
            frame[IMPORT_DOWNWARD_RESERVE_BLOCK]
            if IMPORT_DOWNWARD_RESERVE_BLOCK in frame
            else pd.Series(0.0, index=frame.index)
        )
        if RESERVE_LARGEST_CONTINGENCY_BLOCK not in frame:
            frame[RESERVE_LARGEST_CONTINGENCY_BLOCK] = 0.0
        frame["reserve_upward_requirement_base_mw"] = problem.reserves.upward_requirement_mw
        frame["reserve_downward_requirement_mw"] = problem.reserves.downward_requirement_mw
        frame["reserve_upward_requirement_mw"] = (
            frame["reserve_upward_requirement_base_mw"]
            + reserve_config.largest_online_contingency_fraction
            * frame[RESERVE_LARGEST_CONTINGENCY_BLOCK]
        )
        frame["thermal_upward_reserve_mw"] = thermal_up
        frame["thermal_downward_reserve_mw"] = thermal_down
        frame["storage_upward_reserve_mw"] = storage_up
        frame["storage_downward_reserve_mw"] = storage_down
        frame["demand_upward_reserve_mw"] = demand_up
        frame["import_upward_reserve_mw"] = import_up
        frame["import_downward_reserve_mw"] = import_down
        frame["reserve_upward_procured_mw"] = thermal_up + storage_up + demand_up + import_up
        frame["reserve_downward_procured_mw"] = thermal_down + storage_down + import_down
        frame["reserve_upward_residual_mw"] = (
            frame["reserve_upward_procured_mw"]
            + frame[RESERVE_UPWARD_SHORTFALL_BLOCK]
            - frame["reserve_upward_requirement_mw"]
        )
        frame["reserve_downward_residual_mw"] = (
            frame["reserve_downward_procured_mw"]
            + frame[RESERVE_DOWNWARD_SHORTFALL_BLOCK]
            - frame["reserve_downward_requirement_mw"]
        )
        dt = self.config.simulation.time_step_hours
        frame["reserve_procurement_cost_eur"] = dt * (
            thermal_up * reserve_config.thermal_upward_cost_eur_per_mw_hour
            + thermal_down * reserve_config.thermal_downward_cost_eur_per_mw_hour
            + storage_up * reserve_config.storage_upward_cost_eur_per_mw_hour
            + storage_down * reserve_config.storage_downward_cost_eur_per_mw_hour
            + demand_up * reserve_config.demand_response_upward_cost_eur_per_mw_hour
            + import_up * reserve_config.import_upward_cost_eur_per_mw_hour
            + import_down * reserve_config.import_downward_cost_eur_per_mw_hour
        )
        frame["reserve_shortfall_cost_eur"] = dt * (
            frame[RESERVE_UPWARD_SHORTFALL_BLOCK]
            * reserve_config.upward_shortfall_penalty_eur_per_mw_hour
            + frame[RESERVE_DOWNWARD_SHORTFALL_BLOCK]
            * reserve_config.downward_shortfall_penalty_eur_per_mw_hour
        )

    @staticmethod
    def _sum_prefixed_frame_columns(frame: pd.DataFrame, prefix: str) -> pd.Series:
        columns = [column for column in frame.columns if column.startswith(prefix)]
        return frame[columns].sum(axis=1) if columns else pd.Series(0.0, index=frame.index)

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
        hydro_units: tuple[HydroUnit, ...] | None = None,
        demand_units: tuple[DemandUnit, ...] | None = None,
    ) -> float:
        columns = [
            "renewable_used_mw",
            "imports_mw",
            "source_load_shed_mw",
            "renewable_curtailed_mw",
        ]
        columns.extend(
            column
            for column in (
                RESERVE_UPWARD_SHORTFALL_BLOCK,
                RESERVE_DOWNWARD_SHORTFALL_BLOCK,
                RESERVE_LARGEST_CONTINGENCY_BLOCK,
                IMPORT_UPWARD_RESERVE_BLOCK,
                IMPORT_DOWNWARD_RESERVE_BLOCK,
                "reserve_upward_procured_mw",
                "reserve_downward_procured_mw",
            )
            if column in frame
        )
        columns.extend(
            column
            for column in frame.columns
            if column.startswith(f"{THERMAL_UPWARD_RESERVE_BLOCK}__")
            or column.startswith(f"{THERMAL_DOWNWARD_RESERVE_BLOCK}__")
            or column.startswith(f"{STORAGE_UPWARD_RESERVE_BLOCK}__")
            or column.startswith(f"{STORAGE_DOWNWARD_RESERVE_BLOCK}__")
            or column.startswith(f"{DEMAND_UPWARD_RESERVE_BLOCK}__")
        )
        renewable_asset_columns = [
            column
            for column in frame.columns
            if column.startswith("renewable_used_mw__")
            or column.startswith("renewable_curtailed_mw__")
        ]
        columns.extend(renewable_asset_columns)
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
        if hydro_units is not None:
            for hydro in hydro_units:
                columns.extend(
                    [
                        f"hydro_generation_mw__{hydro.id}",
                        f"hydro_release_mw__{hydro.id}",
                        f"hydro_spill_mw__{hydro.id}",
                        f"hydro_reservoir_mwh__{hydro.id}",
                    ]
                )
        if demand_units is not None:
            for demand in demand_units:
                columns.append(f"demand_involuntary_shed_mw__{demand.id}")
                for block in (
                    "demand_voluntary_curtailment_mw",
                    "demand_shift_down_mw",
                    "demand_shift_up_mw",
                    "demand_task_charge_mw",
                    "demand_task_unserved_mwh",
                ):
                    column = f"{block}__{demand.id}"
                    if column in frame:
                        columns.append(column)
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
        if hydro_units is not None:
            self._add_hydro_aggregate_columns(frame, hydro_units)
        if demand_units is not None:
            self._add_demand_aggregate_columns(
                frame,
                demand_units,
                frame["demand_baseline_mw"].to_numpy(dtype=np.float64),
            )
            if demand_units:
                frame["source_load_shed_mw"] = frame["demand_involuntary_shed_mw"]
        return max_clipped

    def _cost_components(
        self,
        frame: pd.DataFrame,
        problem: FormulationProblem,
    ) -> dict[str, float]:
        dt = self.config.simulation.time_step_hours
        imports = self.config.imports
        penalties = self.config.penalties
        network_efficiency = 1.0 - self.config.network.loss_fraction
        thermal_units = problem.thermal_units
        storage_units = problem.storage_units
        demand_shed_cost_columns = [
            column
            for column in frame.columns
            if column.startswith("demand_involuntary_shed_cost_eur__")
        ]
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
                np.dot(
                    frame["imports_mw"].to_numpy(dtype=np.float64),
                    problem.import_prices_eur_per_mwh,
                )
                * dt
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
            "hydro_terminal_value_eur": -float(frame["hydro_terminal_value_eur"].sum()),
            "demand_voluntary_curtailment_cost_eur": float(
                frame["demand_voluntary_curtailment_cost_eur"].sum()
            ),
            "demand_shift_cost_eur": float(frame["demand_shift_cost_eur"].sum()),
            "demand_task_unserved_cost_eur": float(frame["demand_task_unserved_cost_eur"].sum()),
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
            "reserve_procurement_cost_eur": float(
                frame["reserve_procurement_cost_eur"].sum()
                if "reserve_procurement_cost_eur" in frame
                else 0.0
            ),
            "reserve_shortfall_cost_eur": float(
                frame["reserve_shortfall_cost_eur"].sum()
                if "reserve_shortfall_cost_eur" in frame
                else 0.0
            ),
            "dispatch_load_shedding_cost_eur": float(
                frame[demand_shed_cost_columns].sum().sum()
                if demand_shed_cost_columns
                else frame["source_load_shed_mw"].sum()
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
