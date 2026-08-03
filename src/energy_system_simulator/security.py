from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linprog

from energy_system_simulator.config import (
    ModelConfig,
    ThermalGeneratorConfig,
    TransmissionLineConfig,
)
from energy_system_simulator.exceptions import EnergySystemError

ContingencyKind = Literal["line", "generator", "import"]


class SecurityError(EnergySystemError):
    """Invalid security-analysis request or unsupported dispatch output."""


class SecurityDispatchResult(Protocol):
    """Solved dispatch output required by the security evaluator."""

    @property
    def timeseries(self) -> pd.DataFrame:
        """Period-indexed solved dispatch table."""
        ...

    @property
    def objective_eur(self) -> float:
        """Base-case dispatch objective."""
        ...


@dataclass(frozen=True)
class Contingency:
    """One selected N-1 outage to evaluate."""

    id: str
    kind: ContingencyKind
    asset_id: str


@dataclass(frozen=True)
class SecurityOptions:
    """Post-contingency redispatch policy."""

    allow_emergency_actions: bool = True
    emergency_shed_penalty_eur_per_mwh: float = 1_000_000.0
    emergency_overload_penalty_eur_per_mw_hour: float = 1_000_000.0
    require_committed_reserves: bool = True


@dataclass(frozen=True)
class SecurityEvaluation:
    """Security-study outputs separate from base dispatch accounting."""

    records: pd.DataFrame
    summary: dict[str, object]

    @property
    def secure(self) -> bool:
        """Whether every checked contingency is feasible without emergency actions."""
        return bool(self.summary["secure"])

    def write(self, directory: str | Path) -> None:
        """Write security diagnostics to a directory."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.records.to_csv(output / "security_contingencies.csv", index=False)
        (output / "security_summary.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def default_contingencies(
    config: ModelConfig,
    *,
    include_lines: bool = True,
    include_generators: bool = True,
    include_imports: bool = False,
) -> tuple[Contingency, ...]:
    """Build the default explicit N-1 contingency set."""
    contingencies: list[Contingency] = []
    if include_lines:
        contingencies.extend(
            Contingency(id=f"line:{line.id}", kind="line", asset_id=line.id)
            for line in config.portfolio.lines
        )
    if include_generators:
        contingencies.extend(
            Contingency(id=f"generator:{unit.id}", kind="generator", asset_id=unit.id)
            for unit in config.portfolio.thermal_generators
        )
    if include_imports and config.portfolio.imports:
        contingencies.append(Contingency(id="import:imports", kind="import", asset_id="imports"))
    return tuple(contingencies)


def evaluate_security(
    config: ModelConfig,
    result: SecurityDispatchResult,
    *,
    contingencies: tuple[Contingency, ...] | None = None,
    options: SecurityOptions | None = None,
) -> SecurityEvaluation:
    """Evaluate explicit post-contingency redispatch for a solved nodal dispatch."""
    if config.network.network_mode != "nodal":
        raise SecurityError("Security evaluation requires nodal dispatch output")
    if not config.portfolio.lines:
        raise SecurityError("Security evaluation requires at least one transmission line")
    selected = contingencies or default_contingencies(config)
    policy = options or SecurityOptions()
    if not selected:
        raise SecurityError("Security evaluation requires at least one contingency")

    records = [
        _evaluate_one(config, result, contingency, period, policy)
        for period in range(len(result.timeseries))
        for contingency in selected
    ]
    frame = pd.DataFrame.from_records(records)
    security_costs = frame["security_cost_eur"].to_numpy(dtype=np.float64)
    finite_security_costs = security_costs[np.isfinite(security_costs)]
    has_nonfinite_security_cost = len(finite_security_costs) != len(security_costs)
    max_security_cost = (
        None
        if has_nonfinite_security_cost
        else float(finite_security_costs.max())
        if len(finite_security_costs)
        else 0.0
    )
    total_security_cost = (
        None
        if has_nonfinite_security_cost
        else float(finite_security_costs.sum())
        if len(finite_security_costs)
        else 0.0
    )
    binding = frame.sort_values(
        ["security_cost_eur", "emergency_load_shed_mw", "emergency_overload_mw"],
        ascending=False,
    ).iloc[0]
    summary: dict[str, object] = {
        "schema_version": 1,
        "formulation": "explicit_post_contingency_dc_redispatch",
        "contingencies_checked": len(selected),
        "periods_checked": len(result.timeseries),
        "secure": bool(frame["secure"].all()),
        "total_security_cost_eur": total_security_cost,
        "max_security_cost_eur": max_security_cost,
        "infeasible_checks": int((frame["solver_status"] != "optimal").sum()),
        "max_emergency_load_shed_mw": float(frame["emergency_load_shed_mw"].max()),
        "max_emergency_overload_mw": float(frame["emergency_overload_mw"].max()),
        "binding_contingency": str(binding["contingency_id"]),
        "binding_period": int(binding["period"]),
        "binding_overloaded_element": str(binding["overloaded_element"]),
        "base_objective_eur": float(result.objective_eur),
        "base_costs_are_separate": True,
    }
    return SecurityEvaluation(records=frame, summary=summary)


def lodf_line_outage_flows(
    config: ModelConfig,
    base_flows_mw: dict[str, float],
    *,
    failed_line_id: str,
) -> dict[str, float]:
    """Estimate post-outage flows using DC LODF factors for one line outage."""
    base_buses = tuple(bus.id for bus in config.portfolio.buses)
    lines = config.portfolio.lines
    line_index = {line.id: index for index, line in enumerate(lines)}
    if failed_line_id not in line_index:
        raise SecurityError(f"Unknown failed line for LODF: {failed_line_id}")
    ptdf = _ptdf_matrix(base_buses, lines, config.network.slack_bus_id)
    outage_column = line_index[failed_line_id]
    denominator = 1.0 - ptdf[outage_column, outage_column]
    if abs(denominator) < 1e-9:
        raise SecurityError(f"Line outage islanding or singular LODF denominator: {failed_line_id}")
    failed_flow = float(base_flows_mw[failed_line_id])
    estimates: dict[str, float] = {}
    for line in lines:
        if line.id == failed_line_id:
            estimates[line.id] = 0.0
            continue
        row = line_index[line.id]
        lodf = ptdf[row, outage_column] / denominator
        estimates[line.id] = float(base_flows_mw[line.id] + lodf * failed_flow)
    return estimates


def explicit_line_outage_flows(
    config: ModelConfig,
    bus_injections_mw: dict[str, float],
    *,
    failed_line_id: str,
) -> dict[str, float]:
    """Solve explicit DC flows for fixed injections after one line outage."""
    buses = tuple(bus.id for bus in config.portfolio.buses)
    active_lines = tuple(line for line in config.portfolio.lines if line.id != failed_line_id)
    return _solve_dc_flows(buses, active_lines, bus_injections_mw, config.network.slack_bus_id)


def _evaluate_one(
    config: ModelConfig,
    result: SecurityDispatchResult,
    contingency: Contingency,
    period: int,
    options: SecurityOptions,
) -> dict[str, object]:
    builder = _RedispatchProblem(config, result, contingency, period, options)
    solution = builder.solve()
    solver_status = str(solution["solver_status"])
    load_shed = float(cast(float, solution["emergency_load_shed_mw"]))
    overload = float(cast(float, solution["emergency_overload_mw"]))
    secure = bool(solver_status == "optimal" and load_shed <= 1e-6 and overload <= 1e-6)
    return {
        "schema_version": 1,
        "period": period,
        "contingency_id": contingency.id,
        "contingency_kind": contingency.kind,
        "asset_id": contingency.asset_id,
        "secure": secure,
        **solution,
    }


class _RedispatchProblem:
    def __init__(
        self,
        config: ModelConfig,
        result: SecurityDispatchResult,
        contingency: Contingency,
        period: int,
        options: SecurityOptions,
    ) -> None:
        self.config = config
        self.result = result
        self.frame = result.timeseries
        self.contingency = contingency
        self.period = period
        self.options = options
        self.dt = config.simulation.time_step_hours
        self.buses = tuple(bus.id for bus in config.portfolio.buses)
        self.bus_index = {bus_id: index for index, bus_id in enumerate(self.buses)}
        self.lines = config.portfolio.lines
        self.thermal = config.portfolio.thermal_generators
        self.slack_bus_id = config.network.slack_bus_id or self.buses[0]
        self.names: list[str] = []
        self.bounds: list[tuple[float | None, float | None]] = []
        self.objective: list[float] = []

    def solve(self) -> dict[str, object]:
        self._validate_columns()
        fixed_injection = self._base_fixed_injection()
        thermal_indices = self._add_thermal_variables()
        import_index = self._add_import_variable()
        shed_indices = self._add_shed_variables()
        theta_indices = self._add_theta_variables()
        flow_indices = self._add_flow_variables()
        overload_pos, overload_neg = self._add_overload_variables()

        a_eq: list[list[float]] = []
        b_eq: list[float] = []
        self._add_line_equations(a_eq, b_eq, theta_indices, flow_indices)
        self._add_bus_balances(
            a_eq,
            b_eq,
            fixed_injection,
            thermal_indices,
            import_index,
            shed_indices,
            flow_indices,
        )
        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        self._add_line_capacity_limits(a_ub, b_ub, flow_indices, overload_pos, overload_neg)

        result = linprog(
            c=np.asarray(self.objective, dtype=np.float64),
            A_ub=np.asarray(a_ub, dtype=np.float64) if a_ub else None,
            b_ub=np.asarray(b_ub, dtype=np.float64) if b_ub else None,
            A_eq=np.asarray(a_eq, dtype=np.float64),
            b_eq=np.asarray(b_eq, dtype=np.float64),
            bounds=self.bounds,
            method="highs",
        )
        if not result.success or result.x is None:
            return self._failed_solution(result.message)
        return self._success_solution(
            result.x, thermal_indices, shed_indices, overload_pos, overload_neg
        )

    def _validate_columns(self) -> None:
        for bus_id in self.buses:
            self._require_column(f"bus_net_injection_mw__{bus_id}")
        for line in self.lines:
            self._require_column(f"line_flow_mw__{line.id}")
            self._require_column(f"line_capacity_available_mw__{line.id}")
        for unit in self.thermal:
            self._require_column(f"thermal_output_mw__{unit.id}")
            self._require_column(f"thermal_on__{unit.id}")

    def _base_fixed_injection(self) -> dict[str, float]:
        injection = {
            bus_id: self._value(f"bus_net_injection_mw__{bus_id}") for bus_id in self.buses
        }
        for unit in self.thermal:
            injection[unit.bus_id] -= self._value(f"thermal_output_mw__{unit.id}")
        if self.config.portfolio.imports:
            injection[self.config.portfolio.imports[0].bus_id] -= self._value("imports_mw")
        return injection

    def _add_thermal_variables(self) -> dict[str, int]:
        indices: dict[str, int] = {}
        for unit in self.thermal:
            lower, upper = self._thermal_bounds(unit)
            indices[unit.id] = self._add_variable(
                f"thermal_output_mw__{unit.id}", lower, upper, 0.0
            )
        return indices

    def _thermal_bounds(self, unit: ThermalGeneratorConfig) -> tuple[float, float]:
        if self.contingency.kind == "generator" and self.contingency.asset_id == unit.id:
            return 0.0, 0.0
        base = self._value(f"thermal_output_mw__{unit.id}")
        online = self._value(f"thermal_on__{unit.id}") >= 0.5
        if not online:
            return 0.0, 0.0
        config = unit.config
        capacity = self._value(
            f"thermal_capacity_available_mw__{unit.id}",
            fallback=config.maximum_output_mw,
        )
        minimum = config.minimum_output_mw
        upward = self._reserve_or_zero(f"thermal_upward_reserve_mw__{unit.id}")
        downward = self._reserve_or_zero(f"thermal_downward_reserve_mw__{unit.id}")
        if not self.options.require_committed_reserves:
            upward = min(config.ramp_up_mw_per_hour * self.dt, capacity - base)
            downward = min(config.ramp_down_mw_per_hour * self.dt, base - minimum)
        lower = max(minimum, base - downward)
        upper = min(capacity, base + upward)
        return max(0.0, lower), max(0.0, upper)

    def _add_import_variable(self) -> int | None:
        if not self.config.portfolio.imports:
            return None
        base = self._value("imports_mw")
        if self.contingency.kind == "import":
            lower, upper = 0.0, 0.0
        else:
            upward = self._reserve_or_zero("import_upward_reserve_mw")
            downward = self._reserve_or_zero("import_downward_reserve_mw")
            if not self.options.require_committed_reserves:
                upward = self.config.imports.maximum_power_mw - base
                downward = base
            lower, upper = max(0.0, base - downward), max(0.0, base + upward)
        return self._add_variable("imports_mw", lower, upper, 0.0)

    def _add_shed_variables(self) -> dict[str, int]:
        indices: dict[str, int] = {}
        upper = None if self.options.allow_emergency_actions else 0.0
        penalty = self.options.emergency_shed_penalty_eur_per_mwh * self.dt
        for demand in self.config.portfolio.demand:
            indices[demand.id] = self._add_variable(
                f"emergency_shed_mw__{demand.id}", 0.0, upper, penalty
            )
        return indices

    def _add_theta_variables(self) -> dict[str, int]:
        indices: dict[str, int] = {}
        for bus_id in self.buses:
            if bus_id == self.slack_bus_id:
                continue
            indices[bus_id] = self._add_variable(f"theta__{bus_id}", None, None, 0.0)
        return indices

    def _add_flow_variables(self) -> dict[str, int]:
        return {
            line.id: self._add_variable(f"line_flow_mw__{line.id}", None, None, 0.0)
            for line in self.lines
        }

    def _add_overload_variables(self) -> tuple[dict[str, int], dict[str, int]]:
        pos: dict[str, int] = {}
        neg: dict[str, int] = {}
        penalty = self.options.emergency_overload_penalty_eur_per_mw_hour * self.dt
        for line in self.lines:
            upper = None if self.options.allow_emergency_actions else 0.0
            if self.contingency.kind == "line" and self.contingency.asset_id == line.id:
                upper = 0.0
            pos[line.id] = self._add_variable(
                f"emergency_overload_pos_mw__{line.id}", 0.0, upper, penalty
            )
            neg[line.id] = self._add_variable(
                f"emergency_overload_neg_mw__{line.id}", 0.0, upper, penalty
            )
        return pos, neg

    def _add_line_equations(
        self,
        a_eq: list[list[float]],
        b_eq: list[float],
        theta_indices: dict[str, int],
        flow_indices: dict[str, int],
    ) -> None:
        for line in self.lines:
            row = self._row()
            row[flow_indices[line.id]] = 1.0
            if self.contingency.kind == "line" and self.contingency.asset_id == line.id:
                a_eq.append(row)
                b_eq.append(0.0)
                continue
            if line.from_bus_id != self.slack_bus_id:
                row[theta_indices[line.from_bus_id]] -= line.susceptance
            if line.to_bus_id != self.slack_bus_id:
                row[theta_indices[line.to_bus_id]] += line.susceptance
            a_eq.append(row)
            b_eq.append(0.0)

    def _add_bus_balances(
        self,
        a_eq: list[list[float]],
        b_eq: list[float],
        fixed_injection: dict[str, float],
        thermal_indices: dict[str, int],
        import_index: int | None,
        shed_indices: dict[str, int],
        flow_indices: dict[str, int],
    ) -> None:
        for bus_id in self.buses:
            row = self._row()
            for unit in self.thermal:
                if unit.bus_id == bus_id:
                    row[thermal_indices[unit.id]] += 1.0
            if import_index is not None and self.config.portfolio.imports[0].bus_id == bus_id:
                row[import_index] += 1.0
            for demand in self.config.portfolio.demand:
                if demand.bus_id == bus_id:
                    row[shed_indices[demand.id]] += 1.0
            for line in self.lines:
                if line.from_bus_id == bus_id:
                    row[flow_indices[line.id]] -= 1.0
                if line.to_bus_id == bus_id:
                    row[flow_indices[line.id]] += 1.0
            a_eq.append(row)
            b_eq.append(-fixed_injection[bus_id])

    def _add_line_capacity_limits(
        self,
        a_ub: list[list[float]],
        b_ub: list[float],
        flow_indices: dict[str, int],
        overload_pos: dict[str, int],
        overload_neg: dict[str, int],
    ) -> None:
        for line in self.lines:
            capacity = self._line_capacity(line)
            row = self._row()
            row[flow_indices[line.id]] = 1.0
            row[overload_pos[line.id]] = -1.0
            a_ub.append(row)
            b_ub.append(capacity)
            row = self._row()
            row[flow_indices[line.id]] = -1.0
            row[overload_neg[line.id]] = -1.0
            a_ub.append(row)
            b_ub.append(capacity)

    def _line_capacity(self, line: TransmissionLineConfig) -> float:
        if self.contingency.kind == "line" and self.contingency.asset_id == line.id:
            return 0.0
        return self._value(f"line_capacity_available_mw__{line.id}", fallback=line.capacity_mw)

    def _success_solution(
        self,
        x: npt.NDArray[np.float64],
        thermal_indices: dict[str, int],
        shed_indices: dict[str, int],
        overload_pos: dict[str, int],
        overload_neg: dict[str, int],
    ) -> dict[str, object]:
        redispatch_up = 0.0
        redispatch_down = 0.0
        for unit_id, index in thermal_indices.items():
            base = self._value(f"thermal_output_mw__{unit_id}")
            delta = float(x[index] - base)
            redispatch_up += max(0.0, delta)
            redispatch_down += max(0.0, -delta)
        shed = sum(float(x[index]) for index in shed_indices.values())
        overloads = {
            line.id: max(float(x[overload_pos[line.id]]), float(x[overload_neg[line.id]]))
            for line in self.lines
        }
        overloaded = max(overloads, key=lambda line_id: overloads[line_id]) if overloads else ""
        overload = overloads[overloaded] if overloaded else 0.0
        cost = (
            shed * self.options.emergency_shed_penalty_eur_per_mwh
            + overload * self.options.emergency_overload_penalty_eur_per_mw_hour
        ) * self.dt
        return {
            "solver_status": "optimal",
            "redispatch_up_mw": redispatch_up,
            "redispatch_down_mw": redispatch_down,
            "emergency_load_shed_mw": shed,
            "emergency_overload_mw": overload,
            "security_cost_eur": cost,
            "overloaded_element": overloaded if overload > 1e-6 else "",
        }

    def _failed_solution(self, message: str) -> dict[str, object]:
        return {
            "solver_status": "infeasible",
            "redispatch_up_mw": 0.0,
            "redispatch_down_mw": 0.0,
            "emergency_load_shed_mw": 0.0,
            "emergency_overload_mw": 0.0,
            "security_cost_eur": float("inf"),
            "overloaded_element": "",
            "message": message,
        }

    def _reserve_or_zero(self, column: str) -> float:
        if column not in self.frame.columns:
            return 0.0
        return self._value(column)

    def _value(self, column: str, *, fallback: float | None = None) -> float:
        if column not in self.frame.columns:
            if fallback is not None:
                return fallback
            raise SecurityError(f"Security evaluation requires output column: {column}")
        return float(self.frame[column].iloc[self.period])

    def _require_column(self, column: str) -> None:
        if column not in self.frame.columns:
            raise SecurityError(f"Security evaluation requires output column: {column}")

    def _add_variable(
        self,
        name: str,
        lower: float | None,
        upper: float | None,
        cost: float,
    ) -> int:
        index = len(self.names)
        self.names.append(name)
        self.bounds.append((lower if lower is not None else -np.inf, upper))
        self.objective.append(cost)
        return index

    def _row(self) -> list[float]:
        return [0.0] * len(self.names)


def _ptdf_matrix(
    buses: tuple[str, ...],
    lines: tuple[TransmissionLineConfig, ...],
    slack_bus_id: str | None,
) -> npt.NDArray[np.float64]:
    ptdf = np.zeros((len(lines), len(lines)), dtype=np.float64)
    for column, line in enumerate(lines):
        injections = {bus_id: 0.0 for bus_id in buses}
        injections[line.from_bus_id] = 1.0
        injections[line.to_bus_id] = -1.0
        flows = _solve_dc_flows(buses, lines, injections, slack_bus_id)
        for row, candidate in enumerate(lines):
            ptdf[row, column] = flows[candidate.id]
    return ptdf


def _solve_dc_flows(
    buses: tuple[str, ...],
    lines: tuple[TransmissionLineConfig, ...],
    bus_injections_mw: dict[str, float],
    slack_bus_id: str | None,
) -> dict[str, float]:
    slack = slack_bus_id or buses[0]
    unknown_buses = [bus for bus in buses if bus != slack]
    matrix = np.zeros((len(unknown_buses), len(unknown_buses)), dtype=np.float64)
    rhs = np.array([bus_injections_mw[bus] for bus in unknown_buses], dtype=np.float64)
    bus_position = {bus: index for index, bus in enumerate(unknown_buses)}
    for line in lines:
        for bus_id, sign in ((line.from_bus_id, 1.0), (line.to_bus_id, -1.0)):
            if bus_id == slack:
                continue
            row = bus_position[bus_id]
            matrix[row, row] += line.susceptance
            other = line.to_bus_id if sign > 0 else line.from_bus_id
            if other != slack:
                matrix[row, bus_position[other]] -= line.susceptance
    try:
        theta_unknown = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise SecurityError("Post-contingency DC network is singular") from error
    theta = {slack: 0.0}
    theta.update({bus: float(theta_unknown[index]) for bus, index in bus_position.items()})
    return {
        line.id: float(line.susceptance * (theta[line.from_bus_id] - theta[line.to_bus_id]))
        for line in lines
    }
