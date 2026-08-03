from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_system_simulator.config import ModelConfig, TransmissionLineConfig
from energy_system_simulator.exceptions import EnergySystemError

PeriodPolicy = Literal["peak_demand", "peak_renewable", "congestion"]


class ACValidationError(EnergySystemError):
    """Invalid AC validation request or unsupported dispatch output."""


class ACDispatchResult(Protocol):
    """Solved dispatch output required by the AC validation bridge."""

    @property
    def timeseries(self) -> pd.DataFrame:
        """Period-indexed solved dispatch table."""
        ...

    @property
    def objective_eur(self) -> float:
        """Base-case dispatch objective."""
        ...


@dataclass(frozen=True)
class ACValidationOptions:
    policies: tuple[PeriodPolicy, ...] = ("peak_demand", "peak_renewable", "congestion")
    periods: tuple[int, ...] = ()
    timestamps: tuple[str, ...] = ()
    tolerance_mva: float = 1e-6
    mismatch_tolerance_pu: float = 1e-8
    max_iterations: int = 30


@dataclass(frozen=True)
class ACValidation:
    records: pd.DataFrame
    summary: dict[str, object]

    @property
    def valid(self) -> bool:
        return bool(self.summary["valid"])

    def write(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.records.to_csv(output / "ac_validation.csv", index=False)
        (output / "ac_validation_summary.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def select_ac_validation_periods(
    config: ModelConfig,
    result: ACDispatchResult,
    options: ACValidationOptions | None = None,
) -> tuple[int, ...]:
    policy = options or ACValidationOptions()
    frame = result.timeseries
    selected: set[int] = set()
    for period in policy.periods:
        if period < 0 or period >= len(frame):
            raise ACValidationError(f"Selected AC validation period is out of range: {period}")
        selected.add(period)
    if policy.timestamps:
        if "timestamp" not in frame:
            raise ACValidationError("Timestamp-based AC validation requires a timestamp column")
        timestamps = frame["timestamp"].astype(str)
        for timestamp in policy.timestamps:
            matches = timestamps[timestamps == timestamp]
            if matches.empty:
                raise ACValidationError(f"Selected AC validation timestamp not found: {timestamp}")
            selected.add(int(matches.index[0]))
    for item in policy.policies:
        if item == "peak_demand":
            selected.add(_idxmax(frame, "demand_adjusted_mw", "demand_baseline_mw"))
        elif item == "peak_renewable":
            selected.add(_idxmax(frame, "renewable_available_mw", "renewable_used_mw"))
        elif item == "congestion":
            selected.add(
                _idxmax(frame, "line_max_abs_utilisation", _first_line_flow_column(config))
            )
        else:
            raise ACValidationError(f"Unsupported AC validation period policy: {item}")
    return tuple(sorted(selected))


def validate_ac_power_flow(
    config: ModelConfig,
    result: ACDispatchResult,
    options: ACValidationOptions | None = None,
) -> ACValidation:
    """Validate selected DC-dispatch periods with a nonlinear AC power-flow solve."""
    if config.network.network_mode != "nodal":
        raise ACValidationError("AC validation requires nodal dispatch output")
    if not config.portfolio.lines:
        raise ACValidationError("AC validation requires at least one transmission line")
    policy = options or ACValidationOptions()
    frame = result.timeseries
    _validate_columns(config, frame)
    periods = select_ac_validation_periods(config, result, policy)
    records = [_solve_period(config, frame, period, policy) for period in periods]
    records_frame = pd.DataFrame.from_records(records)
    valid = bool(
        records_frame["converged"].all()
        and (records_frame["max_voltage_violation_pu"] <= policy.tolerance_mva).all()
        and (records_frame["max_branch_overload_mva"] <= policy.tolerance_mva).all()
        and (records_frame["max_reactive_limit_violation_mvar"] <= policy.tolerance_mva).all()
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "formulation": "post_dispatch_newton_raphson_ac_validation",
        "dependency_decision": (
            "Uses a narrow internal Newton-Raphson solver to avoid adding a heavy "
            "AC-OPF dependency; this is validation only, not AC optimal power flow."
        ),
        "validated_periods": [int(period) for period in periods],
        "validated_period_count": len(periods),
        "unvalidated_periods_have_no_ac_feasibility_claim": True,
        "valid": valid,
        "non_converged_periods": int((~records_frame["converged"]).sum()),
        "max_voltage_violation_pu": float(records_frame["max_voltage_violation_pu"].max()),
        "max_branch_overload_mva": float(records_frame["max_branch_overload_mva"].max()),
        "max_reactive_limit_violation_mvar": float(
            records_frame["max_reactive_limit_violation_mvar"].max()
        ),
        "max_active_loss_mw": float(records_frame["active_losses_mw"].max()),
        "max_dc_active_flow_mismatch_mw": float(
            records_frame["max_dc_active_flow_mismatch_mw"].max()
        ),
        "base_objective_eur": float(result.objective_eur),
        "base_costs_are_separate": True,
    }
    return ACValidation(records=records_frame, summary=summary)


def _solve_period(
    config: ModelConfig,
    frame: pd.DataFrame,
    period: int,
    options: ACValidationOptions,
) -> dict[str, object]:
    model = _ACModel(config, frame, period)
    solution = _newton_solve(model, options)
    diagnostics = _diagnostics(model, solution, options)
    return {"schema_version": 1, "period": period, **diagnostics}


@dataclass(frozen=True)
class _ACModel:
    config: ModelConfig
    frame: pd.DataFrame
    period: int

    @property
    def bus_ids(self) -> tuple[str, ...]:
        return tuple(bus.id for bus in self.config.portfolio.buses)

    @property
    def slack_bus_id(self) -> str:
        return self.config.network.slack_bus_id or self.bus_ids[0]

    @property
    def base_mva(self) -> float:
        return self.config.network.ac_base_mva

    @property
    def ybus(self) -> npt.NDArray[np.complex128]:
        buses = self.bus_ids
        index = {bus_id: idx for idx, bus_id in enumerate(buses)}
        matrix = np.zeros((len(buses), len(buses)), dtype=np.complex128)
        for line in self.config.portfolio.lines:
            from_idx = index[line.from_bus_id]
            to_idx = index[line.to_bus_id]
            admittance = _line_admittance(line)
            charging = 1j * line.ac_line_charging_pu / 2.0
            tap = line.transformer_tap_ratio
            matrix[from_idx, from_idx] += admittance / (tap * tap) + charging
            matrix[to_idx, to_idx] += admittance + charging
            matrix[from_idx, to_idx] -= admittance / tap
            matrix[to_idx, from_idx] -= admittance / tap
        return matrix

    @property
    def p_spec(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                _value(self.frame, f"bus_net_injection_mw__{bus_id}", self.period) / self.base_mva
                for bus_id in self.bus_ids
            ],
            dtype=np.float64,
        )

    @property
    def q_load_mvar(self) -> npt.NDArray[np.float64]:
        loads = np.zeros(len(self.bus_ids), dtype=np.float64)
        bus_index = {bus_id: index for index, bus_id in enumerate(self.bus_ids)}
        for demand in self.config.portfolio.demand:
            column = f"demand_served_mw__{demand.id}"
            if column in self.frame:
                loads[bus_index[demand.bus_id]] += (
                    _value(self.frame, column, self.period) * demand.reactive_demand_mvar_per_mw
                )
        return loads

    @property
    def q_spec(self) -> npt.NDArray[np.float64]:
        shunts = np.array([bus.shunt_mvar for bus in self.config.portfolio.buses], dtype=np.float64)
        return -(self.q_load_mvar + shunts) / self.base_mva

    @property
    def bus_types(self) -> dict[str, str]:
        return {
            bus_id: "slack"
            if bus_id == self.slack_bus_id
            else "pv"
            if self._has_online_generator(bus_id)
            else "pq"
            for bus_id in self.bus_ids
        }

    def initial_voltage(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        magnitudes = np.array(
            [bus.voltage_initial_pu for bus in self.config.portfolio.buses],
            dtype=np.float64,
        )
        angles = np.array(
            [
                np.deg2rad(bus.voltage_angle_initial_deg)
                if f"bus_voltage_angle_rad__{bus.id}" not in self.frame
                else _value(self.frame, f"bus_voltage_angle_rad__{bus.id}", self.period)
                for bus in self.config.portfolio.buses
            ],
            dtype=np.float64,
        )
        return magnitudes, angles

    def _has_online_generator(self, bus_id: str) -> bool:
        for thermal_unit in self.config.portfolio.thermal_generators:
            if (
                thermal_unit.bus_id == bus_id
                and _value(self.frame, f"thermal_on__{thermal_unit.id}", self.period) >= 0.5
            ):
                return True
        for hydro_unit in self.config.portfolio.hydro_units:
            column = f"hydro_generation_mw__{hydro_unit.id}"
            if (
                hydro_unit.bus_id == bus_id
                and column in self.frame
                and _value(self.frame, column, self.period) > 1e-9
            ):
                return True
        for renewable_unit in self.config.portfolio.renewable_generators:
            column = f"renewable_used_mw__{renewable_unit.id}"
            if (
                renewable_unit.bus_id == bus_id
                and column in self.frame
                and _value(self.frame, column, self.period) > 1e-9
            ):
                return True
        return False


@dataclass(frozen=True)
class _ACSolution:
    voltage_magnitude_pu: npt.NDArray[np.float64]
    voltage_angle_rad: npt.NDArray[np.float64]
    converged: bool
    iterations: int
    max_mismatch_pu: float


def _newton_solve(model: _ACModel, options: ACValidationOptions) -> _ACSolution:
    voltage, angle = model.initial_voltage()
    variable_buses = [bus for bus in model.bus_ids if bus != model.slack_bus_id]
    pq_buses = [bus for bus in variable_buses if model.bus_types[bus] == "pq"]
    x = _pack_state(model, angle, voltage, variable_buses, pq_buses)
    mismatch = _mismatch(model, x, variable_buses, pq_buses)
    for iteration in range(options.max_iterations + 1):
        max_mismatch = float(np.max(np.abs(mismatch))) if len(mismatch) else 0.0
        if max_mismatch <= options.mismatch_tolerance_pu:
            angle, voltage = _unpack_state(model, x, variable_buses, pq_buses)
            return _ACSolution(voltage, angle, True, iteration, max_mismatch)
        if iteration == options.max_iterations:
            break
        jacobian = _finite_difference_jacobian(model, x, mismatch, variable_buses, pq_buses)
        try:
            step = np.linalg.solve(jacobian, mismatch)
        except np.linalg.LinAlgError:
            break
        x = x - step
        mismatch = _mismatch(model, x, variable_buses, pq_buses)
    angle, voltage = _unpack_state(model, x, variable_buses, pq_buses)
    max_mismatch = float(np.max(np.abs(mismatch))) if len(mismatch) else 0.0
    return _ACSolution(voltage, angle, False, options.max_iterations, max_mismatch)


def _mismatch(
    model: _ACModel,
    x: npt.NDArray[np.float64],
    variable_buses: list[str],
    pq_buses: list[str],
) -> npt.NDArray[np.float64]:
    angle, voltage = _unpack_state(model, x, variable_buses, pq_buses)
    p_calc, q_calc = _power_injections_pu(model.ybus, voltage, angle)
    bus_index = {bus_id: index for index, bus_id in enumerate(model.bus_ids)}
    mismatch: list[float] = []
    for bus_id in variable_buses:
        index = bus_index[bus_id]
        mismatch.append(model.p_spec[index] - p_calc[index])
    for bus_id in pq_buses:
        index = bus_index[bus_id]
        mismatch.append(model.q_spec[index] - q_calc[index])
    return np.array(mismatch, dtype=np.float64)


def _finite_difference_jacobian(
    model: _ACModel,
    x: npt.NDArray[np.float64],
    base_mismatch: npt.NDArray[np.float64],
    variable_buses: list[str],
    pq_buses: list[str],
) -> npt.NDArray[np.float64]:
    epsilon = 1e-6
    jacobian = np.zeros((len(base_mismatch), len(x)), dtype=np.float64)
    for column in range(len(x)):
        trial = x.copy()
        trial[column] += epsilon
        jacobian[:, column] = (
            _mismatch(model, trial, variable_buses, pq_buses) - base_mismatch
        ) / epsilon
    return jacobian


def _pack_state(
    model: _ACModel,
    angle: npt.NDArray[np.float64],
    voltage: npt.NDArray[np.float64],
    variable_buses: list[str],
    pq_buses: list[str],
) -> npt.NDArray[np.float64]:
    bus_index = {bus_id: index for index, bus_id in enumerate(model.bus_ids)}
    return np.array(
        [angle[bus_index[bus_id]] for bus_id in variable_buses]
        + [voltage[bus_index[bus_id]] for bus_id in pq_buses],
        dtype=np.float64,
    )


def _unpack_state(
    model: _ACModel,
    x: npt.NDArray[np.float64],
    variable_buses: list[str],
    pq_buses: list[str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    voltage, angle = model.initial_voltage()
    bus_index = {bus_id: index for index, bus_id in enumerate(model.bus_ids)}
    for offset, bus_id in enumerate(variable_buses):
        angle[bus_index[bus_id]] = x[offset]
    for offset, bus_id in enumerate(pq_buses, start=len(variable_buses)):
        voltage[bus_index[bus_id]] = x[offset]
    return angle, voltage


def _power_injections_pu(
    ybus: npt.NDArray[np.complex128],
    voltage: npt.NDArray[np.float64],
    angle: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    phasor = voltage * np.exp(1j * angle)
    injection = phasor * np.conj(ybus @ phasor)
    return injection.real.astype(np.float64), injection.imag.astype(np.float64)


def _diagnostics(
    model: _ACModel,
    solution: _ACSolution,
    options: ACValidationOptions,
) -> dict[str, object]:
    voltage_violation = _max_voltage_violation(model, solution.voltage_magnitude_pu)
    branch = _branch_diagnostics(model, solution)
    reactive_violation = _reactive_limit_violation(model, solution)
    branch_overload = float(cast(float, branch["max_branch_overload_mva"]))
    return {
        "converged": solution.converged,
        "iterations": solution.iterations,
        "max_power_mismatch_pu": solution.max_mismatch_pu,
        "max_voltage_pu": float(solution.voltage_magnitude_pu.max()),
        "min_voltage_pu": float(solution.voltage_magnitude_pu.min()),
        "max_voltage_violation_pu": voltage_violation,
        "max_branch_mva": branch["max_branch_mva"],
        "max_branch_overload_mva": branch["max_branch_overload_mva"],
        "binding_branch_id": branch["binding_branch_id"],
        "active_losses_mw": branch["active_losses_mw"],
        "max_dc_active_flow_mismatch_mw": branch["max_dc_active_flow_mismatch_mw"],
        "max_reactive_limit_violation_mvar": reactive_violation,
        "valid": bool(
            solution.converged
            and voltage_violation <= options.tolerance_mva
            and branch_overload <= options.tolerance_mva
            and reactive_violation <= options.tolerance_mva
        ),
    }


def _max_voltage_violation(
    model: _ACModel,
    voltage: npt.NDArray[np.float64],
) -> float:
    violation = 0.0
    for index, bus in enumerate(model.config.portfolio.buses):
        violation = max(violation, bus.voltage_min_pu - float(voltage[index]))
        violation = max(violation, float(voltage[index]) - bus.voltage_max_pu)
    return max(0.0, violation)


def _branch_diagnostics(model: _ACModel, solution: _ACSolution) -> dict[str, object]:
    phasor = solution.voltage_magnitude_pu * np.exp(1j * solution.voltage_angle_rad)
    bus_index = {bus_id: index for index, bus_id in enumerate(model.bus_ids)}
    max_mva = 0.0
    max_overload = 0.0
    active_losses = 0.0
    max_dc_mismatch = 0.0
    binding = ""
    for line in model.config.portfolio.lines:
        from_idx = bus_index[line.from_bus_id]
        to_idx = bus_index[line.to_bus_id]
        y = _line_admittance(line)
        b = 1j * line.ac_line_charging_pu / 2.0
        tap = line.transformer_tap_ratio
        i_from = ((phasor[from_idx] / tap) - phasor[to_idx]) * y / tap + phasor[from_idx] * b
        i_to = (phasor[to_idx] - phasor[from_idx] / tap) * y + phasor[to_idx] * b
        s_from = phasor[from_idx] * np.conj(i_from) * model.base_mva
        s_to = phasor[to_idx] * np.conj(i_to) * model.base_mva
        branch_mva = max(abs(s_from), abs(s_to))
        rating = line.ac_rating_mva if line.ac_rating_mva is not None else line.capacity_mw
        overload = max(0.0, float(branch_mva) - rating)
        if overload > max_overload:
            binding = line.id
        max_mva = max(max_mva, float(branch_mva))
        max_overload = max(max_overload, overload)
        active_losses += float(s_from.real + s_to.real)
        dc_column = f"line_flow_mw__{line.id}"
        if dc_column in model.frame:
            max_dc_mismatch = max(
                max_dc_mismatch,
                abs(float(s_from.real) - _value(model.frame, dc_column, model.period)),
            )
    return {
        "max_branch_mva": max_mva,
        "max_branch_overload_mva": max_overload,
        "binding_branch_id": binding,
        "active_losses_mw": active_losses,
        "max_dc_active_flow_mismatch_mw": max_dc_mismatch,
    }


def _reactive_limit_violation(model: _ACModel, solution: _ACSolution) -> float:
    _, q_calc = _power_injections_pu(
        model.ybus,
        solution.voltage_magnitude_pu,
        solution.voltage_angle_rad,
    )
    q_injection_mvar = q_calc * model.base_mva
    max_violation = 0.0
    shunts = np.array([bus.shunt_mvar for bus in model.config.portfolio.buses], dtype=np.float64)
    required_reactive_generation = q_injection_mvar + model.q_load_mvar + shunts
    for bus_id, required in zip(model.bus_ids, required_reactive_generation, strict=True):
        lower, upper = _reactive_limits_at_bus(model, bus_id)
        if lower is not None:
            max_violation = max(max_violation, lower - float(required))
        if upper is not None:
            max_violation = max(max_violation, float(required) - upper)
    return max(0.0, max_violation)


def _reactive_limits_at_bus(model: _ACModel, bus_id: str) -> tuple[float | None, float | None]:
    minimum = 0.0
    maximum = 0.0
    has_minimum = False
    has_maximum = False
    for thermal_unit in model.config.portfolio.thermal_generators:
        if (
            thermal_unit.bus_id == bus_id
            and _value(model.frame, f"thermal_on__{thermal_unit.id}", model.period) >= 0.5
        ):
            if thermal_unit.reactive_power_min_mvar is not None:
                minimum += thermal_unit.reactive_power_min_mvar
                has_minimum = True
            if thermal_unit.reactive_power_max_mvar is not None:
                maximum += thermal_unit.reactive_power_max_mvar
                has_maximum = True
    for hydro_unit in model.config.portfolio.hydro_units:
        column = f"hydro_generation_mw__{hydro_unit.id}"
        if (
            hydro_unit.bus_id == bus_id
            and column in model.frame
            and _value(model.frame, column, model.period) > 1e-9
        ):
            if hydro_unit.reactive_power_min_mvar is not None:
                minimum += hydro_unit.reactive_power_min_mvar
                has_minimum = True
            if hydro_unit.reactive_power_max_mvar is not None:
                maximum += hydro_unit.reactive_power_max_mvar
                has_maximum = True
    for renewable_unit in model.config.portfolio.renewable_generators:
        column = f"renewable_used_mw__{renewable_unit.id}"
        if (
            renewable_unit.bus_id == bus_id
            and column in model.frame
            and _value(model.frame, column, model.period) > 1e-9
        ):
            if renewable_unit.reactive_power_min_mvar is not None:
                minimum += renewable_unit.reactive_power_min_mvar
                has_minimum = True
            if renewable_unit.reactive_power_max_mvar is not None:
                maximum += renewable_unit.reactive_power_max_mvar
                has_maximum = True
    return (minimum if has_minimum else None, maximum if has_maximum else None)


def _line_admittance(line: TransmissionLineConfig) -> complex:
    reactance = line.ac_reactance_pu
    if reactance is None:
        if line.susceptance <= 0.0:
            raise ACValidationError(f"Line {line.id} needs positive AC reactance")
        reactance = 1.0 / line.susceptance
    impedance = complex(line.ac_resistance_pu, reactance)
    if abs(impedance) <= 0.0:
        raise ACValidationError(f"Line {line.id} has zero AC impedance")
    return 1.0 / impedance


def _validate_columns(config: ModelConfig, frame: pd.DataFrame) -> None:
    for bus in config.portfolio.buses:
        _require_column(frame, f"bus_net_injection_mw__{bus.id}")
    for demand in config.portfolio.demand:
        if demand.reactive_demand_mvar_per_mw != 0.0:
            _require_column(frame, f"demand_served_mw__{demand.id}")
    for unit in config.portfolio.thermal_generators:
        _require_column(frame, f"thermal_on__{unit.id}")


def _idxmax(frame: pd.DataFrame, preferred: str, fallback: str) -> int:
    column = preferred if preferred in frame else fallback
    if column not in frame:
        return 0
    return int(frame[column].astype(float).idxmax())


def _first_line_flow_column(config: ModelConfig) -> str:
    return f"line_flow_mw__{config.portfolio.lines[0].id}"


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame:
        raise ACValidationError(f"AC validation requires output column: {column}")


def _value(frame: pd.DataFrame, column: str, period: int) -> float:
    return float(frame[column].iloc[period])
