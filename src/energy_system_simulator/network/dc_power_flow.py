from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Line:
    """A lossless transmission line for the DC power-flow approximation."""

    from_bus: int
    to_bus: int
    susceptance: float
    capacity_mw: float
    line_id: str | None = None


@dataclass(frozen=True)
class LineFlowDiagnostic:
    """Post-solution line loading diagnostic."""

    line_id: str
    from_bus: int
    to_bus: int
    flow_mw: float
    capacity_mw: float
    utilisation: float
    overload_mw: float
    overloaded: bool


@dataclass(frozen=True)
class DCPowerFlowResult:
    """Voltage angles, line flows, and diagnostics from a DC power-flow calculation."""

    voltage_angles_rad: FloatArray
    line_flows_mw: FloatArray
    capacity_utilisation: FloatArray
    line_diagnostics: tuple[LineFlowDiagnostic, ...]
    has_overloads: bool
    overload_policy: Literal["report", "raise"]


def solve_dc_power_flow(
    injections_mw: npt.ArrayLike,
    lines: Sequence[Line],
    *,
    slack_bus: int = 0,
    overload_policy: Literal["report", "raise"] = "report",
) -> DCPowerFlowResult:
    """Solve a connected lossless DC power-flow system for fixed injections.

    Positive injections represent generation and negative injections represent load.
    The sum of injections must be zero within numerical tolerance. Line ratings are
    checked after solving but are not enforced by the calculation.
    """
    injections = np.asarray(injections_mw, dtype=np.float64)
    if injections.ndim != 1 or injections.size < 2:
        raise ValueError("injections_mw must be a one-dimensional array with at least two buses")
    if np.any(~np.isfinite(injections)):
        raise ValueError("Injections must be finite")
    if not np.isclose(
        injections.sum(),
        0.0,
        atol=DEFAULT_NUMERICAL_POLICY.dc_power_balance_mw,
    ):
        raise ValueError("Bus injections must sum to zero")
    if not 0 <= slack_bus < injections.size:
        raise ValueError("slack_bus is outside the bus range")
    if not lines:
        raise ValueError("At least one line is required")
    if overload_policy not in {"report", "raise"}:
        raise ValueError("overload_policy must be 'report' or 'raise'")

    bus_count = injections.size
    matrix = np.zeros((bus_count, bus_count), dtype=np.float64)
    line_ids: set[str] = set()
    for line in lines:
        line_id = line.line_id
        if line_id is not None:
            if line_id in line_ids:
                raise ValueError(f"Duplicate line_id: {line_id}")
            line_ids.add(line_id)
        if not 0 <= line.from_bus < bus_count or not 0 <= line.to_bus < bus_count:
            raise ValueError("Line references an unknown bus")
        if line.from_bus == line.to_bus:
            raise ValueError("A line cannot connect a bus to itself")
        if line.susceptance <= 0.0 or line.capacity_mw <= 0.0:
            raise ValueError("Line susceptance and capacity must be positive")
        i, j, b = line.from_bus, line.to_bus, line.susceptance
        matrix[i, i] += b
        matrix[j, j] += b
        matrix[i, j] -= b
        matrix[j, i] -= b

    retained = np.array([bus for bus in range(bus_count) if bus != slack_bus])
    reduced = matrix[np.ix_(retained, retained)]
    angles = np.zeros(bus_count, dtype=np.float64)
    try:
        angles[retained] = np.linalg.solve(reduced, injections[retained])
    except np.linalg.LinAlgError as exc:
        raise ValueError("The network is disconnected or singular") from exc

    flows = np.array(
        [line.susceptance * (angles[line.from_bus] - angles[line.to_bus]) for line in lines],
        dtype=np.float64,
    )
    capacity = np.array([line.capacity_mw for line in lines], dtype=np.float64)
    utilisation = np.abs(flows) / capacity
    diagnostics = tuple(
        LineFlowDiagnostic(
            line_id=line.line_id or str(index),
            from_bus=line.from_bus,
            to_bus=line.to_bus,
            flow_mw=float(flows[index]),
            capacity_mw=float(capacity[index]),
            utilisation=float(utilisation[index]),
            overload_mw=max(0.0, float(abs(flows[index]) - capacity[index])),
            overloaded=bool(abs(flows[index]) > capacity[index]),
        )
        for index, line in enumerate(lines)
    )
    has_overloads = any(diagnostic.overloaded for diagnostic in diagnostics)
    if has_overloads and overload_policy == "raise":
        overloaded_ids = ", ".join(
            diagnostic.line_id for diagnostic in diagnostics if diagnostic.overloaded
        )
        raise ValueError(f"Line overload detected: {overloaded_ids}")
    return DCPowerFlowResult(
        voltage_angles_rad=angles,
        line_flows_mw=flows,
        capacity_utilisation=utilisation,
        line_diagnostics=diagnostics,
        has_overloads=has_overloads,
        overload_policy=overload_policy,
    )
