from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Line:
    """A lossless transmission line for the DC power-flow approximation."""

    from_bus: int
    to_bus: int
    susceptance: float
    capacity_mw: float


@dataclass(frozen=True)
class DCPowerFlowResult:
    """Voltage angles and line flows from a DC power-flow calculation."""

    voltage_angles_rad: FloatArray
    line_flows_mw: FloatArray
    capacity_utilisation: FloatArray


def solve_dc_power_flow(
    injections_mw: npt.ArrayLike,
    lines: Sequence[Line],
    *,
    slack_bus: int = 0,
) -> DCPowerFlowResult:
    """Solve a connected lossless DC power-flow system.

    Positive injections represent generation and negative injections represent load.
    The sum of injections must be zero within numerical tolerance.
    """
    injections = np.asarray(injections_mw, dtype=np.float64)
    if injections.ndim != 1 or injections.size < 2:
        raise ValueError("injections_mw must be a one-dimensional array with at least two buses")
    if np.any(~np.isfinite(injections)):
        raise ValueError("Injections must be finite")
    if not np.isclose(injections.sum(), 0.0, atol=1e-8):
        raise ValueError("Bus injections must sum to zero")
    if not 0 <= slack_bus < injections.size:
        raise ValueError("slack_bus is outside the bus range")
    if not lines:
        raise ValueError("At least one line is required")

    bus_count = injections.size
    matrix = np.zeros((bus_count, bus_count), dtype=np.float64)
    for line in lines:
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
    return DCPowerFlowResult(
        voltage_angles_rad=angles,
        line_flows_mw=flows,
        capacity_utilisation=np.abs(flows) / capacity,
    )
