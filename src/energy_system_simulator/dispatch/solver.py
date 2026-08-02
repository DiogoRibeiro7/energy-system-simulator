from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, vstack

from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY

FloatArray = npt.NDArray[np.float64]

SolverStatus = Literal[
    "optimal",
    "feasible_limit",
    "infeasible",
    "unbounded",
    "infeasible_or_unbounded",
    "solver_error",
    "interrupted",
    "no_incumbent",
]

SCIPY_STATUS_NAMES: Final[dict[int, str]] = {
    0: "optimal",
    1: "limit_reached",
    2: "infeasible",
    3: "unbounded",
    4: "solver_error",
}


@dataclass(frozen=True)
class BackendSolverResult:
    """Raw solver result normalized at the optimization-library boundary."""

    status_code: int | None
    status_name: str
    message: str
    solution: FloatArray | None
    objective_value: float | None
    objective_bound: float | None
    relative_gap: float | None
    node_count: int | None


@dataclass(frozen=True)
class LinearProgramResult:
    """LP result with marginal values mapped back to original constraint rows."""

    status_code: int
    status_name: str
    message: str
    solution: FloatArray | None
    objective_value: float | None
    constraint_marginals: FloatArray


@dataclass(frozen=True)
class SolverInterpretation:
    """Domain interpretation of a raw solver result."""

    status: SolverStatus
    accepted: bool
    status_code: int | None
    backend_status: str
    message: str
    solution: FloatArray | None
    objective_value: float | None
    objective_bound: float | None
    backend_relative_gap: float | None
    node_count: int | None


def solve_milp(
    *,
    objective: FloatArray,
    integrality: npt.NDArray[np.int_],
    bounds: Bounds,
    constraints: LinearConstraint,
    time_limit_seconds: float,
    mip_relative_gap: float,
) -> BackendSolverResult:
    """Solve a SciPy MILP and return a normalized backend result."""
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": time_limit_seconds,
            "mip_rel_gap": mip_relative_gap,
            "presolve": True,
        },
    )
    status_code = int(result.status)
    solution = None if result.x is None else np.asarray(result.x, dtype=np.float64)
    return BackendSolverResult(
        status_code=status_code,
        status_name=SCIPY_STATUS_NAMES.get(status_code, "unknown"),
        message=str(result.message),
        solution=solution,
        objective_value=_finite_optional(result.fun),
        objective_bound=_finite_optional(getattr(result, "mip_dual_bound", None)),
        relative_gap=_finite_optional(getattr(result, "mip_gap", None)),
        node_count=_integer_optional(getattr(result, "mip_node_count", None)),
    )


def solve_linear_program(
    *,
    objective: FloatArray,
    bounds: Bounds,
    constraints: LinearConstraint,
) -> LinearProgramResult:
    """Solve a continuous LP and return row marginals using SciPy/HiGHS conventions."""
    converted = _constraint_to_linprog(constraints)
    result = linprog(
        c=objective,
        A_ub=converted["A_ub"],
        b_ub=converted["b_ub"],
        A_eq=converted["A_eq"],
        b_eq=converted["b_eq"],
        bounds=list(zip(bounds.lb, bounds.ub, strict=True)),
        method="highs",
    )
    status_code = int(result.status)
    row_marginals = np.full(constraints.lb.shape, np.nan, dtype=np.float64)
    if result.success:
        equality_rows = converted["equality_rows"]
        equality_marginals = np.asarray(result.eqlin.marginals, dtype=np.float64)
        row_marginals[equality_rows] = equality_marginals
    solution = None if result.x is None else np.asarray(result.x, dtype=np.float64)
    return LinearProgramResult(
        status_code=status_code,
        status_name=SCIPY_STATUS_NAMES.get(status_code, "unknown"),
        message=str(result.message),
        solution=solution,
        objective_value=_finite_optional(result.fun),
        constraint_marginals=row_marginals,
    )


def interpret_solver_result(
    result: BackendSolverResult,
    *,
    allow_non_optimal_solution: bool,
) -> SolverInterpretation:
    """Map a backend result into simulator solver semantics."""
    status = _domain_status(result)
    accepted = status == "optimal" or (status == "feasible_limit" and allow_non_optimal_solution)
    return SolverInterpretation(
        status=status,
        accepted=accepted,
        status_code=result.status_code,
        backend_status=result.status_name,
        message=result.message,
        solution=result.solution,
        objective_value=result.objective_value,
        objective_bound=result.objective_bound,
        backend_relative_gap=result.relative_gap,
        node_count=result.node_count,
    )


def objective_bound_with_constant(
    objective_bound: float | None,
    constant_cost: float,
) -> float | None:
    """Add objective constants to a finite bound when the backend reports one."""
    if objective_bound is None:
        return None
    value = objective_bound + constant_cost
    return value if isfinite(value) else None


def absolute_gap(primal_objective: float, objective_bound: float | None) -> float | None:
    """Return an absolute optimality gap when a finite bound is available."""
    if objective_bound is None:
        return None
    return abs(primal_objective - objective_bound)


def relative_gap(primal_objective: float, objective_bound: float | None) -> float | None:
    """Return a relative gap when the denominator is mathematically meaningful."""
    gap = absolute_gap(primal_objective, objective_bound)
    if gap is None:
        return None
    if gap <= DEFAULT_NUMERICAL_POLICY.objective_reconciliation_eur:
        return 0.0
    denominator = abs(primal_objective)
    if denominator <= DEFAULT_NUMERICAL_POLICY.objective_reconciliation_eur:
        return None
    return gap / denominator


def _domain_status(result: BackendSolverResult) -> SolverStatus:
    status_name = result.status_name.lower()
    message = result.message.lower()
    has_incumbent = result.solution is not None
    if result.status_code == 0:
        return "optimal" if has_incumbent else "no_incumbent"
    if result.status_code == 1:
        return "feasible_limit" if has_incumbent else "no_incumbent"
    if result.status_code == 2:
        return (
            "infeasible_or_unbounded" if status_name == "infeasible_or_unbounded" else "infeasible"
        )
    if result.status_code == 3:
        return "unbounded"
    if "interrupt" in status_name or "interrupt" in message:
        return "interrupted"
    return "solver_error"


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _integer_optional(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _constraint_to_linprog(constraints: LinearConstraint) -> dict[str, Any]:
    matrix = constraints.A.tocsr()
    lower = np.asarray(constraints.lb, dtype=np.float64)
    upper = np.asarray(constraints.ub, dtype=np.float64)
    equality_rows: list[int] = []
    equality_rhs: list[float] = []
    upper_rows: list[csr_matrix] = []
    upper_rhs: list[float] = []

    for row in range(matrix.shape[0]):
        lb = lower[row]
        ub = upper[row]
        row_matrix = matrix.getrow(row)
        has_lower = np.isfinite(lb)
        has_upper = np.isfinite(ub)
        if has_lower and has_upper and np.isclose(lb, ub, atol=0.0, rtol=0.0):
            equality_rows.append(row)
            equality_rhs.append(float(lb))
            continue
        if has_upper:
            upper_rows.append(row_matrix)
            upper_rhs.append(float(ub))
        if has_lower:
            upper_rows.append(-row_matrix)
            upper_rhs.append(float(-lb))

    return {
        "A_eq": vstack([matrix.getrow(row) for row in equality_rows]).tocsr()
        if equality_rows
        else None,
        "b_eq": np.asarray(equality_rhs, dtype=np.float64) if equality_rows else None,
        "A_ub": vstack(upper_rows).tocsr() if upper_rows else None,
        "b_ub": np.asarray(upper_rhs, dtype=np.float64) if upper_rows else None,
        "equality_rows": np.asarray(equality_rows, dtype=np.int64),
    }
