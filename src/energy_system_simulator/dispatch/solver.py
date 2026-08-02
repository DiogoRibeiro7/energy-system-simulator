from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, vstack

from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY

FloatArray = npt.NDArray[np.float64]
IntegerArray = npt.NDArray[np.int_]

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
class VariableBounds:
    """Backend-neutral lower and upper bounds for optimisation variables."""

    lower: FloatArray
    upper: FloatArray

    @property
    def lb(self) -> FloatArray:
        """Compatibility alias for SciPy-style lower bounds."""
        return self.lower

    @property
    def ub(self) -> FloatArray:
        """Compatibility alias for SciPy-style upper bounds."""
        return self.upper


@dataclass(frozen=True)
class LinearConstraintData:
    """Backend-neutral sparse linear constraints."""

    matrix: csr_matrix
    lower: FloatArray
    upper: FloatArray
    names: tuple[str, ...]

    @property
    def A(self) -> csr_matrix:
        """Compatibility alias for SciPy-style constraint matrices."""
        return self.matrix

    @property
    def lb(self) -> FloatArray:
        """Compatibility alias for SciPy-style lower bounds."""
        return self.lower

    @property
    def ub(self) -> FloatArray:
        """Compatibility alias for SciPy-style upper bounds."""
        return self.upper


@dataclass(frozen=True)
class SolverProblem:
    """Complete backend-neutral optimisation problem passed to solver backends."""

    objective: FloatArray
    integrality: IntegerArray
    bounds: VariableBounds
    constraints: LinearConstraintData
    variable_names: tuple[str, ...]


@dataclass(frozen=True)
class SolverCapabilities:
    """Truthful feature flags for an optimisation backend."""

    backend: str
    milp: bool
    lp_duals: bool
    warm_starts: bool
    time_limits: bool
    mip_gaps: bool
    node_counts: bool
    infeasibility_diagnostics: bool
    solution_pools: bool
    lp_export: bool
    mps_export: bool


class SolverBackend(Protocol):
    """Minimal backend interface used by dispatch and pricing."""

    name: str
    capabilities: SolverCapabilities

    def solve_milp(
        self,
        problem: SolverProblem,
        *,
        time_limit_seconds: float,
        mip_relative_gap: float,
    ) -> BackendSolverResult:
        """Solve a mixed-integer linear problem."""

    def solve_linear_program(self, problem: SolverProblem) -> LinearProgramResult:
        """Solve a continuous linear problem and return row marginals when available."""


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
    backend: str = "scipy"


@dataclass(frozen=True)
class LinearProgramResult:
    """LP result with marginal values mapped back to original constraint rows."""

    status_code: int
    status_name: str
    message: str
    solution: FloatArray | None
    objective_value: float | None
    constraint_marginals: FloatArray
    backend: str = "scipy"


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


class ScipyBackend:
    """SciPy/HiGHS backend used as the default open-source solver."""

    name = "scipy"
    capabilities = SolverCapabilities(
        backend=name,
        milp=True,
        lp_duals=True,
        warm_starts=False,
        time_limits=True,
        mip_gaps=True,
        node_counts=True,
        infeasibility_diagnostics=False,
        solution_pools=False,
        lp_export=True,
        mps_export=False,
    )

    def solve_milp(
        self,
        problem: SolverProblem,
        *,
        time_limit_seconds: float,
        mip_relative_gap: float,
    ) -> BackendSolverResult:
        """Solve a SciPy MILP and return a normalized backend result."""
        result = milp(
            c=problem.objective,
            integrality=problem.integrality,
            bounds=_scipy_bounds(problem.bounds),
            constraints=_scipy_constraints(problem.constraints),
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
            backend=self.name,
        )

    def solve_linear_program(self, problem: SolverProblem) -> LinearProgramResult:
        """Solve a continuous LP and return row marginals using SciPy/HiGHS conventions."""
        converted = _constraint_to_linprog(problem.constraints)
        result = linprog(
            c=problem.objective,
            A_ub=converted["A_ub"],
            b_ub=converted["b_ub"],
            A_eq=converted["A_eq"],
            b_eq=converted["b_eq"],
            bounds=list(zip(problem.bounds.lower, problem.bounds.upper, strict=True)),
            method="highs",
        )
        status_code = int(result.status)
        row_marginals = np.full(problem.constraints.lower.shape, np.nan, dtype=np.float64)
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
            backend=self.name,
        )


def available_solver_backends() -> tuple[str, ...]:
    """Return installed solver backend names."""
    return ("scipy",)


def solver_capability_matrix() -> dict[str, SolverCapabilities]:
    """Return backend capabilities without implying unsupported features."""
    backend = ScipyBackend()
    return {backend.name: backend.capabilities}


def get_solver_backend(name: str = "scipy") -> SolverBackend:
    """Resolve a backend by name with a clear optional-backend error."""
    if name == "scipy":
        return ScipyBackend()
    available = ", ".join(available_solver_backends())
    raise ValueError(
        f"Solver backend {name!r} is not installed. Available backend: {available}. "
        "Optional backends must be installed through a supported Poetry extra before use."
    )


def solve_milp(
    problem: SolverProblem,
    *,
    time_limit_seconds: float,
    mip_relative_gap: float,
    backend: str = "scipy",
) -> BackendSolverResult:
    """Solve a MILP through the selected backend."""
    return get_solver_backend(backend).solve_milp(
        problem,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
    )


def solve_linear_program(
    problem: SolverProblem,
    *,
    backend: str = "scipy",
) -> LinearProgramResult:
    """Solve a continuous LP through the selected backend."""
    return get_solver_backend(backend).solve_linear_program(problem)


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


def export_problem_lp(problem: SolverProblem, path: Path) -> None:
    """Write a deterministic LP-format debug export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["\\ Model exported by energy-system-simulator", "Minimize"]
    lines.extend(_lp_expression_lines(" obj", problem.objective, problem.variable_names))
    lines.append("Subject To")
    matrix = problem.constraints.matrix.tocsr()
    for row_index, name in enumerate(problem.constraints.names):
        row = matrix.getrow(row_index)
        lower = problem.constraints.lower[row_index]
        upper = problem.constraints.upper[row_index]
        if np.isfinite(lower) and np.isfinite(upper) and np.isclose(lower, upper, atol=0, rtol=0):
            lines.extend(_lp_row_lines(_lp_name(name), row, problem.variable_names, "=", lower))
        else:
            if np.isfinite(upper):
                lines.extend(
                    _lp_row_lines(f"{_lp_name(name)}_ub", row, problem.variable_names, "<=", upper)
                )
            if np.isfinite(lower):
                lines.extend(
                    _lp_row_lines(f"{_lp_name(name)}_lb", row, problem.variable_names, ">=", lower)
                )
    lines.append("Bounds")
    for index, name in enumerate(problem.variable_names):
        lower = problem.bounds.lower[index]
        upper = problem.bounds.upper[index]
        lp_name = _lp_name(name)
        if np.isneginf(lower) and np.isposinf(upper):
            lines.append(f" {lp_name} free")
        elif np.isneginf(lower):
            lines.append(f" {lp_name} <= {_lp_number(upper)}")
        elif np.isposinf(upper):
            lines.append(f" {_lp_number(lower)} <= {lp_name}")
        else:
            lines.append(f" {_lp_number(lower)} <= {lp_name} <= {_lp_number(upper)}")
    binary_names = [
        _lp_name(problem.variable_names[index])
        for index, value in enumerate(problem.integrality)
        if value != 0
    ]
    if binary_names:
        lines.append("Binary")
        lines.extend(f" {name}" for name in binary_names)
    lines.append("End")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _scipy_bounds(bounds: VariableBounds) -> Bounds:
    return Bounds(bounds.lower, bounds.upper)


def _scipy_constraints(constraints: LinearConstraintData) -> LinearConstraint:
    return LinearConstraint(constraints.matrix, constraints.lower, constraints.upper)


def _finite_optional(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _integer_optional(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _constraint_to_linprog(constraints: LinearConstraintData) -> dict[str, Any]:
    matrix = constraints.matrix.tocsr()
    lower = np.asarray(constraints.lower, dtype=np.float64)
    upper = np.asarray(constraints.upper, dtype=np.float64)
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


def _lp_expression_lines(
    prefix: str,
    coefficients: FloatArray,
    names: tuple[str, ...],
) -> list[str]:
    terms = [
        f"{_lp_signed_number(float(value))} {_lp_name(names[index])}"
        for index, value in enumerate(coefficients)
        if value != 0.0
    ]
    if not terms:
        terms = ["+ 0"]
    return _wrap_lp_terms(prefix + ":", terms)


def _lp_row_lines(
    name: str,
    row: csr_matrix,
    variable_names: tuple[str, ...],
    sense: str,
    rhs: float,
) -> list[str]:
    terms = [
        f"{_lp_signed_number(float(value))} {_lp_name(variable_names[index])}"
        for index, value in zip(row.indices, row.data, strict=True)
        if value != 0.0
    ]
    if not terms:
        terms = ["+ 0"]
    wrapped = _wrap_lp_terms(f" {name}:", terms)
    wrapped[-1] = f"{wrapped[-1]} {sense} {_lp_number(rhs)}"
    return wrapped


def _wrap_lp_terms(prefix: str, terms: list[str], *, width: int = 98) -> list[str]:
    lines: list[str] = []
    current = prefix
    for term in terms:
        candidate = f"{current} {term}"
        if len(candidate) > width and current != prefix:
            lines.append(current)
            current = f"  {term}"
        else:
            current = candidate
    lines.append(current)
    return lines


def _lp_name(name: str) -> str:
    return "".join(
        character if character.isalnum() or character == "_" else "_" for character in name
    )


def _lp_number(value: float) -> str:
    return f"{float(value):.12g}"


def _lp_signed_number(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {_lp_number(abs(value))}"
