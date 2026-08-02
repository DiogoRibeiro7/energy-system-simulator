from __future__ import annotations

import numpy as np
import pytest

from energy_system_simulator.dispatch.solver import (
    BackendSolverResult,
    absolute_gap,
    available_solver_backends,
    get_solver_backend,
    interpret_solver_result,
    objective_bound_with_constant,
    relative_gap,
    solver_capability_matrix,
)


def _backend_result(
    *,
    status_code: int | None,
    status_name: str,
    message: str = "",
    has_incumbent: bool = True,
    objective_value: float | None = 10.0,
    objective_bound: float | None = 9.0,
    relative_gap_value: float | None = 0.1,
) -> BackendSolverResult:
    return BackendSolverResult(
        status_code=status_code,
        status_name=status_name,
        message=message,
        solution=np.array([1.0]) if has_incumbent else None,
        objective_value=objective_value,
        objective_bound=objective_bound,
        relative_gap=relative_gap_value,
        node_count=3,
    )


@pytest.mark.parametrize(
    ("backend", "allow_non_optimal", "expected_status", "expected_accepted"),
    [
        (_backend_result(status_code=0, status_name="optimal"), False, "optimal", True),
        (
            _backend_result(status_code=1, status_name="limit_reached"),
            False,
            "feasible_limit",
            False,
        ),
        (
            _backend_result(status_code=1, status_name="limit_reached"),
            True,
            "feasible_limit",
            True,
        ),
        (
            _backend_result(status_code=2, status_name="infeasible", has_incumbent=False),
            True,
            "infeasible",
            False,
        ),
        (
            _backend_result(status_code=3, status_name="unbounded", has_incumbent=False),
            True,
            "unbounded",
            False,
        ),
        (
            _backend_result(
                status_code=2,
                status_name="infeasible_or_unbounded",
                has_incumbent=False,
            ),
            True,
            "infeasible_or_unbounded",
            False,
        ),
        (
            _backend_result(status_code=4, status_name="solver_error", has_incumbent=False),
            True,
            "solver_error",
            False,
        ),
        (
            _backend_result(
                status_code=4,
                status_name="solver_error",
                message="Solve interrupted by callback",
                has_incumbent=False,
            ),
            True,
            "interrupted",
            False,
        ),
        (
            _backend_result(status_code=1, status_name="limit_reached", has_incumbent=False),
            True,
            "no_incumbent",
            False,
        ),
    ],
)
def test_solver_result_interpretation_covers_domain_statuses(
    backend: BackendSolverResult,
    allow_non_optimal: bool,
    expected_status: str,
    expected_accepted: bool,
) -> None:
    interpretation = interpret_solver_result(
        backend,
        allow_non_optimal_solution=allow_non_optimal,
    )

    assert interpretation.status == expected_status
    assert interpretation.accepted is expected_accepted
    assert interpretation.backend_status == backend.status_name
    assert interpretation.message == backend.message


def test_limit_reached_without_incumbent_is_rejected_even_when_allowed() -> None:
    interpretation = interpret_solver_result(
        _backend_result(status_code=1, status_name="limit_reached", has_incumbent=False),
        allow_non_optimal_solution=True,
    )

    assert interpretation.status == "no_incumbent"
    assert interpretation.accepted is False
    assert interpretation.solution is None


def test_objective_bound_is_reported_only_when_finite() -> None:
    assert objective_bound_with_constant(9.0, 2.0) == pytest.approx(11.0)
    assert objective_bound_with_constant(None, 2.0) is None
    assert objective_bound_with_constant(float("inf"), 2.0) is None


@pytest.mark.parametrize(
    ("primal", "bound", "expected_absolute", "expected_relative"),
    [
        (100.0, 90.0, 10.0, 0.1),
        (-100.0, -110.0, 10.0, 0.1),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, None),
        (10.0, None, None, None),
    ],
)
def test_objective_gaps_handle_zero_negative_and_missing_bounds(
    primal: float,
    bound: float | None,
    expected_absolute: float | None,
    expected_relative: float | None,
) -> None:
    assert absolute_gap(primal, bound) == (
        None if expected_absolute is None else pytest.approx(expected_absolute)
    )
    assert relative_gap(primal, bound) == (
        None if expected_relative is None else pytest.approx(expected_relative)
    )


def test_scipy_backend_capabilities_are_explicit() -> None:
    capabilities = solver_capability_matrix()["scipy"]

    assert available_solver_backends() == ("scipy",)
    assert capabilities.milp is True
    assert capabilities.lp_duals is True
    assert capabilities.time_limits is True
    assert capabilities.mip_gaps is True
    assert capabilities.node_counts is True
    assert capabilities.warm_starts is False
    assert capabilities.solution_pools is False


def test_unknown_backend_fails_with_installation_message() -> None:
    with pytest.raises(ValueError, match="not installed"):
        get_solver_backend("highs-direct")
