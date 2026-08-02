from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_runner() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    runner_path = root / "scripts" / "run_verification_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_verification_benchmarks", runner_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_benchmark_output_schema_is_stable(tmp_path: Path) -> None:
    runner = _load_runner()

    payload = runner.run_verification_benchmarks(tmp_path)
    first = payload["benchmarks"][0]

    assert set(payload) == {"schema_version", "tolerance_policy", "benchmarks"}
    assert set(payload["tolerance_policy"]) == {
        "power_abs_tolerance_mw",
        "energy_abs_tolerance_mwh",
        "objective_abs_tolerance_eur",
    }
    assert set(first) == {
        "case_id",
        "periods",
        "continuous_variables",
        "integer_variables",
        "binary_variables",
        "linear_constraints",
        "matrix_nonzeros",
        "variable_counts_by_block",
        "constraint_counts_by_component",
        "build_time_seconds",
        "solve_time_seconds",
        "solver_status",
        "objective_eur",
    }
    assert isinstance(first["variable_counts_by_block"], dict)
    assert isinstance(first["constraint_counts_by_component"], dict)


def test_verification_benchmarks_stay_within_ci_budgets(tmp_path: Path) -> None:
    runner = _load_runner()

    payload = runner.run_verification_benchmarks(tmp_path)

    assert len(payload["benchmarks"]) >= 3
    for row in payload["benchmarks"]:
        assert row["solver_status"] == "optimal"
        assert row["build_time_seconds"] < 5.0
        assert row["solve_time_seconds"] < 20.0
        assert row["linear_constraints"] < 20_000
