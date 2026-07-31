from __future__ import annotations

import json
from math import isclose
from pathlib import Path

from energy_system_simulator.config import load_config
from energy_system_simulator.simulation import SimulationEngine

EXCLUDED_MANIFEST_FIELDS = {
    "generated_at_utc",
    "git_commit",
    "configuration_file",
    "input_file",
}


def current_baseline_metrics() -> dict[str, float | int]:
    """Run the committed example and return deterministic baseline metrics."""
    config = load_config(Path("configs") / "example.yaml")
    result = SimulationEngine(config).run()
    reconciliation = result.summary["energy_reconciliation"]
    return {
        "periods": int(result.summary["periods"]),
        "objective_eur": float(result.objective_eur),
        "renewable_share_of_primary_generation": float(
            result.summary["renewable_share_of_primary_generation"]
        ),
        "total_emissions_tonnes": float(result.summary["total_emissions_tonnes"]),
        "imports_mwh": float(result.summary["imports_mwh"]),
        "unserved_energy_mwh": float(result.summary["unserved_energy_mwh"]),
        "continuous_variables": result.formulation_statistics.continuous_variables,
        "integer_variables": result.formulation_statistics.integer_variables,
        "binary_variables": result.formulation_statistics.binary_variables,
        "linear_constraints": result.formulation_statistics.linear_constraints,
        "matrix_nonzeros": result.formulation_statistics.matrix_nonzeros,
        "max_abs_source_balance_residual_mw": float(
            reconciliation["max_abs_source_balance_residual_mw"]
        ),
        "max_abs_delivered_demand_residual_mw": float(
            reconciliation["max_abs_delivered_demand_residual_mw"]
        ),
        "max_abs_battery_energy_residual_mwh": float(
            reconciliation["max_abs_battery_energy_residual_mwh"]
        ),
        "objective_reconciliation_error_eur": float(
            result.summary["objective_reconciliation_error_eur"]
        ),
    }


def compare_baseline(fixture_path: Path) -> None:
    """Compare the current deterministic model metrics with a baseline fixture."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = fixture["metrics"]
    actual = current_baseline_metrics()
    absolute_tolerance = float(fixture["absolute_tolerance"])
    relative_tolerance = float(fixture["relative_tolerance"])

    for key, expected_value in expected.items():
        if key not in actual:
            raise AssertionError(f"Missing current baseline metric: {key}")
        actual_value = actual[key]
        if isinstance(expected_value, int):
            if actual_value != expected_value:
                raise AssertionError(
                    f"{key} mismatch: expected {expected_value}, got {actual_value}"
                )
            continue
        if not _close(
            float(actual_value), float(expected_value), absolute_tolerance, relative_tolerance
        ):
            raise AssertionError(f"{key} mismatch: expected {expected_value}, got {actual_value}")


def _close(
    actual: float, expected: float, absolute_tolerance: float, relative_tolerance: float
) -> bool:
    return isclose(
        actual,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def main() -> None:
    """Run the baseline comparison using the committed fixture."""
    fixture_path = Path("tests") / "fixtures" / "baseline_0_1_0.json"
    compare_baseline(fixture_path)
    print("baseline comparison ok")


if __name__ == "__main__":
    main()
