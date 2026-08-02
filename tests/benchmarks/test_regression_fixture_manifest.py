from __future__ import annotations

import json
from pathlib import Path


def test_regression_fixture_manifest_is_complete_and_resolvable() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "tests" / "benchmarks" / "regression_fixtures.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    fixture_ids = {fixture["id"] for fixture in manifest["fixtures"]}
    assert {
        "baseline_0_1_0",
        "subhourly_thermal_energy_cost",
        "subhourly_ramp_limit",
        "battery_state_throughput_scaling",
        "terminal_commitment_residual",
        "invalid_transition_ramp",
    } <= fixture_ids
    for fixture in manifest["fixtures"]:
        assert (root / fixture["path"]).exists()
        assert (root / fixture["covered_by"]).exists()
        assert fixture["component"]
        assert fixture["risk"]
