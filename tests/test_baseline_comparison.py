from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_comparator() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    comparator_path = root / "scripts" / "compare_baseline.py"
    spec = importlib.util.spec_from_file_location("compare_baseline", comparator_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_comparison_passes_twice() -> None:
    comparator = _load_comparator()
    fixture_path = Path("tests") / "fixtures" / "baseline_0_1_0.json"
    comparator.compare_baseline(fixture_path)
    comparator.compare_baseline(fixture_path)


def test_modified_expected_objective_fails_with_metric_name(tmp_path: Path) -> None:
    comparator = _load_comparator()
    fixture_path = Path("tests") / "fixtures" / "baseline_0_1_0.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["metrics"]["objective_eur"] += 1000.0
    modified = tmp_path / "baseline.json"
    modified.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(AssertionError, match="objective_eur mismatch"):
        comparator.compare_baseline(modified)


def test_nondeterministic_manifest_fields_are_excluded_from_baseline() -> None:
    comparator = _load_comparator()
    fixture_path = Path("tests") / "fixtures" / "baseline_0_1_0.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert set(fixture["manifest_fields_excluded"]) == comparator.EXCLUDED_MANIFEST_FIELDS
    assert not comparator.EXCLUDED_MANIFEST_FIELDS.intersection(fixture["metrics"])
