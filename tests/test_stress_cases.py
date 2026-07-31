from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_runner() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "scripts" / "run_stress_cases.py"
    spec = importlib.util.spec_from_file_location("run_stress_cases", runner_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_stress_cases_run_and_write_summary(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "stress-summary.csv"
    rows = runner.run_stress_cases(output)

    assert output.is_file()
    assert len(rows) == len(runner.load_cases())
    assert any(row["case"] == "invalid_transition_ramp" for row in rows)
    assert any(float(row["terminal_residual_up_hours"]) > 0.0 for row in rows)
