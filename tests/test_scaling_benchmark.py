from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_scaling() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "benchmark_scaling.py"
    spec = importlib.util.spec_from_file_location("benchmark_scaling", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaling_benchmark_writes_requested_axes(tmp_path: Path) -> None:
    scaling = _load_scaling()

    rows = scaling.run_scaling_benchmarks(tmp_path)

    assert {"periods", "thermal_units", "storage_assets", "buses", "scenarios"} <= {
        row["axis"] for row in rows
    }
    assert (tmp_path / "scaling.csv").is_file()
    assert (tmp_path / "scaling.md").is_file()
    for row in rows:
        assert row["solver_status"] == "optimal"
        assert row["build_time_seconds"] >= 0.0
        assert row["solve_time_seconds"] >= 0.0
