from __future__ import annotations

from pathlib import Path


def test_unit_commitment_does_not_import_scipy_optimize_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src" / "energy_system_simulator" / "dispatch" / "unit_commitment.py"
    ).read_text(encoding="utf-8")

    assert "scipy.optimize" not in source
