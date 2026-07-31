from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path
from types import ModuleType


def _load_generator() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    generator_path = root / "scripts" / "generate_example_data.py"
    spec = importlib.util.spec_from_file_location("generate_example_data", generator_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_data_is_reproducible() -> None:
    root = Path(__file__).resolve().parents[1]
    buffer = StringIO()
    _load_generator().generate_example_data().to_csv(
        buffer,
        index=False,
        lineterminator="\n",
    )
    assert buffer.getvalue() == (root / "data" / "example_hourly.csv").read_text(encoding="utf-8")
