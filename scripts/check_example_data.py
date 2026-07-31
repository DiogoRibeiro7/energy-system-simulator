from __future__ import annotations

import importlib.util
from io import StringIO
from pathlib import Path
from types import ModuleType


def _load_generator() -> ModuleType:
    generator_path = Path(__file__).with_name("generate_example_data.py")
    spec = importlib.util.spec_from_file_location("generate_example_data", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load example data generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Verify that the committed example data matches the deterministic generator."""
    root = Path(__file__).resolve().parents[1]
    expected_path = root / "data" / "example_hourly.csv"
    buffer = StringIO()
    _load_generator().generate_example_data().to_csv(
        buffer,
        index=False,
        lineterminator="\n",
    )
    generated = buffer.getvalue()
    committed = expected_path.read_text(encoding="utf-8")
    if generated != committed:
        raise SystemExit("data/example_hourly.csv does not match scripts/generate_example_data.py")
    print("example data ok")


if __name__ == "__main__":
    main()
