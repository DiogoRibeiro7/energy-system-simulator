from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

from energy_system_simulator.metadata import get_package_version

ROOT = Path(__file__).resolve().parents[1]


def authoritative_version(root: Path = ROOT) -> str:
    """Return the project version declared in pyproject.toml."""
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    value = pyproject.get("tool", {}).get("poetry", {}).get("version")
    if not isinstance(value, str) or not value:
        raise AssertionError("pyproject.toml must declare tool.poetry.version")
    return value


def validate_version_consistency(root: Path = ROOT) -> None:
    """Validate all repository metadata uses the pyproject version."""
    expected = authoritative_version(root)
    checks: dict[str, Any] = {
        "source_tree": get_package_version(
            distribution_name="energy-system-simulator-not-installed",
            project_root=root,
        ),
        "licensing/metadata.json": json.loads(
            (root / "licensing" / "metadata.json").read_text(encoding="utf-8")
        ).get("version"),
        "CITATION.cff": str(
            yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8")).get("version")
        ),
    }
    for source, actual in checks.items():
        if actual != expected:
            raise AssertionError(f"{source} version {actual!r} does not match {expected!r}")


def main() -> int:
    validate_version_consistency()
    print("version metadata ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
