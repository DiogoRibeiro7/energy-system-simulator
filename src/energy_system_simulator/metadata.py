from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGE_NAME = "energy-system-simulator"
UNKNOWN_VERSION = "0.0.0+unknown"


def get_package_version(
    *,
    distribution_name: str = PACKAGE_NAME,
    project_root: Path | None = None,
    prefer_installed: bool = True,
) -> str:
    """Return the installed package version or a deterministic source-tree fallback."""
    if prefer_installed:
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            pass

    pyproject_path = _find_pyproject(project_root or Path(__file__).resolve())
    if pyproject_path is None:
        return UNKNOWN_VERSION
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    raw_version = pyproject.get("tool", {}).get("poetry", {}).get("version")
    return raw_version if isinstance(raw_version, str) and raw_version else UNKNOWN_VERSION


def _find_pyproject(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    for candidate_root in (current, *current.parents):
        candidate = candidate_root / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None
