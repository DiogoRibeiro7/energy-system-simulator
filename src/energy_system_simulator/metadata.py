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
    """Return the source-tree version or installed package metadata fallback."""
    if project_root is None:
        pyproject_path = _find_pyproject(Path(__file__).resolve())
    else:
        pyproject_path = _pyproject_at_root(project_root)
    if pyproject_path is not None:
        return _version_from_pyproject(pyproject_path)

    if prefer_installed:
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            pass
    return UNKNOWN_VERSION


def _pyproject_at_root(root: Path) -> Path | None:
    pyproject_path = root.expanduser().resolve() / "pyproject.toml"
    return pyproject_path if pyproject_path.is_file() else None


def _find_pyproject(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    for candidate_root in (current, *current.parents):
        candidate = candidate_root / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def _version_from_pyproject(path: Path) -> str:
    pyproject = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_version = pyproject.get("tool", {}).get("poetry", {}).get("version")
    return raw_version if isinstance(raw_version, str) and raw_version else UNKNOWN_VERSION
