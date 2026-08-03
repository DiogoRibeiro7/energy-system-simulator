from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from energy_system_simulator.metadata import UNKNOWN_VERSION, get_package_version


def _load_version_validator():
    root = Path(__file__).resolve().parents[1]
    validator_path = root / "scripts" / "validate_version.py"
    spec = importlib.util.spec_from_file_location("validate_version", validator_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_package_version_matches_authority() -> None:
    validator = _load_version_validator()
    assert get_package_version() == validator.authoritative_version()


def test_source_tree_fallback_matches_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    validator = _load_version_validator()
    assert (
        get_package_version(
            distribution_name="energy-system-simulator-not-installed",
            project_root=root,
        )
        == validator.authoritative_version()
    )


def test_unknown_version_is_returned_without_installed_or_project_metadata(
    tmp_path: Path,
) -> None:
    assert (
        get_package_version(
            distribution_name="energy-system-simulator-not-installed",
            project_root=tmp_path,
            prefer_installed=False,
        )
        == UNKNOWN_VERSION
    )


def test_mismatched_duplicate_version_fails(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    validator = _load_version_validator()
    _copy_version_metadata(root, tmp_path)
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text(encoding="utf-8").replace("version: 1.0.0", "version: 9.9.9"),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match=r"CITATION\.cff version"):
        validator.validate_version_consistency(tmp_path)


def _copy_version_metadata(root: Path, target: Path) -> None:
    for relative_path in ("pyproject.toml", "CITATION.cff"):
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative_path, destination)
    shutil.copytree(root / "licensing", target / "licensing")
