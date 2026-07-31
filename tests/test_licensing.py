from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml


def _load_validator():
    root = Path(__file__).resolve().parents[1]
    validator_path = root / "scripts" / "validate_licensing.py"
    spec = importlib.util.spec_from_file_location("validate_licensing", validator_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_licensing_metadata_is_consistent() -> None:
    validator = _load_validator()
    validator.validate_required_files()
    validator.validate_no_unresolved_release_placeholders()
    validator.validate_package_metadata()
    validator.validate_owner_contact_metadata()
    validator.validate_readme_links()
    validator.validate_no_superseded_current_license_claims()
    validator.validate_release_manifest()


def test_placeholder_fixture_fails_validation() -> None:
    validator = _load_validator()
    root = Path(__file__).resolve().parents[1]
    text = (root / "tests" / "fixtures" / "release_metadata_with_placeholders.txt").read_text(
        encoding="utf-8"
    )

    with pytest.raises(AssertionError, match="Unresolved placeholder token"):
        validator.validate_text_has_no_placeholders(text, "fixture")


@pytest.mark.parametrize(
    "token",
    [
        "[FULL LEGAL NAME]",
        "[CONTACT EMAIL]",
        "[PROJECT NAME]",
        "[START YEAR]",
        "[CURRENT YEAR]",
        "TODO",
        "TBD",
        "OWNER",
        "example.org",
    ],
)
def test_each_prohibited_placeholder_token_fails(token: str) -> None:
    validator = _load_validator()

    with pytest.raises(AssertionError, match="Unresolved placeholder token"):
        validator.validate_text_has_no_placeholders(f"value: {token}", "fixture")


@pytest.mark.parametrize(
    ("metadata_key", "replacement", "expected_message"),
    [
        ("copyright_holder", "Someone Else", "missing metadata value"),
        ("commercial_contact_email", "other@example.net", "missing metadata value"),
        ("version", "9.9.9", "version"),
        ("license", "MIT", "license must be BUSL-1.1"),
    ],
)
def test_inconsistent_release_metadata_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_key: str,
    replacement: str,
    expected_message: str,
) -> None:
    validator = _load_validator()
    root = Path(__file__).resolve().parents[1]
    _copy_release_metadata_tree(root, tmp_path)

    metadata_path = tmp_path / "licensing" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[metadata_key] = replacement
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(validator, "RELEASES_PATH", tmp_path / "licensing" / "releases.json")

    with pytest.raises(AssertionError, match=expected_message):
        validator.validate_owner_contact_metadata()


def test_citation_structure_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["title"] == "Energy System Simulator"
    assert citation["license"] == "BUSL-1.1"
    assert citation["authors"][0]["given-names"] == "Diogo"
    assert citation["authors"][0]["family-names"] == "Ribeiro"


def _copy_release_metadata_tree(root: Path, target: Path) -> None:
    for relative_path in (
        "LICENSE",
        "COMMERCIAL-LICENSE.md",
        "LICENSING.md",
        "NOTICE",
        "README.md",
        "RELEASES.md",
        "CITATION.cff",
        "pyproject.toml",
    ):
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative_path, destination)
    shutil.copytree(root / "licensing", target / "licensing")
