from __future__ import annotations

import importlib.util
from pathlib import Path


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
    validator.validate_package_metadata()
    validator.validate_readme_links()
    validator.validate_no_superseded_current_license_claims()
    validator.validate_release_manifest()
