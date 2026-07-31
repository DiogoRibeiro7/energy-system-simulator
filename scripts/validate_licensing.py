from __future__ import annotations

import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASES_PATH = ROOT / "licensing" / "releases.json"
PACKAGE_LICENSE = "BUSL-1.1"
CHANGE_LICENSE = "Apache-2.0"
REQUIRED_FILES = (
    "LICENSE",
    "COMMERCIAL-LICENSE.md",
    "LICENSING.md",
    "NOTICE",
)
README_LICENSE_LINKS = (
    "LICENSE",
    "LICENSING.md",
    "COMMERCIAL-LICENSE.md",
)


def add_calendar_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def load_releases() -> list[dict[str, Any]]:
    payload = json.loads(RELEASES_PATH.read_text(encoding="utf-8"))
    releases = payload.get("releases")
    if not isinstance(releases, list) or not releases:
        raise AssertionError("licensing/releases.json must contain at least one release")
    return releases


def validate_release_manifest() -> None:
    seen_versions: set[str] = set()
    for entry in load_releases():
        version = entry.get("version")
        release_date_raw = entry.get("release_date")
        change_date_raw = entry.get("change_date")
        change_license = entry.get("change_license")

        if not isinstance(version, str) or not version:
            raise AssertionError("Release entry is missing version")
        if version in seen_versions:
            raise AssertionError(f"Duplicate release entry: {version}")
        seen_versions.add(version)

        if not isinstance(release_date_raw, str) or not isinstance(change_date_raw, str):
            raise AssertionError(f"Release {version} is missing dates")

        release_date = date.fromisoformat(release_date_raw)
        change_date = date.fromisoformat(change_date_raw)
        expected_change_date = add_calendar_years(release_date, 4)

        if change_date != expected_change_date:
            raise AssertionError(
                f"Release {version} change_date must be {expected_change_date.isoformat()}"
            )
        if change_date > expected_change_date:
            raise AssertionError(f"Release {version} change_date is more than four years later")
        if change_license != CHANGE_LICENSE:
            raise AssertionError(f"Release {version} change_license must be {CHANGE_LICENSE}")


def validate_package_metadata() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    poetry = pyproject["tool"]["poetry"]
    if poetry.get("license") != PACKAGE_LICENSE:
        raise AssertionError("pyproject.toml must declare BUSL-1.1")

    include_entries = poetry.get("include", [])
    include_paths = {
        entry if isinstance(entry, str) else entry.get("path") for entry in include_entries
    }
    required_include_paths = {
        "LICENSE",
        "COMMERCIAL-LICENSE.md",
        "LICENSING.md",
        "NOTICE",
        "RELEASES.md",
        "licensing/releases.json",
    }
    missing = required_include_paths - include_paths
    if missing:
        raise AssertionError(f"pyproject.toml missing license include entries: {sorted(missing)}")


def validate_required_files() -> None:
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            raise AssertionError(f"Missing required licensing file: {relative_path}")


def validate_readme_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative_path in README_LICENSE_LINKS:
        if f"]({relative_path})" not in readme:
            raise AssertionError(f"README.md does not link to {relative_path}")


def validate_no_superseded_current_license_claims() -> None:
    checked_files = (
        "README.md",
        "pyproject.toml",
        "CITATION.cff",
        "NOTICE",
        "COMMERCIAL-LICENSE.md",
        "LICENSING.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
    )
    forbidden_patterns = (
        re.compile(r"license\s*[:=]\s*[\"']?MIT[\"']?", re.IGNORECASE),
        re.compile(r"license\s*[:=]\s*[\"']?Apache-?2\.0[\"']?", re.IGNORECASE),
        re.compile(r"OSI Approved", re.IGNORECASE),
    )
    for relative_path in checked_files:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern.search(text):
                raise AssertionError(f"Superseded license claim in {relative_path}")


def main() -> int:
    checks = (
        validate_required_files,
        validate_package_metadata,
        validate_readme_links,
        validate_no_superseded_current_license_claims,
        validate_release_manifest,
    )
    for check in checks:
        check()
    print("licensing validation ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
