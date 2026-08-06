from __future__ import annotations

import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RELEASES_PATH = ROOT / "licensing" / "releases.json"
METADATA_PATH = ROOT / "licensing" / "metadata.json"
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
RELEASE_METADATA_FILES = (
    "LICENSE",
    "COMMERCIAL-LICENSE.md",
    "LICENSING.md",
    "NOTICE",
    "README.md",
    "RELEASES.md",
    "CITATION.cff",
    ".zenodo.json",
    "pyproject.toml",
    "licensing/releases.json",
    "licensing/metadata.json",
)
PROHIBITED_PLACEHOLDER_PATTERNS = (
    re.compile(r"\[(FULL LEGAL NAME|CONTACT EMAIL|PROJECT NAME|START YEAR|CURRENT YEAR)\]"),
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\bOWNER\b"),
    re.compile(r"example\.(com|org)", re.IGNORECASE),
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


def load_release_metadata() -> dict[str, str]:
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    required = {
        "project_name",
        "package_name",
        "copyright_holder",
        "commercial_contact_email",
        "version",
        "license",
        "release_date",
    }
    missing = required - payload.keys()
    if missing:
        raise AssertionError(f"licensing/metadata.json missing keys: {sorted(missing)}")
    metadata: dict[str, str] = {}
    for key in required:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise AssertionError(f"licensing/metadata.json {key} must be a non-empty string")
        metadata[key] = value
    return metadata


def validate_release_manifest() -> None:
    metadata = load_release_metadata()
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

        if version == metadata["version"] and release_date_raw != metadata["release_date"]:
            raise AssertionError("licensing/releases.json release_date does not match metadata")


def validate_package_metadata() -> None:
    metadata = load_release_metadata()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    poetry = pyproject["tool"]["poetry"]
    if poetry.get("license") != PACKAGE_LICENSE:
        raise AssertionError("pyproject.toml must declare BUSL-1.1")
    if poetry.get("name") != metadata["package_name"]:
        raise AssertionError("pyproject.toml package name does not match metadata")
    if poetry.get("version") != metadata["version"]:
        raise AssertionError("pyproject.toml version does not match metadata")
    expected_author = f"{metadata['copyright_holder']} <{metadata['commercial_contact_email']}>"
    if expected_author not in poetry.get("authors", []):
        raise AssertionError("pyproject.toml author does not match release metadata")

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
        "licensing/metadata.json",
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


def validate_no_unresolved_release_placeholders() -> None:
    for relative_path in RELEASE_METADATA_FILES:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        validate_text_has_no_placeholders(text, relative_path)


def validate_text_has_no_placeholders(text: str, source: str) -> None:
    for pattern in PROHIBITED_PLACEHOLDER_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            raise AssertionError(f"Unresolved placeholder token {match.group(0)!r} in {source}")


def validate_owner_contact_metadata() -> None:
    metadata = load_release_metadata()
    holder = metadata["copyright_holder"]
    email = metadata["commercial_contact_email"]
    project_name = metadata["project_name"]
    version = metadata["version"]
    license_id = metadata["license"]

    if license_id != PACKAGE_LICENSE:
        raise AssertionError("licensing/metadata.json license must be BUSL-1.1")
    date.fromisoformat(metadata["release_date"])

    required_text = {
        "LICENSE": (holder, project_name),
        "COMMERCIAL-LICENSE.md": (email, project_name),
        "LICENSING.md": (email, project_name),
        "NOTICE": (holder, email, project_name, PACKAGE_LICENSE),
        "README.md": (project_name, "CITATION.cff", PACKAGE_LICENSE),
    }
    for relative_path, expected_values in required_text.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for value in expected_values:
            if value not in text:
                raise AssertionError(f"{relative_path} missing metadata value {value!r}")

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    if not isinstance(citation, dict):
        raise AssertionError("CITATION.cff must be a mapping")
    if citation.get("title") != project_name:
        raise AssertionError("CITATION.cff title does not match metadata")
    if str(citation.get("version")) != version:
        raise AssertionError("CITATION.cff version does not match metadata")
    if citation.get("license") != PACKAGE_LICENSE:
        raise AssertionError("CITATION.cff license does not match metadata")
    if str(citation.get("date-released")) != metadata["release_date"]:
        raise AssertionError("CITATION.cff date-released does not match metadata")
    authors = citation.get("authors")
    if not isinstance(authors, list) or not authors:
        raise AssertionError("CITATION.cff must include at least one author")
    first_author = authors[0]
    if not isinstance(first_author, dict):
        raise AssertionError("CITATION.cff author entry must be a mapping")
    if first_author.get("email") != email:
        raise AssertionError("CITATION.cff author email does not match metadata")
    given = first_author.get("given-names")
    family = first_author.get("family-names")
    if f"{given} {family}".strip() != holder:
        raise AssertionError("CITATION.cff author name does not match metadata")


def main() -> int:
    checks = (
        validate_required_files,
        validate_no_unresolved_release_placeholders,
        validate_package_metadata,
        validate_owner_contact_metadata,
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
