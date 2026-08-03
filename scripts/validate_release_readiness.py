from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DOCS = (
    "docs/index.md",
    "docs/architecture.md",
    "docs/model.md",
    "docs/configuration.md",
    "docs/data-contract.md",
    "docs/data-provenance-inventory.md",
    "docs/reporting.md",
    "docs/verification.md",
    "docs/market-model.md",
    "docs/reliability.md",
    "docs/research-experiments.md",
    "docs/compatibility.md",
    "docs/release-checklist.md",
    "docs/release-validation-1.0.md",
    "case_studies/iberia/README.md",
)
REQUIRED_GITHUB_FILES = (
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
)


def project_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject.get("tool", {}).get("poetry", {}).get("version")
    if not isinstance(version, str):
        raise AssertionError("pyproject.toml is missing tool.poetry.version")
    return version


def validate_required_files() -> None:
    missing = [
        relative_path
        for relative_path in (*REQUIRED_DOCS, *REQUIRED_GITHUB_FILES)
        if not (ROOT / relative_path).is_file()
    ]
    if missing:
        raise AssertionError(f"Missing release-readiness files: {missing}")


def validate_release_docs() -> None:
    version = project_version()
    expected = {
        "README.md": ("Quick", "limitations", "CITATION.cff", "docs/index.md"),
        "CHANGELOG.md": (f"## [{version}]", "Breaking Changes From 0.1.0", "Model Scope"),
        "docs/index.md": ("architecture.md", "model.md", "configuration.md", "case_studies"),
        "docs/data-provenance-inventory.md": (
            "data/example_hourly.csv",
            "experiments/storage_value/data/storage_value_hourly.csv",
            "case_studies/iberia/provenance.md",
        ),
        "docs/compatibility.md": ("3.11", "3.12", "3.13", "scipy.optimize.milp"),
        "docs/release-validation-1.0.md": (
            "Commands Run",
            "validate_examples.py",
            "Known Limitations",
            "Unresolved Risks",
        ),
        "docs/reporting.md": ("schema version 1", "cost_components_v1.csv"),
        "docs/configuration.md": ("schema 1", "schema 2", "unknown"),
    }
    for relative_path, needles in expected.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise AssertionError(f"{relative_path} missing release text {needle!r}")


def main() -> int:
    validate_required_files()
    validate_release_docs()
    print("release readiness ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
