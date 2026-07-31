from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> None:
    """Smoke-test the editable Poetry environment and CLI."""
    root = Path(__file__).resolve().parents[1]
    expected_version = _authoritative_version(root)
    commands = (
        [
            sys.executable,
            "-c",
            (f"import energy_system_simulator as e; assert e.__version__ == {expected_version!r}"),
        ],
        [sys.executable, "-m", "energy_system_simulator", "--version"],
        [
            sys.executable,
            "-m",
            "energy_system_simulator",
            "validate",
            "--config",
            str(root / "configs" / "example.yaml"),
        ],
    )
    for command in commands:
        subprocess.run(command, check=True, cwd=root)
    print("editable install smoke ok")


def _authoritative_version(root: Path) -> str:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["tool"]["poetry"]["version"]
    if not isinstance(version, str) or not version:
        raise SystemExit("pyproject.toml does not declare a valid version")
    return version


if __name__ == "__main__":
    main()
