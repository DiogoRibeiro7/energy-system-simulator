from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
import venv
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    """Install the built wheel in a fresh environment and run a reduced simulation."""
    root = Path(__file__).resolve().parents[1]
    expected_version = _authoritative_version(root)
    wheel = _latest_wheel(root / "dist")
    with tempfile.TemporaryDirectory(prefix="energy-sim-wheel-") as temp_raw:
        temp = Path(temp_raw)
        venv_dir = temp / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _venv_python(venv_dir)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--retries",
                "0",
                "--timeout",
                "30",
                "--only-binary",
                ":all:",
                str(wheel),
            ],
            check=True,
            timeout=180,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import energy_system_simulator as e; "
                    f"assert e.__version__ == {expected_version!r}"
                ),
            ],
            check=True,
            timeout=30,
        )
        subprocess.run(
            [str(python), "-m", "energy_system_simulator", "--version"],
            check=True,
            timeout=30,
        )
        reduced_config = _write_reduced_case(root, temp)
        subprocess.run(
            [
                str(python),
                "-m",
                "energy_system_simulator",
                "validate",
                "--config",
                str(reduced_config),
            ],
            check=True,
            timeout=60,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "energy_system_simulator",
                "simulate",
                "--config",
                str(reduced_config),
                "--no-plots",
            ],
            check=True,
            timeout=180,
        )
    print("wheel install smoke ok")


def _authoritative_version(root: Path) -> str:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["tool"]["poetry"]["version"]
    if not isinstance(version, str) or not version:
        raise SystemExit("pyproject.toml does not declare a valid version")
    return version


def _latest_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("energy_system_simulator-*.whl"))
    if not wheels:
        raise SystemExit("No built wheel found. Run `poetry build` first.")
    return wheels[-1]


def _venv_python(venv_dir: Path) -> Path:
    candidate = (
        venv_dir
        / ("Scripts" if sys.platform == "win32" else "bin")
        / ("python.exe" if sys.platform == "win32" else "python")
    )
    if not candidate.is_file():
        raise SystemExit(f"Virtual environment Python not found: {candidate}")
    return candidate


def _write_reduced_case(root: Path, temp: Path) -> Path:
    data = pd.read_csv(root / "data" / "example_hourly.csv").head(24)
    data_path = temp / "example_24h.csv"
    data.to_csv(data_path, index=False, lineterminator="\n")

    raw_config = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw_config["paths"]["input_csv"] = str(data_path)
    raw_config["paths"]["output_directory"] = str(temp / "outputs")
    config_path = temp / "example.yaml"
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=True), encoding="utf-8")
    return config_path


if __name__ == "__main__":
    main()
