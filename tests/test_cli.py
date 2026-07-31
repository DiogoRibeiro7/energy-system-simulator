from __future__ import annotations

from pathlib import Path

import yaml

from energy_system_simulator.cli import main


def _write_config_with_output(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["paths"]["input_csv"] = str(root / "data" / "example_hourly.csv")
    raw["paths"]["output_directory"] = str(tmp_path / "outputs")
    config_path = tmp_path / "example.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def test_cli_validate_reports_period_count(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_output(tmp_path)
    main(["validate", "--config", str(config_path)])
    captured = capsys.readouterr()
    assert "Input contains 336 periods" in captured.out


def test_cli_simulate_writes_outputs_without_plots(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_output(tmp_path)
    main(["simulate", "--config", str(config_path), "--no-plots"])
    captured = capsys.readouterr()
    output = tmp_path / "outputs"
    assert "Simulation complete" in captured.out
    assert (output / "timeseries.csv").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "manifest.json").is_file()
    assert not (output / "dispatch.png").exists()
