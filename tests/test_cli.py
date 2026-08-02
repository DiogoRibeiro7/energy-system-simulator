from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import energy_system_simulator.cli as cli
from energy_system_simulator.cli import main
from energy_system_simulator.metadata import get_package_version


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


def test_cli_validate_can_report_json_success(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_output(tmp_path)
    main(["validate", "--config", str(config_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload == {"ok": True, "periods": 336}


def test_cli_validate_can_report_json_error(tmp_path: Path, capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["thermal"]["maximum_ouput_mw"] = raw["thermal"].pop("maximum_output_mw")
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        main(["validate", "--config", str(config_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert raised.value.code == 2
    assert payload["ok"] is False
    assert payload["error"]["type"] == "ConfigurationError"
    assert "thermal.maximum_ouput_mw" in payload["error"]["message"]


def test_cli_version_reports_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    captured = capsys.readouterr()
    assert raised.value.code == 0
    assert get_package_version() in captured.out


def test_cli_simulate_writes_outputs_without_plots(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_output(tmp_path)
    main(["simulate", "--config", str(config_path), "--no-plots"])
    captured = capsys.readouterr()
    output = tmp_path / "outputs"
    assert "Simulation complete" in captured.out
    assert (output / "timeseries.csv").is_file()
    assert (output / "asset_timeseries.csv").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "manifest.json").is_file()
    assert not (output / "dispatch.png").exists()


def test_cli_simulate_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    config_path = _write_config_with_output(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        main(["simulate", "--config", str(config_path), "--no-plots"])

    assert raised.value.code == 2


def test_cli_simulate_allows_existing_output_with_overwrite(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_output(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    main(["simulate", "--config", str(config_path), "--no-plots", "--overwrite"])

    assert (output / "summary.json").is_file()
    assert "Simulation complete" in capsys.readouterr().out


def test_cli_dry_run_reports_model_size_without_solving(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config_with_output(tmp_path)

    def fail_solve(*args, **kwargs):
        raise AssertionError("dry-run must not solve")

    monkeypatch.setattr(cli, "solve", fail_solve)

    main(["simulate", "--config", str(config_path), "--dry-run"])

    captured = capsys.readouterr()
    assert "Dry run complete" in captured.out
    assert "Linear constraints" in captured.out
    assert not (tmp_path / "outputs").exists()


def test_cli_validated_overrides_are_recorded_in_manifest(tmp_path: Path) -> None:
    config_path = _write_config_with_output(tmp_path)

    main(
        [
            "simulate",
            "--config",
            str(config_path),
            "--no-plots",
            "--set",
            "penalties.carbon_price_eur_per_tonne=5.0",
        ]
    )

    manifest = json.loads((tmp_path / "outputs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["command_line_overrides"] == {"penalties.carbon_price_eur_per_tonne": 5.0}


def test_cli_rejects_unknown_override_path(tmp_path: Path) -> None:
    config_path = _write_config_with_output(tmp_path)

    with pytest.raises(SystemExit) as raised:
        main(["simulate", "--config", str(config_path), "--set", "thermal.missing=1"])

    assert raised.value.code == 2


def test_cli_export_model_writes_lp_file(tmp_path: Path, capsys) -> None:
    config_path = _write_config_with_output(tmp_path)
    output_path = tmp_path / "debug" / "model.lp"

    main(["export-model", "--config", str(config_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert "Model exported" in captured.out
    exported = output_path.read_text(encoding="utf-8")
    assert "Minimize" in exported
    assert "Binary" in exported


def test_cli_export_model_refuses_existing_file_without_overwrite(tmp_path: Path) -> None:
    config_path = _write_config_with_output(tmp_path)
    output_path = tmp_path / "model.lp"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        main(["export-model", "--config", str(config_path), "--output", str(output_path)])

    assert raised.value.code == 2


def test_cli_compare_outputs_command_writes_report(tmp_path: Path) -> None:
    config_path = _write_config_with_output(tmp_path)
    main(["simulate", "--config", str(config_path), "--no-plots"])
    output = tmp_path / "comparison.md"

    main(
        [
            "compare-outputs",
            str(tmp_path / "outputs"),
            str(tmp_path / "outputs"),
            "--output",
            str(output),
        ]
    )

    assert "Scenario Output Comparison" in output.read_text(encoding="utf-8")


def test_cli_capabilities_lists_public_commands(capsys) -> None:
    main(["capabilities"])

    payload = json.loads(capsys.readouterr().out)
    assert "simulate" in payload["commands"]
    assert "compare-outputs" in payload["commands"]
    assert payload["exit_codes"]["invalid_configuration"] == 2


def test_cli_subprocess_invalid_data_uses_data_exit_code(tmp_path: Path) -> None:
    config_path = _write_config_with_output(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = tmp_path / "bad.csv"
    data_path.write_text("timestamp,demand_mw\nnot-a-date,1\n", encoding="utf-8")
    raw["paths"]["input_csv"] = str(data_path)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "energy_system_simulator",
            "validate-data",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 3
    assert "Data validation failed" in completed.stderr


def test_cli_migrate_config_writes_portfolio_schema(tmp_path: Path, capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "portfolio.yaml"

    main(
        [
            "migrate-config",
            "--config",
            str(root / "configs" / "example.yaml"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert "Migrated configuration written" in captured.out
    assert payload["schema_version"] == 2
    assert payload["thermal_generators"][0]["id"] == "thermal_1"
