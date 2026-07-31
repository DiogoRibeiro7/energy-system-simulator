from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from energy_system_simulator.config import (
    CURRENT_SCHEMA_VERSION,
    load_config,
    migrate_config_file,
    resolved_config_yaml,
)
from energy_system_simulator.exceptions import ConfigurationError


def test_example_configuration_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    assert config.thermal.maximum_output_mw == 220.0
    assert config.paths.input_csv.name == "example_hourly.csv"
    assert config.portfolio.thermal_generators[0].id == "thermal_1"


def test_invalid_network_loss_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text())
    raw["network"]["loss_fraction"] = 1.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_initial_state_time_fields_must_match_commitment_state(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["thermal"]["initial_on"] = False
    raw["thermal"]["initial_output_mw"] = 0.0
    raw["thermal"]["initial_up_time_hours"] = 1.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="initial_up_time_hours"):
        load_config(path)


def test_invalid_terminal_soc_mode_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["battery"]["terminal_soc_mode"] = "eventual"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="terminal_soc_mode"):
        load_config(path)


def test_invalid_terminal_commitment_mode_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["thermal"]["terminal_commitment_mode"] = "eventual"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="terminal_commitment_mode"):
        load_config(path)


def test_fixed_terminal_commitment_requires_terminal_state(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["thermal"]["terminal_commitment_mode"] = "fixed_terminal_commitment"
    raw["thermal"].pop("terminal_on", None)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="terminal_on"):
        load_config(path)


def test_terminal_state_is_only_valid_for_fixed_commitment(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["thermal"]["terminal_commitment_mode"] = "forbid_incomplete_transitions"
    raw["thermal"]["terminal_on"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="terminal_on"):
        load_config(path)


def test_misspelled_nested_field_fails_with_suggestion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["thermal"]["maximum_ouput_mw"] = raw["thermal"].pop("maximum_output_mw")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError) as raised:
        load_config(path)

    message = str(raised.value)
    assert "thermal.maximum_ouput_mw" in message
    assert "maximum_output_mw" in message
    assert "Allowed keys:" in message


def test_unknown_root_field_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["simulaton"] = {}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="simulaton"):
        load_config(path)


def test_unknown_nested_field_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["battery"]["max_soc_mwh"] = raw["battery"]["maximum_soc_mwh"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"battery\.max_soc_mwh"):
        load_config(path)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """
schema_version: 1
simulation:
  time_step_hours: 1.0
simulation:
  time_step_hours: 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Duplicate configuration field: simulation"):
        load_config(path)


def test_boolean_values_do_not_pass_numeric_validation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["simulation"]["time_step_hours"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="time_step_hours"):
        load_config(path)


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["schema_version"] = 99
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unsupported configuration schema_version"):
        load_config(path)


def test_relative_paths_support_windows_and_posix_separators(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    raw["paths"]["input_csv"] = "data/input.csv"
    raw["paths"]["output_directory"] = "outputs\\example"
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.paths.input_csv == (config_path.parent / "data" / "input.csv").resolve()
    assert str(config.paths.output_directory).endswith("outputs\\example") or str(
        config.paths.output_directory
    ).endswith("outputs/example")


def _portfolio_raw() -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs" / "portfolio_two_thermal.yaml").read_text(encoding="utf-8")
    )


def _write_portfolio(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_portfolio_configuration_loads_with_typed_assets() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "portfolio_two_thermal.yaml")

    assert config.schema_version == CURRENT_SCHEMA_VERSION
    assert config.solver.solver_time_limit_seconds == 60.0
    assert config.portfolio.scenario.id == "two-thermal-example"
    assert [generator.id for generator in config.portfolio.thermal_generators] == [
        "north-ccgt",
        "south-peaker",
    ]
    assert [generator.kind for generator in config.portfolio.renewable_generators] == [
        "solar",
        "wind",
    ]
    assert [fuel.id for fuel in config.portfolio.fuels] == ["gas"]
    assert config.portfolio.thermal_generators[0].config.heat_rate_segments[0].id == "efficient"
    assert config.portfolio.thermal_generators[1].config.startup_categories[-1].id == "cold"
    assert config.thermal.name == "North combined-cycle gas plant"


def test_duplicate_portfolio_ids_are_rejected(tmp_path: Path) -> None:
    raw = _portfolio_raw()
    raw["thermal_generators"][1]["id"] = raw["thermal_generators"][0]["id"]

    with pytest.raises(ConfigurationError, match=r"thermal_generators\[1\]\.id"):
        load_config(_write_portfolio(tmp_path, raw))


def test_missing_bus_reference_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "tests" / "fixtures" / "invalid_portfolio_missing_bus.yaml"

    with pytest.raises(ConfigurationError, match=r"renewable_generators\[0\]\.bus_id"):
        load_config(path)


def test_invalid_portfolio_efficiency_reports_path(tmp_path: Path) -> None:
    raw = _portfolio_raw()
    raw["storage_units"][0]["charge_efficiency"] = 1.2

    with pytest.raises(ConfigurationError, match=r"storage_units\[0\]\.charge_efficiency"):
        load_config(_write_portfolio(tmp_path, raw))


def test_invalid_portfolio_initial_state_reports_path(tmp_path: Path) -> None:
    raw = _portfolio_raw()
    raw["thermal_generators"][1]["initial_output_mw"] = 5.0

    with pytest.raises(ConfigurationError, match=r"thermal_generators\[1\]\.initial_output_mw"):
        load_config(_write_portfolio(tmp_path, raw))


def test_unknown_portfolio_field_is_rejected(tmp_path: Path) -> None:
    raw = _portfolio_raw()
    raw["thermal_generators"][1]["max_output"] = raw["thermal_generators"][1]["maximum_output_mw"]

    with pytest.raises(ConfigurationError, match=r"thermal_generators\[1\]\.max_output"):
        load_config(_write_portfolio(tmp_path, raw))


def test_legacy_config_migration_preserves_dispatch_fields(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    migrated = migrate_config_file(root / "configs" / "example.yaml")

    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert migrated["simulation"]["time_step_hours"] == 1.0
    assert migrated["solver"]["mip_relative_gap"] == 0.001
    assert migrated["thermal_generators"][0]["id"] == "thermal_1"
    assert migrated["thermal_generators"][0]["maximum_output_mw"] == 220.0
    assert migrated["storage_units"][0]["charge_efficiency"] == 0.94

    config = load_config(_write_portfolio(tmp_path, migrated))
    assert config.schema_version == CURRENT_SCHEMA_VERSION
    assert config.thermal.maximum_output_mw == 220.0


def test_resolved_config_yaml_serializes_canonical_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "portfolio_two_thermal.yaml")

    text = resolved_config_yaml(config)

    assert "portfolio:" in text
    assert "schema_version: 2" in text
    assert str(config.paths.input_csv) in text
