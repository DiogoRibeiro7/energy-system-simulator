from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from energy_system_simulator.config import load_config
from energy_system_simulator.exceptions import ConfigurationError


def test_example_configuration_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    assert config.thermal.maximum_output_mw == 220.0
    assert config.paths.input_csv.name == "example_hourly.csv"


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
