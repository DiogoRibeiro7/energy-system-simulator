from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from energy_system_simulator.config import ModelConfig, ThermalConfig, load_config
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.exceptions import ConfigurationError


def _example_config() -> ModelConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "example.yaml")


def _thermal(
    *,
    initial_on: bool,
    initial_output_mw: float,
    minimum_output_mw: float = 35.0,
    startup_ramp_mw: float = 35.0,
    shutdown_ramp_mw: float = 35.0,
    ramp_mw_per_hour: float = 1_000.0,
) -> ThermalConfig:
    config = _example_config()
    return replace(
        config.thermal,
        minimum_output_mw=minimum_output_mw,
        maximum_output_mw=100.0,
        ramp_up_mw_per_hour=ramp_mw_per_hour,
        ramp_down_mw_per_hour=ramp_mw_per_hour,
        startup_ramp_mw=startup_ramp_mw,
        shutdown_ramp_mw=shutdown_ramp_mw,
        variable_cost_eur_per_mwh=1.0,
        no_load_cost_eur_per_hour=1.0,
        startup_cost_eur=0.0,
        shutdown_cost_eur=0.0,
        emission_factor_tonnes_per_mwh=0.0,
        minimum_up_hours=1.0,
        minimum_down_hours=1.0,
        initial_on=initial_on,
        initial_output_mw=initial_output_mw,
        initial_up_time_hours=10.0 if initial_on else 0.0,
        initial_down_time_hours=0.0 if initial_on else 10.0,
        terminal_commitment_mode="forbid_incomplete_transitions",
    )


def _dispatch_config(thermal: ThermalConfig) -> ModelConfig:
    config = _example_config()
    battery = replace(
        config.battery,
        energy_capacity_mwh=0.0,
        power_capacity_mw=0.0,
        minimum_soc_mwh=0.0,
        maximum_soc_mwh=0.0,
        initial_soc_mwh=0.0,
        minimum_final_soc_mwh=0.0,
        terminal_soc_mode="free",
    )
    return replace(
        config,
        simulation=replace(config.simulation, time_step_hours=1.0),
        thermal=thermal,
        battery=battery,
        network=replace(config.network, loss_fraction=0.0),
        imports=replace(config.imports, maximum_power_mw=0.0),
        penalties=replace(config.penalties, lost_load_eur_per_mwh=10_000.0),
    )


def test_off_unit_can_start_at_minimum_stable_output() -> None:
    config = _dispatch_config(
        _thermal(
            initial_on=False,
            initial_output_mw=0.0,
            startup_ramp_mw=35.0,
        )
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([35.0]),
    )

    assert result.frame["thermal_startup"].tolist() == [1]
    assert result.frame["thermal_output_mw"].iloc[0] == pytest.approx(35.0)


def test_unit_at_minimum_stable_output_can_shutdown() -> None:
    config = _dispatch_config(
        _thermal(
            initial_on=True,
            initial_output_mw=35.0,
            shutdown_ramp_mw=35.0,
        )
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.zeros(1),
    )

    assert result.frame["thermal_shutdown"].tolist() == [1]
    assert result.frame["thermal_output_mw"].iloc[0] == pytest.approx(0.0)


def test_startup_limit_below_minimum_stable_output_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "example.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["thermal"]["startup_ramp_mw"] = raw["thermal"]["minimum_output_mw"] - 1.0
    path = tmp_path / "invalid-startup.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="startup_ramp_mw"):
        load_config(path)


def test_shutdown_limit_below_minimum_stable_output_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "example.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["thermal"]["shutdown_ramp_mw"] = raw["thermal"]["minimum_output_mw"] - 1.0
    path = tmp_path / "invalid-shutdown.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="shutdown_ramp_mw"):
        load_config(path)


def test_startup_period_output_then_normal_ramp_are_both_binding() -> None:
    config = _dispatch_config(
        _thermal(
            initial_on=False,
            initial_output_mw=0.0,
            minimum_output_mw=10.0,
            startup_ramp_mw=20.0,
            shutdown_ramp_mw=20.0,
            ramp_mw_per_hour=15.0,
        )
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(2),
        gross_demand_mw=np.array([20.0, 35.0]),
    )

    # The startup period is capped at 20 MW; the next period can add 15 MW/h.
    assert result.frame["thermal_startup"].tolist() == [1, 0]
    assert result.frame["thermal_output_mw"].tolist() == pytest.approx([20.0, 35.0])


def test_initially_on_period_zero_output_uses_normal_ramp_limit() -> None:
    config = _dispatch_config(
        _thermal(
            initial_on=True,
            initial_output_mw=50.0,
            minimum_output_mw=10.0,
            startup_ramp_mw=100.0,
            shutdown_ramp_mw=100.0,
            ramp_mw_per_hour=10.0,
        )
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(1),
        gross_demand_mw=np.array([65.0]),
    )

    # From an initial 50 MW, a 10 MW/h ramp allows 60 MW in period zero.
    assert result.frame["thermal_on"].tolist() == [1]
    assert result.frame["thermal_output_mw"].iloc[0] == pytest.approx(60.0)
    assert result.frame["source_load_shed_mw"].iloc[0] == pytest.approx(5.0)


def test_startup_and_shutdown_are_never_simultaneous() -> None:
    config = _dispatch_config(
        _thermal(
            initial_on=False,
            initial_output_mw=0.0,
            minimum_output_mw=10.0,
            startup_ramp_mw=100.0,
            shutdown_ramp_mw=100.0,
        )
    )

    result = UnitCommitment(config).solve(
        renewable_available_mw=np.zeros(3),
        gross_demand_mw=np.array([50.0, 0.0, 50.0]),
    )

    transitions = result.frame["thermal_startup"] + result.frame["thermal_shutdown"]
    assert transitions.max() <= 1
