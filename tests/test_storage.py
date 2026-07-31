from __future__ import annotations

from pathlib import Path

from energy_system_simulator.config import load_config
from energy_system_simulator.storage import BatteryLimits


def test_battery_limits_are_derived_from_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")

    limits = BatteryLimits.from_config(config.battery)

    assert limits.minimum_soc_mwh == config.battery.minimum_soc_mwh
    assert limits.maximum_soc_mwh == config.battery.maximum_soc_mwh
    assert limits.power_capacity_mw == config.battery.power_capacity_mw
    assert limits.charge_efficiency == config.battery.charge_efficiency
    assert limits.discharge_efficiency == config.battery.discharge_efficiency
