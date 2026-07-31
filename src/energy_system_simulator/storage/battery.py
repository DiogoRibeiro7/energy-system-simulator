from __future__ import annotations

from dataclasses import dataclass

from energy_system_simulator.config import BatteryConfig


@dataclass(frozen=True)
class BatteryLimits:
    """Validated battery limits used by the optimisation model."""

    minimum_soc_mwh: float
    maximum_soc_mwh: float
    power_capacity_mw: float
    charge_efficiency: float
    discharge_efficiency: float

    @classmethod
    def from_config(cls, config: BatteryConfig) -> BatteryLimits:
        """Construct limits from a validated model configuration."""
        return cls(
            minimum_soc_mwh=config.minimum_soc_mwh,
            maximum_soc_mwh=config.maximum_soc_mwh,
            power_capacity_mw=config.power_capacity_mw,
            charge_efficiency=config.charge_efficiency,
            discharge_efficiency=config.discharge_efficiency,
        )
