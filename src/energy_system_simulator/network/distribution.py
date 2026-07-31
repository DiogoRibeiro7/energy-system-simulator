from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from energy_system_simulator.config import NetworkConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DistributionDemand:
    """Demand quantities after applying network efficiency and capacity."""

    end_user_demand_mw: FloatArray
    deliverable_demand_mw: FloatArray
    gross_demand_mw: FloatArray
    network_capacity_shed_mw: FloatArray


class DistributionNetwork:
    """Aggregated distribution network with fixed losses and transfer capacity."""

    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        self.efficiency = 1.0 - config.loss_fraction

    def prepare_demand(self, end_user_demand_mw: npt.ArrayLike) -> DistributionDemand:
        """Convert end-user demand to source-side demand for dispatch."""
        demand = np.asarray(end_user_demand_mw, dtype=np.float64)
        if np.any(~np.isfinite(demand)) or np.any(demand < 0.0):
            raise ValueError("Demand must be finite and non-negative")

        maximum_deliverable = self.config.transfer_capacity_mw * self.efficiency
        deliverable = np.minimum(demand, maximum_deliverable)
        capacity_shed = demand - deliverable
        gross = deliverable / self.efficiency
        return DistributionDemand(
            end_user_demand_mw=demand,
            deliverable_demand_mw=deliverable,
            gross_demand_mw=gross,
            network_capacity_shed_mw=capacity_shed,
        )
