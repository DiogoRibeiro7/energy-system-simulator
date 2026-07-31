from __future__ import annotations

import numpy as np

from energy_system_simulator.config import NetworkConfig
from energy_system_simulator.network import DistributionNetwork, Line, solve_dc_power_flow


def test_distribution_applies_losses_and_capacity() -> None:
    network = DistributionNetwork(NetworkConfig(loss_fraction=0.1, transfer_capacity_mw=100.0))
    prepared = network.prepare_demand([45.0, 120.0])
    assert np.allclose(prepared.deliverable_demand_mw, [45.0, 90.0])
    assert np.allclose(prepared.gross_demand_mw, [50.0, 100.0])
    assert np.allclose(prepared.network_capacity_shed_mw, [0.0, 30.0])


def test_dc_power_flow_balances_three_bus_network() -> None:
    lines = [
        Line(0, 1, susceptance=10.0, capacity_mw=100.0),
        Line(1, 2, susceptance=8.0, capacity_mw=100.0),
        Line(0, 2, susceptance=5.0, capacity_mw=100.0),
    ]
    result = solve_dc_power_flow([80.0, -30.0, -50.0], lines)
    assert result.voltage_angles_rad[0] == 0.0
    assert result.line_flows_mw.shape == (3,)
    assert np.all(result.capacity_utilisation >= 0.0)
