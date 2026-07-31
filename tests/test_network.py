from __future__ import annotations

import numpy as np
import pytest

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
    assert result.has_overloads is False


def test_dc_power_flow_two_bus_hand_calculation() -> None:
    result = solve_dc_power_flow(
        [50.0, -50.0],
        [Line(0, 1, susceptance=10.0, capacity_mw=100.0, line_id="tie")],
    )

    assert result.voltage_angles_rad.tolist() == pytest.approx([0.0, -5.0])
    assert result.line_flows_mw.tolist() == pytest.approx([50.0])
    assert result.line_diagnostics[0].line_id == "tie"
    assert result.line_diagnostics[0].utilisation == pytest.approx(0.5)


def test_dc_power_flow_three_bus_sign_convention() -> None:
    lines = [
        Line(0, 1, susceptance=10.0, capacity_mw=200.0, line_id="0-1"),
        Line(1, 2, susceptance=10.0, capacity_mw=200.0, line_id="1-2"),
    ]

    result = solve_dc_power_flow([100.0, -40.0, -60.0], lines)

    assert result.voltage_angles_rad.tolist() == pytest.approx([0.0, -10.0, -16.0])
    assert result.line_flows_mw.tolist() == pytest.approx([100.0, 60.0])


def test_dc_power_flow_reports_overload_without_enforcing_capacity() -> None:
    result = solve_dc_power_flow(
        [50.0, -50.0],
        [Line(0, 1, susceptance=10.0, capacity_mw=40.0, line_id="limited")],
    )

    diagnostic = result.line_diagnostics[0]
    assert result.has_overloads is True
    assert diagnostic.overloaded is True
    assert diagnostic.flow_mw == pytest.approx(50.0)
    assert diagnostic.capacity_mw == pytest.approx(40.0)
    assert diagnostic.overload_mw == pytest.approx(10.0)


def test_dc_power_flow_can_raise_on_overload_when_requested() -> None:
    with pytest.raises(ValueError, match="Line overload detected: limited"):
        solve_dc_power_flow(
            [50.0, -50.0],
            [Line(0, 1, susceptance=10.0, capacity_mw=40.0, line_id="limited")],
            overload_policy="raise",
        )


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([Line(0, 0, susceptance=10.0, capacity_mw=100.0)], "itself"),
        ([Line(0, 1, susceptance=0.0, capacity_mw=100.0)], "positive"),
        ([Line(0, 1, susceptance=10.0, capacity_mw=0.0)], "positive"),
        (
            [
                Line(0, 1, susceptance=10.0, capacity_mw=100.0, line_id="a"),
                Line(1, 0, susceptance=10.0, capacity_mw=100.0, line_id="a"),
            ],
            "Duplicate line_id",
        ),
    ],
)
def test_dc_power_flow_rejects_invalid_lines(lines: list[Line], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_dc_power_flow([10.0, -10.0], lines)


def test_dc_power_flow_rejects_disconnected_network() -> None:
    with pytest.raises(ValueError, match="disconnected"):
        solve_dc_power_flow(
            [10.0, -10.0, 0.0],
            [Line(0, 1, susceptance=10.0, capacity_mw=100.0)],
        )


def test_dc_power_flow_rejects_imbalanced_injections() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        solve_dc_power_flow(
            [10.0, -9.0],
            [Line(0, 1, susceptance=10.0, capacity_mw=100.0)],
        )
