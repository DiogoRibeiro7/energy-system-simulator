from __future__ import annotations

from pathlib import Path

import numpy as np

from energy_system_simulator.config import load_config
from energy_system_simulator.dispatch import UnitCommitment


def test_unit_commitment_respects_balance_and_bounds() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    renewable = np.array([0.0, 30.0, 130.0, 10.0, 0.0], dtype=float)
    demand = np.array([120.0, 120.0, 120.0, 120.0, 120.0], dtype=float)
    result = UnitCommitment(config).solve(renewable, demand)
    frame = result.frame

    left = (
        frame["renewable_used_mw"]
        + frame["thermal_output_mw"]
        + frame["battery_discharge_mw"]
        + frame["imports_mw"]
        + frame["source_load_shed_mw"]
    )
    right = frame["gross_demand_mw"] + frame["battery_charge_mw"]
    assert np.allclose(left, right, atol=1e-6)
    assert (frame["thermal_output_mw"] <= config.thermal.maximum_output_mw + 1e-7).all()
    assert (frame["renewable_used_mw"] <= renewable + 1e-7).all()
    assert frame["battery_soc_mwh"].iloc[-1] >= config.battery.minimum_final_soc_mwh - 1e-7
    assert result.formulation_statistics.continuous_variables == 35
    assert result.formulation_statistics.integer_variables == 20
    assert result.formulation_statistics.linear_constraints > 0
    assert result.formulation_statistics.matrix_nonzeros > 0
