from __future__ import annotations

from pathlib import Path

from energy_system_simulator.config import load_config
from energy_system_simulator.simulation import SimulationEngine


def test_example_simulation_runs_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "example.yaml")
    result = SimulationEngine(config).run()
    assert len(result.timeseries) == 336
    assert result.summary["total_demand_mwh"] > 0.0
    assert result.summary["served_demand_mwh"] <= result.summary["total_demand_mwh"] + 1e-6
    assert result.summary["renewable_available_mwh"] > 0.0
