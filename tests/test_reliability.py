from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_system_simulator.config import load_config
from energy_system_simulator.simulation import (
    OutageModel,
    ReliabilityStudy,
    ReliabilityStudyConfig,
    SimulationEngine,
)
from energy_system_simulator.simulation.reliability import (
    _confidence_interval,
    _sample_two_state_path,
)


def _firm_capacity_config(tmp_path: Path, periods: int = 4) -> Path:
    input_path = tmp_path / "firm-capacity.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC"),
            "demand_mw": np.full(periods, 50.0),
            "irradiance_w_m2": np.zeros(periods),
            "ambient_temperature_c": np.full(periods, 15.0),
            "wind_speed_m_s": np.zeros(periods),
        }
    ).to_csv(input_path, index=False)

    config_path = tmp_path / "firm-capacity.yaml"
    config_path.write_text(
        f"""
schema_version: 1

simulation:
  time_step_hours: 1.0
  solver_time_limit_seconds: 20.0
  mip_relative_gap: 0.001

solar:
  capacity_mw: 1.0
  performance_ratio: 0.86
  reference_irradiance_w_m2: 1000.0
  temperature_coefficient_per_c: -0.004
  nominal_operating_cell_temperature_c: 45.0

wind:
  capacity_mw: 1.0
  cut_in_speed_m_s: 3.0
  rated_speed_m_s: 12.0
  cut_out_speed_m_s: 25.0

thermal:
  name: Firm unit
  minimum_output_mw: 0.0
  maximum_output_mw: 60.0
  ramp_up_mw_per_hour: 60.0
  ramp_down_mw_per_hour: 60.0
  startup_ramp_mw: 60.0
  shutdown_ramp_mw: 60.0
  variable_cost_eur_per_mwh: 50.0
  no_load_cost_eur_per_hour: 0.0
  startup_cost_eur: 0.0
  shutdown_cost_eur: 0.0
  emission_factor_tonnes_per_mwh: 0.4
  minimum_up_hours: 1.0
  minimum_down_hours: 1.0
  initial_on: true
  initial_output_mw: 50.0
  initial_up_time_hours: 2.0
  initial_down_time_hours: 0.0
  terminal_commitment_mode: carry_residual_obligations

battery:
  energy_capacity_mwh: 0.0
  power_capacity_mw: 0.0
  minimum_soc_mwh: 0.0
  maximum_soc_mwh: 0.0
  initial_soc_mwh: 0.0
  charge_efficiency: 1.0
  discharge_efficiency: 1.0
  throughput_cost_eur_per_mwh: 0.0
  minimum_final_soc_mwh: 0.0
  terminal_soc_mode: free

network:
  loss_fraction: 0.0
  transfer_capacity_mw: 100.0

imports:
  maximum_power_mw: 0.0
  price_eur_per_mwh: 100.0
  emission_factor_tonnes_per_mwh: 0.0

penalties:
  renewable_curtailment_eur_per_mwh: 1.0
  lost_load_eur_per_mwh: 10000.0
  carbon_price_eur_per_tonne: 0.0

paths:
  input_csv: {input_path.as_posix()}
  output_directory: {(tmp_path / "outputs").as_posix()}
""",
        encoding="utf-8",
    )
    return config_path


def test_outage_paths_are_seed_reproducible(tmp_path: Path) -> None:
    config = load_config(_firm_capacity_config(tmp_path))
    study = ReliabilityStudy(
        config,
        ReliabilityStudyConfig(
            replications=2,
            seed=42,
            outage_models=(
                OutageModel("thermal_1", "thermal", 0.25, 2.0),
                OutageModel("imports", "import", 0.10, 1.0),
            ),
        ),
    )

    first = study.sample_outage_paths(replication_index=0, periods=12)
    second = study.sample_outage_paths(replication_index=0, periods=12)

    assert first.keys() == second.keys()
    assert all(np.array_equal(first[key], second[key]) for key in first)


def test_zero_outage_reproduces_deterministic_availability(tmp_path: Path) -> None:
    config = load_config(_firm_capacity_config(tmp_path))
    deterministic = SimulationEngine(config).run()
    study = ReliabilityStudy(
        config,
        ReliabilityStudyConfig(
            replications=1,
            seed=1,
            outage_models=(OutageModel("thermal_1", "thermal", 0.0, 1.0),),
        ),
    )

    result = study.run()

    assert result.metrics["expected_unserved_energy_mwh"] == pytest.approx(
        deterministic.summary["unserved_energy_mwh"]
    )
    assert result.metrics["loss_of_load_probability"] == pytest.approx(0.0)


def test_complete_thermal_outage_creates_unserved_energy(tmp_path: Path) -> None:
    config = load_config(_firm_capacity_config(tmp_path))
    study = ReliabilityStudy(
        config,
        ReliabilityStudyConfig(
            replications=1,
            seed=1,
            outage_models=(OutageModel("thermal_1", "thermal", 1.0, 1.0),),
        ),
    )

    result = study.run()

    assert result.metrics["expected_unserved_energy_mwh"] == pytest.approx(200.0)
    assert result.metrics["expected_demand_not_served_mw"] == pytest.approx(50.0)
    assert result.attributed_unserved_energy_mwh_by_type["thermal"] == pytest.approx(200.0)


def test_repair_transitions_are_seeded() -> None:
    path = _sample_two_state_path(
        periods=80,
        dt_hours=1.0,
        forced_outage_rate=0.5,
        mean_time_to_repair_hours=1.0,
        rng=np.random.default_rng(7),
    )

    assert path.any()
    assert (~path).any()


def test_confidence_interval_calculation() -> None:
    interval = _confidence_interval([1.0, 2.0, 3.0], z=1.959963984540054)

    assert interval["mean"] == pytest.approx(2.0)
    assert interval["half_width"] == pytest.approx(1.1315857340761715)


def test_reliability_distribution_and_failed_counts(tmp_path: Path) -> None:
    config = load_config(_firm_capacity_config(tmp_path))
    study = ReliabilityStudy(
        config,
        ReliabilityStudyConfig(
            replications=2,
            seed=1,
            outage_models=(OutageModel("thermal_1", "thermal", 0.0, 1.0),),
            parallel_workers=2,
        ),
    )

    result = study.run()

    assert result.metrics["replications"] == pytest.approx(2.0)
    assert result.metrics["failed_replications"] == pytest.approx(0.0)
    assert result.metric_distribution["expected_unserved_energy_mwh"] == (0.0, 0.0)
