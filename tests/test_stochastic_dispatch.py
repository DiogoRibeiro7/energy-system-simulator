from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_system_simulator.config import load_config
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.simulation import (
    StochasticDispatch,
    StochasticDispatchConfig,
    StochasticScenario,
    SyntheticScenarioConfig,
    generate_synthetic_scenarios,
)


def _stochastic_config(tmp_path: Path, *, no_load_cost: float = 0.0) -> Path:
    input_path = tmp_path / "stochastic.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "demand_mw": [0.0, 0.0],
            "irradiance_w_m2": [0.0, 0.0],
            "ambient_temperature_c": [15.0, 15.0],
            "wind_speed_m_s": [0.0, 0.0],
        }
    ).to_csv(input_path, index=False)
    config_path = tmp_path / "stochastic.yaml"
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
  name: Stochastic unit
  minimum_output_mw: 0.0
  maximum_output_mw: 100.0
  ramp_up_mw_per_hour: 100.0
  ramp_down_mw_per_hour: 100.0
  startup_ramp_mw: 100.0
  shutdown_ramp_mw: 100.0
  variable_cost_eur_per_mwh: 0.0
  no_load_cost_eur_per_hour: {no_load_cost}
  startup_cost_eur: 0.0
  shutdown_cost_eur: 0.0
  emission_factor_tonnes_per_mwh: 0.0
  minimum_up_hours: 1.0
  minimum_down_hours: 1.0
  initial_on: false
  initial_output_mw: 0.0
  initial_up_time_hours: 0.0
  initial_down_time_hours: 2.0
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
  lost_load_eur_per_mwh: 1000.0
  carbon_price_eur_per_tonne: 0.0

paths:
  input_csv: {input_path.as_posix()}
  output_directory: {(tmp_path / "outputs").as_posix()}
""",
        encoding="utf-8",
    )
    return config_path


def test_scenario_probabilities_must_sum_to_one(tmp_path: Path) -> None:
    config = load_config(_stochastic_config(tmp_path))
    scenarios = (
        StochasticScenario("a", 0.6, [0.0], [10.0]),
        StochasticScenario("b", 0.3, [0.0], [20.0]),
    )

    with pytest.raises(ValueError, match="sum to one"):
        StochasticDispatch(
            config,
            scenarios,
            StochasticDispatchConfig(commitment_horizon_periods=1),
        )


def test_one_scenario_reproduces_deterministic_dispatch(tmp_path: Path) -> None:
    config = load_config(_stochastic_config(tmp_path))
    renewable = np.array([0.0, 0.0])
    demand = np.array([40.0, 30.0])
    deterministic = UnitCommitment(config).solve(renewable, demand)
    stochastic = StochasticDispatch(
        config,
        (StochasticScenario("only", 1.0, renewable, demand),),
        StochasticDispatchConfig(commitment_horizon_periods=2),
    ).run()

    assert stochastic.expected_objective_eur == pytest.approx(deterministic.objective_eur)
    assert stochastic.scenario_cost_distribution_eur["only"] == pytest.approx(
        deterministic.objective_eur
    )


def test_first_stage_commitment_is_shared_across_scenarios(tmp_path: Path) -> None:
    config = load_config(_stochastic_config(tmp_path))
    result = StochasticDispatch(
        config,
        (
            StochasticScenario("low", 0.5, [0.0, 0.0], [0.0, 0.0]),
            StochasticScenario("high", 0.5, [0.0, 0.0], [50.0, 50.0]),
        ),
        StochasticDispatchConfig(commitment_horizon_periods=1),
    ).run()

    expected = result.first_stage_commitment_by_unit
    assert all(
        scenario.first_stage_commitment_by_unit == expected for scenario in result.scenario_results
    )
    weighted_cost = sum(
        scenario.probability * result.scenario_cost_distribution_eur[scenario.id]
        for scenario in result.scenario_results
    )
    assert result.expected_objective_eur == pytest.approx(weighted_cost)


def test_value_of_information_identities_are_reported(tmp_path: Path) -> None:
    config = load_config(_stochastic_config(tmp_path, no_load_cost=10.0))
    result = StochasticDispatch(
        config,
        (
            StochasticScenario("low", 0.5, [0.0], [0.0]),
            StochasticScenario("high", 0.5, [0.0], [50.0]),
        ),
        StochasticDispatchConfig(commitment_horizon_periods=1),
    ).run()
    benchmarks = result.benchmarks

    assert benchmarks.value_of_stochastic_solution_eur == pytest.approx(
        benchmarks.expected_result_using_expected_value_solution_eur - result.expected_objective_eur
    )
    assert benchmarks.expected_value_of_perfect_information_eur == pytest.approx(
        result.expected_objective_eur - benchmarks.wait_and_see_expected_objective_eur
    )


def test_synthetic_scenario_generation_is_seed_reproducible() -> None:
    config = SyntheticScenarioConfig(
        count=3,
        seed=7,
        demand_multiplier_std=0.1,
        renewable_multiplier_std=0.1,
    )

    first = generate_synthetic_scenarios([10.0, 20.0], [30.0, 40.0], config)
    second = generate_synthetic_scenarios([10.0, 20.0], [30.0, 40.0], config)

    assert [scenario.id for scenario in first] == ["s000", "s001", "s002"]
    assert all(
        np.allclose(a.gross_demand_mw, b.gross_demand_mw)
        and np.allclose(a.renewable_available_mw, b.renewable_available_mw)
        for a, b in zip(first, second, strict=True)
    )


def test_cvar_weight_selects_high_cost_protection(tmp_path: Path) -> None:
    config = load_config(_stochastic_config(tmp_path, no_load_cost=100.0))
    scenarios = (
        StochasticScenario("ordinary", 0.999, [0.0], [0.0]),
        StochasticScenario("scarcity", 0.001, [0.0], [50.0]),
    )

    neutral = StochasticDispatch(
        config,
        scenarios,
        StochasticDispatchConfig(commitment_horizon_periods=1),
    ).run()
    risk_averse = StochasticDispatch(
        config,
        scenarios,
        StochasticDispatchConfig(
            commitment_horizon_periods=1,
            cvar_confidence_level=0.99,
            cvar_weight=0.02,
        ),
    ).run()

    assert neutral.first_stage_commitment_by_unit["thermal_1"] == (0,)
    assert risk_averse.first_stage_commitment_by_unit["thermal_1"] == (1,)
    assert risk_averse.cvar_eur is not None
