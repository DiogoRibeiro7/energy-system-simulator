from __future__ import annotations

import numpy as np
import pytest

from energy_system_simulator.planning import (
    CapacityExpansionPlanner,
    CapacityExpansionProblem,
    GenerationCandidate,
    PlanningBlock,
    PlanningPolicy,
    StorageCandidate,
)


def test_investment_threshold_builds_when_capacity_is_cheaper_than_unserved_energy() -> None:
    problem = CapacityExpansionProblem(
        demand_mw={"system": np.array([1.0])},
        representative_weights_hours=np.array([1.0]),
        annual_hours=1.0,
        reliability_penalty_eur_per_mwh=1_000.0,
        generation_candidates=(
            GenerationCandidate(
                id="peaker",
                technology="thermal",
                max_build_mw=1.0,
                annualized_capital_cost_eur_per_mw_year=100.0,
            ),
        ),
    )

    result = CapacityExpansionPlanner().solve(problem)

    assert result.selected_generation_capacity_mw["peaker"] == pytest.approx(1.0)
    assert result.unserved_energy_mwh == pytest.approx(0.0)
    assert result.annual_costs_eur["annualized_capital_cost_eur"] == pytest.approx(100.0)


def test_carbon_price_shifts_build_from_thermal_to_wind() -> None:
    base = {
        "demand_mw": {"system": np.array([1.0])},
        "representative_weights_hours": np.array([1.0]),
        "annual_hours": 1.0,
        "generation_candidates": (
            GenerationCandidate(
                id="gas",
                technology="thermal",
                max_build_mw=1.0,
                annualized_capital_cost_eur_per_mw_year=1.0,
                variable_cost_eur_per_mwh=10.0,
                emission_tonnes_per_mwh=1.0,
            ),
            GenerationCandidate(
                id="wind",
                technology="wind",
                max_build_mw=1.0,
                annualized_capital_cost_eur_per_mw_year=50.0,
                availability_profile=np.array([1.0]),
            ),
        ),
    }

    no_policy = CapacityExpansionPlanner().solve(CapacityExpansionProblem(**base))
    carbon_policy = CapacityExpansionPlanner().solve(
        CapacityExpansionProblem(
            **base,
            policy=PlanningPolicy(carbon_price_eur_per_tonne=100.0),
        )
    )

    assert no_policy.selected_generation_capacity_mw["gas"] == pytest.approx(1.0)
    assert no_policy.selected_generation_capacity_mw["wind"] == pytest.approx(0.0)
    assert carbon_policy.selected_generation_capacity_mw["gas"] == pytest.approx(0.0)
    assert carbon_policy.selected_generation_capacity_mw["wind"] == pytest.approx(1.0)
    assert carbon_policy.emissions_tonnes == pytest.approx(0.0)


def test_storage_selects_power_and_energy_to_shift_renewable_generation() -> None:
    problem = CapacityExpansionProblem(
        demand_mw={"system": np.array([0.0, 1.0])},
        representative_weights_hours=np.array([1.0, 1.0]),
        annual_hours=2.0,
        blocks=(PlanningBlock("day", 0, 2),),
        generation_candidates=(
            GenerationCandidate(
                id="solar",
                technology="solar",
                max_build_mw=1.0,
                availability_profile=np.array([1.0, 0.0]),
            ),
        ),
        storage_candidates=(
            StorageCandidate(
                id="battery",
                max_power_build_mw=1.0,
                max_energy_build_mwh=1.0,
                annualized_power_cost_eur_per_mw_year=1.0,
                annualized_energy_cost_eur_per_mwh_year=1.0,
            ),
        ),
        reliability_penalty_eur_per_mwh=1_000.0,
    )

    result = CapacityExpansionPlanner().solve(problem)

    assert result.selected_generation_capacity_mw["solar"] == pytest.approx(1.0)
    assert result.selected_storage_power_mw["battery"] == pytest.approx(1.0)
    assert result.selected_storage_energy_mwh["battery"] == pytest.approx(1.0)
    assert result.dispatch["storage_charge_mw__battery"].tolist() == pytest.approx([1.0, 0.0])
    assert result.dispatch["storage_discharge_mw__battery"].tolist() == pytest.approx([0.0, 1.0])
    assert result.unserved_energy_mwh == pytest.approx(0.0)


def test_representative_period_weights_scale_generation_and_costs() -> None:
    problem = CapacityExpansionProblem(
        demand_mw={"system": np.array([1.0, 1.0])},
        representative_weights_hours=np.array([100.0, 200.0]),
        annual_hours=300.0,
        generation_candidates=(
            GenerationCandidate(
                id="existing-gas",
                technology="thermal",
                existing_capacity_mw=1.0,
                variable_cost_eur_per_mwh=2.0,
            ),
        ),
    )

    result = CapacityExpansionPlanner().solve(problem)

    assert result.generation_mix_mwh["existing-gas"] == pytest.approx(300.0)
    assert result.annual_costs_eur["variable_operation_cost_eur"] == pytest.approx(600.0)


def test_zero_candidate_build_reproduces_fixed_capacity_operation() -> None:
    problem = CapacityExpansionProblem(
        demand_mw={"system": np.array([0.5, 1.0])},
        representative_weights_hours=np.array([1.0, 1.0]),
        annual_hours=2.0,
        generation_candidates=(
            GenerationCandidate(
                id="fixed-gas",
                technology="thermal",
                existing_capacity_mw=1.0,
                max_build_mw=0.0,
                variable_cost_eur_per_mwh=5.0,
            ),
        ),
    )

    result = CapacityExpansionPlanner().solve(problem)

    assert result.selected_generation_capacity_mw["fixed-gas"] == pytest.approx(0.0)
    assert result.dispatch["generation_mw__fixed-gas"].tolist() == pytest.approx([0.5, 1.0])
    assert result.annual_costs_eur["variable_operation_cost_eur"] == pytest.approx(7.5)
    assert result.unserved_energy_mwh == pytest.approx(0.0)


def test_existing_fixed_om_is_reported_once() -> None:
    problem = CapacityExpansionProblem(
        demand_mw={"system": np.array([1.0])},
        representative_weights_hours=np.array([1.0]),
        annual_hours=1.0,
        generation_candidates=(
            GenerationCandidate(
                id="fixed-gas",
                technology="thermal",
                existing_capacity_mw=1.0,
                fixed_om_cost_eur_per_mw_year=7.0,
                variable_cost_eur_per_mwh=5.0,
            ),
        ),
    )

    result = CapacityExpansionPlanner().solve(problem)

    assert result.annual_costs_eur["fixed_om_cost_eur"] == pytest.approx(7.0)
    assert result.annual_costs_eur["total_annual_cost_eur"] == pytest.approx(result.objective_eur)


def test_representative_weights_must_match_intended_year_hours() -> None:
    problem = CapacityExpansionProblem(
        demand_mw={"system": np.array([1.0])},
        representative_weights_hours=np.array([1.0]),
        annual_hours=2.0,
    )

    with pytest.raises(ValueError, match="annual_hours"):
        CapacityExpansionPlanner().solve(problem)
