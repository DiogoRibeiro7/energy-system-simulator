from __future__ import annotations

import numpy as np

from energy_system_simulator.planning import (
    CapacityExpansionPlanner,
    CapacityExpansionProblem,
    GenerationCandidate,
    PlanningBlock,
    PlanningPolicy,
    StorageCandidate,
)


def main() -> None:
    problem = CapacityExpansionProblem(
        demand_mw={"system": np.array([0.2, 1.0, 0.8, 0.4], dtype=float)},
        representative_weights_hours=np.array([1000.0, 2000.0, 3000.0, 2760.0], dtype=float),
        annual_hours=8760.0,
        blocks=(PlanningBlock("representative-day", 0, 4),),
        generation_candidates=(
            GenerationCandidate(
                id="solar",
                technology="solar",
                max_build_mw=2.0,
                annualized_capital_cost_eur_per_mw_year=70_000.0,
                fixed_om_cost_eur_per_mw_year=8_000.0,
                availability_profile=np.array([0.8, 0.9, 0.2, 0.0], dtype=float),
            ),
            GenerationCandidate(
                id="gas",
                technology="thermal",
                max_build_mw=2.0,
                annualized_capital_cost_eur_per_mw_year=45_000.0,
                fixed_om_cost_eur_per_mw_year=6_000.0,
                variable_cost_eur_per_mwh=45.0,
                fuel_cost_eur_per_mwh=35.0,
                emission_tonnes_per_mwh=0.36,
                capacity_credit=0.95,
            ),
        ),
        storage_candidates=(
            StorageCandidate(
                id="battery",
                max_power_build_mw=1.0,
                max_energy_build_mwh=2.0,
                annualized_power_cost_eur_per_mw_year=35_000.0,
                annualized_energy_cost_eur_per_mwh_year=12_000.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                capacity_credit=0.8,
            ),
        ),
        policy=PlanningPolicy(
            carbon_price_eur_per_tonne=90.0,
            renewable_share_min=0.45,
            planning_reserve_margin_fraction=0.15,
        ),
        reliability_penalty_eur_per_mwh=25_000.0,
    )

    result = CapacityExpansionPlanner().solve(problem)
    print("Selected generation MW:", result.selected_generation_capacity_mw)
    print("Selected storage power MW:", result.selected_storage_power_mw)
    print("Selected storage energy MWh:", result.selected_storage_energy_mwh)
    print("Annual costs EUR:", result.annual_costs_eur)
    print("Generation mix MWh:", result.generation_mix_mwh)
    print("Emissions tonnes:", result.emissions_tonnes)


if __name__ == "__main__":
    main()
