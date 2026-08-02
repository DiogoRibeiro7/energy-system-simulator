# Capacity Expansion

Capacity expansion is exposed as a separate model entry point through
`CapacityExpansionPlanner`. Operational simulation, deterministic unit
commitment, stochastic dispatch, and market settlement are unchanged.

The first planning implementation is a continuous single-year model. It selects
new capacity and dispatches representative periods in one LP:

- solar, wind, and thermal generation capacity in MW
- storage power capacity in MW and energy capacity in MWh
- interconnector import capacity in MW
- transmission transfer capacity in MW between zones
- unserved energy as an explicit reliability-penalty variable

Existing assets are represented by fixed existing capacity on the same candidate
objects. New build is reported separately from existing capacity.

## Objective

The annual objective separates:

- annualized capital cost
- fixed operation and maintenance cost
- variable operation cost
- fuel cost
- carbon cost
- reliability penalty

Variable operation, fuel, carbon, emissions, generation, curtailment, imports,
and unserved energy are multiplied by representative-period weights in hours.
Capital and fixed O&M costs are already annual values.

## Representative Periods

`representative_weights_hours` must be positive and must sum to `annual_hours`.
This makes the intended year length explicit, for example 8760 for a full
non-leap year or a smaller hand-checkable value in tests.

Storage chronology is preserved inside each `PlanningBlock`. The current model
starts each block with zero state of charge and requires zero terminal state of
charge at the end of the block. This prevents energy leakage across unrelated
representative days.

Representative-period methods are limited for seasonal storage, hydro reservoir
carryover, long fuel constraints, and multi-week weather persistence. Use
chronological operational simulation for those studies unless the
representative-period construction explicitly preserves the relevant chronology.

## Policy Constraints

The first implementation supports:

- minimum renewable-energy share
- emissions cap
- firm-capacity planning reserve margin
- maximum build by generation technology
- minimum domestic generation share
- carbon price

Policy shadow prices are reported from the continuous LP relaxation when the
solver exposes inequality marginals. They use solver sign conventions for
`A_ub x <= b_ub` constraints.

## Scope Limits

The model intentionally does not yet include discounted multi-year build years,
asset retirement by lifetime, integer unit commitment, minimum up/down times,
startup decisions, network losses, or market settlements. Thermal startup costs
remain part of the operational unit-commitment model; they are not represented
in this continuous first planning layer.

This boundary keeps the first planning model reviewable. A future multi-year
planner can add build vintages, retirements, discounting, and integer operations
on top of this single-year core.
