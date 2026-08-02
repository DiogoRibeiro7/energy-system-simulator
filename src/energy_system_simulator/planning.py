from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from energy_system_simulator.exceptions import OptimisationError

FloatArray = npt.NDArray[np.float64]


Technology = Literal["solar", "wind", "thermal"]
ConstraintMatrices = tuple[
    tuple[coo_matrix | None, FloatArray | None],
    tuple[coo_matrix, FloatArray],
    list[str],
]


@dataclass(frozen=True)
class PlanningBlock:
    """Contiguous representative block whose storage chronology is preserved."""

    id: str
    start_period: int
    end_period: int


@dataclass(frozen=True)
class GenerationCandidate:
    """Existing and buildable generation capacity for one technology at one zone."""

    id: str
    technology: Technology
    zone_id: str = "system"
    existing_capacity_mw: float = 0.0
    max_build_mw: float = 0.0
    annualized_capital_cost_eur_per_mw_year: float = 0.0
    fixed_om_cost_eur_per_mw_year: float = 0.0
    variable_cost_eur_per_mwh: float = 0.0
    fuel_cost_eur_per_mwh: float = 0.0
    emission_tonnes_per_mwh: float = 0.0
    capacity_credit: float = 1.0
    availability_profile: npt.ArrayLike | None = None


@dataclass(frozen=True)
class StorageCandidate:
    """Existing and buildable storage power and energy capacity."""

    id: str
    zone_id: str = "system"
    existing_power_mw: float = 0.0
    existing_energy_mwh: float = 0.0
    max_power_build_mw: float = 0.0
    max_energy_build_mwh: float = 0.0
    annualized_power_cost_eur_per_mw_year: float = 0.0
    annualized_energy_cost_eur_per_mwh_year: float = 0.0
    fixed_om_cost_eur_per_mw_year: float = 0.0
    variable_cost_eur_per_mwh: float = 0.0
    charge_efficiency: float = 1.0
    discharge_efficiency: float = 1.0
    capacity_credit: float = 1.0


@dataclass(frozen=True)
class InterconnectorCandidate:
    """Import capacity candidate feeding one zone."""

    id: str
    zone_id: str = "system"
    existing_capacity_mw: float = 0.0
    max_build_mw: float = 0.0
    annualized_capital_cost_eur_per_mw_year: float = 0.0
    fixed_om_cost_eur_per_mw_year: float = 0.0
    variable_cost_eur_per_mwh: float = 0.0
    emission_tonnes_per_mwh: float = 0.0
    capacity_credit: float = 0.0
    availability_profile: npt.ArrayLike | None = None


@dataclass(frozen=True)
class TransmissionCandidate:
    """Transfer capacity between zones. Positive flow is from from_zone_id to to_zone_id."""

    id: str
    from_zone_id: str
    to_zone_id: str
    existing_capacity_mw: float = 0.0
    max_build_mw: float = 0.0
    annualized_capital_cost_eur_per_mw_year: float = 0.0
    fixed_om_cost_eur_per_mw_year: float = 0.0


@dataclass(frozen=True)
class PlanningPolicy:
    """Single-year policy constraints for the continuous expansion model."""

    renewable_share_min: float | None = None
    emissions_cap_tonnes: float | None = None
    planning_reserve_margin_fraction: float | None = None
    max_build_mw_by_technology: dict[Technology, float] = field(default_factory=dict)
    minimum_domestic_generation_share: float | None = None
    carbon_price_eur_per_tonne: float = 0.0


@dataclass(frozen=True)
class CapacityExpansionProblem:
    """Single-year capacity-expansion problem with representative-period operation."""

    demand_mw: dict[str, npt.ArrayLike]
    representative_weights_hours: npt.ArrayLike
    annual_hours: float = 8760.0
    generation_candidates: tuple[GenerationCandidate, ...] = ()
    storage_candidates: tuple[StorageCandidate, ...] = ()
    interconnector_candidates: tuple[InterconnectorCandidate, ...] = ()
    transmission_candidates: tuple[TransmissionCandidate, ...] = ()
    policy: PlanningPolicy = field(default_factory=PlanningPolicy)
    reliability_penalty_eur_per_mwh: float = 100_000.0
    blocks: tuple[PlanningBlock, ...] = ()


@dataclass(frozen=True)
class CapacityExpansionResult:
    """Selected capacities, weighted operations, costs, and policy shadow prices."""

    selected_generation_capacity_mw: dict[str, float]
    selected_storage_power_mw: dict[str, float]
    selected_storage_energy_mwh: dict[str, float]
    selected_interconnector_capacity_mw: dict[str, float]
    selected_transmission_capacity_mw: dict[str, float]
    dispatch: pd.DataFrame
    annual_costs_eur: dict[str, float]
    generation_mix_mwh: dict[str, float]
    curtailment_mwh: dict[str, float]
    emissions_tonnes: float
    unserved_energy_mwh: float
    planning_reserve_margin: float | None
    policy_shadow_prices: dict[str, float]
    objective_eur: float
    solver_message: str


class _Registry:
    def __init__(self) -> None:
        self._offsets: dict[tuple[str, str | None], tuple[int, int]] = {}
        self.size = 0

    def add(self, name: str, size: int = 1, *, asset_id: str | None = None) -> None:
        key = (name, asset_id)
        if key in self._offsets:
            raise ValueError(f"Duplicate planning variable block: {name}, {asset_id}")
        self._offsets[key] = (self.size, size)
        self.size += size

    def at(self, name: str, period: int = 0, *, asset_id: str | None = None) -> int:
        offset, size = self._offsets[(name, asset_id)]
        if period < 0 or period >= size:
            raise IndexError(f"Period {period} is outside planning variable block {name}")
        return offset + period

    def values(self, solution: FloatArray, name: str, *, asset_id: str | None = None) -> FloatArray:
        offset, size = self._offsets[(name, asset_id)]
        return solution[offset : offset + size]


class CapacityExpansionPlanner:
    """Continuous single-year capacity-expansion model."""

    def solve(self, problem: CapacityExpansionProblem) -> CapacityExpansionResult:
        data = _PreparedProblem.from_problem(problem)
        registry = self._registry(data)
        objective, constant_costs = self._objective(data, registry)
        lower, upper = self._bounds(data, registry)
        equalities, inequalities, row_names = self._constraints(data, registry)

        result = linprog(
            c=objective,
            A_ub=inequalities[0],
            b_ub=inequalities[1],
            A_eq=equalities[0],
            b_eq=equalities[1],
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        if result.status != 0 or result.x is None:
            raise OptimisationError(f"Capacity expansion failed: {result.message}")
        solution = np.asarray(result.x, dtype=np.float64)
        objective_value = float(result.fun + sum(constant_costs.values()))
        return self._result(
            data,
            registry,
            solution,
            objective_value,
            constant_costs,
            row_names,
            np.asarray(result.ineqlin.marginals, dtype=np.float64),
            str(result.message),
        )

    def _registry(self, data: _PreparedProblem) -> _Registry:
        registry = _Registry()
        for candidate in data.generation_candidates:
            registry.add("generation_build_mw", asset_id=candidate.id)
            registry.add("generation_mw", data.periods, asset_id=candidate.id)
        for storage in data.storage_candidates:
            registry.add("storage_power_build_mw", asset_id=storage.id)
            registry.add("storage_energy_build_mwh", asset_id=storage.id)
            registry.add("storage_charge_mw", data.periods, asset_id=storage.id)
            registry.add("storage_discharge_mw", data.periods, asset_id=storage.id)
            registry.add("storage_soc_mwh", data.periods, asset_id=storage.id)
        for interconnector in data.interconnector_candidates:
            registry.add("interconnector_build_mw", asset_id=interconnector.id)
            registry.add("interconnector_import_mw", data.periods, asset_id=interconnector.id)
        for line in data.transmission_candidates:
            registry.add("transmission_build_mw", asset_id=line.id)
            registry.add("line_flow_mw", data.periods, asset_id=line.id)
        for zone_id in data.zone_ids:
            registry.add("unserved_mw", data.periods, asset_id=zone_id)
        return registry

    def _objective(
        self,
        data: _PreparedProblem,
        registry: _Registry,
    ) -> tuple[FloatArray, dict[str, float]]:
        coefficients = np.zeros(registry.size, dtype=np.float64)
        constants = {
            "existing_fixed_om_cost_eur": 0.0,
        }
        policy = data.problem.policy
        for candidate in data.generation_candidates:
            build = registry.at("generation_build_mw", asset_id=candidate.id)
            coefficients[build] = candidate.annualized_capital_cost_eur_per_mw_year
            coefficients[build] += candidate.fixed_om_cost_eur_per_mw_year
            constants["existing_fixed_om_cost_eur"] += (
                candidate.existing_capacity_mw * candidate.fixed_om_cost_eur_per_mw_year
            )
            variable_cost = (
                candidate.variable_cost_eur_per_mwh
                + candidate.fuel_cost_eur_per_mwh
                + candidate.emission_tonnes_per_mwh * policy.carbon_price_eur_per_tonne
            )
            for t in range(data.periods):
                coefficients[registry.at("generation_mw", t, asset_id=candidate.id)] = (
                    variable_cost * data.weights[t]
                )
        for storage in data.storage_candidates:
            power_build = registry.at("storage_power_build_mw", asset_id=storage.id)
            energy_build = registry.at("storage_energy_build_mwh", asset_id=storage.id)
            coefficients[power_build] = (
                storage.annualized_power_cost_eur_per_mw_year
                + storage.fixed_om_cost_eur_per_mw_year
            )
            coefficients[energy_build] = storage.annualized_energy_cost_eur_per_mwh_year
            constants["existing_fixed_om_cost_eur"] += (
                storage.existing_power_mw * storage.fixed_om_cost_eur_per_mw_year
            )
            for t in range(data.periods):
                coefficients[registry.at("storage_charge_mw", t, asset_id=storage.id)] = (
                    storage.variable_cost_eur_per_mwh * data.weights[t]
                )
                coefficients[registry.at("storage_discharge_mw", t, asset_id=storage.id)] = (
                    storage.variable_cost_eur_per_mwh * data.weights[t]
                )
        for interconnector in data.interconnector_candidates:
            build = registry.at("interconnector_build_mw", asset_id=interconnector.id)
            coefficients[build] = interconnector.annualized_capital_cost_eur_per_mw_year
            coefficients[build] += interconnector.fixed_om_cost_eur_per_mw_year
            constants["existing_fixed_om_cost_eur"] += (
                interconnector.existing_capacity_mw * interconnector.fixed_om_cost_eur_per_mw_year
            )
            variable_cost = (
                interconnector.variable_cost_eur_per_mwh
                + interconnector.emission_tonnes_per_mwh * policy.carbon_price_eur_per_tonne
            )
            for t in range(data.periods):
                coefficients[
                    registry.at("interconnector_import_mw", t, asset_id=interconnector.id)
                ] = variable_cost * data.weights[t]
        for line in data.transmission_candidates:
            build = registry.at("transmission_build_mw", asset_id=line.id)
            coefficients[build] = line.annualized_capital_cost_eur_per_mw_year
            coefficients[build] += line.fixed_om_cost_eur_per_mw_year
            constants["existing_fixed_om_cost_eur"] += (
                line.existing_capacity_mw * line.fixed_om_cost_eur_per_mw_year
            )
        for zone_id in data.zone_ids:
            for t in range(data.periods):
                coefficients[registry.at("unserved_mw", t, asset_id=zone_id)] = (
                    data.problem.reliability_penalty_eur_per_mwh * data.weights[t]
                )
        return coefficients, constants

    def _bounds(self, data: _PreparedProblem, registry: _Registry) -> tuple[FloatArray, FloatArray]:
        lower = np.zeros(registry.size, dtype=np.float64)
        upper = np.full(registry.size, np.inf, dtype=np.float64)
        for candidate in data.generation_candidates:
            upper[registry.at("generation_build_mw", asset_id=candidate.id)] = (
                candidate.max_build_mw
            )
        for storage in data.storage_candidates:
            upper[registry.at("storage_power_build_mw", asset_id=storage.id)] = (
                storage.max_power_build_mw
            )
            upper[registry.at("storage_energy_build_mwh", asset_id=storage.id)] = (
                storage.max_energy_build_mwh
            )
        for interconnector in data.interconnector_candidates:
            upper[registry.at("interconnector_build_mw", asset_id=interconnector.id)] = (
                interconnector.max_build_mw
            )
        for line in data.transmission_candidates:
            upper[registry.at("transmission_build_mw", asset_id=line.id)] = line.max_build_mw
            for t in range(data.periods):
                lower[registry.at("line_flow_mw", t, asset_id=line.id)] = -np.inf
        return lower, upper

    def _constraints(
        self,
        data: _PreparedProblem,
        registry: _Registry,
    ) -> ConstraintMatrices:
        equalities = _RowBuilder(registry.size)
        inequalities = _RowBuilder(registry.size)
        self._balance_constraints(data, registry, equalities)
        self._capacity_constraints(data, registry, inequalities)
        self._storage_constraints(data, registry, equalities, inequalities)
        self._policy_constraints(data, registry, inequalities)
        return equalities.build_optional(), inequalities.build_required(), inequalities.names

    def _balance_constraints(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        rows: _RowBuilder,
    ) -> None:
        for zone_id in data.zone_ids:
            for t in range(data.periods):
                coefficients: dict[int, float] = {
                    registry.at("unserved_mw", t, asset_id=zone_id): 1.0,
                }
                for candidate in data.generation_candidates:
                    if candidate.zone_id == zone_id:
                        coefficients[registry.at("generation_mw", t, asset_id=candidate.id)] = 1.0
                for storage in data.storage_candidates:
                    if storage.zone_id == zone_id:
                        coefficients[
                            registry.at("storage_discharge_mw", t, asset_id=storage.id)
                        ] = 1.0
                        coefficients[registry.at("storage_charge_mw", t, asset_id=storage.id)] = (
                            -1.0
                        )
                for interconnector in data.interconnector_candidates:
                    if interconnector.zone_id == zone_id:
                        coefficients[
                            registry.at(
                                "interconnector_import_mw",
                                t,
                                asset_id=interconnector.id,
                            )
                        ] = 1.0
                for line in data.transmission_candidates:
                    if line.from_zone_id == zone_id:
                        coefficients[registry.at("line_flow_mw", t, asset_id=line.id)] = -1.0
                    elif line.to_zone_id == zone_id:
                        coefficients[registry.at("line_flow_mw", t, asset_id=line.id)] = 1.0
                rows.add(coefficients, data.demand_by_zone[zone_id][t], f"balance[{zone_id}]")

    def _capacity_constraints(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        rows: _RowBuilder,
    ) -> None:
        for candidate in data.generation_candidates:
            profile = data.generation_profiles[candidate.id]
            build = registry.at("generation_build_mw", asset_id=candidate.id)
            for t in range(data.periods):
                rows.add(
                    {
                        registry.at("generation_mw", t, asset_id=candidate.id): 1.0,
                        build: -profile[t],
                    },
                    candidate.existing_capacity_mw * profile[t],
                    f"generation_capacity[{candidate.id}]",
                )
        for interconnector in data.interconnector_candidates:
            profile = data.interconnector_profiles[interconnector.id]
            build = registry.at("interconnector_build_mw", asset_id=interconnector.id)
            for t in range(data.periods):
                rows.add(
                    {
                        registry.at("interconnector_import_mw", t, asset_id=interconnector.id): 1.0,
                        build: -profile[t],
                    },
                    interconnector.existing_capacity_mw * profile[t],
                    f"interconnector_capacity[{interconnector.id}]",
                )
        for line in data.transmission_candidates:
            build = registry.at("transmission_build_mw", asset_id=line.id)
            for t in range(data.periods):
                flow = registry.at("line_flow_mw", t, asset_id=line.id)
                rows.add(
                    {flow: 1.0, build: -1.0},
                    line.existing_capacity_mw,
                    f"transmission_capacity[{line.id}]",
                )
                rows.add(
                    {flow: -1.0, build: -1.0},
                    line.existing_capacity_mw,
                    f"transmission_capacity[{line.id}]",
                )

    def _storage_constraints(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        equalities: _RowBuilder,
        inequalities: _RowBuilder,
    ) -> None:
        for storage in data.storage_candidates:
            power_build = registry.at("storage_power_build_mw", asset_id=storage.id)
            energy_build = registry.at("storage_energy_build_mwh", asset_id=storage.id)
            for t in range(data.periods):
                charge = registry.at("storage_charge_mw", t, asset_id=storage.id)
                discharge = registry.at("storage_discharge_mw", t, asset_id=storage.id)
                soc = registry.at("storage_soc_mwh", t, asset_id=storage.id)
                inequalities.add(
                    {charge: 1.0, power_build: -1.0},
                    storage.existing_power_mw,
                    f"storage_charge_power[{storage.id}]",
                )
                inequalities.add(
                    {discharge: 1.0, power_build: -1.0},
                    storage.existing_power_mw,
                    f"storage_discharge_power[{storage.id}]",
                )
                inequalities.add(
                    {soc: 1.0, energy_build: -1.0},
                    storage.existing_energy_mwh,
                    f"storage_energy[{storage.id}]",
                )
            for block in data.blocks:
                previous_soc: int | None = None
                for t in range(block.start_period, block.end_period):
                    soc = registry.at("storage_soc_mwh", t, asset_id=storage.id)
                    charge = registry.at("storage_charge_mw", t, asset_id=storage.id)
                    discharge = registry.at("storage_discharge_mw", t, asset_id=storage.id)
                    coefficients = {
                        soc: 1.0,
                        charge: -storage.charge_efficiency,
                        discharge: 1.0 / storage.discharge_efficiency,
                    }
                    if previous_soc is not None:
                        coefficients[previous_soc] = -1.0
                    equalities.add(coefficients, 0.0, f"storage_balance[{storage.id}]")
                    previous_soc = soc
                if previous_soc is not None:
                    equalities.add(
                        {previous_soc: 1.0},
                        0.0,
                        f"storage_terminal[{storage.id}]",
                    )

    def _policy_constraints(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        rows: _RowBuilder,
    ) -> None:
        policy = data.problem.policy
        total_demand_mwh = data.weighted_demand_mwh
        if policy.renewable_share_min is not None:
            coefficients: dict[int, float] = {}
            for candidate in data.generation_candidates:
                if candidate.technology not in {"solar", "wind"}:
                    continue
                for t in range(data.periods):
                    coefficients[registry.at("generation_mw", t, asset_id=candidate.id)] = (
                        -data.weights[t]
                    )
            rows.add(
                coefficients,
                -policy.renewable_share_min * total_demand_mwh,
                "policy_renewable_share_min",
            )
        if policy.emissions_cap_tonnes is not None:
            rows.add(
                self._emission_coefficients(data, registry),
                policy.emissions_cap_tonnes,
                "policy_emissions_cap",
            )
        if policy.planning_reserve_margin_fraction is not None:
            coefficients = self._firm_capacity_coefficients(data, registry)
            rows.add(
                coefficients,
                -data.peak_demand_mw * (1.0 + policy.planning_reserve_margin_fraction),
                "policy_planning_reserve_margin",
            )
        for technology, maximum in policy.max_build_mw_by_technology.items():
            coefficients = {
                registry.at("generation_build_mw", asset_id=candidate.id): 1.0
                for candidate in data.generation_candidates
                if candidate.technology == technology
            }
            rows.add(coefficients, maximum, f"policy_max_build[{technology}]")
        if policy.minimum_domestic_generation_share is not None:
            coefficients = {}
            for candidate in data.generation_candidates:
                for t in range(data.periods):
                    coefficients[registry.at("generation_mw", t, asset_id=candidate.id)] = (
                        -data.weights[t]
                    )
            rows.add(
                coefficients,
                -policy.minimum_domestic_generation_share * total_demand_mwh,
                "policy_minimum_domestic_generation",
            )

    def _emission_coefficients(
        self,
        data: _PreparedProblem,
        registry: _Registry,
    ) -> dict[int, float]:
        coefficients: dict[int, float] = {}
        for candidate in data.generation_candidates:
            for t in range(data.periods):
                coefficients[registry.at("generation_mw", t, asset_id=candidate.id)] = (
                    candidate.emission_tonnes_per_mwh * data.weights[t]
                )
        for interconnector in data.interconnector_candidates:
            for t in range(data.periods):
                coefficients[
                    registry.at("interconnector_import_mw", t, asset_id=interconnector.id)
                ] = interconnector.emission_tonnes_per_mwh * data.weights[t]
        return coefficients

    def _firm_capacity_coefficients(
        self,
        data: _PreparedProblem,
        registry: _Registry,
    ) -> dict[int, float]:
        coefficients: dict[int, float] = {}
        existing_firm = 0.0
        for candidate in data.generation_candidates:
            coefficients[registry.at("generation_build_mw", asset_id=candidate.id)] = (
                -candidate.capacity_credit
            )
            existing_firm += candidate.existing_capacity_mw * candidate.capacity_credit
        for storage in data.storage_candidates:
            coefficients[registry.at("storage_power_build_mw", asset_id=storage.id)] = (
                -storage.capacity_credit
            )
            existing_firm += storage.existing_power_mw * storage.capacity_credit
        for interconnector in data.interconnector_candidates:
            coefficients[registry.at("interconnector_build_mw", asset_id=interconnector.id)] = (
                -interconnector.capacity_credit
            )
            existing_firm += interconnector.existing_capacity_mw * interconnector.capacity_credit
        coefficients[-1] = -existing_firm
        return coefficients

    def _result(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        solution: FloatArray,
        objective_eur: float,
        constant_costs: dict[str, float],
        inequality_names: list[str],
        inequality_marginals: FloatArray,
        solver_message: str,
    ) -> CapacityExpansionResult:
        generation_build = {
            candidate.id: float(solution[registry.at("generation_build_mw", asset_id=candidate.id)])
            for candidate in data.generation_candidates
        }
        storage_power = {
            storage.id: float(solution[registry.at("storage_power_build_mw", asset_id=storage.id)])
            for storage in data.storage_candidates
        }
        storage_energy = {
            storage.id: float(
                solution[registry.at("storage_energy_build_mwh", asset_id=storage.id)]
            )
            for storage in data.storage_candidates
        }
        interconnector_build = {
            candidate.id: float(
                solution[registry.at("interconnector_build_mw", asset_id=candidate.id)]
            )
            for candidate in data.interconnector_candidates
        }
        transmission_build = {
            line.id: float(solution[registry.at("transmission_build_mw", asset_id=line.id)])
            for line in data.transmission_candidates
        }
        dispatch = self._dispatch_frame(data, registry, solution)
        generation_mix = {
            candidate.id: float(
                np.dot(
                    registry.values(solution, "generation_mw", asset_id=candidate.id),
                    data.weights,
                )
            )
            for candidate in data.generation_candidates
        }
        curtailment = self._curtailment(data, generation_build, generation_mix)
        emissions = self._emissions(data, registry, solution)
        unserved = float(
            sum(
                np.dot(registry.values(solution, "unserved_mw", asset_id=zone_id), data.weights)
                for zone_id in data.zone_ids
            )
        )
        annual_costs = self._annual_costs(
            data,
            registry,
            solution,
            generation_build,
            storage_power,
            storage_energy,
            interconnector_build,
            transmission_build,
            constant_costs,
        )
        return CapacityExpansionResult(
            selected_generation_capacity_mw=generation_build,
            selected_storage_power_mw=storage_power,
            selected_storage_energy_mwh=storage_energy,
            selected_interconnector_capacity_mw=interconnector_build,
            selected_transmission_capacity_mw=transmission_build,
            dispatch=dispatch,
            annual_costs_eur=annual_costs,
            generation_mix_mwh=generation_mix,
            curtailment_mwh=curtailment,
            emissions_tonnes=emissions,
            unserved_energy_mwh=unserved,
            planning_reserve_margin=self._planning_reserve_margin(data, generation_build),
            policy_shadow_prices=self._policy_shadow_prices(
                inequality_names,
                inequality_marginals,
            ),
            objective_eur=objective_eur,
            solver_message=solver_message,
        )

    def _dispatch_frame(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        solution: FloatArray,
    ) -> pd.DataFrame:
        frame = pd.DataFrame({"period": np.arange(data.periods), "weight_hours": data.weights})
        for zone_id in data.zone_ids:
            frame[f"demand_mw__{zone_id}"] = data.demand_by_zone[zone_id]
            frame[f"unserved_mw__{zone_id}"] = registry.values(
                solution,
                "unserved_mw",
                asset_id=zone_id,
            )
        for candidate in data.generation_candidates:
            frame[f"generation_mw__{candidate.id}"] = registry.values(
                solution,
                "generation_mw",
                asset_id=candidate.id,
            )
        for storage in data.storage_candidates:
            for block in ("storage_charge_mw", "storage_discharge_mw", "storage_soc_mwh"):
                frame[f"{block}__{storage.id}"] = registry.values(
                    solution,
                    block,
                    asset_id=storage.id,
                )
        for interconnector in data.interconnector_candidates:
            frame[f"interconnector_import_mw__{interconnector.id}"] = registry.values(
                solution,
                "interconnector_import_mw",
                asset_id=interconnector.id,
            )
        for line in data.transmission_candidates:
            frame[f"line_flow_mw__{line.id}"] = registry.values(
                solution,
                "line_flow_mw",
                asset_id=line.id,
            )
        return frame

    def _curtailment(
        self,
        data: _PreparedProblem,
        generation_build: dict[str, float],
        generation_mix: dict[str, float],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for candidate in data.generation_candidates:
            if candidate.technology not in {"solar", "wind"}:
                continue
            capacity = candidate.existing_capacity_mw + generation_build[candidate.id]
            potential = float(
                np.dot(data.generation_profiles[candidate.id] * capacity, data.weights)
            )
            result[candidate.id] = max(0.0, potential - generation_mix[candidate.id])
        return result

    def _emissions(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        solution: FloatArray,
    ) -> float:
        total = 0.0
        for candidate in data.generation_candidates:
            generation = registry.values(solution, "generation_mw", asset_id=candidate.id)
            total += float(np.dot(generation, data.weights) * candidate.emission_tonnes_per_mwh)
        for interconnector in data.interconnector_candidates:
            imports = registry.values(
                solution,
                "interconnector_import_mw",
                asset_id=interconnector.id,
            )
            total += float(np.dot(imports, data.weights) * interconnector.emission_tonnes_per_mwh)
        return total

    def _annual_costs(
        self,
        data: _PreparedProblem,
        registry: _Registry,
        solution: FloatArray,
        generation_build: dict[str, float],
        storage_power: dict[str, float],
        storage_energy: dict[str, float],
        interconnector_build: dict[str, float],
        transmission_build: dict[str, float],
        constant_costs: dict[str, float],
    ) -> dict[str, float]:
        capital = 0.0
        fixed_om = constant_costs["existing_fixed_om_cost_eur"]
        variable = 0.0
        fuel = 0.0
        carbon = 0.0
        policy = data.problem.policy
        for candidate in data.generation_candidates:
            build = generation_build[candidate.id]
            capital += build * candidate.annualized_capital_cost_eur_per_mw_year
            fixed_om += build * candidate.fixed_om_cost_eur_per_mw_year
            generation = registry.values(solution, "generation_mw", asset_id=candidate.id)
            weighted_generation = float(np.dot(generation, data.weights))
            variable += weighted_generation * candidate.variable_cost_eur_per_mwh
            fuel += weighted_generation * candidate.fuel_cost_eur_per_mwh
            carbon += (
                weighted_generation
                * candidate.emission_tonnes_per_mwh
                * policy.carbon_price_eur_per_tonne
            )
        for storage in data.storage_candidates:
            capital += (
                storage_power[storage.id] * storage.annualized_power_cost_eur_per_mw_year
                + storage_energy[storage.id] * storage.annualized_energy_cost_eur_per_mwh_year
            )
            fixed_om += storage_power[storage.id] * storage.fixed_om_cost_eur_per_mw_year
            charge = registry.values(solution, "storage_charge_mw", asset_id=storage.id)
            discharge = registry.values(solution, "storage_discharge_mw", asset_id=storage.id)
            variable += float(np.dot(charge + discharge, data.weights)) * (
                storage.variable_cost_eur_per_mwh
            )
        for interconnector in data.interconnector_candidates:
            build = interconnector_build[interconnector.id]
            capital += build * interconnector.annualized_capital_cost_eur_per_mw_year
            fixed_om += build * interconnector.fixed_om_cost_eur_per_mw_year
            imports = registry.values(
                solution,
                "interconnector_import_mw",
                asset_id=interconnector.id,
            )
            weighted_imports = float(np.dot(imports, data.weights))
            variable += weighted_imports * interconnector.variable_cost_eur_per_mwh
            carbon += (
                weighted_imports
                * interconnector.emission_tonnes_per_mwh
                * policy.carbon_price_eur_per_tonne
            )
        for line in data.transmission_candidates:
            build = transmission_build[line.id]
            capital += build * line.annualized_capital_cost_eur_per_mw_year
            fixed_om += build * line.fixed_om_cost_eur_per_mw_year
        reliability = float(
            sum(
                np.dot(registry.values(solution, "unserved_mw", asset_id=zone_id), data.weights)
                for zone_id in data.zone_ids
            )
            * data.problem.reliability_penalty_eur_per_mwh
        )
        total = capital + fixed_om + variable + fuel + carbon + reliability
        return {
            "annualized_capital_cost_eur": capital,
            "fixed_om_cost_eur": fixed_om,
            "variable_operation_cost_eur": variable,
            "fuel_cost_eur": fuel,
            "carbon_cost_eur": carbon,
            "reliability_penalty_eur": reliability,
            "total_annual_cost_eur": total,
        }

    def _planning_reserve_margin(
        self,
        data: _PreparedProblem,
        generation_build: dict[str, float],
    ) -> float | None:
        if data.peak_demand_mw <= 0.0:
            return None
        firm = 0.0
        for candidate in data.generation_candidates:
            firm += (
                candidate.existing_capacity_mw + generation_build[candidate.id]
            ) * candidate.capacity_credit
        return firm / data.peak_demand_mw - 1.0

    @staticmethod
    def _policy_shadow_prices(
        inequality_names: list[str],
        marginals: FloatArray,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, marginal in zip(inequality_names, marginals, strict=True):
            if name.startswith("policy_"):
                result[name] = float(marginal)
        return result


class _RowBuilder:
    def __init__(self, variable_count: int) -> None:
        self.variable_count = variable_count
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.rhs: list[float] = []
        self.names: list[str] = []

    def add(self, coefficients: dict[int, float], rhs: float, name: str) -> None:
        row = len(self.rhs)
        constant = coefficients.pop(-1, 0.0)
        for column, value in coefficients.items():
            if value != 0.0:
                self.rows.append(row)
                self.columns.append(column)
                self.values.append(value)
        self.rhs.append(rhs - constant)
        self.names.append(name)

    def build_optional(self) -> tuple[coo_matrix | None, FloatArray | None]:
        if not self.rhs:
            return None, None
        return self._matrix(), np.asarray(self.rhs, dtype=np.float64)

    def build_required(self) -> tuple[coo_matrix, FloatArray]:
        return self._matrix(), np.asarray(self.rhs, dtype=np.float64)

    def _matrix(self) -> coo_matrix:
        return coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(len(self.rhs), self.variable_count),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class _PreparedProblem:
    problem: CapacityExpansionProblem
    demand_by_zone: dict[str, FloatArray]
    weights: FloatArray
    periods: int
    zone_ids: tuple[str, ...]
    blocks: tuple[PlanningBlock, ...]
    generation_candidates: tuple[GenerationCandidate, ...]
    storage_candidates: tuple[StorageCandidate, ...]
    interconnector_candidates: tuple[InterconnectorCandidate, ...]
    transmission_candidates: tuple[TransmissionCandidate, ...]
    generation_profiles: dict[str, FloatArray]
    interconnector_profiles: dict[str, FloatArray]
    weighted_demand_mwh: float
    peak_demand_mw: float

    @classmethod
    def from_problem(cls, problem: CapacityExpansionProblem) -> _PreparedProblem:
        weights = _as_vector(problem.representative_weights_hours, "representative_weights_hours")
        if np.any(weights <= 0.0):
            raise ValueError("Representative weights must be positive hours")
        if not np.isclose(weights.sum(), problem.annual_hours, rtol=0.0, atol=1e-6):
            raise ValueError("Representative weights must sum to annual_hours")
        demand_by_zone = {
            zone_id: _as_vector(values, f"demand_mw[{zone_id}]")
            for zone_id, values in problem.demand_mw.items()
        }
        if not demand_by_zone:
            raise ValueError("At least one demand zone is required")
        periods = weights.size
        for zone_id, demand in demand_by_zone.items():
            if demand.shape != (periods,):
                raise ValueError(f"Demand profile for {zone_id} must match representative weights")
            if np.any(demand < 0.0):
                raise ValueError(f"Demand profile for {zone_id} must be non-negative")
        blocks = problem.blocks or (PlanningBlock("year", 0, periods),)
        _validate_blocks(blocks, periods)
        zone_ids = tuple(sorted(demand_by_zone))
        generation_profiles = {
            candidate.id: _availability(candidate.availability_profile, periods, candidate.id)
            for candidate in problem.generation_candidates
        }
        interconnector_profiles = {
            candidate.id: _availability(candidate.availability_profile, periods, candidate.id)
            for candidate in problem.interconnector_candidates
        }
        _validate_candidates(problem)
        weighted_demand = float(
            sum(np.dot(values, weights) for values in demand_by_zone.values())
        )
        aggregate_demand = np.sum(
            np.vstack(list(demand_by_zone.values())),
            axis=0,
            dtype=np.float64,
        )
        peak_demand = float(aggregate_demand.max())
        return cls(
            problem=problem,
            demand_by_zone=demand_by_zone,
            weights=weights,
            periods=periods,
            zone_ids=zone_ids,
            blocks=blocks,
            generation_candidates=problem.generation_candidates,
            storage_candidates=problem.storage_candidates,
            interconnector_candidates=problem.interconnector_candidates,
            transmission_candidates=problem.transmission_candidates,
            generation_profiles=generation_profiles,
            interconnector_profiles=interconnector_profiles,
            weighted_demand_mwh=weighted_demand,
            peak_demand_mw=peak_demand,
        )


def _as_vector(values: npt.ArrayLike, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return result


def _availability(values: npt.ArrayLike | None, periods: int, asset_id: str) -> FloatArray:
    if values is None:
        return np.ones(periods, dtype=np.float64)
    result = _as_vector(values, f"availability_profile[{asset_id}]")
    if result.shape != (periods,):
        raise ValueError(f"Availability profile for {asset_id} must match representative weights")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"Availability profile for {asset_id} must be in [0, 1]")
    return result


def _validate_blocks(blocks: tuple[PlanningBlock, ...], periods: int) -> None:
    covered: list[int] = []
    for block in blocks:
        if (
            block.start_period < 0
            or block.end_period > periods
            or block.start_period >= block.end_period
        ):
            raise ValueError(f"Invalid representative block {block.id}")
        covered.extend(range(block.start_period, block.end_period))
    if sorted(covered) != list(range(periods)):
        raise ValueError("Representative blocks must cover each period exactly once")


def _validate_candidates(problem: CapacityExpansionProblem) -> None:
    ids: set[str] = set()
    for candidate_id in (
        *(candidate.id for candidate in problem.generation_candidates),
        *(candidate.id for candidate in problem.storage_candidates),
        *(candidate.id for candidate in problem.interconnector_candidates),
        *(candidate.id for candidate in problem.transmission_candidates),
    ):
        if candidate_id in ids:
            raise ValueError(f"Duplicate planning candidate id: {candidate_id}")
        ids.add(candidate_id)
    for candidate in problem.generation_candidates:
        _nonnegative(candidate.existing_capacity_mw, f"{candidate.id}.existing_capacity_mw")
        _nonnegative(candidate.max_build_mw, f"{candidate.id}.max_build_mw")
        _nonnegative(candidate.capacity_credit, f"{candidate.id}.capacity_credit")
    for storage in problem.storage_candidates:
        _nonnegative(storage.existing_power_mw, f"{storage.id}.existing_power_mw")
        _nonnegative(storage.existing_energy_mwh, f"{storage.id}.existing_energy_mwh")
        _nonnegative(storage.max_power_build_mw, f"{storage.id}.max_power_build_mw")
        _nonnegative(storage.max_energy_build_mwh, f"{storage.id}.max_energy_build_mwh")
        if storage.charge_efficiency <= 0.0 or storage.discharge_efficiency <= 0.0:
            raise ValueError(f"{storage.id} storage efficiencies must be positive")
    for interconnector in problem.interconnector_candidates:
        _nonnegative(
            interconnector.existing_capacity_mw,
            f"{interconnector.id}.existing_capacity_mw",
        )
        _nonnegative(interconnector.max_build_mw, f"{interconnector.id}.max_build_mw")
    for line in problem.transmission_candidates:
        _nonnegative(line.existing_capacity_mw, f"{line.id}.existing_capacity_mw")
        _nonnegative(line.max_build_mw, f"{line.id}.max_build_mw")


def _nonnegative(value: float, name: str) -> None:
    if value < 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and non-negative")
