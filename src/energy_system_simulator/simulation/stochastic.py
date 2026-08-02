from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_system_simulator.config import ModelConfig
from energy_system_simulator.dispatch import DispatchResult, UnitCommitment

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class StochasticScenario:
    """One explicit uncertainty scenario for stochastic dispatch."""

    id: str
    probability: float
    renewable_available_mw: npt.ArrayLike
    gross_demand_mw: npt.ArrayLike
    import_price_eur_per_mwh: npt.ArrayLike | None = None
    thermal_availability_factors: dict[str, npt.ArrayLike] | None = None
    storage_availability_factors: dict[str, npt.ArrayLike] | None = None
    line_availability_factors: dict[str, npt.ArrayLike] | None = None
    import_availability_factors: npt.ArrayLike | None = None
    hydro_inflows_mw: dict[str, npt.ArrayLike] | None = None


@dataclass(frozen=True)
class SyntheticScenarioConfig:
    count: int
    seed: int
    demand_multiplier_std: float = 0.0
    renewable_multiplier_std: float = 0.0
    import_price_multiplier_std: float = 0.0


@dataclass(frozen=True)
class StochasticDispatchConfig:
    commitment_horizon_periods: int
    probability_tolerance: float = 1e-8
    cvar_confidence_level: float | None = None
    cvar_weight: float = 0.0


@dataclass(frozen=True)
class StochasticScenarioResult:
    id: str
    probability: float
    objective_eur: float
    first_stage_commitment_by_unit: dict[str, tuple[int, ...]]
    frame: pd.DataFrame


@dataclass(frozen=True)
class StochasticBenchmarks:
    expected_value_solution_objective_eur: float
    expected_result_using_expected_value_solution_eur: float
    wait_and_see_expected_objective_eur: float
    value_of_stochastic_solution_eur: float
    expected_value_of_perfect_information_eur: float


@dataclass(frozen=True)
class StochasticDispatchResult:
    expected_objective_eur: float
    risk_adjusted_objective_eur: float
    cvar_eur: float | None
    first_stage_commitment_by_unit: dict[str, tuple[int, ...]]
    scenario_results: tuple[StochasticScenarioResult, ...]
    scenario_cost_distribution_eur: dict[str, float]
    benchmarks: StochasticBenchmarks
    candidate_count: int


@dataclass(frozen=True)
class _ResolvedScenario:
    id: str
    probability: float
    renewable_available_mw: FloatArray
    gross_demand_mw: FloatArray
    import_price_eur_per_mwh: FloatArray
    thermal_availability_factors: dict[str, FloatArray]
    storage_availability_factors: dict[str, FloatArray]
    line_availability_factors: dict[str, FloatArray]
    import_availability_factors: FloatArray | None
    hydro_inflows_mw: dict[str, FloatArray]


class StochasticDispatch:
    """Scenario-based dispatch with shared first-stage thermal commitments."""

    def __init__(
        self,
        model_config: ModelConfig,
        scenarios: tuple[StochasticScenario, ...],
        config: StochasticDispatchConfig,
    ) -> None:
        self.model_config = model_config
        self.config = config
        self.scenarios = self._resolve_scenarios(scenarios)

    def run(self) -> StochasticDispatchResult:
        expected_scenario = self._expected_scenario()
        expected_value_result = self._solve(expected_scenario)
        expected_value_commitment = self._first_stage_commitment(expected_value_result)

        wait_and_see = {scenario.id: self._solve(scenario) for scenario in self.scenarios}
        candidate_commitments = self._candidate_commitments(
            expected_value_commitment,
            [self._first_stage_commitment(result) for result in wait_and_see.values()],
        )

        evaluated = [self._evaluate_commitment(candidate) for candidate in candidate_commitments]
        best = min(evaluated, key=lambda item: (item[0], item[1]))
        risk_adjusted, expected_objective, cvar, commitment, scenario_results = best

        expected_value_evaluation = self._evaluate_commitment(expected_value_commitment)
        wait_and_see_expected = float(
            sum(
                scenario.probability * wait_and_see[scenario.id].objective_eur
                for scenario in self.scenarios
            )
        )
        benchmarks = StochasticBenchmarks(
            expected_value_solution_objective_eur=expected_value_result.objective_eur,
            expected_result_using_expected_value_solution_eur=expected_value_evaluation[1],
            wait_and_see_expected_objective_eur=wait_and_see_expected,
            value_of_stochastic_solution_eur=max(
                0.0, expected_value_evaluation[1] - expected_objective
            ),
            expected_value_of_perfect_information_eur=max(
                0.0, expected_objective - wait_and_see_expected
            ),
        )
        return StochasticDispatchResult(
            expected_objective_eur=expected_objective,
            risk_adjusted_objective_eur=risk_adjusted,
            cvar_eur=cvar,
            first_stage_commitment_by_unit=commitment,
            scenario_results=tuple(scenario_results),
            scenario_cost_distribution_eur={
                result.id: result.objective_eur for result in scenario_results
            },
            benchmarks=benchmarks,
            candidate_count=len(candidate_commitments),
        )

    def _resolve_scenarios(
        self,
        scenarios: tuple[StochasticScenario, ...],
    ) -> tuple[_ResolvedScenario, ...]:
        if not scenarios:
            raise ValueError("At least one stochastic scenario is required")
        ids = [scenario.id for scenario in scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("Stochastic scenario IDs must be unique")
        probabilities = np.asarray(
            [scenario.probability for scenario in scenarios], dtype=np.float64
        )
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("Scenario probabilities must be non-negative finite values")
        total_probability = float(probabilities.sum())
        if abs(total_probability - 1.0) > self.config.probability_tolerance:
            raise ValueError("Scenario probabilities must sum to one")
        if self.config.commitment_horizon_periods <= 0:
            raise ValueError("commitment_horizon_periods must be positive")
        if self.config.cvar_confidence_level is not None and not (
            0.0 < self.config.cvar_confidence_level < 1.0
        ):
            raise ValueError("cvar_confidence_level must be in (0, 1)")
        if self.config.cvar_weight < 0.0:
            raise ValueError("cvar_weight must be non-negative")

        first_renewable = _as_vector(scenarios[0].renewable_available_mw, "renewable_available_mw")
        first_demand = _as_vector(scenarios[0].gross_demand_mw, "gross_demand_mw")
        if first_renewable.shape != first_demand.shape:
            raise ValueError("Scenario renewable and demand arrays must have the same shape")
        periods = first_renewable.size
        if self.config.commitment_horizon_periods > periods:
            raise ValueError("commitment_horizon_periods cannot exceed scenario horizon")

        resolved: list[_ResolvedScenario] = []
        for scenario in scenarios:
            renewable = _as_vector(scenario.renewable_available_mw, "renewable_available_mw")
            demand = _as_vector(scenario.gross_demand_mw, "gross_demand_mw")
            if renewable.shape != (periods,) or demand.shape != (periods,):
                raise ValueError("All stochastic scenarios must share the same horizon")
            import_prices = (
                np.full(periods, self.model_config.imports.price_eur_per_mwh, dtype=np.float64)
                if scenario.import_price_eur_per_mwh is None
                else _as_vector(scenario.import_price_eur_per_mwh, "import_price_eur_per_mwh")
            )
            if import_prices.shape != (periods,):
                raise ValueError("Import price series must match scenario horizon")
            resolved.append(
                _ResolvedScenario(
                    id=scenario.id,
                    probability=scenario.probability,
                    renewable_available_mw=renewable,
                    gross_demand_mw=demand,
                    import_price_eur_per_mwh=import_prices,
                    thermal_availability_factors=_as_factor_mapping(
                        scenario.thermal_availability_factors or {},
                        periods,
                        "thermal_availability_factors",
                    ),
                    storage_availability_factors=_as_factor_mapping(
                        scenario.storage_availability_factors or {},
                        periods,
                        "storage_availability_factors",
                    ),
                    line_availability_factors=_as_factor_mapping(
                        scenario.line_availability_factors or {},
                        periods,
                        "line_availability_factors",
                    ),
                    import_availability_factors=(
                        None
                        if scenario.import_availability_factors is None
                        else _as_factor(
                            scenario.import_availability_factors,
                            periods,
                            "import_availability_factors",
                        )
                    ),
                    hydro_inflows_mw=_as_nonnegative_mapping(
                        scenario.hydro_inflows_mw or {},
                        periods,
                        "hydro_inflows_mw",
                    ),
                )
            )
        return tuple(resolved)

    def _solve(
        self,
        scenario: _ResolvedScenario,
        fixed_commitment: dict[str, tuple[int, ...]] | None = None,
    ) -> DispatchResult:
        return UnitCommitment(self.model_config).solve(
            scenario.renewable_available_mw,
            scenario.gross_demand_mw,
            thermal_availability_factors=scenario.thermal_availability_factors,
            storage_availability_factors=scenario.storage_availability_factors,
            hydro_inflows_mw=scenario.hydro_inflows_mw,
            line_availability_factors=scenario.line_availability_factors,
            import_availability_factors=scenario.import_availability_factors,
            import_price_series=scenario.import_price_eur_per_mwh,
            fixed_thermal_commitment=(
                None if fixed_commitment is None else _commitment_arrays(fixed_commitment)
            ),
        )

    def _evaluate_commitment(
        self,
        commitment: dict[str, tuple[int, ...]],
    ) -> tuple[
        float,
        float,
        float | None,
        dict[str, tuple[int, ...]],
        list[StochasticScenarioResult],
    ]:
        scenario_results: list[StochasticScenarioResult] = []
        for scenario in self.scenarios:
            dispatch = self._solve(scenario, commitment)
            scenario_results.append(
                StochasticScenarioResult(
                    id=scenario.id,
                    probability=scenario.probability,
                    objective_eur=dispatch.objective_eur,
                    first_stage_commitment_by_unit=self._first_stage_commitment(dispatch),
                    frame=dispatch.frame,
                )
            )
        expected = float(
            sum(result.probability * result.objective_eur for result in scenario_results)
        )
        cvar = self._cvar({result.id: result.objective_eur for result in scenario_results})
        risk_adjusted = expected + self.config.cvar_weight * (cvar if cvar is not None else 0.0)
        return risk_adjusted, expected, cvar, commitment, scenario_results

    def _expected_scenario(self) -> _ResolvedScenario:
        return _ResolvedScenario(
            id="expected-value",
            probability=1.0,
            renewable_available_mw=_weighted_vector(
                [scenario.renewable_available_mw for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
            ),
            gross_demand_mw=_weighted_vector(
                [scenario.gross_demand_mw for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
            ),
            import_price_eur_per_mwh=_weighted_vector(
                [scenario.import_price_eur_per_mwh for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
            ),
            thermal_availability_factors=_weighted_factor_mapping(
                [scenario.thermal_availability_factors for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
                len(self.scenarios[0].renewable_available_mw),
            ),
            storage_availability_factors=_weighted_factor_mapping(
                [scenario.storage_availability_factors for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
                len(self.scenarios[0].renewable_available_mw),
            ),
            line_availability_factors=_weighted_factor_mapping(
                [scenario.line_availability_factors for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
                len(self.scenarios[0].renewable_available_mw),
            ),
            import_availability_factors=_weighted_optional_factor(
                [scenario.import_availability_factors for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
                len(self.scenarios[0].renewable_available_mw),
            ),
            hydro_inflows_mw=_weighted_nonnegative_mapping(
                [scenario.hydro_inflows_mw for scenario in self.scenarios],
                [scenario.probability for scenario in self.scenarios],
                len(self.scenarios[0].renewable_available_mw),
            ),
        )

    def _first_stage_commitment(self, dispatch: DispatchResult) -> dict[str, tuple[int, ...]]:
        horizon = self.config.commitment_horizon_periods
        commitment: dict[str, tuple[int, ...]] = {}
        for unit in self.model_config.portfolio.thermal_generators:
            column = f"thermal_on__{unit.id}"
            values = dispatch.frame[column].iloc[:horizon].to_numpy(dtype=np.int64)
            commitment[unit.id] = tuple(int(value) for value in values)
        return commitment

    def _candidate_commitments(
        self,
        expected_value_commitment: dict[str, tuple[int, ...]],
        wait_and_see_commitments: list[dict[str, tuple[int, ...]]],
    ) -> list[dict[str, tuple[int, ...]]]:
        candidates = [expected_value_commitment, *wait_and_see_commitments]
        unique: dict[tuple[tuple[str, tuple[int, ...]], ...], dict[str, tuple[int, ...]]] = {}
        for candidate in candidates:
            key = tuple(sorted(candidate.items()))
            unique[key] = candidate
        return list(unique.values())

    def _cvar(self, costs_by_id: dict[str, float]) -> float | None:
        alpha = self.config.cvar_confidence_level
        if alpha is None:
            return None
        probabilities = {scenario.id: scenario.probability for scenario in self.scenarios}
        candidates = sorted(set(costs_by_id.values()))
        scale = 1.0 / (1.0 - alpha)
        return min(
            eta
            + scale
            * sum(
                probabilities[scenario_id] * max(cost - eta, 0.0)
                for scenario_id, cost in costs_by_id.items()
            )
            for eta in candidates
        )


def generate_synthetic_scenarios(
    base_renewable_available_mw: npt.ArrayLike,
    base_gross_demand_mw: npt.ArrayLike,
    config: SyntheticScenarioConfig,
    *,
    base_import_price_eur_per_mwh: npt.ArrayLike | None = None,
) -> tuple[StochasticScenario, ...]:
    """Generate deterministic perturbation scenarios from an explicit seed."""
    if config.count <= 0:
        raise ValueError("Synthetic scenario count must be positive")
    for name, value in (
        ("demand_multiplier_std", config.demand_multiplier_std),
        ("renewable_multiplier_std", config.renewable_multiplier_std),
        ("import_price_multiplier_std", config.import_price_multiplier_std),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    renewable = _as_vector(base_renewable_available_mw, "base_renewable_available_mw")
    demand = _as_vector(base_gross_demand_mw, "base_gross_demand_mw")
    if renewable.shape != demand.shape:
        raise ValueError("Base renewable and demand arrays must have the same shape")
    import_price = (
        None
        if base_import_price_eur_per_mwh is None
        else _as_vector(base_import_price_eur_per_mwh, "base_import_price_eur_per_mwh")
    )
    if import_price is not None and import_price.shape != renewable.shape:
        raise ValueError("Base import price series must match base horizon")

    rng = np.random.default_rng(config.seed)
    probability = 1.0 / config.count
    scenarios: list[StochasticScenario] = []
    for index in range(config.count):
        demand_multiplier = _clipped_multiplier(rng, config.demand_multiplier_std, demand.size)
        renewable_multiplier = _clipped_multiplier(
            rng,
            config.renewable_multiplier_std,
            renewable.size,
        )
        price_multiplier = _clipped_multiplier(
            rng,
            config.import_price_multiplier_std,
            renewable.size,
        )
        scenarios.append(
            StochasticScenario(
                id=f"s{index:03d}",
                probability=probability,
                renewable_available_mw=renewable * renewable_multiplier,
                gross_demand_mw=demand * demand_multiplier,
                import_price_eur_per_mwh=(
                    None if import_price is None else import_price * price_multiplier
                ),
            )
        )
    return tuple(scenarios)


def _as_vector(values: npt.ArrayLike, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain non-negative finite values")
    return result.astype(np.float64)


def _as_factor(values: npt.ArrayLike, periods: int, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (periods,):
        raise ValueError(f"{name} must match scenario horizon")
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return result.astype(np.float64)


def _as_factor_mapping(
    values: dict[str, npt.ArrayLike],
    periods: int,
    name: str,
) -> dict[str, FloatArray]:
    return {
        asset_id: _as_factor(series, periods, f"{name}.{asset_id}")
        for asset_id, series in values.items()
    }


def _as_nonnegative_mapping(
    values: dict[str, npt.ArrayLike],
    periods: int,
    name: str,
) -> dict[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    for asset_id, series in values.items():
        vector = _as_vector(series, f"{name}.{asset_id}")
        if vector.shape != (periods,):
            raise ValueError(f"{name}.{asset_id} must match scenario horizon")
        result[asset_id] = vector
    return result


def _weighted_vector(values: list[FloatArray], probabilities: list[float]) -> FloatArray:
    result = np.zeros_like(values[0], dtype=np.float64)
    for probability, value in zip(probabilities, values, strict=True):
        result += probability * value
    return result


def _weighted_factor_mapping(
    values: list[dict[str, FloatArray]],
    probabilities: list[float],
    periods: int,
) -> dict[str, FloatArray]:
    keys = set().union(*(value.keys() for value in values))
    result: dict[str, FloatArray] = {}
    for key in keys:
        weighted = np.zeros(periods, dtype=np.float64)
        for probability, value in zip(probabilities, values, strict=True):
            weighted += probability * value.get(key, np.ones(periods, dtype=np.float64))
        result[key] = weighted
    return result


def _weighted_nonnegative_mapping(
    values: list[dict[str, FloatArray]],
    probabilities: list[float],
    periods: int,
) -> dict[str, FloatArray]:
    keys = set().union(*(value.keys() for value in values))
    result: dict[str, FloatArray] = {}
    for key in keys:
        weighted = np.zeros(periods, dtype=np.float64)
        for probability, value in zip(probabilities, values, strict=True):
            weighted += probability * value.get(key, np.zeros(periods, dtype=np.float64))
        result[key] = weighted
    return result


def _weighted_optional_factor(
    values: list[FloatArray | None],
    probabilities: list[float],
    periods: int,
) -> FloatArray | None:
    if not any(value is not None for value in values):
        return None
    weighted = np.zeros(periods, dtype=np.float64)
    for probability, value in zip(probabilities, values, strict=True):
        weighted += probability * (
            value if value is not None else np.ones(periods, dtype=np.float64)
        )
    return weighted


def _commitment_arrays(commitment: dict[str, tuple[int, ...]]) -> dict[str, FloatArray]:
    return {unit_id: np.asarray(values, dtype=np.float64) for unit_id, values in commitment.items()}


def _clipped_multiplier(
    rng: np.random.Generator,
    standard_deviation: float,
    periods: int,
) -> FloatArray:
    if standard_deviation == 0.0:
        return np.ones(periods, dtype=np.float64)
    return np.maximum(
        rng.normal(loc=1.0, scale=standard_deviation, size=periods),
        0.0,
    ).astype(np.float64)
