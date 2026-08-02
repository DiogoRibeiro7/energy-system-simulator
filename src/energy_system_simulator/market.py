from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_system_simulator.config import ModelConfig
from energy_system_simulator.dispatch.solver import (
    SolverProblem,
    VariableBounds,
    solve_linear_program,
)
from energy_system_simulator.dispatch.unit_commitment import (
    THERMAL_DOWNWARD_RESERVE_BLOCK,
    THERMAL_STARTUP_CATEGORY_BLOCK,
    THERMAL_UPWARD_RESERVE_BLOCK,
    DispatchResult,
    FormulationProblem,
    ThermalUnit,
    _startup_category_asset_id,
)
from energy_system_simulator.exceptions import OptimisationError

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GeneratorSettlement:
    """Per-generator market settlement independent of production-cost accounting."""

    unit_id: str
    energy_revenue_eur: float
    reserve_revenue_eur: float
    variable_cost_eur: float
    startup_cost_eur: float
    no_load_cost_eur: float
    gross_margin_eur: float
    committed_cost_eur: float
    make_whole_payment_eur: float


@dataclass(frozen=True)
class MarketSettlement:
    """Dual-based price outputs and settlement reconciliation."""

    prices: pd.DataFrame
    nodal_prices: pd.DataFrame
    generator_settlements: tuple[GeneratorSettlement, ...]
    consumer_payment_eur: float
    generator_energy_revenue_eur: float
    generator_reserve_revenue_eur: float
    import_energy_revenue_eur: float
    congestion_rent_eur: float
    scarcity_rent_eur: float
    make_whole_payment_eur: float
    uplift_eur: float
    settlement_residual_eur: float
    pricing_objective_eur: float
    pricing_status: str
    dual_sign_convention: str


class MarketAnalyzer:
    """Post-dispatch pricing and settlement calculations."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def settle(
        self,
        problem: FormulationProblem,
        dispatch: DispatchResult,
    ) -> MarketSettlement:
        """Price an accepted dispatch by fixing integer decisions and solving an LP."""
        pricing_solution, row_duals, pricing_objective = self._price_fixed_commitment(
            problem,
            dispatch,
        )
        del pricing_solution
        prices, nodal_prices = self._prices_from_duals(problem, row_duals)
        generator_settlements = self._generator_settlements(problem, dispatch.frame, prices)
        consumer_payment = self._consumer_payment(problem, prices)
        generator_energy = float(
            sum(settlement.energy_revenue_eur for settlement in generator_settlements)
        )
        generator_reserve = float(
            sum(settlement.reserve_revenue_eur for settlement in generator_settlements)
        )
        import_revenue = self._import_energy_revenue(problem, dispatch.frame, prices)
        congestion_rent = self._congestion_rent(problem, dispatch.frame, prices)
        scarcity_rent = self._scarcity_rent(problem, dispatch.frame, prices)
        make_whole = float(
            sum(settlement.make_whole_payment_eur for settlement in generator_settlements)
        )
        residual = (
            consumer_payment - generator_energy - import_revenue - congestion_rent - make_whole
        )
        return MarketSettlement(
            prices=prices,
            nodal_prices=nodal_prices,
            generator_settlements=generator_settlements,
            consumer_payment_eur=consumer_payment,
            generator_energy_revenue_eur=generator_energy,
            generator_reserve_revenue_eur=generator_reserve,
            import_energy_revenue_eur=import_revenue,
            congestion_rent_eur=congestion_rent,
            scarcity_rent_eur=scarcity_rent,
            make_whole_payment_eur=make_whole,
            uplift_eur=make_whole,
            settlement_residual_eur=float(residual),
            pricing_objective_eur=pricing_objective,
            pricing_status="optimal",
            dual_sign_convention=(
                "SciPy HiGHS equality marginals for Ax=b minimization; balance rows are "
                "generation plus unserved energy minus load, so price is marginal divided by "
                "time_step_hours."
            ),
        )

    def _price_fixed_commitment(
        self,
        problem: FormulationProblem,
        dispatch: DispatchResult,
    ) -> tuple[FloatArray, FloatArray, float]:
        fixed_bounds = self._fixed_integer_bounds(problem, dispatch.frame)
        pricing_problem = SolverProblem(
            objective=problem.objective,
            integrality=np.zeros_like(problem.integrality),
            bounds=fixed_bounds,
            constraints=problem.constraints,
            variable_names=tuple(variable.name for variable in problem.variable_metadata),
        )
        result = solve_linear_program(
            pricing_problem,
        )
        if result.status_code != 0 or result.solution is None or result.objective_value is None:
            raise OptimisationError(
                f"Fixed-commitment pricing LP failed with status {result.status_name}: "
                f"{result.message}"
            )
        return result.solution, result.constraint_marginals, float(result.objective_value)

    def _fixed_integer_bounds(
        self,
        problem: FormulationProblem,
        frame: pd.DataFrame,
    ) -> VariableBounds:
        lower = np.asarray(problem.bounds.lower, dtype=np.float64).copy()
        upper = np.asarray(problem.bounds.upper, dtype=np.float64).copy()
        for t in range(problem.gross_demand_mw.size):
            for unit in problem.thermal_units:
                self._fix_binary(lower, upper, problem, frame, "thermal_on", t, unit.id)
                self._fix_binary(lower, upper, problem, frame, "thermal_startup", t, unit.id)
                self._fix_binary(lower, upper, problem, frame, "thermal_shutdown", t, unit.id)
                for category in unit.config.startup_categories:
                    category_asset_id = _startup_category_asset_id(unit.id, category.id)
                    column = f"thermal_startup_category__{unit.id}__{category.id}"
                    self._fix_binary(
                        lower,
                        upper,
                        problem,
                        frame,
                        THERMAL_STARTUP_CATEGORY_BLOCK,
                        t,
                        category_asset_id,
                        column=column,
                    )
            for storage in problem.storage_units:
                self._fix_binary(lower, upper, problem, frame, "storage_charge_mode", t, storage.id)
                self._fix_binary(
                    lower,
                    upper,
                    problem,
                    frame,
                    "storage_discharge_mode",
                    t,
                    storage.id,
                )
        return VariableBounds(lower, upper)

    @staticmethod
    def _fix_binary(
        lower: FloatArray,
        upper: FloatArray,
        problem: FormulationProblem,
        frame: pd.DataFrame,
        block: str,
        period: int,
        asset_id: str,
        *,
        column: str | None = None,
    ) -> None:
        column_name = column or f"{block}__{asset_id}"
        value = float(np.rint(frame[column_name].iloc[period]))
        index = problem.registry.at(block, period, asset_id=asset_id)
        lower[index] = value
        upper[index] = value

    def _prices_from_duals(
        self,
        problem: FormulationProblem,
        row_duals: FloatArray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        periods = problem.gross_demand_mw.size
        dt = self.config.simulation.time_step_hours
        components = np.asarray(problem.constraint_components, dtype=object)
        if problem.network.enabled:
            balance_rows = np.flatnonzero(components == "nodal_balance")
            expected = periods * len(problem.network.bus_ids)
            if balance_rows.size != expected:
                raise OptimisationError("Pricing LP did not expose all nodal balance duals")
            records: list[dict[str, float | int | str]] = []
            system_price = np.zeros(periods, dtype=np.float64)
            slack_bus = problem.network.slack_bus_id or problem.network.bus_ids[0]
            for t in range(periods):
                period_rows = balance_rows[
                    t * len(problem.network.bus_ids) : (t + 1) * len(problem.network.bus_ids)
                ]
                period_prices = row_duals[period_rows] / dt
                energy_component = float(period_prices[problem.network.bus_ids.index(slack_bus)])
                system_price[t] = energy_component
                for bus_id, price in zip(problem.network.bus_ids, period_prices, strict=True):
                    records.append(
                        {
                            "period": t,
                            "bus_id": bus_id,
                            "lmp_eur_per_mwh": float(price),
                            "energy_component_eur_per_mwh": energy_component,
                            "congestion_component_eur_per_mwh": float(price - energy_component),
                        }
                    )
            nodal_prices = pd.DataFrame.from_records(records)
            prices = pd.DataFrame(
                {
                    "period": np.arange(periods),
                    "energy_price_eur_per_mwh": system_price,
                    "system_price_eur_per_mwh": system_price,
                    "scarcity_price_eur_per_mwh": self._scarcity_price_series(system_price),
                }
            )
            return attach_wide_nodal_prices(prices, nodal_prices), nodal_prices

        balance_rows = np.flatnonzero(components == "balance")
        if balance_rows.size != periods:
            raise OptimisationError("Pricing LP did not expose all balance duals")
        energy_price = row_duals[balance_rows] / dt
        prices = pd.DataFrame(
            {
                "period": np.arange(periods),
                "energy_price_eur_per_mwh": energy_price,
                "system_price_eur_per_mwh": energy_price,
                "scarcity_price_eur_per_mwh": self._scarcity_price_series(energy_price),
            }
        )
        return prices, pd.DataFrame(
            columns=[
                "period",
                "bus_id",
                "lmp_eur_per_mwh",
                "energy_component_eur_per_mwh",
                "congestion_component_eur_per_mwh",
            ]
        )

    def _scarcity_price_series(self, price: FloatArray) -> FloatArray:
        periods = price.size
        scarcity = np.zeros(periods, dtype=np.float64)
        cap = self.config.penalties.lost_load_eur_per_mwh
        scarcity[price >= cap - 1e-7] = cap
        return scarcity

    def _generator_settlements(
        self,
        problem: FormulationProblem,
        frame: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> tuple[GeneratorSettlement, ...]:
        return tuple(
            self._generator_settlement(problem, frame, prices, unit)
            for unit in problem.thermal_units
        )

    def _generator_settlement(
        self,
        problem: FormulationProblem,
        frame: pd.DataFrame,
        prices: pd.DataFrame,
        unit: ThermalUnit,
    ) -> GeneratorSettlement:
        dt = self.config.simulation.time_step_hours
        price = self._price_for_asset(problem, prices, unit.id)
        output = frame[f"thermal_output_mw__{unit.id}"].to_numpy(dtype=np.float64)
        energy_revenue = float(np.dot(output, price) * dt)
        reserve_revenue = self._reserve_revenue(frame, unit.id)
        variable_cost = float(frame[f"thermal_variable_cost_eur__{unit.id}"].sum())
        startup_cost = float(frame[f"thermal_startup_cost_eur__{unit.id}"].sum())
        no_load_cost = float(frame[f"thermal_no_load_cost_eur__{unit.id}"].sum())
        committed_cost = variable_cost + startup_cost + no_load_cost
        gross_margin = energy_revenue + reserve_revenue - variable_cost
        make_whole = max(0.0, committed_cost - energy_revenue - reserve_revenue)
        return GeneratorSettlement(
            unit_id=unit.id,
            energy_revenue_eur=energy_revenue,
            reserve_revenue_eur=reserve_revenue,
            variable_cost_eur=variable_cost,
            startup_cost_eur=startup_cost,
            no_load_cost_eur=no_load_cost,
            gross_margin_eur=gross_margin,
            committed_cost_eur=committed_cost,
            make_whole_payment_eur=make_whole,
        )

    def _price_for_asset(
        self,
        problem: FormulationProblem,
        prices: pd.DataFrame,
        asset_id: str,
    ) -> FloatArray:
        if not problem.network.enabled:
            return prices["energy_price_eur_per_mwh"].to_numpy(dtype=np.float64)
        bus_by_thermal = {unit.id: unit.bus_id for unit in self.config.portfolio.thermal_generators}
        bus_id = bus_by_thermal[asset_id]
        return prices[f"lmp_eur_per_mwh__{bus_id}"].to_numpy(dtype=np.float64)

    def _reserve_revenue(self, frame: pd.DataFrame, unit_id: str) -> float:
        dt = self.config.simulation.time_step_hours
        revenue = 0.0
        upward = f"{THERMAL_UPWARD_RESERVE_BLOCK}__{unit_id}"
        downward = f"{THERMAL_DOWNWARD_RESERVE_BLOCK}__{unit_id}"
        if upward in frame:
            revenue += (
                float(frame[upward].sum())
                * dt
                * self.config.reserves.thermal_upward_cost_eur_per_mw_hour
            )
        if downward in frame:
            revenue += (
                float(frame[downward].sum())
                * dt
                * self.config.reserves.thermal_downward_cost_eur_per_mw_hour
            )
        return revenue

    def _consumer_payment(self, problem: FormulationProblem, prices: pd.DataFrame) -> float:
        dt = self.config.simulation.time_step_hours
        if not problem.network.enabled:
            return float(np.dot(problem.gross_demand_mw, prices["energy_price_eur_per_mwh"]) * dt)
        payment = 0.0
        demand_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.demand}
        for demand_unit in problem.demand_units:
            bus_id = demand_bus[demand_unit.id]
            price = prices[f"lmp_eur_per_mwh__{bus_id}"].to_numpy(dtype=np.float64)
            payment += float(np.dot(problem.demand_profiles_mw[demand_unit.id], price) * dt)
        return payment

    def _import_energy_revenue(
        self,
        problem: FormulationProblem,
        frame: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> float:
        dt = self.config.simulation.time_step_hours
        if not problem.network.enabled:
            price = prices["energy_price_eur_per_mwh"].to_numpy(dtype=np.float64)
        else:
            import_bus = self.config.portfolio.imports[0].bus_id
            price = prices[f"lmp_eur_per_mwh__{import_bus}"].to_numpy(dtype=np.float64)
        return float(np.dot(frame["imports_mw"].to_numpy(dtype=np.float64), price) * dt)

    def _congestion_rent(
        self,
        problem: FormulationProblem,
        frame: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> float:
        if not problem.network.enabled:
            return 0.0
        dt = self.config.simulation.time_step_hours
        rent = 0.0
        for line in problem.network.lines:
            from_price = prices[f"lmp_eur_per_mwh__{line.from_bus_id}"].to_numpy(dtype=np.float64)
            to_price = prices[f"lmp_eur_per_mwh__{line.to_bus_id}"].to_numpy(dtype=np.float64)
            flow = frame[f"line_flow_mw__{line.id}"].to_numpy(dtype=np.float64)
            rent += float(np.dot(flow, to_price - from_price) * dt)
        return rent

    def _scarcity_rent(
        self,
        problem: FormulationProblem,
        frame: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> float:
        dt = self.config.simulation.time_step_hours
        if not problem.network.enabled:
            price = prices["energy_price_eur_per_mwh"].to_numpy(dtype=np.float64)
            source_shed = frame["source_load_shed_mw"].to_numpy(dtype=np.float64)
            return float(np.dot(source_shed, price) * dt)
        rent = 0.0
        demand_bus = {unit.id: unit.bus_id for unit in self.config.portfolio.demand}
        for demand_unit in problem.demand_units:
            column = f"demand_involuntary_shed_mw__{demand_unit.id}"
            if column not in frame:
                continue
            bus_id = demand_bus[demand_unit.id]
            price = prices[f"lmp_eur_per_mwh__{bus_id}"].to_numpy(dtype=np.float64)
            rent += float(np.dot(frame[column].to_numpy(dtype=np.float64), price) * dt)
        return rent


def wide_nodal_prices(nodal_prices: pd.DataFrame) -> pd.DataFrame:
    """Return one LMP column per bus for settlement calculations."""
    if nodal_prices.empty:
        return pd.DataFrame()
    wide = nodal_prices.pivot(index="period", columns="bus_id", values="lmp_eur_per_mwh")
    wide.columns = [f"lmp_eur_per_mwh__{column}" for column in wide.columns]
    return wide.reset_index(drop=True)


def attach_wide_nodal_prices(prices: pd.DataFrame, nodal_prices: pd.DataFrame) -> pd.DataFrame:
    """Combine system prices with per-bus LMP columns."""
    if nodal_prices.empty:
        return prices
    return pd.concat([prices.reset_index(drop=True), wide_nodal_prices(nodal_prices)], axis=1)
