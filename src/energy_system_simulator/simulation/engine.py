from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from energy_system_simulator.config import ModelConfig, StorageUnitConfig
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.data import load_input_data
from energy_system_simulator.dispatch import (
    DispatchResult,
    FormulationStatistics,
    TerminalCommitmentState,
    UnitCommitment,
)
from energy_system_simulator.exceptions import OptimisationError
from energy_system_simulator.network import DistributionDemand, DistributionNetwork
from energy_system_simulator.simulation.assets import (
    AssetRegistry,
    AssetTimeSeries,
    FloatArray,
    RenewableAvailability,
    allocate_renewable_dispatch,
)


@dataclass(frozen=True)
class SimulationResult:
    """Complete simulation output and aggregate metrics."""

    timeseries: pd.DataFrame
    asset_timeseries: pd.DataFrame
    summary: dict[str, Any]
    objective_eur: float
    solver_message: str
    solver_status: str
    backend_solver_status: str
    backend_solver_status_code: int | None
    mip_gap: float | None
    objective_bound_eur: float | None
    absolute_gap_eur: float | None
    relative_gap: float | None
    solver_runtime_seconds: float
    solver_node_count: int | None
    formulation_statistics: FormulationStatistics
    terminal_commitment_state: TerminalCommitmentState
    terminal_commitment_by_unit: dict[str, TerminalCommitmentState]
    numerical_diagnostics: dict[str, float]


class SimulationEngine:
    """Coordinate data loading, generation models, dispatch, and accounting."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def run(self) -> SimulationResult:
        """Execute the configured simulation."""
        data = self._load_data()
        registry = self._resolve_assets()
        renewable = self._calculate_renewable_availability(registry, data)
        demand = self._calculate_demand(registry, data)
        network = DistributionNetwork(self.config.network)
        distribution = network.prepare_demand(demand)
        dispatch = self._solve_dispatch(registry, data, renewable, distribution.gross_demand_mw)
        frame = self._assemble_timeseries(data, renewable, distribution, dispatch)
        asset_timeseries = self._asset_timeseries(data, renewable, frame)

        efficiency = network.efficiency
        frame["dispatch_load_shed_mw"] = efficiency * frame["source_load_shed_mw"]
        frame["total_load_shed_mw"] = (
            frame["network_capacity_shed_mw"] + frame["dispatch_load_shed_mw"]
        )
        frame["served_demand_mw"] = frame["end_user_demand_mw"] - frame["total_load_shed_mw"]
        source_power_to_load = frame["gross_demand_mw"] - frame["source_load_shed_mw"]
        frame["network_losses_mw"] = source_power_to_load * self.config.network.loss_fraction
        thermal_emission_columns = [
            column for column in frame.columns if column.startswith("thermal_emissions_tonnes__")
        ]
        frame["thermal_emissions_tonnes"] = (
            frame[thermal_emission_columns].sum(axis=1)
            if thermal_emission_columns
            else frame["thermal_output_mw"]
            * self.config.thermal.emission_factor_tonnes_per_mwh
            * self.config.simulation.time_step_hours
        )
        frame["import_emissions_tonnes"] = (
            frame["imports_mw"]
            * self.config.imports.emission_factor_tonnes_per_mwh
            * self.config.simulation.time_step_hours
        )
        self._add_energy_reconciliation(frame)

        fixed_network_shedding_cost = (
            frame["network_capacity_shed_mw"].sum()
            * self.config.penalties.lost_load_eur_per_mwh
            * self.config.simulation.time_step_hours
        )
        cost_components = {
            **dispatch.cost_components_eur,
            "network_capacity_load_shedding_cost_eur": float(fixed_network_shedding_cost),
        }
        reconciled_objective_eur = float(sum(cost_components.values()))
        total_objective_eur = dispatch.objective_eur + float(fixed_network_shedding_cost)
        if (
            abs(total_objective_eur - reconciled_objective_eur)
            > DEFAULT_NUMERICAL_POLICY.objective_reconciliation_eur
        ):
            raise OptimisationError("Total cost components do not reconcile with objective")
        summary = self._summary(
            frame,
            asset_timeseries,
            total_objective_eur,
            cost_components,
            dispatch.terminal_commitment_state,
            dispatch.terminal_commitment_by_unit,
            dispatch.numerical_diagnostics,
        )
        return SimulationResult(
            timeseries=frame,
            asset_timeseries=asset_timeseries.table,
            summary=summary,
            objective_eur=total_objective_eur,
            solver_message=dispatch.solver_message,
            solver_status=dispatch.solver_status,
            backend_solver_status=dispatch.backend_solver_status,
            backend_solver_status_code=dispatch.backend_solver_status_code,
            mip_gap=dispatch.mip_gap,
            objective_bound_eur=(
                dispatch.objective_bound_eur + fixed_network_shedding_cost
                if dispatch.objective_bound_eur is not None
                else None
            ),
            absolute_gap_eur=dispatch.absolute_gap_eur,
            relative_gap=dispatch.relative_gap,
            solver_runtime_seconds=dispatch.solver_runtime_seconds,
            solver_node_count=dispatch.solver_node_count,
            formulation_statistics=dispatch.formulation_statistics,
            terminal_commitment_state=dispatch.terminal_commitment_state,
            terminal_commitment_by_unit=dispatch.terminal_commitment_by_unit,
            numerical_diagnostics=dispatch.numerical_diagnostics,
        )

    def _load_data(self) -> pd.DataFrame:
        return load_input_data(
            self.config.paths.input_csv,
            self.config.simulation.time_step_hours,
        )

    def _resolve_assets(self) -> AssetRegistry:
        return AssetRegistry.from_config(self.config)

    def _calculate_renewable_availability(
        self,
        registry: AssetRegistry,
        data: pd.DataFrame,
    ) -> RenewableAvailability:
        return registry.renewable_availability(data)

    def _calculate_demand(self, registry: AssetRegistry, data: pd.DataFrame) -> FloatArray:
        return registry.demand_mw(data)

    def _solve_dispatch(
        self,
        registry: AssetRegistry,
        data: pd.DataFrame,
        renewable: RenewableAvailability,
        gross_demand_mw: FloatArray,
    ) -> DispatchResult:
        return UnitCommitment(self.config).solve(
            renewable.aggregate_mw,
            gross_demand_mw,
            thermal_availability_factors=registry.thermal_availability_factors(data),
            fuel_price_series=self._fuel_price_series(data),
            storage_availability_factors=registry.storage_availability_factors(data),
            hydro_inflows_mw=registry.hydro_inflows_mw(data),
        )

    def _fuel_price_series(self, data: pd.DataFrame) -> dict[str, FloatArray]:
        prices: dict[str, FloatArray] = {}
        for fuel in self.config.portfolio.fuels:
            if fuel.price_time_series_key is None:
                continue
            if fuel.price_time_series_key not in data.columns:
                raise OptimisationError(
                    f"Fuel {fuel.id!r} references missing input column "
                    f"{fuel.price_time_series_key!r}"
                )
            values = pd.to_numeric(data[fuel.price_time_series_key], errors="coerce").to_numpy(
                dtype="float64"
            )
            if not pd.notna(values).all():
                raise OptimisationError(
                    f"Fuel price column {fuel.price_time_series_key!r} contains non-finite values"
                )
            if (values < 0.0).any():
                raise OptimisationError(
                    f"Fuel price column {fuel.price_time_series_key!r} must be non-negative"
                )
            prices[fuel.id] = values
        return prices

    def _assemble_timeseries(
        self,
        data: pd.DataFrame,
        renewable: RenewableAvailability,
        distribution: DistributionDemand,
        dispatch: DispatchResult,
    ) -> pd.DataFrame:
        frame = dispatch.frame.copy()
        frame.insert(0, "timestamp", data["timestamp"].to_numpy())
        frame.insert(1, "end_user_demand_mw", distribution.end_user_demand_mw)
        frame.insert(2, "deliverable_demand_mw", distribution.deliverable_demand_mw)
        frame.insert(3, "network_capacity_shed_mw", distribution.network_capacity_shed_mw)
        for asset_id, values in renewable.by_asset_mw.items():
            frame[f"renewable_available_mw__{asset_id}"] = values
        frame["solar_available_mw"] = self._renewable_kind_available(renewable, "solar")
        frame["wind_available_mw"] = self._renewable_kind_available(renewable, "wind")
        return frame

    def _asset_timeseries(
        self,
        data: pd.DataFrame,
        renewable: RenewableAvailability,
        frame: pd.DataFrame,
    ) -> AssetTimeSeries:
        dispatch = allocate_renewable_dispatch(
            data["timestamp"],
            renewable.by_asset_mw,
            renewable.aggregate_mw,
            frame["renewable_used_mw"].to_numpy(dtype="float64"),
        )
        return (
            renewable.asset_table.append(dispatch)
            .append(self._thermal_asset_timeseries(data["timestamp"], frame))
            .append(self._storage_asset_timeseries(data["timestamp"], frame))
            .append(self._hydro_asset_timeseries(data["timestamp"], frame))
        )

    def _thermal_asset_timeseries(
        self,
        timestamps: pd.Series,
        frame: pd.DataFrame,
    ) -> AssetTimeSeries:
        pieces: list[pd.DataFrame] = []
        variable_map = {
            "thermal_output_mw": ("output_mw", "MW"),
            "thermal_on": ("commitment", "binary"),
            "thermal_startup": ("startup", "binary"),
            "thermal_shutdown": ("shutdown", "binary"),
            "thermal_capacity_available_mw": ("capacity_available_mw", "MW"),
            "thermal_capacity_factor": ("capacity_factor", "fraction"),
            "thermal_variable_cost_eur": ("variable_cost_eur", "EUR"),
            "thermal_no_load_cost_eur": ("no_load_cost_eur", "EUR"),
            "thermal_startup_cost_eur": ("startup_cost_eur", "EUR"),
            "thermal_shutdown_cost_eur": ("shutdown_cost_eur", "EUR"),
            "thermal_carbon_cost_eur": ("carbon_cost_eur", "EUR"),
            "thermal_emissions_tonnes": ("emissions_tonnes", "tonnes"),
            "thermal_direct_co2_emissions_tonnes": ("direct_co2_emissions_tonnes", "tonnes"),
            "thermal_methane_emissions_tonnes": ("methane_emissions_tonnes", "tonnes"),
            "thermal_nox_emissions_kg": ("nox_emissions_kg", "kg"),
            "thermal_sox_emissions_kg": ("sox_emissions_kg", "kg"),
            "thermal_fuel_input_mwh_thermal": ("fuel_input_mwh_thermal", "MWh-thermal"),
            "thermal_running_fuel_input_mwh_thermal": (
                "running_fuel_input_mwh_thermal",
                "MWh-thermal",
            ),
            "thermal_startup_fuel_input_mwh_thermal": (
                "startup_fuel_input_mwh_thermal",
                "MWh-thermal",
            ),
            "thermal_fuel_cost_eur": ("fuel_cost_eur", "EUR"),
            "thermal_efficiency": ("efficiency", "fraction"),
        }
        for generator in self.config.portfolio.thermal_generators:
            for prefix, (variable, unit) in variable_map.items():
                column = f"{prefix}__{generator.id}"
                if column not in frame:
                    continue
                pieces.append(
                    pd.DataFrame(
                        {
                            "timestamp": timestamps.to_numpy(),
                            "asset_id": generator.id,
                            "variable": variable,
                            "value": frame[column].to_numpy(),
                            "unit": unit,
                        }
                    )
                )
        if not pieces:
            return AssetTimeSeries.empty()
        return AssetTimeSeries(pd.concat(pieces, ignore_index=True))

    def _storage_asset_timeseries(
        self,
        timestamps: pd.Series,
        frame: pd.DataFrame,
    ) -> AssetTimeSeries:
        pieces: list[pd.DataFrame] = []
        variable_map = {
            "storage_charge_mw": ("charge_mw", "MW"),
            "storage_discharge_mw": ("discharge_mw", "MW"),
            "storage_charge_mode": ("charge_mode", "binary"),
            "storage_discharge_mode": ("discharge_mode", "binary"),
            "storage_soc_mwh": ("stored_energy_mwh", "MWh"),
            "storage_throughput_mwh": ("throughput_mwh", "MWh"),
            "storage_throughput_cost_eur": ("throughput_cost_eur", "EUR"),
            "storage_degradation_cost_eur": ("degradation_cost_eur", "EUR"),
            "storage_energy_residual_mwh": ("energy_residual_mwh", "MWh"),
            "storage_round_trip_losses_mwh": ("round_trip_losses_mwh", "MWh"),
            "storage_depth_of_discharge": ("depth_of_discharge", "fraction"),
            "storage_equivalent_full_cycles": ("equivalent_full_cycles", "cycles"),
        }
        for storage in self._storage_units_for_reporting():
            for prefix, (variable, unit) in variable_map.items():
                column = f"{prefix}__{storage.id}"
                if column not in frame:
                    continue
                pieces.append(
                    pd.DataFrame(
                        {
                            "timestamp": timestamps.to_numpy(),
                            "asset_id": storage.id,
                            "variable": variable,
                            "value": frame[column].to_numpy(),
                            "unit": unit,
                        }
                    )
                )
        if not pieces:
            return AssetTimeSeries.empty()
        return AssetTimeSeries(pd.concat(pieces, ignore_index=True))

    def _hydro_asset_timeseries(
        self,
        timestamps: pd.Series,
        frame: pd.DataFrame,
    ) -> AssetTimeSeries:
        pieces: list[pd.DataFrame] = []
        variable_map = {
            "hydro_generation_mw": ("generation_mw", "MW"),
            "hydro_release_mw": ("release_mw_water_equivalent", "MW-water"),
            "hydro_spill_mw": ("spill_mw_water_equivalent", "MW-water"),
            "hydro_reservoir_mwh": ("reservoir_mwh_water_equivalent", "MWh-water"),
            "hydro_inflow_mw": ("inflow_mw_water_equivalent", "MW-water"),
            "hydro_capacity_factor": ("capacity_factor", "fraction"),
            "hydro_water_loss_mwh": ("water_loss_mwh_water_equivalent", "MWh-water"),
            "hydro_water_balance_residual_mwh": (
                "water_balance_residual_mwh_water_equivalent",
                "MWh-water",
            ),
            "hydro_terminal_value_eur": ("terminal_value_eur", "EUR"),
        }
        for hydro in self.config.portfolio.hydro_units:
            for prefix, (variable, unit) in variable_map.items():
                column = f"{prefix}__{hydro.id}"
                if column not in frame:
                    continue
                pieces.append(
                    pd.DataFrame(
                        {
                            "timestamp": timestamps.to_numpy(),
                            "asset_id": hydro.id,
                            "variable": variable,
                            "value": frame[column].to_numpy(),
                            "unit": unit,
                        }
                    )
                )
        if not pieces:
            return AssetTimeSeries.empty()
        return AssetTimeSeries(pd.concat(pieces, ignore_index=True))

    def _renewable_kind_available(
        self,
        renewable: RenewableAvailability,
        kind: str,
    ) -> pd.Series:
        values = [
            renewable.by_asset_mw[asset.id]
            for asset in self.config.portfolio.renewable_generators
            if asset.kind == kind
        ]
        if not values:
            return pd.Series([0.0] * len(renewable.aggregate_mw), dtype="float64")
        return pd.Series(pd.DataFrame(values).sum(axis=0).to_numpy(), dtype="float64")

    def _summary(
        self,
        frame: pd.DataFrame,
        asset_timeseries: AssetTimeSeries,
        objective_eur: float,
        cost_components: dict[str, float],
        terminal_commitment_state: TerminalCommitmentState,
        terminal_commitment_by_unit: dict[str, TerminalCommitmentState],
        numerical_diagnostics: dict[str, float],
    ) -> dict[str, Any]:
        dt = self.config.simulation.time_step_hours

        def energy(column: str) -> float:
            return float(frame[column].sum() * dt)

        demand_mwh = energy("end_user_demand_mw")
        served_mwh = energy("served_demand_mw")
        renewable_used_mwh = energy("renewable_used_mw")
        primary_source_generation_mwh = (
            renewable_used_mwh
            + energy("thermal_output_mw")
            + energy("hydro_generation_mw")
            + energy("imports_mw")
        )
        return {
            "periods": len(frame),
            "time_step_hours": dt,
            "objective_eur": objective_eur,
            "cost_components_eur": cost_components,
            "numerical_diagnostics": numerical_diagnostics,
            "objective_reconciliation_error_eur": abs(
                objective_eur - sum(cost_components.values())
            ),
            "total_demand_mwh": demand_mwh,
            "served_demand_mwh": served_mwh,
            "unserved_energy_mwh": energy("total_load_shed_mw"),
            "loss_of_load_probability": float(
                (
                    frame["total_load_shed_mw"] > DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw
                ).mean()
            ),
            "renewable_available_mwh": energy("renewable_available_mw"),
            "renewable_used_mwh": renewable_used_mwh,
            "renewable_curtailed_mwh": energy("renewable_curtailed_mw"),
            "renewable_assets": self._renewable_asset_summary(asset_timeseries),
            "renewable_share_of_primary_generation": (
                renewable_used_mwh / primary_source_generation_mwh
                if primary_source_generation_mwh > 0.0
                else 0.0
            ),
            "thermal_generation_mwh": energy("thermal_output_mw"),
            "thermal_starts": int(frame["thermal_startup"].sum()),
            "thermal_shutdowns": int(frame["thermal_shutdown"].sum()),
            "terminal_commitment": asdict(terminal_commitment_state),
            "terminal_commitment_by_unit": {
                unit_id: asdict(state) for unit_id, state in terminal_commitment_by_unit.items()
            },
            "thermal_generation_by_unit_mwh": self._thermal_generation_by_unit(frame),
            "average_online_thermal_capacity_mw": float(frame["online_thermal_capacity_mw"].mean()),
            "unused_committed_capacity_mwh": energy("unused_committed_capacity_mw"),
            "thermal_fleet_capacity_factor": (
                energy("thermal_output_mw") / energy("available_thermal_capacity_mw")
                if energy("available_thermal_capacity_mw") > 0.0
                else 0.0
            ),
            "thermal_operating_cost_summary": self._thermal_operating_cost_summary(frame),
            "thermal_fuel_input_mwh_thermal": self._sum_prefixed_columns(
                frame,
                "thermal_fuel_input_mwh_thermal__",
            ),
            "thermal_fuel_cost_eur": self._sum_prefixed_columns(frame, "thermal_fuel_cost_eur__"),
            "thermal_direct_co2_emissions_tonnes": self._sum_prefixed_columns(
                frame,
                "thermal_direct_co2_emissions_tonnes__",
            ),
            "thermal_methane_emissions_tonnes": self._sum_prefixed_columns(
                frame,
                "thermal_methane_emissions_tonnes__",
            ),
            "thermal_nox_emissions_kg": self._sum_prefixed_columns(
                frame,
                "thermal_nox_emissions_kg__",
            ),
            "thermal_sox_emissions_kg": self._sum_prefixed_columns(
                frame,
                "thermal_sox_emissions_kg__",
            ),
            "imports_mwh": energy("imports_mw"),
            "battery_charge_mwh": energy("battery_charge_mw"),
            "battery_discharge_mwh": energy("battery_discharge_mw"),
            "final_battery_soc_mwh": float(frame["battery_soc_mwh"].iloc[-1]),
            "storage_assets": self._storage_asset_summary(frame),
            "hydro_generation_mwh": energy("hydro_generation_mw"),
            "hydro_spill_mwh_water_equivalent": energy("hydro_spill_mw"),
            "hydro_water_losses_mwh_water_equivalent": float(frame["hydro_water_loss_mwh"].sum()),
            "hydro_terminal_value_eur": float(frame["hydro_terminal_value_eur"].sum()),
            "hydro_assets": self._hydro_asset_summary(frame),
            "network_losses_mwh": energy("network_losses_mw"),
            "thermal_emissions_tonnes": float(frame["thermal_emissions_tonnes"].sum()),
            "import_emissions_tonnes": float(frame["import_emissions_tonnes"].sum()),
            "total_emissions_tonnes": float(
                frame["thermal_emissions_tonnes"].sum() + frame["import_emissions_tonnes"].sum()
            ),
            "peak_demand_mw": float(frame["end_user_demand_mw"].max()),
            "peak_thermal_output_mw": float(frame["thermal_output_mw"].max()),
            "energy_reconciliation": {
                "max_abs_source_balance_residual_mw": float(
                    frame["source_balance_residual_mw"].abs().max()
                ),
                "max_abs_delivered_demand_residual_mw": float(
                    frame["delivered_demand_balance_residual_mw"].abs().max()
                ),
                "max_abs_battery_energy_residual_mwh": float(
                    frame["battery_energy_residual_mwh"].abs().max()
                ),
                "max_abs_curtailment_residual_mw": float(
                    frame["curtailment_residual_mw"].abs().max()
                ),
                "total_network_losses_mwh": energy("network_losses_mw"),
                "total_unserved_energy_mwh": energy("total_load_shed_mw"),
                "storage_assets": self._storage_reconciliation_summary(frame),
                "hydro_assets": self._hydro_reconciliation_summary(frame),
                "residual_summaries": {
                    "source_balance": self._residual_summary(
                        frame,
                        "source_balance",
                        "source_balance_residual_mw",
                        (
                            "renewable_used_mw",
                            "thermal_output_mw",
                            "hydro_generation_mw",
                            "battery_discharge_mw",
                            "imports_mw",
                            "source_load_shed_mw",
                            "gross_demand_mw",
                            "battery_charge_mw",
                        ),
                    ),
                    "delivered_demand_balance": self._residual_summary(
                        frame,
                        "delivered_demand_balance",
                        "delivered_demand_balance_residual_mw",
                        ("served_demand_mw", "total_load_shed_mw", "end_user_demand_mw"),
                    ),
                    "battery_energy": self._residual_summary(
                        frame,
                        "battery_energy",
                        "battery_energy_residual_mwh",
                        ("battery_energy_change_mwh", "battery_charge_mw", "battery_discharge_mw"),
                    ),
                    "curtailment": self._residual_summary(
                        frame,
                        "curtailment",
                        "curtailment_residual_mw",
                        ("renewable_available_mw", "renewable_used_mw", "renewable_curtailed_mw"),
                    ),
                },
            },
        }

    @staticmethod
    def _sum_prefixed_columns(frame: pd.DataFrame, prefix: str) -> float:
        columns = [column for column in frame.columns if column.startswith(prefix)]
        if not columns:
            return 0.0
        return float(frame[columns].sum().sum())

    def _renewable_asset_summary(self, asset_timeseries: AssetTimeSeries) -> dict[str, Any]:
        dt = self.config.simulation.time_step_hours
        table = asset_timeseries.table[
            asset_timeseries.table["variable"].isin({"available_mw", "used_mw", "curtailed_mw"})
        ]
        if table.empty:
            return {}
        pivot = (
            table.groupby(["asset_id", "variable"], sort=True)["value"].sum().unstack(fill_value=0)
        )
        result: dict[str, Any] = {}
        for asset_id, row in pivot.iterrows():
            result[str(asset_id)] = {
                "available_mwh": float(row.get("available_mw", 0.0) * dt),
                "used_mwh": float(row.get("used_mw", 0.0) * dt),
                "curtailed_mwh": float(row.get("curtailed_mw", 0.0) * dt),
            }
        return result

    def _thermal_generation_by_unit(self, frame: pd.DataFrame) -> dict[str, float]:
        dt = self.config.simulation.time_step_hours
        return {
            generator.id: float(frame[f"thermal_output_mw__{generator.id}"].sum() * dt)
            for generator in self.config.portfolio.thermal_generators
            if f"thermal_output_mw__{generator.id}" in frame
        }

    def _thermal_operating_cost_summary(self, frame: pd.DataFrame) -> dict[str, float]:
        generators = [
            generator
            for generator in self.config.portfolio.thermal_generators
            if f"thermal_output_mw__{generator.id}" in frame
        ]
        generation_mwh = {
            generator.id: float(
                frame[f"thermal_output_mw__{generator.id}"].sum()
                * self.config.simulation.time_step_hours
            )
            for generator in generators
        }
        total_generation = sum(generation_mwh.values())
        if total_generation <= 0.0:
            return {
                "minimum_variable_cost_eur_per_mwh": 0.0,
                "maximum_variable_cost_eur_per_mwh": 0.0,
                "generation_weighted_variable_cost_eur_per_mwh": 0.0,
            }
        costs = {}
        for generator in generators:
            cost_column = f"thermal_variable_cost_eur__{generator.id}"
            costs[generator.id] = (
                float(frame[cost_column].sum()) / generation_mwh[generator.id]
                if cost_column in frame and generation_mwh[generator.id] > 0.0
                else generator.config.variable_cost_eur_per_mwh
            )
        return {
            "minimum_variable_cost_eur_per_mwh": min(costs.values()),
            "maximum_variable_cost_eur_per_mwh": max(costs.values()),
            "generation_weighted_variable_cost_eur_per_mwh": sum(
                generation_mwh[unit_id] * costs[unit_id] for unit_id in costs
            )
            / total_generation,
        }

    def _storage_asset_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        dt = self.config.simulation.time_step_hours
        result: dict[str, Any] = {}
        for storage in self._storage_units_for_reporting():
            battery = storage.config
            usable_energy = battery.maximum_soc_mwh - battery.minimum_soc_mwh
            throughput = float(frame[f"storage_throughput_mwh__{storage.id}"].sum())
            result[storage.id] = {
                "technology": battery.technology,
                "charged_mwh": float(frame[f"storage_charged_mwh__{storage.id}"].sum()),
                "discharged_mwh": float(frame[f"storage_discharged_mwh__{storage.id}"].sum()),
                "throughput_mwh": throughput,
                "equivalent_full_cycles": (
                    throughput / (2.0 * usable_energy) if usable_energy > 0.0 else 0.0
                ),
                "round_trip_losses_mwh": float(
                    frame[f"storage_round_trip_losses_mwh__{storage.id}"].sum()
                ),
                "average_depth_of_discharge": float(
                    frame[f"storage_depth_of_discharge__{storage.id}"].mean()
                ),
                "maximum_depth_of_discharge": float(
                    frame[f"storage_depth_of_discharge__{storage.id}"].max()
                ),
                "time_at_min_soc_hours": float(
                    frame[f"storage_at_min_soc__{storage.id}"].sum() * dt
                ),
                "time_at_max_soc_hours": float(
                    frame[f"storage_at_max_soc__{storage.id}"].sum() * dt
                ),
                "throughput_cost_eur": float(
                    frame[f"storage_throughput_cost_eur__{storage.id}"].sum()
                ),
                "degradation_cost_eur": float(
                    frame[f"storage_degradation_cost_eur__{storage.id}"].sum()
                ),
                "max_abs_energy_residual_mwh": float(
                    frame[f"storage_energy_residual_mwh__{storage.id}"].abs().max()
                ),
            }
        return result

    def _hydro_asset_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        dt = self.config.simulation.time_step_hours
        result: dict[str, Any] = {}
        for hydro in self.config.portfolio.hydro_units:
            generation_column = f"hydro_generation_mw__{hydro.id}"
            result[hydro.id] = {
                "kind": hydro.kind,
                "generation_mwh": float(frame[generation_column].sum() * dt),
                "inflow_mwh_water_equivalent": float(
                    frame[f"hydro_inflow_mw__{hydro.id}"].sum() * dt
                ),
                "release_mwh_water_equivalent": float(
                    frame[f"hydro_release_mw__{hydro.id}"].sum() * dt
                ),
                "spill_mwh_water_equivalent": float(
                    frame[f"hydro_spill_mw__{hydro.id}"].sum() * dt
                ),
                "water_losses_mwh_water_equivalent": float(
                    frame[f"hydro_water_loss_mwh__{hydro.id}"].sum()
                ),
                "final_reservoir_mwh_water_equivalent": float(
                    frame[f"hydro_reservoir_mwh__{hydro.id}"].iloc[-1]
                ),
                "capacity_factor": (
                    float(frame[generation_column].mean()) / hydro.turbine_capacity_mw
                    if hydro.turbine_capacity_mw > 0.0
                    else 0.0
                ),
                "terminal_value_eur": float(frame[f"hydro_terminal_value_eur__{hydro.id}"].sum()),
                "max_abs_water_balance_residual_mwh": float(
                    frame[f"hydro_water_balance_residual_mwh__{hydro.id}"].abs().max()
                ),
            }
        return result

    def _storage_reconciliation_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for storage in self._storage_units_for_reporting():
            residual_column = f"storage_energy_residual_mwh__{storage.id}"
            result[storage.id] = self._residual_summary(
                frame,
                f"storage_energy[{storage.id}]",
                residual_column,
                (
                    f"storage_soc_mwh__{storage.id}",
                    f"storage_charge_mw__{storage.id}",
                    f"storage_discharge_mw__{storage.id}",
                ),
            )
        return result

    def _hydro_reconciliation_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for hydro in self.config.portfolio.hydro_units:
            residual_column = f"hydro_water_balance_residual_mwh__{hydro.id}"
            result[hydro.id] = self._residual_summary(
                frame,
                f"hydro_water_balance[{hydro.id}]",
                residual_column,
                (
                    f"hydro_reservoir_mwh__{hydro.id}",
                    f"hydro_inflow_mw__{hydro.id}",
                    f"hydro_release_mw__{hydro.id}",
                    f"hydro_spill_mw__{hydro.id}",
                ),
            )
        return result

    def _storage_units_for_reporting(self) -> tuple[StorageUnitConfig, ...]:
        units = self.config.portfolio.storage_units
        if len(units) == 1:
            unit = units[0]
            return (StorageUnitConfig(id=unit.id, bus_id=unit.bus_id, config=self.config.battery),)
        return units

    def _add_energy_reconciliation(self, frame: pd.DataFrame) -> None:
        dt = self.config.simulation.time_step_hours
        source_left = (
            frame["renewable_used_mw"]
            + frame["thermal_output_mw"]
            + frame["hydro_generation_mw"]
            + frame["battery_discharge_mw"]
            + frame["imports_mw"]
            + frame["source_load_shed_mw"]
        )
        source_right = frame["gross_demand_mw"] + frame["battery_charge_mw"]
        frame["source_balance_residual_mw"] = source_left - source_right
        frame["delivered_demand_balance_residual_mw"] = (
            frame["served_demand_mw"] + frame["total_load_shed_mw"] - frame["end_user_demand_mw"]
        )
        storage_units = self._storage_units_for_reporting()
        storage_residual_columns: list[str] = []
        storage_change_columns: list[str] = []
        for storage in storage_units:
            battery = storage.config
            soc = frame[f"storage_soc_mwh__{storage.id}"]
            previous_soc = soc.shift(1)
            previous_soc.iloc[0] = battery.initial_soc_mwh
            retention = (1.0 - battery.self_discharge_rate_per_hour) ** dt
            frame[f"storage_energy_change_mwh__{storage.id}"] = soc - previous_soc
            expected_soc = (
                retention * previous_soc
                + battery.charge_efficiency * frame[f"storage_charge_mw__{storage.id}"] * dt
                - frame[f"storage_discharge_mw__{storage.id}"] * dt / battery.discharge_efficiency
            )
            residual = f"storage_energy_residual_mwh__{storage.id}"
            frame[residual] = soc - expected_soc
            storage_residual_columns.append(residual)
            storage_change_columns.append(f"storage_energy_change_mwh__{storage.id}")
        frame["battery_energy_change_mwh"] = (
            frame[storage_change_columns].sum(axis=1) if storage_change_columns else 0.0
        )
        frame["battery_energy_residual_mwh"] = (
            frame[storage_residual_columns].sum(axis=1) if storage_residual_columns else 0.0
        )
        frame["curtailment_residual_mw"] = (
            frame["renewable_available_mw"]
            - frame["renewable_used_mw"]
            - frame["renewable_curtailed_mw"]
        )
        source_summary = self._residual_summary(
            frame,
            "source_balance",
            "source_balance_residual_mw",
            (
                "renewable_used_mw",
                "thermal_output_mw",
                "hydro_generation_mw",
                "battery_discharge_mw",
                "imports_mw",
                "source_load_shed_mw",
                "gross_demand_mw",
                "battery_charge_mw",
            ),
        )
        if (
            float(source_summary["max_abs_residual"])
            > DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw
        ):
            raise OptimisationError(
                "source_balance residual exceeds tolerance at period "
                f"{source_summary['period_index']}: {source_summary['max_abs_residual']} MW"
            )

    @staticmethod
    def _residual_summary(
        frame: pd.DataFrame,
        equation_family: str,
        residual_column: str,
        scale_columns: tuple[str, ...],
    ) -> dict[str, Any]:
        residuals = frame[residual_column].abs()
        period_index = int(residuals.idxmax())
        max_abs_residual = float(residuals.loc[period_index])
        scale = sum(abs(float(frame[column].iloc[period_index])) for column in scale_columns)
        denominator = max(scale, DEFAULT_NUMERICAL_POLICY.primal_feasibility_mw)
        timestamp = frame["timestamp"].iloc[period_index] if "timestamp" in frame else None
        return {
            "equation_family": equation_family,
            "period_index": period_index,
            "timestamp": str(timestamp) if timestamp is not None else None,
            "max_abs_residual": max_abs_residual,
            "scale": scale,
            "scale_normalized_residual": max_abs_residual / denominator,
        }

    @staticmethod
    def ensure_output_directory(path: Path) -> None:
        """Create the configured output directory when needed."""
        path.mkdir(parents=True, exist_ok=True)
