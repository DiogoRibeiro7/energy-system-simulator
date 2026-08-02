from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
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
        if self.config.rolling_horizon.enabled:
            return self._run_rolling_horizon(data)
        return self._run_full_horizon_data(data)

    def _run_full_horizon_data(self, data: pd.DataFrame) -> SimulationResult:
        """Execute one optimisation over the provided data frame."""
        registry = self._resolve_assets()
        renewable = self._calculate_renewable_availability(registry, data)
        demand = self._calculate_demand(registry, data)
        network = DistributionNetwork(self.config.network)
        distribution = network.prepare_demand(demand)
        dispatch = self._solve_dispatch(
            registry,
            data,
            renewable,
            distribution.gross_demand_mw,
            self._source_side_demand_profiles(registry, data, demand, distribution),
        )
        frame = self._assemble_timeseries(data, renewable, distribution, dispatch)
        asset_timeseries = self._asset_timeseries(data, renewable, frame)

        efficiency = network.efficiency
        frame["dispatch_load_shed_mw"] = efficiency * frame["source_load_shed_mw"]
        frame["total_load_shed_mw"] = (
            frame["network_capacity_shed_mw"] + frame["dispatch_load_shed_mw"]
        )
        frame["demand_voluntary_curtailment_end_user_mw"] = (
            efficiency * frame["demand_voluntary_curtailment_mw"]
        )
        frame["demand_shifted_out_end_user_mw"] = efficiency * frame["demand_shift_down_mw"]
        frame["demand_shifted_in_end_user_mw"] = efficiency * frame["demand_shift_up_mw"]
        frame["demand_task_charge_end_user_mw"] = efficiency * frame["demand_task_charge_mw"]
        frame["adjusted_end_user_demand_mw"] = (
            frame["end_user_demand_mw"]
            - frame["network_capacity_shed_mw"]
            - frame["demand_voluntary_curtailment_end_user_mw"]
            - frame["demand_shifted_out_end_user_mw"]
            + frame["demand_shifted_in_end_user_mw"]
            + frame["demand_task_charge_end_user_mw"]
        )
        frame["served_demand_mw"] = (
            frame["adjusted_end_user_demand_mw"] - frame["dispatch_load_shed_mw"]
        )
        source_power_to_load = frame["demand_adjusted_mw"] - frame["source_load_shed_mw"]
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

    def _run_rolling_horizon(self, data: pd.DataFrame) -> SimulationResult:
        rolling = self.config.rolling_horizon
        total_periods = len(data)
        checkpoint_path = self._rolling_checkpoint_path()
        state = self._initial_rolling_state()
        start = 0
        window_records: list[dict[str, Any]] = []
        implemented_frames: list[pd.DataFrame] = []
        implemented_assets: list[pd.DataFrame] = []
        if rolling.resume_from_checkpoint and checkpoint_path.exists():
            checkpoint = self._read_rolling_checkpoint(checkpoint_path)
            start = int(checkpoint["next_start"])
            state = checkpoint["state"]
            window_records = list(checkpoint["windows"])
            implemented_frames = [
                self._records_frame(checkpoint["timeseries"], checkpoint["timeseries_columns"])
            ]
            implemented_assets = [
                self._records_frame(
                    checkpoint["asset_timeseries"],
                    checkpoint["asset_timeseries_columns"],
                )
            ]

        last_result: SimulationResult | None = None
        while start < total_periods:
            implementation_end = min(start + rolling.implementation_periods, total_periods)
            solve_end = min(start + rolling.optimisation_window_periods, total_periods)
            if solve_end < implementation_end:
                raise OptimisationError("Rolling horizon window does not cover implementation step")
            is_final_window = implementation_end >= total_periods
            window_config = self._rolling_window_config(
                state,
                window_start=start,
                is_final_window=is_final_window,
            )
            window_data = data.iloc[start:solve_end].reset_index(drop=True)
            window_result = SimulationEngine(window_config)._run_full_horizon_data(window_data)
            last_result = window_result
            implemented_count = implementation_end - start
            implemented_frame = window_result.timeseries.iloc[:implemented_count].copy()
            retained_timestamps = set(implemented_frame["timestamp"])
            implemented_asset = window_result.asset_timeseries[
                window_result.asset_timeseries["timestamp"].isin(retained_timestamps)
            ].copy()
            state = self._rolling_state_after_segment(
                state,
                window_config,
                implemented_frame,
                window_start=start,
            )
            window_record = {
                "window_index": len(window_records),
                "start_period": start,
                "solve_end_period": solve_end,
                "implementation_end_period": implementation_end,
                "implemented_periods": implemented_count,
                "lookahead_periods": solve_end - implementation_end,
                "solver_status": window_result.solver_status,
                "backend_solver_status": window_result.backend_solver_status,
                "backend_solver_status_code": window_result.backend_solver_status_code,
                "objective_eur": window_result.objective_eur,
                "solver_runtime_seconds": window_result.solver_runtime_seconds,
                "solver_node_count": window_result.solver_node_count,
                "mip_gap": window_result.mip_gap,
                "absolute_gap_eur": window_result.absolute_gap_eur,
                "formulation_statistics": asdict(window_result.formulation_statistics),
                "transferred_state": state,
                "fallback": None,
            }
            window_records.append(window_record)
            implemented_frames.append(implemented_frame)
            implemented_assets.append(implemented_asset)
            start = implementation_end
            self._write_rolling_checkpoint(
                checkpoint_path,
                start,
                state,
                window_records,
                pd.concat(implemented_frames, ignore_index=True),
                pd.concat(implemented_assets, ignore_index=True),
            )

        if last_result is None and not window_records:
            raise OptimisationError("Rolling horizon requires at least one period")
        last_window = window_records[-1]
        formulation_statistics = (
            last_result.formulation_statistics
            if last_result is not None
            else FormulationStatistics(**last_window["formulation_statistics"])
        )
        frame = pd.concat(implemented_frames, ignore_index=True)
        asset_frame = pd.concat(implemented_assets, ignore_index=True)
        if frame["timestamp"].duplicated().any():
            raise OptimisationError("Rolling horizon produced duplicated timestamps")
        if len(frame) != total_periods:
            raise OptimisationError("Rolling horizon did not cover the complete input horizon")

        cost_components = self._rolling_cost_components(frame)
        objective_eur = float(sum(cost_components.values()))
        asset_timeseries = AssetTimeSeries(asset_frame)
        terminal_by_unit = self._terminal_commitment_from_state(state)
        terminal_state = terminal_by_unit[self.config.portfolio.thermal_generators[0].id]
        numerical_diagnostics = self._combine_window_diagnostics(window_records)
        summary = self._summary(
            frame,
            asset_timeseries,
            objective_eur,
            cost_components,
            terminal_state,
            terminal_by_unit,
            numerical_diagnostics,
        )
        summary["rolling_horizon"] = {
            "enabled": True,
            "optimisation_window_periods": rolling.optimisation_window_periods,
            "implementation_periods": rolling.implementation_periods,
            "lookahead_periods": rolling.lookahead_periods,
            "terminal_treatment": rolling.terminal_treatment,
            "forecast_mode": rolling.forecast_mode,
            "checkpoint_path": str(checkpoint_path),
            "windows": window_records,
        }
        if rolling.compare_full_horizon:
            summary["rolling_horizon"]["full_horizon_comparison"] = self._full_horizon_comparison(
                data, frame, objective_eur
            )

        return SimulationResult(
            timeseries=frame,
            asset_timeseries=asset_frame,
            summary=summary,
            objective_eur=objective_eur,
            solver_message=f"Rolling horizon solved {len(window_records)} windows",
            solver_status=(
                "optimal"
                if all(window["solver_status"] == "optimal" for window in window_records)
                else "accepted"
            ),
            backend_solver_status=str(last_window["backend_solver_status"]),
            backend_solver_status_code=(
                int(last_window["backend_solver_status_code"])
                if last_window["backend_solver_status_code"] is not None
                else None
            ),
            mip_gap=max(
                (window["mip_gap"] for window in window_records if window["mip_gap"] is not None),
                default=None,
            ),
            objective_bound_eur=None,
            absolute_gap_eur=None,
            relative_gap=None,
            solver_runtime_seconds=float(
                sum(window["solver_runtime_seconds"] for window in window_records)
            ),
            solver_node_count=(
                int(last_window["solver_node_count"])
                if last_window["solver_node_count"] is not None
                else None
            ),
            formulation_statistics=formulation_statistics,
            terminal_commitment_state=terminal_state,
            terminal_commitment_by_unit=terminal_by_unit,
            numerical_diagnostics=numerical_diagnostics,
        )

    def _load_data(self) -> pd.DataFrame:
        return load_input_data(
            self.config.paths.input_csv,
            self.config.simulation.time_step_hours,
        )

    def _rolling_checkpoint_path(self) -> Path:
        directory = (
            self.config.rolling_horizon.checkpoint_directory
            if self.config.rolling_horizon.checkpoint_directory is not None
            else self.config.paths.output_directory / "rolling-checkpoints"
        )
        return directory / "rolling_checkpoint.json"

    def _initial_rolling_state(self) -> dict[str, Any]:
        return {
            "thermal": {
                unit.id: {
                    "initial_on": unit.config.initial_on,
                    "initial_output_mw": unit.config.initial_output_mw,
                    "initial_up_time_hours": unit.config.initial_up_time_hours,
                    "initial_down_time_hours": unit.config.initial_down_time_hours,
                }
                for unit in self.config.portfolio.thermal_generators
            },
            "storage": {
                unit.id: {"initial_soc_mwh": unit.config.initial_soc_mwh}
                for unit in self.config.portfolio.storage_units
            },
            "hydro": {
                unit.id: {"initial_reservoir_mwh": unit.initial_reservoir_mwh}
                for unit in self.config.portfolio.hydro_units
            },
            "demand": {
                demand.id: {"remaining_task_energy_mwh": demand.task_required_energy_mwh}
                for demand in self.config.portfolio.demand
                if demand.kind in {"deferrable", "ev_charging"}
            },
        }

    def _rolling_window_config(
        self,
        state: dict[str, Any],
        *,
        window_start: int,
        is_final_window: bool,
    ) -> ModelConfig:
        rolling = self.config.rolling_horizon
        relax_terminal = rolling.terminal_treatment == "relaxed" and not is_final_window
        thermal_generators = tuple(
            replace(
                unit,
                config=replace(
                    unit.config,
                    initial_on=bool(state["thermal"][unit.id]["initial_on"]),
                    initial_output_mw=float(state["thermal"][unit.id]["initial_output_mw"]),
                    initial_up_time_hours=float(state["thermal"][unit.id]["initial_up_time_hours"]),
                    initial_down_time_hours=float(
                        state["thermal"][unit.id]["initial_down_time_hours"]
                    ),
                    terminal_commitment_mode=(
                        "carry_residual_obligations"
                        if relax_terminal
                        else unit.config.terminal_commitment_mode
                    ),
                    terminal_on=None if relax_terminal else unit.config.terminal_on,
                ),
            )
            for unit in self.config.portfolio.thermal_generators
        )
        storage_units = tuple(
            replace(
                unit,
                config=replace(
                    unit.config,
                    initial_soc_mwh=float(state["storage"][unit.id]["initial_soc_mwh"]),
                    terminal_soc_mode="free" if relax_terminal else unit.config.terminal_soc_mode,
                ),
            )
            for unit in self.config.portfolio.storage_units
        )
        hydro_units = tuple(
            replace(
                unit,
                initial_reservoir_mwh=float(state["hydro"][unit.id]["initial_reservoir_mwh"]),
                terminal_reservoir_mode=(
                    "free" if relax_terminal else unit.terminal_reservoir_mode
                ),
                water_value_eur_per_mwh=(0.0 if relax_terminal else unit.water_value_eur_per_mwh),
            )
            for unit in self.config.portfolio.hydro_units
        )
        demand_units = tuple(
            self._rolling_window_demand(demand, state, window_start)
            for demand in self.config.portfolio.demand
        )
        portfolio = replace(
            self.config.portfolio,
            thermal_generators=thermal_generators,
            storage_units=storage_units,
            hydro_units=hydro_units,
            demand=demand_units,
        )
        return replace(
            self.config,
            portfolio=portfolio,
            thermal=thermal_generators[0].config,
            battery=storage_units[0].config,
            rolling_horizon=replace(self.config.rolling_horizon, enabled=False),
        )

    def _rolling_window_demand(
        self,
        demand: Any,
        state: dict[str, Any],
        window_start: int,
    ) -> Any:
        if demand.kind not in {"deferrable", "ev_charging"}:
            return demand
        remaining = float(state["demand"][demand.id]["remaining_task_energy_mwh"])
        task_end_period = (
            max(0, demand.task_end_period - window_start)
            if demand.task_end_period is not None
            else None
        )
        return replace(
            demand,
            task_required_energy_mwh=remaining,
            task_start_period=max(0, demand.task_start_period - window_start),
            task_end_period=task_end_period,
        )

    def _rolling_state_after_segment(
        self,
        previous_state: dict[str, Any],
        window_config: ModelConfig,
        frame: pd.DataFrame,
        *,
        window_start: int,
    ) -> dict[str, Any]:
        state = cast(dict[str, Any], json.loads(json.dumps(previous_state)))
        for thermal_unit in window_config.portfolio.thermal_generators:
            state["thermal"][thermal_unit.id] = self._thermal_state_after_segment(
                thermal_unit.id, thermal_unit.config, frame
            )
        for storage_unit in window_config.portfolio.storage_units:
            state["storage"][storage_unit.id] = {
                "initial_soc_mwh": float(frame[f"storage_soc_mwh__{storage_unit.id}"].iloc[-1])
            }
        for hydro_unit in window_config.portfolio.hydro_units:
            state["hydro"][hydro_unit.id] = {
                "initial_reservoir_mwh": float(
                    frame[f"hydro_reservoir_mwh__{hydro_unit.id}"].iloc[-1]
                )
            }
        dt = self.config.simulation.time_step_hours
        for demand in window_config.portfolio.demand:
            if demand.kind not in {"deferrable", "ev_charging"}:
                continue
            charge_column = f"demand_task_charge_mw__{demand.id}"
            charged = float(frame[charge_column].sum() * dt) if charge_column in frame else 0.0
            remaining = max(
                0.0,
                float(previous_state["demand"][demand.id]["remaining_task_energy_mwh"]) - charged,
            )
            state["demand"][demand.id] = {
                "remaining_task_energy_mwh": remaining,
                "last_implemented_period": window_start + len(frame) - 1,
            }
        return state

    def _thermal_state_after_segment(
        self,
        unit_id: str,
        thermal: Any,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        dt = self.config.simulation.time_step_hours
        on_values = frame[f"thermal_on__{unit_id}"].to_numpy(dtype=np.float64)
        output = float(frame[f"thermal_output_mw__{unit_id}"].iloc[-1])
        terminal_on = bool(round(float(on_values[-1])))
        matching = 0
        for value in reversed(on_values):
            if bool(round(float(value))) != terminal_on:
                break
            matching += 1
        if terminal_on:
            elapsed_up = matching * dt
            if matching == len(on_values):
                elapsed_up += thermal.initial_up_time_hours
            return {
                "initial_on": True,
                "initial_output_mw": output,
                "initial_up_time_hours": elapsed_up,
                "initial_down_time_hours": 0.0,
            }
        elapsed_down = matching * dt
        if matching == len(on_values):
            elapsed_down += thermal.initial_down_time_hours
        return {
            "initial_on": False,
            "initial_output_mw": 0.0,
            "initial_up_time_hours": 0.0,
            "initial_down_time_hours": elapsed_down,
        }

    def _terminal_commitment_from_state(
        self,
        state: dict[str, Any],
    ) -> dict[str, TerminalCommitmentState]:
        result: dict[str, TerminalCommitmentState] = {}
        for unit in self.config.portfolio.thermal_generators:
            thermal_state = state["thermal"][unit.id]
            result[unit.id] = TerminalCommitmentState(
                thermal_on=bool(thermal_state["initial_on"]),
                thermal_output_mw=float(thermal_state["initial_output_mw"]),
                consecutive_on_hours=float(thermal_state["initial_up_time_hours"]),
                consecutive_off_hours=float(thermal_state["initial_down_time_hours"]),
                residual_minimum_up_hours=max(
                    0.0,
                    unit.config.minimum_up_hours - float(thermal_state["initial_up_time_hours"]),
                )
                if bool(thermal_state["initial_on"])
                else 0.0,
                residual_minimum_down_hours=max(
                    0.0,
                    unit.config.minimum_down_hours
                    - float(thermal_state["initial_down_time_hours"]),
                )
                if not bool(thermal_state["initial_on"])
                else 0.0,
                terminal_commitment_mode=unit.config.terminal_commitment_mode,
            )
        return result

    def _write_rolling_checkpoint(
        self,
        path: Path,
        next_start: int,
        state: dict[str, Any],
        windows: list[dict[str, Any]],
        timeseries: pd.DataFrame,
        asset_timeseries: pd.DataFrame,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "next_start": next_start,
            "state": state,
            "windows": windows,
            "timeseries_columns": list(timeseries.columns),
            "timeseries": timeseries.to_dict(orient="records"),
            "asset_timeseries_columns": list(asset_timeseries.columns),
            "asset_timeseries": asset_timeseries.to_dict(orient="records"),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def _read_rolling_checkpoint(self, path: Path) -> dict[str, Any]:
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        if payload.get("schema_version") != 1:
            raise OptimisationError("Unsupported rolling checkpoint schema version")
        return payload

    @staticmethod
    def _records_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(records)
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame[columns]
        return frame

    def _rolling_cost_components(self, frame: pd.DataFrame) -> dict[str, float]:
        dt = self.config.simulation.time_step_hours
        imports = self.config.imports
        penalties = self.config.penalties
        network_efficiency = 1.0 - self.config.network.loss_fraction
        thermal_units = self.config.portfolio.thermal_generators
        storage_units = self.config.portfolio.storage_units
        demand_shed_cost_columns = [
            column
            for column in frame.columns
            if column.startswith("demand_involuntary_shed_cost_eur__")
        ]
        fixed_network_shedding_cost = (
            frame["network_capacity_shed_mw"].sum() * penalties.lost_load_eur_per_mwh * dt
        )
        return {
            "thermal_variable_cost_eur": float(
                sum(frame[f"thermal_variable_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "thermal_no_load_cost_eur": float(
                sum(frame[f"thermal_no_load_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "startup_cost_eur": float(
                sum(frame[f"thermal_startup_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "shutdown_cost_eur": float(
                sum(frame[f"thermal_shutdown_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "import_energy_cost_eur": float(
                frame["imports_mw"].sum() * dt * imports.price_eur_per_mwh
            ),
            "battery_throughput_cost_eur": float(
                sum(
                    frame[f"storage_throughput_cost_eur__{unit.id}"].sum() for unit in storage_units
                )
            ),
            "storage_degradation_cost_eur": float(
                sum(
                    frame[f"storage_degradation_cost_eur__{unit.id}"].sum()
                    for unit in storage_units
                )
            ),
            "hydro_terminal_value_eur": -float(frame["hydro_terminal_value_eur"].sum()),
            "demand_voluntary_curtailment_cost_eur": float(
                frame["demand_voluntary_curtailment_cost_eur"].sum()
            ),
            "demand_shift_cost_eur": float(frame["demand_shift_cost_eur"].sum()),
            "demand_task_unserved_cost_eur": float(frame["demand_task_unserved_cost_eur"].sum()),
            "thermal_carbon_cost_eur": float(
                sum(frame[f"thermal_carbon_cost_eur__{unit.id}"].sum() for unit in thermal_units)
            ),
            "import_carbon_cost_eur": float(
                frame["imports_mw"].sum()
                * dt
                * imports.emission_factor_tonnes_per_mwh
                * penalties.carbon_price_eur_per_tonne
            ),
            "renewable_curtailment_cost_eur": float(
                frame["renewable_curtailed_mw"].sum()
                * dt
                * penalties.renewable_curtailment_eur_per_mwh
            ),
            "reserve_procurement_cost_eur": float(
                frame["reserve_procurement_cost_eur"].sum()
                if "reserve_procurement_cost_eur" in frame
                else 0.0
            ),
            "reserve_shortfall_cost_eur": float(
                frame["reserve_shortfall_cost_eur"].sum()
                if "reserve_shortfall_cost_eur" in frame
                else 0.0
            ),
            "dispatch_load_shedding_cost_eur": float(
                frame[demand_shed_cost_columns].sum().sum()
                if demand_shed_cost_columns
                else frame["source_load_shed_mw"].sum()
                * network_efficiency
                * dt
                * penalties.lost_load_eur_per_mwh
            ),
            "network_capacity_load_shedding_cost_eur": float(fixed_network_shedding_cost),
        }

    @staticmethod
    def _combine_window_diagnostics(windows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "rolling_window_count": float(len(windows)),
            "rolling_total_solver_runtime_seconds": float(
                sum(window["solver_runtime_seconds"] for window in windows)
            ),
        }

    def _full_horizon_comparison(
        self,
        data: pd.DataFrame,
        rolling_frame: pd.DataFrame,
        rolling_objective_eur: float,
    ) -> dict[str, float]:
        full = SimulationEngine(
            replace(
                self.config, rolling_horizon=replace(self.config.rolling_horizon, enabled=False)
            )
        )._run_full_horizon_data(data)
        common_columns = [
            "renewable_used_mw",
            "thermal_output_mw",
            "battery_soc_mwh",
            "hydro_reservoir_mwh",
            "served_demand_mw",
        ]
        max_abs_difference = 0.0
        for column in common_columns:
            if column in rolling_frame and column in full.timeseries:
                max_abs_difference = max(
                    max_abs_difference,
                    float((rolling_frame[column] - full.timeseries[column]).abs().max()),
                )
        return {
            "full_horizon_objective_eur": full.objective_eur,
            "rolling_objective_eur": rolling_objective_eur,
            "objective_delta_eur": rolling_objective_eur - full.objective_eur,
            "max_abs_common_timeseries_difference": max_abs_difference,
        }

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
        demand_profiles_mw: dict[str, FloatArray] | None,
    ) -> DispatchResult:
        return UnitCommitment(self.config).solve(
            renewable.aggregate_mw,
            gross_demand_mw,
            thermal_availability_factors=registry.thermal_availability_factors(data),
            fuel_price_series=self._fuel_price_series(data),
            storage_availability_factors=registry.storage_availability_factors(data),
            hydro_inflows_mw=registry.hydro_inflows_mw(data),
            demand_profiles_mw=demand_profiles_mw,
            renewable_availability_by_asset_mw=renewable.by_asset_mw,
            line_availability_factors=self._line_availability_factors(data),
        )

    def _source_side_demand_profiles(
        self,
        registry: AssetRegistry,
        data: pd.DataFrame,
        end_user_demand_mw: FloatArray,
        distribution: DistributionDemand,
    ) -> dict[str, FloatArray] | None:
        if self.config.network.network_mode != "nodal" and not any(
            demand.kind != "fixed" or demand.value_of_lost_load_eur_per_mwh is not None
            for demand in self.config.portfolio.demand
        ):
            return None
        profiles = registry.demand_profiles_mw(data)
        scale = np.divide(
            distribution.gross_demand_mw,
            end_user_demand_mw,
            out=np.zeros_like(distribution.gross_demand_mw, dtype=np.float64),
            where=end_user_demand_mw > 0.0,
        )
        return {asset_id: values * scale for asset_id, values in profiles.items()}

    def _line_availability_factors(self, data: pd.DataFrame) -> dict[str, FloatArray]:
        factors: dict[str, FloatArray] = {}
        for line in self.config.portfolio.lines:
            if line.availability_factor_key is None:
                continue
            if line.availability_factor_key not in data.columns:
                raise OptimisationError(
                    f"Line {line.id!r} references missing input column "
                    f"{line.availability_factor_key!r}"
                )
            values = pd.to_numeric(data[line.availability_factor_key], errors="coerce").to_numpy(
                dtype="float64"
            )
            if not pd.notna(values).all():
                raise OptimisationError(
                    f"Line availability column {line.availability_factor_key!r} "
                    "contains non-finite values"
                )
            if ((values < 0.0) | (values > 1.0)).any():
                raise OptimisationError(
                    f"Line availability column {line.availability_factor_key!r} must be in [0, 1]"
                )
            factors[line.id] = values
        return factors

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
        dispatch = self._renewable_dispatch_timeseries(data["timestamp"], renewable, frame)
        return (
            renewable.asset_table.append(dispatch)
            .append(self._thermal_asset_timeseries(data["timestamp"], frame))
            .append(self._storage_asset_timeseries(data["timestamp"], frame))
            .append(self._hydro_asset_timeseries(data["timestamp"], frame))
            .append(self._demand_asset_timeseries(data["timestamp"], frame))
        )

    def _renewable_dispatch_timeseries(
        self,
        timestamps: pd.Series,
        renewable: RenewableAvailability,
        frame: pd.DataFrame,
    ) -> AssetTimeSeries:
        if all(f"renewable_used_mw__{asset_id}" in frame for asset_id in renewable.by_asset_mw):
            used_by_asset = {
                asset_id: frame[f"renewable_used_mw__{asset_id}"].to_numpy(dtype="float64")
                for asset_id in renewable.by_asset_mw
            }
            curtailed_by_asset = {
                asset_id: renewable.by_asset_mw[asset_id] - used_by_asset[asset_id]
                for asset_id in renewable.by_asset_mw
            }
            return AssetTimeSeries.from_variable_matrix(
                timestamps,
                used_by_asset,
                "used_mw",
                "MW",
            ).append(
                AssetTimeSeries.from_variable_matrix(
                    timestamps,
                    curtailed_by_asset,
                    "curtailed_mw",
                    "MW",
                )
            )
        return allocate_renewable_dispatch(
            timestamps,
            renewable.by_asset_mw,
            renewable.aggregate_mw,
            frame["renewable_used_mw"].to_numpy(dtype="float64"),
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

    def _demand_asset_timeseries(
        self,
        timestamps: pd.Series,
        frame: pd.DataFrame,
    ) -> AssetTimeSeries:
        pieces: list[pd.DataFrame] = []
        variable_map = {
            "demand_baseline_mw": ("baseline_mw_source", "MW-source"),
            "demand_adjusted_mw": ("adjusted_mw_source", "MW-source"),
            "demand_served_mw": ("served_mw_source", "MW-source"),
            "demand_involuntary_shed_mw": ("involuntary_shed_mw_source", "MW-source"),
            "demand_voluntary_curtailment_mw": ("voluntary_curtailment_mw_source", "MW-source"),
            "demand_shift_down_mw": ("shifted_out_mw_source", "MW-source"),
            "demand_shift_up_mw": ("shifted_in_mw_source", "MW-source"),
            "demand_task_charge_mw": ("task_charge_mw_source", "MW-source"),
            "demand_task_unserved_mwh": ("task_unserved_mwh", "MWh"),
        }
        for demand in self.config.portfolio.demand:
            for prefix, (variable, unit) in variable_map.items():
                column = f"{prefix}__{demand.id}"
                if column not in frame:
                    continue
                pieces.append(
                    pd.DataFrame(
                        {
                            "timestamp": timestamps.to_numpy(),
                            "asset_id": demand.id,
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
            "adjusted_demand_mwh": energy("adjusted_end_user_demand_mw"),
            "served_demand_mwh": served_mwh,
            "unserved_energy_mwh": energy("total_load_shed_mw"),
            "voluntary_demand_curtailment_mwh": energy("demand_voluntary_curtailment_end_user_mw"),
            "demand_shifted_out_mwh": energy("demand_shifted_out_end_user_mw"),
            "demand_shifted_in_mwh": energy("demand_shifted_in_end_user_mw"),
            "demand_task_charge_mwh": energy("demand_task_charge_end_user_mw"),
            "demand_task_unserved_mwh": float(frame["demand_task_unserved_mwh"].sum()),
            "demand_assets": self._demand_asset_summary(frame),
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
            "reserves": self._reserve_summary(frame),
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
            "network": self._network_summary(frame),
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
                            "demand_adjusted_mw",
                            "battery_charge_mw",
                        ),
                    ),
                    "delivered_demand_balance": self._residual_summary(
                        frame,
                        "delivered_demand_balance",
                        "delivered_demand_balance_residual_mw",
                        (
                            "served_demand_mw",
                            "total_load_shed_mw",
                            "demand_voluntary_curtailment_end_user_mw",
                            "demand_shifted_out_end_user_mw",
                            "demand_shifted_in_end_user_mw",
                            "demand_task_charge_end_user_mw",
                            "end_user_demand_mw",
                        ),
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

    def _network_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        if "line_max_abs_utilisation" not in frame:
            return {"mode": self.config.network.network_mode}
        return {
            "mode": self.config.network.network_mode,
            "congested_hours": float(
                frame["line_congested"].sum() * self.config.simulation.time_step_hours
            ),
            "max_abs_line_utilisation": float(frame["line_max_abs_utilisation"].max()),
            "max_line_overload_residual_mw": float(frame["line_overload_residual_mw"].max()),
            "max_abs_bus_balance_residual_mw": float(frame["bus_balance_residual_mw"].abs().max()),
        }

    def _reserve_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        if "reserve_upward_requirement_mw" not in frame:
            return {"enabled": False}
        dt = self.config.simulation.time_step_hours
        return {
            "enabled": True,
            "upward_requirement_mwh": float(frame["reserve_upward_requirement_mw"].sum() * dt),
            "downward_requirement_mwh": float(frame["reserve_downward_requirement_mw"].sum() * dt),
            "upward_procured_mwh": float(frame["reserve_upward_procured_mw"].sum() * dt),
            "downward_procured_mwh": float(frame["reserve_downward_procured_mw"].sum() * dt),
            "upward_shortfall_mwh": float(frame["reserve_upward_shortfall_mw"].sum() * dt),
            "downward_shortfall_mwh": float(frame["reserve_downward_shortfall_mw"].sum() * dt),
            "hours_with_upward_shortfall": float(
                (frame["reserve_upward_shortfall_mw"] > 0.0).sum() * dt
            ),
            "hours_with_downward_shortfall": float(
                (frame["reserve_downward_shortfall_mw"] > 0.0).sum() * dt
            ),
            "max_abs_upward_residual_mw": float(frame["reserve_upward_residual_mw"].abs().max()),
            "max_abs_downward_residual_mw": float(
                frame["reserve_downward_residual_mw"].abs().max()
            ),
            "procurement_cost_eur": float(frame["reserve_procurement_cost_eur"].sum()),
            "shortfall_cost_eur": float(frame["reserve_shortfall_cost_eur"].sum()),
        }

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

    def _demand_asset_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        dt = self.config.simulation.time_step_hours
        result: dict[str, Any] = {}
        for demand in self.config.portfolio.demand:
            if f"demand_baseline_mw__{demand.id}" not in frame:
                continue
            result[demand.id] = {
                "kind": demand.kind,
                "sector": demand.sector,
                "baseline_mwh_source": float(frame[f"demand_baseline_mw__{demand.id}"].sum() * dt),
                "adjusted_mwh_source": float(frame[f"demand_adjusted_mw__{demand.id}"].sum() * dt),
                "served_mwh_source": float(frame[f"demand_served_mw__{demand.id}"].sum() * dt),
                "voluntary_curtailment_mwh_source": self._energy_prefixed_asset(
                    frame,
                    f"demand_voluntary_curtailment_mw__{demand.id}",
                ),
                "shifted_out_mwh_source": self._energy_prefixed_asset(
                    frame,
                    f"demand_shift_down_mw__{demand.id}",
                ),
                "shifted_in_mwh_source": self._energy_prefixed_asset(
                    frame,
                    f"demand_shift_up_mw__{demand.id}",
                ),
                "task_charge_mwh_source": self._energy_prefixed_asset(
                    frame,
                    f"demand_task_charge_mw__{demand.id}",
                ),
                "task_unserved_mwh": self._sum_optional_column(
                    frame,
                    f"demand_task_unserved_mwh__{demand.id}",
                ),
                "involuntary_shed_mwh_source": float(
                    frame[f"demand_involuntary_shed_mw__{demand.id}"].sum() * dt
                ),
            }
        return result

    def _energy_prefixed_asset(self, frame: pd.DataFrame, column: str) -> float:
        if column not in frame:
            return 0.0
        return float(frame[column].sum() * self.config.simulation.time_step_hours)

    @staticmethod
    def _sum_optional_column(frame: pd.DataFrame, column: str) -> float:
        if column not in frame:
            return 0.0
        return float(frame[column].sum())

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
        source_right = frame["demand_adjusted_mw"] + frame["battery_charge_mw"]
        frame["source_balance_residual_mw"] = source_left - source_right
        frame["delivered_demand_balance_residual_mw"] = (
            frame["served_demand_mw"]
            + frame["total_load_shed_mw"]
            + frame["demand_voluntary_curtailment_end_user_mw"]
            + frame["demand_shifted_out_end_user_mw"]
            - frame["demand_shifted_in_end_user_mw"]
            - frame["demand_task_charge_end_user_mw"]
            - frame["end_user_demand_mw"]
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
                "demand_adjusted_mw",
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
