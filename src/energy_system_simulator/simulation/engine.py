from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from energy_system_simulator.config import ModelConfig
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.data import load_input_data
from energy_system_simulator.dispatch import (
    FormulationStatistics,
    TerminalCommitmentState,
    UnitCommitment,
)
from energy_system_simulator.exceptions import OptimisationError
from energy_system_simulator.generation import SolarPlant, WindFarm
from energy_system_simulator.network import DistributionNetwork


@dataclass(frozen=True)
class SimulationResult:
    """Complete simulation output and aggregate metrics."""

    timeseries: pd.DataFrame
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
    numerical_diagnostics: dict[str, float]


class SimulationEngine:
    """Coordinate data loading, generation models, dispatch, and accounting."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def run(self) -> SimulationResult:
        """Execute the configured simulation."""
        data = load_input_data(
            self.config.paths.input_csv,
            self.config.simulation.time_step_hours,
        )
        solar = SolarPlant(self.config.solar).output_mw(
            data["irradiance_w_m2"].to_numpy(),
            data["ambient_temperature_c"].to_numpy(),
        )
        wind = WindFarm(self.config.wind).output_mw(data["wind_speed_m_s"].to_numpy())
        renewable = solar + wind

        network = DistributionNetwork(self.config.network)
        distribution = network.prepare_demand(data["demand_mw"].to_numpy())
        dispatch = UnitCommitment(self.config).solve(
            renewable,
            distribution.gross_demand_mw,
        )

        frame = dispatch.frame.copy()
        frame.insert(0, "timestamp", data["timestamp"].to_numpy())
        frame.insert(1, "end_user_demand_mw", distribution.end_user_demand_mw)
        frame.insert(2, "deliverable_demand_mw", distribution.deliverable_demand_mw)
        frame.insert(3, "network_capacity_shed_mw", distribution.network_capacity_shed_mw)
        frame.insert(4, "solar_available_mw", solar)
        frame.insert(5, "wind_available_mw", wind)

        efficiency = network.efficiency
        frame["dispatch_load_shed_mw"] = efficiency * frame["source_load_shed_mw"]
        frame["total_load_shed_mw"] = (
            frame["network_capacity_shed_mw"] + frame["dispatch_load_shed_mw"]
        )
        frame["served_demand_mw"] = frame["end_user_demand_mw"] - frame["total_load_shed_mw"]
        source_power_to_load = frame["gross_demand_mw"] - frame["source_load_shed_mw"]
        frame["network_losses_mw"] = source_power_to_load * self.config.network.loss_fraction
        frame["thermal_emissions_tonnes"] = (
            frame["thermal_output_mw"]
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
            total_objective_eur,
            cost_components,
            dispatch.terminal_commitment_state,
            dispatch.numerical_diagnostics,
        )
        return SimulationResult(
            timeseries=frame,
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
            numerical_diagnostics=dispatch.numerical_diagnostics,
        )

    def _summary(
        self,
        frame: pd.DataFrame,
        objective_eur: float,
        cost_components: dict[str, float],
        terminal_commitment_state: TerminalCommitmentState,
        numerical_diagnostics: dict[str, float],
    ) -> dict[str, Any]:
        dt = self.config.simulation.time_step_hours

        def energy(column: str) -> float:
            return float(frame[column].sum() * dt)

        demand_mwh = energy("end_user_demand_mw")
        served_mwh = energy("served_demand_mw")
        renewable_used_mwh = energy("renewable_used_mw")
        primary_source_generation_mwh = (
            renewable_used_mwh + energy("thermal_output_mw") + energy("imports_mw")
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
            "renewable_share_of_primary_generation": (
                renewable_used_mwh / primary_source_generation_mwh
                if primary_source_generation_mwh > 0.0
                else 0.0
            ),
            "thermal_generation_mwh": energy("thermal_output_mw"),
            "thermal_starts": int(frame["thermal_startup"].sum()),
            "thermal_shutdowns": int(frame["thermal_shutdown"].sum()),
            "terminal_commitment": asdict(terminal_commitment_state),
            "imports_mwh": energy("imports_mw"),
            "battery_charge_mwh": energy("battery_charge_mw"),
            "battery_discharge_mwh": energy("battery_discharge_mw"),
            "final_battery_soc_mwh": float(frame["battery_soc_mwh"].iloc[-1]),
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
                "residual_summaries": {
                    "source_balance": self._residual_summary(
                        frame,
                        "source_balance",
                        "source_balance_residual_mw",
                        (
                            "renewable_used_mw",
                            "thermal_output_mw",
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

    def _add_energy_reconciliation(self, frame: pd.DataFrame) -> None:
        dt = self.config.simulation.time_step_hours
        battery = self.config.battery
        source_left = (
            frame["renewable_used_mw"]
            + frame["thermal_output_mw"]
            + frame["battery_discharge_mw"]
            + frame["imports_mw"]
            + frame["source_load_shed_mw"]
        )
        source_right = frame["gross_demand_mw"] + frame["battery_charge_mw"]
        frame["source_balance_residual_mw"] = source_left - source_right
        frame["delivered_demand_balance_residual_mw"] = (
            frame["served_demand_mw"] + frame["total_load_shed_mw"] - frame["end_user_demand_mw"]
        )
        previous_soc = frame["battery_soc_mwh"].shift(1)
        previous_soc.iloc[0] = battery.initial_soc_mwh
        frame["battery_energy_change_mwh"] = frame["battery_soc_mwh"] - previous_soc
        expected_change = (
            battery.charge_efficiency * frame["battery_charge_mw"] * dt
            - frame["battery_discharge_mw"] * dt / battery.discharge_efficiency
        )
        frame["battery_energy_residual_mwh"] = frame["battery_energy_change_mwh"] - expected_change
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
