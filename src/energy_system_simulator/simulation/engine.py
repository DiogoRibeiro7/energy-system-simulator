from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from energy_system_simulator.config import ModelConfig
from energy_system_simulator.data import load_input_data
from energy_system_simulator.dispatch import FormulationStatistics, UnitCommitment
from energy_system_simulator.generation import SolarPlant, WindFarm
from energy_system_simulator.network import DistributionNetwork


@dataclass(frozen=True)
class SimulationResult:
    """Complete simulation output and aggregate metrics."""

    timeseries: pd.DataFrame
    summary: dict[str, Any]
    objective_eur: float
    solver_message: str
    mip_gap: float | None
    formulation_statistics: FormulationStatistics


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

        fixed_network_shedding_cost = (
            frame["network_capacity_shed_mw"].sum()
            * self.config.penalties.lost_load_eur_per_mwh
            * self.config.simulation.time_step_hours
        )
        total_objective_eur = dispatch.objective_eur + float(fixed_network_shedding_cost)
        summary = self._summary(frame, total_objective_eur)
        return SimulationResult(
            timeseries=frame,
            summary=summary,
            objective_eur=total_objective_eur,
            solver_message=dispatch.solver_message,
            mip_gap=dispatch.mip_gap,
            formulation_statistics=dispatch.formulation_statistics,
        )

    def _summary(self, frame: pd.DataFrame, objective_eur: float) -> dict[str, Any]:
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
            "total_demand_mwh": demand_mwh,
            "served_demand_mwh": served_mwh,
            "unserved_energy_mwh": energy("total_load_shed_mw"),
            "loss_of_load_probability": float((frame["total_load_shed_mw"] > 1e-7).mean()),
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
        }

    @staticmethod
    def ensure_output_directory(path: Path) -> None:
        """Create the configured output directory when needed."""
        path.mkdir(parents=True, exist_ok=True)
