from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_system_simulator.config import (
    BatteryConfig,
    DemandConfig,
    HydroUnitConfig,
    ImportResourceConfig,
    ModelConfig,
    NetworkConfig,
    RenewableGeneratorConfig,
    SolarConfig,
    StorageUnitConfig,
    ThermalConfig,
    ThermalGeneratorConfig,
    WindConfig,
)
from energy_system_simulator.exceptions import DataValidationError
from energy_system_simulator.generation import SolarPlant, WindFarm

FloatArray = npt.NDArray[np.float64]
AssetRole = Literal["availability", "dispatchable", "intertemporal", "demand", "network"]


@dataclass(frozen=True)
class RenewableProfile:
    """Transparent renewable availability decomposition for one asset."""

    variables: dict[str, tuple[FloatArray, str]]

    @property
    def available_mw(self) -> FloatArray:
        return self.variables["available_mw"][0]

    def to_asset_timeseries(self, timestamps: pd.Series, asset_id: str) -> AssetTimeSeries:
        pieces: list[pd.DataFrame] = []
        for variable, (values, unit) in self.variables.items():
            pieces.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps.to_numpy(),
                        "asset_id": asset_id,
                        "variable": variable,
                        "value": values,
                        "unit": unit,
                    }
                )
            )
        return AssetTimeSeries(pd.concat(pieces, ignore_index=True))


class TimeDependentAsset(Protocol):
    """Minimal runtime contract for configured assets with time-dependent values."""

    asset_id: str
    bus_id: str
    role: AssetRole


class AvailabilityAsset(TimeDependentAsset, Protocol):
    """Asset that contributes exogenous availability before dispatch."""

    def availability_mw(self, data: pd.DataFrame) -> FloatArray:
        """Return source-side available power in MW for each model period."""


@dataclass(frozen=True)
class RenewableAsset:
    """Runtime renewable resource resolved from portfolio configuration."""

    asset_id: str
    bus_id: str
    kind: Literal["solar", "wind"]
    time_series_key: str
    ambient_temperature_key: str | None
    availability_model: Literal["simple", "detailed", "power_curve"]
    availability_factor: float
    availability_factor_key: str | None
    maintenance_factor_key: str | None
    dc_capacity_mw: float | None = None
    inverter_ac_capacity_mw: float | None = None
    inverter_efficiency: float = 1.0
    degradation_factor: float = 1.0
    irradiance_basis: Literal["plane_of_array", "global_horizontal"] = "plane_of_array"
    transposition_model: Literal["none", "isotropic_fixed_tilt"] = "none"
    tilt_degrees: float | None = None
    albedo: float = 0.2
    soiling_loss_fraction: float = 0.0
    soiling_loss_key: str | None = None
    snow_loss_fraction: float = 0.0
    snow_loss_key: str | None = None
    measurement_height_m: float | None = None
    hub_height_m: float | None = None
    wind_speed_adjustment: Literal["none", "power_law", "logarithmic"] = "none"
    wind_shear_exponent: float = 1.0 / 7.0
    roughness_length_m: float | None = None
    air_density_correction: bool = False
    air_temperature_key: str | None = None
    air_pressure_key: str | None = None
    turbine_count: int | None = None
    turbine_rated_capacity_mw: float | None = None
    power_curve_speed_m_s: FloatArray | None = None
    power_curve_power_mw: FloatArray | None = None
    wake_loss_fraction: float = 0.0
    wake_loss_key: str | None = None
    electrical_loss_fraction: float = 0.0
    electrical_loss_key: str | None = None
    solar: SolarConfig | None = None
    wind: WindConfig | None = None
    role: AssetRole = "availability"

    @classmethod
    def from_config(cls, config: RenewableGeneratorConfig) -> RenewableAsset:
        if config.kind == "solar":
            return cls(
                asset_id=config.id,
                bus_id=config.bus_id,
                kind="solar",
                time_series_key=config.time_series_key,
                ambient_temperature_key=config.ambient_temperature_key,
                availability_model=config.availability_model,
                availability_factor=config.availability_factor,
                availability_factor_key=config.availability_factor_key,
                maintenance_factor_key=config.maintenance_factor_key,
                dc_capacity_mw=config.dc_capacity_mw,
                inverter_ac_capacity_mw=config.inverter_ac_capacity_mw,
                inverter_efficiency=config.inverter_efficiency,
                degradation_factor=config.degradation_factor,
                irradiance_basis=config.irradiance_basis,
                transposition_model=config.transposition_model,
                tilt_degrees=config.tilt_degrees,
                albedo=config.albedo,
                soiling_loss_fraction=config.soiling_loss_fraction,
                soiling_loss_key=config.soiling_loss_key,
                snow_loss_fraction=config.snow_loss_fraction,
                snow_loss_key=config.snow_loss_key,
                solar=SolarConfig(
                    capacity_mw=config.capacity_mw,
                    performance_ratio=_required(
                        (
                            config.module_performance_ratio
                            if config.module_performance_ratio is not None
                            else config.performance_ratio
                        ),
                        config.id,
                    ),
                    reference_irradiance_w_m2=_required(
                        config.reference_irradiance_w_m2,
                        config.id,
                    ),
                    temperature_coefficient_per_c=_required(
                        config.temperature_coefficient_per_c,
                        config.id,
                    ),
                    nominal_operating_cell_temperature_c=_required(
                        config.nominal_operating_cell_temperature_c,
                        config.id,
                    ),
                ),
            )
        return cls(
            asset_id=config.id,
            bus_id=config.bus_id,
            kind="wind",
            time_series_key=config.time_series_key,
            ambient_temperature_key=None,
            availability_model=config.availability_model,
            availability_factor=config.availability_factor,
            availability_factor_key=config.availability_factor_key,
            maintenance_factor_key=config.maintenance_factor_key,
            measurement_height_m=config.measurement_height_m,
            hub_height_m=config.hub_height_m,
            wind_speed_adjustment=config.wind_speed_adjustment,
            wind_shear_exponent=config.wind_shear_exponent,
            roughness_length_m=config.roughness_length_m,
            air_density_correction=config.air_density_correction,
            air_temperature_key=config.air_temperature_key,
            air_pressure_key=config.air_pressure_key,
            turbine_count=config.turbine_count,
            turbine_rated_capacity_mw=config.turbine_rated_capacity_mw,
            power_curve_speed_m_s=(
                np.asarray([point.wind_speed_m_s for point in config.power_curve], dtype=np.float64)
                if config.power_curve
                else None
            ),
            power_curve_power_mw=(
                np.asarray([point.power_mw for point in config.power_curve], dtype=np.float64)
                if config.power_curve
                else None
            ),
            wake_loss_fraction=config.wake_loss_fraction,
            wake_loss_key=config.wake_loss_key,
            electrical_loss_fraction=config.electrical_loss_fraction,
            electrical_loss_key=config.electrical_loss_key,
            wind=WindConfig(
                capacity_mw=config.capacity_mw,
                cut_in_speed_m_s=_required(config.cut_in_speed_m_s, config.id),
                rated_speed_m_s=_required(config.rated_speed_m_s, config.id),
                cut_out_speed_m_s=_required(config.cut_out_speed_m_s, config.id),
            ),
        )

    def availability_mw(self, data: pd.DataFrame) -> FloatArray:
        return self.availability_profile(data).available_mw

    def availability_profile(self, data: pd.DataFrame) -> RenewableProfile:
        if self.kind == "solar":
            return self._solar_profile(data)
        return self._wind_profile(data)

    def _solar_profile(self, data: pd.DataFrame) -> RenewableProfile:
        solar = cast(SolarConfig, self.solar)
        if self.availability_model == "simple":
            if self.ambient_temperature_key is None:
                raise DataValidationError(
                    f"{self.asset_id} is missing ambient temperature data key"
                )
            irradiance = _column(data, self.time_series_key, self.asset_id, nonnegative=True)
            ambient = _column(
                data,
                self.ambient_temperature_key,
                self.asset_id,
                nonnegative=False,
            )
            final = SolarPlant(solar).output_mw(irradiance, ambient)
            cell_temperature = (
                ambient + ((solar.nominal_operating_cell_temperature_c - 20.0) / 800.0) * irradiance
            )
            temperature_factor = 1.0 + solar.temperature_coefficient_per_c * (
                cell_temperature - 25.0
            )
            dc_potential = solar.capacity_mw * irradiance / solar.reference_irradiance_w_m2
            ac_before_clip = dc_potential * solar.performance_ratio * temperature_factor
            clipping_loss = np.maximum(ac_before_clip - solar.capacity_mw, 0.0)
            temperature_loss = np.maximum(
                dc_potential * solar.performance_ratio - ac_before_clip,
                0.0,
            )
            availability_factor = self._availability_factor(data)
            maintenance_factor = self._maintenance_factor(data)
            final = final * availability_factor * maintenance_factor
            availability_loss = SolarPlant(solar).output_mw(irradiance, ambient) - final
            return RenewableProfile(
                {
                    "dc_potential_mw": (dc_potential.astype(np.float64), "MW"),
                    "ac_potential_mw": (
                        np.minimum(ac_before_clip, solar.capacity_mw).astype(np.float64),
                        "MW",
                    ),
                    "clipping_loss_mw": (clipping_loss.astype(np.float64), "MW"),
                    "temperature_loss_mw": (temperature_loss.astype(np.float64), "MW"),
                    "other_derating_loss_mw": (availability_loss.astype(np.float64), "MW"),
                    "available_mw": (final.astype(np.float64), "MW"),
                }
            )

        if self.ambient_temperature_key is None:
            raise DataValidationError(f"{self.asset_id} is missing ambient temperature data key")
        irradiance = _column(data, self.time_series_key, self.asset_id, nonnegative=True)
        ambient = _column(data, self.ambient_temperature_key, self.asset_id, nonnegative=False)
        plane_irradiance = self._solar_plane_irradiance(irradiance)
        dc_capacity = self.dc_capacity_mw if self.dc_capacity_mw is not None else solar.capacity_mw
        inverter_ac = (
            self.inverter_ac_capacity_mw
            if self.inverter_ac_capacity_mw is not None
            else solar.capacity_mw
        )
        module_ratio = self.degradation_factor
        cell_temperature = (
            ambient
            + ((solar.nominal_operating_cell_temperature_c - 20.0) / 800.0) * plane_irradiance
        )
        temperature_factor = np.maximum(
            0.0,
            1.0 + solar.temperature_coefficient_per_c * (cell_temperature - 25.0),
        )
        dc_potential = (
            dc_capacity
            * solar.performance_ratio
            * plane_irradiance
            / solar.reference_irradiance_w_m2
        )
        temperature_adjusted = dc_potential * temperature_factor
        ac_before_clip = temperature_adjusted * self.inverter_efficiency
        ac_potential = np.minimum(ac_before_clip, inverter_ac)
        clipping_loss = np.maximum(ac_before_clip - inverter_ac, 0.0)
        soiling_factor = self._loss_factor(data, self.soiling_loss_fraction, self.soiling_loss_key)
        snow_factor = self._loss_factor(data, self.snow_loss_fraction, self.snow_loss_key)
        availability_factor = self._availability_factor(data)
        maintenance_factor = self._maintenance_factor(data)
        derating_factor = (
            module_ratio * soiling_factor * snow_factor * availability_factor * maintenance_factor
        )
        final = ac_potential * derating_factor
        temperature_loss = np.maximum(dc_potential - temperature_adjusted, 0.0)
        other_derating_loss = ac_potential - final
        return RenewableProfile(
            {
                "dc_potential_mw": (dc_potential.astype(np.float64), "MW"),
                "ac_potential_mw": (ac_potential.astype(np.float64), "MW"),
                "clipping_loss_mw": (clipping_loss.astype(np.float64), "MW"),
                "temperature_loss_mw": (temperature_loss.astype(np.float64), "MW"),
                "other_derating_loss_mw": (other_derating_loss.astype(np.float64), "MW"),
                "available_mw": (final.astype(np.float64), "MW"),
            }
        )

    def _wind_profile(self, data: pd.DataFrame) -> RenewableProfile:
        wind = cast(WindConfig, self.wind)
        speed = _column(data, self.time_series_key, self.asset_id, nonnegative=True)
        hub_speed = self._hub_height_wind_speed(speed)
        if self.availability_model == "power_curve":
            speed_curve = cast(FloatArray, self.power_curve_speed_m_s)
            power_curve = cast(FloatArray, self.power_curve_power_mw)
            if (
                hub_speed.min(initial=0.0) < speed_curve[0]
                or hub_speed.max(initial=0.0) > speed_curve[-1]
            ):
                raise DataValidationError(
                    f"Input column {self.time_series_key!r} for asset {self.asset_id!r} "
                    "falls outside the configured wind power curve range"
                )
            gross_per_curve = np.interp(hub_speed, speed_curve, power_curve)
            gross = gross_per_curve * (self.turbine_count or 1)
            rated_capacity = self._wind_rated_capacity_mw()
            gross = np.minimum(gross, rated_capacity)
            shutdown_loss = np.zeros_like(gross)
            after_shutdown = gross
        else:
            simple = WindFarm(wind).output_mw(hub_speed)
            rated_capacity = self._wind_rated_capacity_mw()
            partial = (hub_speed >= wind.cut_in_speed_m_s) & (hub_speed < wind.rated_speed_m_s)
            at_or_above_rated = hub_speed >= wind.rated_speed_m_s
            gross = np.zeros_like(hub_speed, dtype=np.float64)
            numerator = hub_speed[partial] ** 3 - wind.cut_in_speed_m_s**3
            denominator = wind.rated_speed_m_s**3 - wind.cut_in_speed_m_s**3
            gross[partial] = rated_capacity * numerator / denominator
            gross[at_or_above_rated] = rated_capacity
            shutdown = hub_speed >= wind.cut_out_speed_m_s
            shutdown_loss = np.where(shutdown, rated_capacity, 0.0).astype(np.float64)
            after_shutdown = simple * (rated_capacity / wind.capacity_mw)
        density_factor = self._air_density_factor(data)
        after_density = after_shutdown * density_factor
        wake_factor = self._loss_factor(data, self.wake_loss_fraction, self.wake_loss_key)
        after_wake = after_density * wake_factor
        electrical_factor = self._loss_factor(
            data,
            self.electrical_loss_fraction,
            self.electrical_loss_key,
        )
        after_electrical = after_wake * electrical_factor
        availability_factor = self._availability_factor(data) * self._maintenance_factor(data)
        final = after_electrical * availability_factor
        wake_loss = after_density - after_wake
        electrical_loss = after_wake - after_electrical
        availability_loss = after_electrical - final
        return RenewableProfile(
            {
                "hub_height_wind_speed_m_s": (hub_speed.astype(np.float64), "m/s"),
                "gross_potential_mw": (gross.astype(np.float64), "MW"),
                "high_wind_shutdown_loss_mw": (shutdown_loss.astype(np.float64), "MW"),
                "wake_loss_mw": (wake_loss.astype(np.float64), "MW"),
                "electrical_loss_mw": (electrical_loss.astype(np.float64), "MW"),
                "availability_loss_mw": (availability_loss.astype(np.float64), "MW"),
                "available_mw": (final.astype(np.float64), "MW"),
            }
        )

    def _solar_plane_irradiance(self, irradiance: FloatArray) -> FloatArray:
        if self.irradiance_basis == "plane_of_array" or self.transposition_model == "none":
            return irradiance
        tilt = np.deg2rad(_required(self.tilt_degrees, self.asset_id))
        transposition_factor = np.cos(tilt) + self.albedo * (1.0 - np.cos(tilt)) / 2.0
        return cast(FloatArray, (irradiance * max(transposition_factor, 0.0)).astype(np.float64))

    def _hub_height_wind_speed(self, speed: FloatArray) -> FloatArray:
        if self.wind_speed_adjustment == "none":
            return speed
        measurement_height = _required(self.measurement_height_m, self.asset_id)
        hub_height = _required(self.hub_height_m, self.asset_id)
        if self.wind_speed_adjustment == "power_law":
            return cast(
                FloatArray,
                (speed * (hub_height / measurement_height) ** self.wind_shear_exponent).astype(
                    np.float64
                ),
            )
        roughness = _required(self.roughness_length_m, self.asset_id)
        numerator = np.log(hub_height / roughness)
        denominator = np.log(measurement_height / roughness)
        return cast(FloatArray, (speed * numerator / denominator).astype(np.float64))

    def _air_density_factor(self, data: pd.DataFrame) -> FloatArray:
        if not self.air_density_correction:
            return np.ones(len(data), dtype=np.float64)
        temperature = _column(
            data,
            cast(str, self.air_temperature_key),
            self.asset_id,
            nonnegative=False,
        )
        pressure = _column(data, cast(str, self.air_pressure_key), self.asset_id, nonnegative=True)
        if np.any(temperature <= -273.15):
            raise DataValidationError(
                f"Input column {self.air_temperature_key!r} for asset {self.asset_id!r} "
                "must be above absolute zero"
            )
        density = pressure / (287.05 * (temperature + 273.15))
        return (density / 1.225).astype(np.float64)

    def _wind_rated_capacity_mw(self) -> float:
        if self.turbine_count is None or self.turbine_rated_capacity_mw is None:
            return cast(WindConfig, self.wind).capacity_mw
        return float(self.turbine_count * self.turbine_rated_capacity_mw)

    def _availability_factor(self, data: pd.DataFrame) -> FloatArray:
        factor = np.full(len(data), self.availability_factor, dtype=np.float64)
        if self.availability_factor_key is not None:
            factor *= _fraction_column(data, self.availability_factor_key, self.asset_id)
        return factor

    def _maintenance_factor(self, data: pd.DataFrame) -> FloatArray:
        if self.maintenance_factor_key is None:
            return np.ones(len(data), dtype=np.float64)
        return _fraction_column(data, self.maintenance_factor_key, self.asset_id)

    def _loss_factor(
        self,
        data: pd.DataFrame,
        loss_fraction: float,
        loss_key: str | None,
    ) -> FloatArray:
        factor = np.full(len(data), 1.0 - loss_fraction, dtype=np.float64)
        if loss_key is not None:
            factor *= 1.0 - _fraction_column(data, loss_key, self.asset_id)
        return factor


@dataclass(frozen=True)
class DispatchableAsset:
    """Dispatchable resource placeholder for future generator-indexed MILP variables."""

    asset_id: str
    bus_id: str
    fuel_id: str
    config: ThermalConfig
    must_run: bool = False
    availability_factor_key: str | None = None
    role: AssetRole = "dispatchable"

    @classmethod
    def from_config(cls, config: ThermalGeneratorConfig) -> DispatchableAsset:
        return cls(
            asset_id=config.id,
            bus_id=config.bus_id,
            fuel_id=config.fuel_id,
            config=config.config,
            must_run=config.must_run,
            availability_factor_key=config.availability_factor_key,
        )

    def availability_factor(self, data: pd.DataFrame) -> FloatArray | None:
        if self.availability_factor_key is None:
            return None
        values = _column(data, self.availability_factor_key, self.asset_id, nonnegative=True)
        if np.any(values > 1.0):
            raise DataValidationError(
                f"Input column {self.availability_factor_key!r} for asset "
                f"{self.asset_id!r} must be in [0, 1]"
            )
        return values


@dataclass(frozen=True)
class IntertemporalAsset:
    """Intertemporal storage resource placeholder for indexed storage optimisation."""

    asset_id: str
    bus_id: str
    config: BatteryConfig
    availability_factor_key: str | None = None
    role: AssetRole = "intertemporal"

    @classmethod
    def from_config(cls, config: StorageUnitConfig) -> IntertemporalAsset:
        return cls(
            asset_id=config.id,
            bus_id=config.bus_id,
            config=config.config,
            availability_factor_key=config.config.availability_factor_key,
        )

    def availability_factor(self, data: pd.DataFrame) -> FloatArray | None:
        if self.availability_factor_key is None:
            return None
        values = _column(data, self.availability_factor_key, self.asset_id, nonnegative=True)
        if np.any(values > 1.0):
            raise DataValidationError(
                f"Input column {self.availability_factor_key!r} for asset "
                f"{self.asset_id!r} must be in [0, 1]"
            )
        return values


@dataclass(frozen=True)
class HydroAsset:
    """Hydro resource with exogenous inflow data."""

    asset_id: str
    bus_id: str
    config: HydroUnitConfig
    role: AssetRole = "intertemporal"

    @classmethod
    def from_config(cls, config: HydroUnitConfig) -> HydroAsset:
        return cls(asset_id=config.id, bus_id=config.bus_id, config=config)

    def inflow_mw(self, data: pd.DataFrame) -> FloatArray:
        return _column(data, self.config.inflow_time_series_key, self.asset_id, nonnegative=True)


@dataclass(frozen=True)
class DemandAsset:
    """Demand resource resolved from portfolio configuration."""

    asset_id: str
    bus_id: str
    time_series_key: str
    config: DemandConfig
    role: AssetRole = "demand"

    @classmethod
    def from_config(cls, config: DemandConfig) -> DemandAsset:
        return cls(
            asset_id=config.id,
            bus_id=config.bus_id,
            time_series_key=config.time_series_key,
            config=config,
        )

    def demand_mw(self, data: pd.DataFrame) -> FloatArray:
        baseline = _column(data, self.time_series_key, self.asset_id, nonnegative=True)
        if self.config.temperature_time_series_key is None:
            return baseline
        temperature = _column(
            data,
            self.config.temperature_time_series_key,
            self.asset_id,
            nonnegative=False,
        )
        adjustment = np.zeros_like(baseline, dtype=np.float64)
        if self.config.heating_base_temperature_c is not None:
            adjustment += (
                np.maximum(self.config.heating_base_temperature_c - temperature, 0.0)
                * self.config.heating_sensitivity_mw_per_c
            )
        if self.config.cooling_base_temperature_c is not None:
            adjustment += (
                np.maximum(temperature - self.config.cooling_base_temperature_c, 0.0)
                * self.config.cooling_sensitivity_mw_per_c
            )
        return baseline + adjustment


@dataclass(frozen=True)
class NetworkComponent:
    """Aggregate network component resolved from configuration."""

    asset_id: str
    bus_id: str
    config: NetworkConfig
    role: AssetRole = "network"


@dataclass(frozen=True)
class ImportAsset:
    """Import resource placeholder for future import-indexed dispatch."""

    asset_id: str
    bus_id: str
    config: ImportResourceConfig
    role: AssetRole = "dispatchable"


@dataclass(frozen=True)
class AssetTimeSeries:
    """Tidy asset table with stable dimensions and units."""

    table: pd.DataFrame

    @staticmethod
    def empty() -> AssetTimeSeries:
        return AssetTimeSeries(
            pd.DataFrame(columns=["timestamp", "asset_id", "variable", "value", "unit"])
        )

    @classmethod
    def from_variable_matrix(
        cls,
        timestamps: pd.Series,
        values_by_asset: MappingByAsset,
        variable: str,
        unit: str,
    ) -> AssetTimeSeries:
        pieces: list[pd.DataFrame] = []
        for asset_id, values in values_by_asset.items():
            pieces.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps.to_numpy(),
                        "asset_id": asset_id,
                        "variable": variable,
                        "value": np.asarray(values, dtype=np.float64),
                        "unit": unit,
                    }
                )
            )
        if not pieces:
            return cls.empty()
        return cls(pd.concat(pieces, ignore_index=True))

    def append(self, other: AssetTimeSeries) -> AssetTimeSeries:
        if self.table.empty:
            return other
        if other.table.empty:
            return self
        return AssetTimeSeries(pd.concat([self.table, other.table], ignore_index=True))


@dataclass(frozen=True)
class RenewableAvailability:
    """Asset-level and aggregate renewable availability."""

    asset_table: AssetTimeSeries
    by_asset_mw: dict[str, FloatArray]
    aggregate_mw: FloatArray


@dataclass(frozen=True)
class AssetRegistry:
    """Resolved runtime assets grouped by modelling role."""

    renewable_assets: tuple[RenewableAsset, ...]
    dispatchable_assets: tuple[DispatchableAsset, ...]
    intertemporal_assets: tuple[IntertemporalAsset, ...]
    hydro_assets: tuple[HydroAsset, ...]
    import_assets: tuple[ImportAsset, ...]
    demand_assets: tuple[DemandAsset, ...]
    network_components: tuple[NetworkComponent, ...]

    @classmethod
    def from_config(cls, config: ModelConfig) -> AssetRegistry:
        portfolio = config.portfolio
        return cls(
            renewable_assets=tuple(
                RenewableAsset.from_config(asset) for asset in portfolio.renewable_generators
            ),
            dispatchable_assets=tuple(
                DispatchableAsset.from_config(asset) for asset in portfolio.thermal_generators
            ),
            intertemporal_assets=tuple(
                IntertemporalAsset.from_config(asset) for asset in portfolio.storage_units
            ),
            hydro_assets=tuple(HydroAsset.from_config(asset) for asset in portfolio.hydro_units),
            import_assets=tuple(
                ImportAsset(asset_id=asset.id, bus_id=asset.bus_id, config=asset)
                for asset in portfolio.imports
            ),
            demand_assets=tuple(DemandAsset.from_config(asset) for asset in portfolio.demand),
            network_components=(
                NetworkComponent(
                    asset_id="aggregate_network",
                    bus_id=config.portfolio.buses[0].id,
                    config=config.network,
                ),
            ),
        )

    def renewable_availability(self, data: pd.DataFrame) -> RenewableAvailability:
        profiles_by_asset = {
            asset.asset_id: asset.availability_profile(data) for asset in self.renewable_assets
        }
        values_by_asset = {
            asset_id: profile.available_mw for asset_id, profile in profiles_by_asset.items()
        }
        aggregate = _sum_arrays(values_by_asset)
        table = AssetTimeSeries.empty()
        for asset_id, profile in profiles_by_asset.items():
            table = table.append(profile.to_asset_timeseries(data["timestamp"], asset_id))
        return RenewableAvailability(
            asset_table=table,
            by_asset_mw=values_by_asset,
            aggregate_mw=aggregate,
        )

    def demand_mw(self, data: pd.DataFrame) -> FloatArray:
        return _sum_arrays(self.demand_profiles_mw(data))

    def demand_profiles_mw(self, data: pd.DataFrame) -> dict[str, FloatArray]:
        return {asset.asset_id: asset.demand_mw(data) for asset in self.demand_assets}

    def thermal_availability_factors(self, data: pd.DataFrame) -> dict[str, FloatArray]:
        factors: dict[str, FloatArray] = {}
        for asset in self.dispatchable_assets:
            factor = asset.availability_factor(data)
            if factor is not None:
                factors[asset.asset_id] = factor
        return factors

    def storage_availability_factors(self, data: pd.DataFrame) -> dict[str, FloatArray]:
        factors: dict[str, FloatArray] = {}
        for asset in self.intertemporal_assets:
            factor = asset.availability_factor(data)
            if factor is not None:
                factors[asset.asset_id] = factor
        return factors

    def hydro_inflows_mw(self, data: pd.DataFrame) -> dict[str, FloatArray]:
        return {asset.asset_id: asset.inflow_mw(data) for asset in self.hydro_assets}


MappingByAsset = dict[str, FloatArray]


def allocate_renewable_dispatch(
    timestamps: pd.Series,
    availability_by_asset_mw: MappingByAsset,
    aggregate_available_mw: FloatArray,
    aggregate_used_mw: npt.ArrayLike,
) -> AssetTimeSeries:
    """Allocate aggregate renewable dispatch to assets by availability share."""
    used = np.asarray(aggregate_used_mw, dtype=np.float64)
    used_by_asset: MappingByAsset = {}
    curtailed_by_asset: MappingByAsset = {}
    for asset_id, available in availability_by_asset_mw.items():
        share = np.divide(
            available,
            aggregate_available_mw,
            out=np.zeros_like(available, dtype=np.float64),
            where=aggregate_available_mw > 0.0,
        )
        asset_used = used * share
        used_by_asset[asset_id] = asset_used
        curtailed_by_asset[asset_id] = available - asset_used
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


def _required(value: float | None, asset_id: str) -> float:
    if value is None:
        raise DataValidationError(f"Renewable asset {asset_id!r} has unresolved parameters")
    return value


def _column(
    data: pd.DataFrame,
    column: str,
    asset_id: str,
    *,
    nonnegative: bool,
) -> FloatArray:
    if column not in data.columns:
        raise DataValidationError(f"Asset {asset_id!r} references missing input column {column!r}")
    values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise DataValidationError(
            f"Input column {column!r} for asset {asset_id!r} contains non-finite values"
        )
    if nonnegative and np.any(values < 0.0):
        raise DataValidationError(
            f"Input column {column!r} for asset {asset_id!r} must be non-negative"
        )
    return values


def _fraction_column(data: pd.DataFrame, column: str, asset_id: str) -> FloatArray:
    values = _column(data, column, asset_id, nonnegative=True)
    if np.any(values > 1.0):
        raise DataValidationError(
            f"Input column {column!r} for asset {asset_id!r} must be in [0, 1]"
        )
    return values


def _sum_arrays(values_by_asset: MappingByAsset) -> FloatArray:
    if not values_by_asset:
        return np.asarray([], dtype=np.float64)
    values = list(values_by_asset.values())
    return np.asarray(np.sum(np.vstack(values), axis=0, dtype=np.float64), dtype=np.float64)
