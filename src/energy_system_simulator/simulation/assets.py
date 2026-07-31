from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_system_simulator.config import (
    BatteryConfig,
    DemandConfig,
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
                solar=SolarConfig(
                    capacity_mw=config.capacity_mw,
                    performance_ratio=_required(config.performance_ratio, config.id),
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
            wind=WindConfig(
                capacity_mw=config.capacity_mw,
                cut_in_speed_m_s=_required(config.cut_in_speed_m_s, config.id),
                rated_speed_m_s=_required(config.rated_speed_m_s, config.id),
                cut_out_speed_m_s=_required(config.cut_out_speed_m_s, config.id),
            ),
        )

    def availability_mw(self, data: pd.DataFrame) -> FloatArray:
        if self.kind == "solar":
            if self.ambient_temperature_key is None:
                raise DataValidationError(
                    f"{self.asset_id} is missing ambient temperature data key"
                )
            return SolarPlant(cast(SolarConfig, self.solar)).output_mw(
                _column(data, self.time_series_key, self.asset_id, nonnegative=True),
                _column(data, self.ambient_temperature_key, self.asset_id, nonnegative=False),
            )
        return WindFarm(cast(WindConfig, self.wind)).output_mw(
            _column(data, self.time_series_key, self.asset_id, nonnegative=True)
        )


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
class DemandAsset:
    """Demand resource resolved from portfolio configuration."""

    asset_id: str
    bus_id: str
    time_series_key: str
    role: AssetRole = "demand"

    @classmethod
    def from_config(cls, config: DemandConfig) -> DemandAsset:
        return cls(asset_id=config.id, bus_id=config.bus_id, time_series_key=config.time_series_key)

    def demand_mw(self, data: pd.DataFrame) -> FloatArray:
        return _column(data, self.time_series_key, self.asset_id, nonnegative=True)


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
        values_by_asset = {
            asset.asset_id: asset.availability_mw(data) for asset in self.renewable_assets
        }
        aggregate = _sum_arrays(values_by_asset)
        table = AssetTimeSeries.from_variable_matrix(
            data["timestamp"],
            values_by_asset,
            "available_mw",
            "MW",
        )
        return RenewableAvailability(
            asset_table=table,
            by_asset_mw=values_by_asset,
            aggregate_mw=aggregate,
        )

    def demand_mw(self, data: pd.DataFrame) -> FloatArray:
        values_by_asset = {asset.asset_id: asset.demand_mw(data) for asset in self.demand_assets}
        return _sum_arrays(values_by_asset)

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


def _sum_arrays(values_by_asset: MappingByAsset) -> FloatArray:
    if not values_by_asset:
        return np.asarray([], dtype=np.float64)
    values = list(values_by_asset.values())
    return np.asarray(np.sum(np.vstack(values), axis=0, dtype=np.float64), dtype=np.float64)
