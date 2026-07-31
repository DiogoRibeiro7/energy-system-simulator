from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml

from energy_system_simulator.exceptions import ConfigurationError

T = TypeVar("T", int, float)
CURRENT_SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
INPUT_DATA_KEYS = {
    "demand_mw",
    "irradiance_w_m2",
    "ambient_temperature_c",
    "wind_speed_m_s",
}

LEGACY_ROOT_KEYS = {
    "schema_version",
    "simulation",
    "solar",
    "wind",
    "thermal",
    "battery",
    "network",
    "imports",
    "penalties",
    "paths",
}
LEGACY_REQUIRED_ROOT_KEYS = LEGACY_ROOT_KEYS - {"schema_version"}
LEGACY_SECTION_KEYS = {
    "simulation": {
        "time_step_hours",
        "solver_time_limit_seconds",
        "mip_relative_gap",
        "allow_non_optimal_solution",
    },
    "solar": {
        "capacity_mw",
        "performance_ratio",
        "reference_irradiance_w_m2",
        "temperature_coefficient_per_c",
        "nominal_operating_cell_temperature_c",
    },
    "wind": {
        "capacity_mw",
        "cut_in_speed_m_s",
        "rated_speed_m_s",
        "cut_out_speed_m_s",
    },
    "thermal": {
        "name",
        "minimum_output_mw",
        "maximum_output_mw",
        "ramp_up_mw_per_hour",
        "ramp_down_mw_per_hour",
        "startup_ramp_mw",
        "shutdown_ramp_mw",
        "variable_cost_eur_per_mwh",
        "no_load_cost_eur_per_hour",
        "startup_cost_eur",
        "shutdown_cost_eur",
        "emission_factor_tonnes_per_mwh",
        "minimum_up_hours",
        "minimum_down_hours",
        "initial_on",
        "initial_output_mw",
        "initial_up_time_hours",
        "initial_down_time_hours",
        "terminal_commitment_mode",
        "terminal_on",
    },
    "battery": {
        "energy_capacity_mwh",
        "power_capacity_mw",
        "charge_power_capacity_mw",
        "discharge_power_capacity_mw",
        "minimum_soc_mwh",
        "maximum_soc_mwh",
        "initial_soc_mwh",
        "charge_efficiency",
        "discharge_efficiency",
        "self_discharge_rate_per_hour",
        "minimum_charge_mw",
        "minimum_discharge_mw",
        "charge_ramp_mw_per_hour",
        "discharge_ramp_mw_per_hour",
        "throughput_cost_eur_per_mwh",
        "minimum_final_soc_mwh",
        "terminal_soc_mode",
    },
    "network": {"loss_fraction", "transfer_capacity_mw"},
    "imports": {"maximum_power_mw", "price_eur_per_mwh", "emission_factor_tonnes_per_mwh"},
    "penalties": {
        "renewable_curtailment_eur_per_mwh",
        "lost_load_eur_per_mwh",
        "carbon_price_eur_per_tonne",
    },
    "paths": {"input_csv", "output_directory"},
}
OPTIONAL_SECTION_KEYS = {
    "simulation": {"allow_non_optimal_solution"},
    "thermal": {
        "startup_ramp_mw",
        "shutdown_ramp_mw",
        "initial_up_time_hours",
        "initial_down_time_hours",
        "terminal_commitment_mode",
        "terminal_on",
    },
    "battery": {
        "terminal_soc_mode",
        "charge_power_capacity_mw",
        "discharge_power_capacity_mw",
        "self_discharge_rate_per_hour",
        "minimum_charge_mw",
        "minimum_discharge_mw",
        "charge_ramp_mw_per_hour",
        "discharge_ramp_mw_per_hour",
    },
    "aggregate_network": {"network_mode", "slack_bus_id"},
}
LEGACY_REQUIRED_SECTION_KEYS = {
    section: keys - OPTIONAL_SECTION_KEYS.get(section, set())
    for section, keys in LEGACY_SECTION_KEYS.items()
}
ROOT_KEYS = {
    "schema_version",
    "scenario",
    "simulation",
    "solver",
    "fuels",
    "zones",
    "buses",
    "lines",
    "aggregate_network",
    "renewable_generators",
    "thermal_generators",
    "storage_units",
    "hydro_units",
    "imports",
    "demand",
    "penalties",
    "paths",
}
REQUIRED_ROOT_KEYS = ROOT_KEYS - {"fuels", "hydro_units"}
SECTION_KEYS = {
    "scenario": {"id"},
    "simulation": {"time_step_hours"},
    "solver": {
        "solver_time_limit_seconds",
        "mip_relative_gap",
        "allow_non_optimal_solution",
    },
    "aggregate_network": {"loss_fraction", "transfer_capacity_mw", "network_mode", "slack_bus_id"},
    "penalties": LEGACY_SECTION_KEYS["penalties"],
    "paths": LEGACY_SECTION_KEYS["paths"],
}
LIST_SECTION_KEYS = {
    "zones": {"id"},
    "buses": {"id", "zone_id"},
    "lines": {
        "id",
        "from_bus_id",
        "to_bus_id",
        "susceptance",
        "capacity_mw",
        "availability_factor",
        "availability_factor_key",
    },
    "renewable_generators": {
        "id",
        "kind",
        "bus_id",
        "capacity_mw",
        "performance_ratio",
        "reference_irradiance_w_m2",
        "temperature_coefficient_per_c",
        "nominal_operating_cell_temperature_c",
        "cut_in_speed_m_s",
        "rated_speed_m_s",
        "cut_out_speed_m_s",
        "time_series_key",
        "ambient_temperature_key",
    },
    "thermal_generators": LEGACY_SECTION_KEYS["thermal"]
    | {
        "id",
        "bus_id",
        "fuel_id",
        "must_run",
        "availability_factor",
        "availability_factor_key",
        "minimum_fuel_input_mwh_per_hour",
        "heat_rate_segments",
        "startup_categories",
    },
    "fuels": {
        "id",
        "price_eur_per_mwh_thermal",
        "price_time_series_key",
        "lower_heating_value_mj_per_unit",
        "co2_factor_tonnes_per_mwh_thermal",
        "methane_factor_tonnes_per_mwh_thermal",
        "nox_factor_kg_per_mwh_thermal",
        "sox_factor_kg_per_mwh_thermal",
    },
    "storage_units": LEGACY_SECTION_KEYS["battery"]
    | {
        "id",
        "bus_id",
        "technology",
        "availability_factor",
        "availability_factor_key",
        "degradation_bands",
    },
    "hydro_units": {
        "id",
        "bus_id",
        "kind",
        "inflow_time_series_key",
        "minimum_reservoir_mwh",
        "maximum_reservoir_mwh",
        "initial_reservoir_mwh",
        "minimum_final_reservoir_mwh",
        "terminal_reservoir_mode",
        "turbine_capacity_mw",
        "turbine_efficiency",
        "spill_capacity_mw",
        "minimum_release_mw",
        "evaporation_rate_per_hour",
        "water_value_eur_per_mwh",
        "upstream_hydro_id",
        "cascade_delay_hours",
    },
    "imports": LEGACY_SECTION_KEYS["imports"] | {"id", "bus_id"},
    "demand": {
        "id",
        "bus_id",
        "time_series_key",
        "kind",
        "sector",
        "value_of_lost_load_eur_per_mwh",
        "voluntary_curtailment_cost_eur_per_mwh",
        "maximum_curtailment_fraction",
        "maximum_curtailment_mw",
        "shift_up_capacity_mw",
        "shift_down_capacity_mw",
        "shift_window_hours",
        "rebound_fraction",
        "shift_cost_eur_per_mwh",
        "task_power_capacity_mw",
        "task_required_energy_mwh",
        "task_start_period",
        "task_end_period",
        "task_unserved_penalty_eur_per_mwh",
        "temperature_time_series_key",
        "heating_base_temperature_c",
        "cooling_base_temperature_c",
        "heating_sensitivity_mw_per_c",
        "cooling_sensitivity_mw_per_c",
    },
}
LIST_OPTIONAL_KEYS = {
    "renewable_generators": {
        "performance_ratio",
        "reference_irradiance_w_m2",
        "temperature_coefficient_per_c",
        "nominal_operating_cell_temperature_c",
        "cut_in_speed_m_s",
        "rated_speed_m_s",
        "cut_out_speed_m_s",
        "ambient_temperature_key",
    },
    "thermal_generators": OPTIONAL_SECTION_KEYS["thermal"]
    | {
        "must_run",
        "availability_factor",
        "availability_factor_key",
        "minimum_fuel_input_mwh_per_hour",
        "heat_rate_segments",
        "startup_categories",
    },
    "fuels": {
        "price_time_series_key",
        "lower_heating_value_mj_per_unit",
        "methane_factor_tonnes_per_mwh_thermal",
        "nox_factor_kg_per_mwh_thermal",
        "sox_factor_kg_per_mwh_thermal",
    },
    "storage_units": OPTIONAL_SECTION_KEYS["battery"]
    | {
        "technology",
        "charge_power_capacity_mw",
        "discharge_power_capacity_mw",
        "self_discharge_rate_per_hour",
        "minimum_charge_mw",
        "minimum_discharge_mw",
        "charge_ramp_mw_per_hour",
        "discharge_ramp_mw_per_hour",
        "availability_factor",
        "availability_factor_key",
        "degradation_bands",
    },
    "hydro_units": {
        "kind",
        "minimum_reservoir_mwh",
        "maximum_reservoir_mwh",
        "initial_reservoir_mwh",
        "minimum_final_reservoir_mwh",
        "terminal_reservoir_mode",
        "spill_capacity_mw",
        "minimum_release_mw",
        "evaporation_rate_per_hour",
        "water_value_eur_per_mwh",
        "upstream_hydro_id",
        "cascade_delay_hours",
    },
    "imports": set(),
    "lines": {"availability_factor", "availability_factor_key"},
    "demand": {
        "kind",
        "sector",
        "value_of_lost_load_eur_per_mwh",
        "voluntary_curtailment_cost_eur_per_mwh",
        "maximum_curtailment_fraction",
        "maximum_curtailment_mw",
        "shift_up_capacity_mw",
        "shift_down_capacity_mw",
        "shift_window_hours",
        "rebound_fraction",
        "shift_cost_eur_per_mwh",
        "task_power_capacity_mw",
        "task_required_energy_mwh",
        "task_start_period",
        "task_end_period",
        "task_unserved_penalty_eur_per_mwh",
        "temperature_time_series_key",
        "heating_base_temperature_c",
        "cooling_base_temperature_c",
        "heating_sensitivity_mw_per_c",
        "cooling_sensitivity_mw_per_c",
    },
}
LIST_REQUIRED_KEYS = {
    section: keys - LIST_OPTIONAL_KEYS.get(section, set())
    for section, keys in LIST_SECTION_KEYS.items()
}


class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: Any,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationError(f"Duplicate configuration field: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _number(section: Mapping[str, Any], key: str, expected: type[T]) -> T:
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {key}")
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key!r} must be numeric")
    return expected(value)


def _integer(section: Mapping[str, Any], key: str) -> int:
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {key}")
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{key!r} must be an integer")
    return value


def _boolean(section: Mapping[str, Any], key: str) -> bool:
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {key}")
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key!r} must be boolean")
    return value


def _optional_number(
    section: Mapping[str, Any],
    key: str,
    expected: type[T],
    default: T,
) -> T:
    if key not in section:
        return default
    return _number(section, key, expected)


def _optional_boolean(section: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in section:
        return default
    return _boolean(section, key)


def _optional_nullable_boolean(section: Mapping[str, Any], key: str) -> bool | None:
    if key not in section:
        return None
    if section.get(key) is None:
        return None
    return _boolean(section, key)


def _optional_string(section: Mapping[str, Any], key: str, default: str) -> str:
    if key not in section:
        return default
    return _string(section, key)


def _string(section: Mapping[str, Any], key: str) -> str:
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {key}")
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key!r} must be a non-empty string")
    return value


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in data:
        raise ConfigurationError(f"Missing required configuration section: {name}")
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing or invalid section: {name}")
    return value


def _validate_allowed_keys(
    section: Mapping[str, Any],
    path: str,
    allowed_keys: set[str],
) -> None:
    for key in section:
        if key in allowed_keys:
            continue
        field = f"{path}.{key}" if path else str(key)
        allowed = ", ".join(sorted(allowed_keys))
        suggestion = _suggestion(str(key), allowed_keys)
        hint = f"; did you mean {suggestion!r}?" if suggestion is not None else ""
        raise ConfigurationError(
            f"Unknown configuration field: {field}{hint}. Allowed keys: {allowed}"
        )


def _validate_required_keys(
    section: Mapping[str, Any],
    path: str,
    required_keys: set[str],
) -> None:
    missing = sorted(key for key in required_keys if key not in section)
    if missing:
        field = f"{path}.{missing[0]}" if path else missing[0]
        raise ConfigurationError(f"Missing required configuration field: {field}")


def _suggestion(key: str, allowed_keys: set[str]) -> str | None:
    matches = get_close_matches(key, sorted(allowed_keys), n=1, cutoff=0.75)
    return matches[0] if matches else None


def _check_nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise ConfigurationError(f"{name} must be non-negative")


def _check_fraction(name: str, value: float, *, allow_one: bool = True) -> None:
    upper_valid = value <= 1.0 if allow_one else value < 1.0
    if value < 0.0 or not upper_valid:
        operator = "[0, 1]" if allow_one else "[0, 1)"
        raise ConfigurationError(f"{name} must be in {operator}")


def _field_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _number_at(section: Mapping[str, Any], key: str, expected: type[T], path: str) -> T:
    field = _field_path(path, key)
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {field}")
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field!r} must be numeric")
    return expected(value)


def _integer_at(section: Mapping[str, Any], key: str, path: str) -> int:
    field = _field_path(path, key)
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {field}")
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field!r} must be an integer")
    return value


def _boolean_at(section: Mapping[str, Any], key: str, path: str) -> bool:
    field = _field_path(path, key)
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {field}")
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field!r} must be boolean")
    return value


def _optional_boolean_at(
    section: Mapping[str, Any],
    key: str,
    default: bool,
    path: str,
) -> bool:
    if key not in section:
        return default
    return _boolean_at(section, key, path)


def _optional_number_at(
    section: Mapping[str, Any],
    key: str,
    expected: type[T],
    default: T,
    path: str,
) -> T:
    if key not in section:
        return default
    return _number_at(section, key, expected, path)


def _optional_integer_at(section: Mapping[str, Any], key: str, default: int, path: str) -> int:
    if key not in section:
        return default
    return _integer_at(section, key, path)


def _optional_string_at(
    section: Mapping[str, Any],
    key: str,
    default: str,
    path: str,
) -> str:
    if key not in section:
        return default
    return _string_at(section, key, path)


def _optional_nullable_boolean_at(
    section: Mapping[str, Any],
    key: str,
    path: str,
) -> bool | None:
    if key not in section:
        return None
    if section.get(key) is None:
        return None
    return _boolean_at(section, key, path)


def _string_at(section: Mapping[str, Any], key: str, path: str) -> str:
    field = _field_path(path, key)
    if key not in section:
        raise ConfigurationError(f"Missing required configuration field: {field}")
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field!r} must be a non-empty string")
    return value


def _id_at(section: Mapping[str, Any], key: str, path: str) -> str:
    identifier = _string_at(section, key, path)
    if any(character.isspace() for character in identifier):
        raise ConfigurationError(f"{_field_path(path, key)} must not contain whitespace")
    return identifier


def _input_key_at(section: Mapping[str, Any], key: str, path: str) -> str:
    return _string_at(section, key, path)


def _optional_input_key_at(
    section: Mapping[str, Any],
    key: str,
    default: str,
    path: str,
) -> str:
    if key not in section:
        return default
    return _input_key_at(section, key, path)


def _list_section(
    data: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    if name not in data:
        if required:
            raise ConfigurationError(f"Missing required configuration section: {name}")
        return ()
    value = data.get(name)
    if not isinstance(value, list):
        raise ConfigurationError(f"Configuration section must be a list: {name}")
    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ConfigurationError(f"Configuration item must be a mapping: {name}[{index}]")
        items.append(item)
    return tuple(items)


@dataclass(frozen=True)
class SimulationConfig:
    time_step_hours: float
    solver_time_limit_seconds: float
    mip_relative_gap: float
    allow_non_optimal_solution: bool


@dataclass(frozen=True)
class SolverConfig:
    solver_time_limit_seconds: float
    mip_relative_gap: float
    allow_non_optimal_solution: bool


@dataclass(frozen=True)
class ScenarioConfig:
    id: str


@dataclass(frozen=True)
class ZoneConfig:
    id: str


@dataclass(frozen=True)
class BusConfig:
    id: str
    zone_id: str


@dataclass(frozen=True)
class TransmissionLineConfig:
    id: str
    from_bus_id: str
    to_bus_id: str
    susceptance: float
    capacity_mw: float
    availability_factor: float = 1.0
    availability_factor_key: str | None = None


@dataclass(frozen=True)
class SolarConfig:
    capacity_mw: float
    performance_ratio: float
    reference_irradiance_w_m2: float
    temperature_coefficient_per_c: float
    nominal_operating_cell_temperature_c: float


@dataclass(frozen=True)
class WindConfig:
    capacity_mw: float
    cut_in_speed_m_s: float
    rated_speed_m_s: float
    cut_out_speed_m_s: float


@dataclass(frozen=True)
class RenewableGeneratorConfig:
    id: str
    kind: Literal["solar", "wind"]
    bus_id: str
    capacity_mw: float
    time_series_key: str
    ambient_temperature_key: str | None = None
    performance_ratio: float | None = None
    reference_irradiance_w_m2: float | None = None
    temperature_coefficient_per_c: float | None = None
    nominal_operating_cell_temperature_c: float | None = None
    cut_in_speed_m_s: float | None = None
    rated_speed_m_s: float | None = None
    cut_out_speed_m_s: float | None = None


@dataclass(frozen=True)
class FuelConfig:
    id: str
    price_eur_per_mwh_thermal: float
    co2_factor_tonnes_per_mwh_thermal: float
    price_time_series_key: str | None = None
    lower_heating_value_mj_per_unit: float | None = None
    methane_factor_tonnes_per_mwh_thermal: float = 0.0
    nox_factor_kg_per_mwh_thermal: float = 0.0
    sox_factor_kg_per_mwh_thermal: float = 0.0


@dataclass(frozen=True)
class HeatRateSegmentConfig:
    id: str
    capacity_mw: float
    heat_rate_mwh_thermal_per_mwh: float


@dataclass(frozen=True)
class StartupCategoryConfig:
    id: str
    minimum_down_time_hours: float
    startup_cost_eur: float
    startup_fuel_input_mwh_thermal: float = 0.0


TerminalCommitmentMode = Literal[
    "forbid_incomplete_transitions",
    "carry_residual_obligations",
    "fixed_terminal_commitment",
]


@dataclass(frozen=True)
class ThermalConfig:
    name: str
    minimum_output_mw: float
    maximum_output_mw: float
    ramp_up_mw_per_hour: float
    ramp_down_mw_per_hour: float
    startup_ramp_mw: float
    shutdown_ramp_mw: float
    variable_cost_eur_per_mwh: float
    no_load_cost_eur_per_hour: float
    startup_cost_eur: float
    shutdown_cost_eur: float
    emission_factor_tonnes_per_mwh: float
    minimum_up_hours: float
    minimum_down_hours: float
    initial_on: bool
    initial_output_mw: float
    initial_up_time_hours: float
    initial_down_time_hours: float
    terminal_commitment_mode: TerminalCommitmentMode = "forbid_incomplete_transitions"
    terminal_on: bool | None = None
    minimum_fuel_input_mwh_per_hour: float = 0.0
    heat_rate_segments: tuple[HeatRateSegmentConfig, ...] = ()
    startup_categories: tuple[StartupCategoryConfig, ...] = ()


@dataclass(frozen=True)
class ThermalGeneratorConfig:
    id: str
    bus_id: str
    fuel_id: str
    config: ThermalConfig
    must_run: bool = False
    availability_factor: float = 1.0
    availability_factor_key: str | None = None


@dataclass(frozen=True)
class StorageDegradationBandConfig:
    id: str
    capacity_mwh: float
    cost_eur_per_mwh: float


@dataclass(frozen=True)
class BatteryConfig:
    energy_capacity_mwh: float
    power_capacity_mw: float
    minimum_soc_mwh: float
    maximum_soc_mwh: float
    initial_soc_mwh: float
    charge_efficiency: float
    discharge_efficiency: float
    throughput_cost_eur_per_mwh: float
    minimum_final_soc_mwh: float
    terminal_soc_mode: Literal["minimum", "exact", "cyclic", "free"]
    technology: Literal["battery", "pumped_storage"] = "battery"
    charge_power_capacity_mw: float | None = None
    discharge_power_capacity_mw: float | None = None
    self_discharge_rate_per_hour: float = 0.0
    minimum_charge_mw: float = 0.0
    minimum_discharge_mw: float = 0.0
    charge_ramp_mw_per_hour: float | None = None
    discharge_ramp_mw_per_hour: float | None = None
    availability_factor: float = 1.0
    availability_factor_key: str | None = None
    degradation_bands: tuple[StorageDegradationBandConfig, ...] = ()


@dataclass(frozen=True)
class StorageUnitConfig:
    id: str
    bus_id: str
    config: BatteryConfig


@dataclass(frozen=True)
class HydroUnitConfig:
    id: str
    bus_id: str
    kind: Literal["reservoir", "run_of_river"]
    inflow_time_series_key: str
    turbine_capacity_mw: float
    turbine_efficiency: float
    minimum_reservoir_mwh: float = 0.0
    maximum_reservoir_mwh: float = 0.0
    initial_reservoir_mwh: float = 0.0
    minimum_final_reservoir_mwh: float = 0.0
    terminal_reservoir_mode: Literal["minimum", "exact", "cyclic", "free"] = "minimum"
    spill_capacity_mw: float | None = None
    minimum_release_mw: float = 0.0
    evaporation_rate_per_hour: float = 0.0
    water_value_eur_per_mwh: float = 0.0
    upstream_hydro_id: str | None = None
    cascade_delay_hours: float = 0.0


@dataclass(frozen=True)
class NetworkConfig:
    loss_fraction: float
    transfer_capacity_mw: float
    network_mode: Literal["aggregate", "nodal"] = "aggregate"
    slack_bus_id: str | None = None


@dataclass(frozen=True)
class ImportConfig:
    maximum_power_mw: float
    price_eur_per_mwh: float
    emission_factor_tonnes_per_mwh: float


@dataclass(frozen=True)
class ImportResourceConfig:
    id: str
    bus_id: str
    config: ImportConfig


@dataclass(frozen=True)
class DemandConfig:
    id: str
    bus_id: str
    time_series_key: str
    kind: Literal["fixed", "curtailable", "shiftable", "deferrable", "ev_charging"] = "fixed"
    sector: str | None = None
    value_of_lost_load_eur_per_mwh: float | None = None
    voluntary_curtailment_cost_eur_per_mwh: float = 0.0
    maximum_curtailment_fraction: float = 0.0
    maximum_curtailment_mw: float | None = None
    shift_up_capacity_mw: float = 0.0
    shift_down_capacity_mw: float = 0.0
    shift_window_hours: float = 0.0
    rebound_fraction: float = 0.0
    shift_cost_eur_per_mwh: float = 0.0
    task_power_capacity_mw: float = 0.0
    task_required_energy_mwh: float = 0.0
    task_start_period: int = 0
    task_end_period: int | None = None
    task_unserved_penalty_eur_per_mwh: float = 0.0
    temperature_time_series_key: str | None = None
    heating_base_temperature_c: float | None = None
    cooling_base_temperature_c: float | None = None
    heating_sensitivity_mw_per_c: float = 0.0
    cooling_sensitivity_mw_per_c: float = 0.0


@dataclass(frozen=True)
class PenaltyConfig:
    renewable_curtailment_eur_per_mwh: float
    lost_load_eur_per_mwh: float
    carbon_price_eur_per_tonne: float


@dataclass(frozen=True)
class PathConfig:
    input_csv: Path
    output_directory: Path


@dataclass(frozen=True)
class PortfolioConfig:
    scenario: ScenarioConfig
    fuels: tuple[FuelConfig, ...]
    zones: tuple[ZoneConfig, ...]
    buses: tuple[BusConfig, ...]
    lines: tuple[TransmissionLineConfig, ...]
    renewable_generators: tuple[RenewableGeneratorConfig, ...]
    thermal_generators: tuple[ThermalGeneratorConfig, ...]
    storage_units: tuple[StorageUnitConfig, ...]
    hydro_units: tuple[HydroUnitConfig, ...]
    imports: tuple[ImportResourceConfig, ...]
    demand: tuple[DemandConfig, ...]


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    solver: SolverConfig
    portfolio: PortfolioConfig
    simulation: SimulationConfig
    solar: SolarConfig
    wind: WindConfig
    thermal: ThermalConfig
    battery: BatteryConfig
    network: NetworkConfig
    imports: ImportConfig
    penalties: PenaltyConfig
    paths: PathConfig


def _schema_version(raw: Mapping[str, Any]) -> int:
    if "schema_version" not in raw:
        return LEGACY_SCHEMA_VERSION
    return _integer(raw, "schema_version")


def _raw_config(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a mapping")
    return config_path, raw


def load_config(path: str | Path) -> ModelConfig:
    """Load and validate a YAML model configuration."""
    config_path, raw = _raw_config(path)
    schema_version = _schema_version(raw)
    if schema_version == LEGACY_SCHEMA_VERSION:
        return _load_legacy_config(config_path, raw, schema_version)
    if schema_version == CURRENT_SCHEMA_VERSION:
        return _load_schema_v2(config_path, raw)
    raise ConfigurationError(
        f"Unsupported configuration schema_version: {schema_version}. "
        f"Supported versions: {LEGACY_SCHEMA_VERSION}, {CURRENT_SCHEMA_VERSION}"
    )


def _load_legacy_config(
    config_path: Path,
    raw: Mapping[str, Any],
    schema_version: int,
) -> ModelConfig:
    _validate_allowed_keys(raw, "", LEGACY_ROOT_KEYS)
    _validate_required_keys(raw, "", LEGACY_REQUIRED_ROOT_KEYS)
    simulation_raw = _section(raw, "simulation")
    solar_raw = _section(raw, "solar")
    wind_raw = _section(raw, "wind")
    thermal_raw = _section(raw, "thermal")
    battery_raw = _section(raw, "battery")
    network_raw = _section(raw, "network")
    import_raw = _section(raw, "imports")
    penalties_raw = _section(raw, "penalties")
    paths_raw = _section(raw, "paths")
    for section_name, allowed_keys in LEGACY_SECTION_KEYS.items():
        section = _section(raw, section_name)
        _validate_allowed_keys(section, section_name, allowed_keys)
        _validate_required_keys(section, section_name, LEGACY_REQUIRED_SECTION_KEYS[section_name])

    simulation = SimulationConfig(
        time_step_hours=_number(simulation_raw, "time_step_hours", float),
        solver_time_limit_seconds=_number(simulation_raw, "solver_time_limit_seconds", float),
        mip_relative_gap=_number(simulation_raw, "mip_relative_gap", float),
        allow_non_optimal_solution=_optional_boolean(
            simulation_raw,
            "allow_non_optimal_solution",
            False,
        ),
    )
    solar = SolarConfig(
        capacity_mw=_number(solar_raw, "capacity_mw", float),
        performance_ratio=_number(solar_raw, "performance_ratio", float),
        reference_irradiance_w_m2=_number(solar_raw, "reference_irradiance_w_m2", float),
        temperature_coefficient_per_c=_number(solar_raw, "temperature_coefficient_per_c", float),
        nominal_operating_cell_temperature_c=_number(
            solar_raw, "nominal_operating_cell_temperature_c", float
        ),
    )
    wind = WindConfig(
        capacity_mw=_number(wind_raw, "capacity_mw", float),
        cut_in_speed_m_s=_number(wind_raw, "cut_in_speed_m_s", float),
        rated_speed_m_s=_number(wind_raw, "rated_speed_m_s", float),
        cut_out_speed_m_s=_number(wind_raw, "cut_out_speed_m_s", float),
    )
    thermal = ThermalConfig(
        name=_string(thermal_raw, "name"),
        minimum_output_mw=_number(thermal_raw, "minimum_output_mw", float),
        maximum_output_mw=_number(thermal_raw, "maximum_output_mw", float),
        ramp_up_mw_per_hour=_number(thermal_raw, "ramp_up_mw_per_hour", float),
        ramp_down_mw_per_hour=_number(thermal_raw, "ramp_down_mw_per_hour", float),
        startup_ramp_mw=_optional_number(
            thermal_raw,
            "startup_ramp_mw",
            float,
            _number(thermal_raw, "maximum_output_mw", float),
        ),
        shutdown_ramp_mw=_optional_number(
            thermal_raw,
            "shutdown_ramp_mw",
            float,
            _number(thermal_raw, "maximum_output_mw", float),
        ),
        variable_cost_eur_per_mwh=_number(thermal_raw, "variable_cost_eur_per_mwh", float),
        no_load_cost_eur_per_hour=_number(thermal_raw, "no_load_cost_eur_per_hour", float),
        startup_cost_eur=_number(thermal_raw, "startup_cost_eur", float),
        shutdown_cost_eur=_number(thermal_raw, "shutdown_cost_eur", float),
        emission_factor_tonnes_per_mwh=_number(
            thermal_raw, "emission_factor_tonnes_per_mwh", float
        ),
        minimum_up_hours=_number(thermal_raw, "minimum_up_hours", float),
        minimum_down_hours=_number(thermal_raw, "minimum_down_hours", float),
        initial_on=_boolean(thermal_raw, "initial_on"),
        initial_output_mw=_number(thermal_raw, "initial_output_mw", float),
        initial_up_time_hours=_optional_number(
            thermal_raw,
            "initial_up_time_hours",
            float,
            0.0,
        ),
        initial_down_time_hours=_optional_number(
            thermal_raw,
            "initial_down_time_hours",
            float,
            0.0,
        ),
        terminal_commitment_mode=cast(
            TerminalCommitmentMode,
            _optional_string(
                thermal_raw,
                "terminal_commitment_mode",
                "forbid_incomplete_transitions",
            ),
        ),
        terminal_on=_optional_nullable_boolean(thermal_raw, "terminal_on"),
    )
    battery = BatteryConfig(
        energy_capacity_mwh=_number(battery_raw, "energy_capacity_mwh", float),
        power_capacity_mw=_number(battery_raw, "power_capacity_mw", float),
        minimum_soc_mwh=_number(battery_raw, "minimum_soc_mwh", float),
        maximum_soc_mwh=_number(battery_raw, "maximum_soc_mwh", float),
        initial_soc_mwh=_number(battery_raw, "initial_soc_mwh", float),
        charge_efficiency=_number(battery_raw, "charge_efficiency", float),
        discharge_efficiency=_number(battery_raw, "discharge_efficiency", float),
        throughput_cost_eur_per_mwh=_number(battery_raw, "throughput_cost_eur_per_mwh", float),
        minimum_final_soc_mwh=_number(battery_raw, "minimum_final_soc_mwh", float),
        terminal_soc_mode=cast(
            Literal["minimum", "exact", "cyclic", "free"],
            _optional_string(battery_raw, "terminal_soc_mode", "minimum"),
        ),
        charge_power_capacity_mw=(
            _number(battery_raw, "charge_power_capacity_mw", float)
            if "charge_power_capacity_mw" in battery_raw
            else None
        ),
        discharge_power_capacity_mw=(
            _number(battery_raw, "discharge_power_capacity_mw", float)
            if "discharge_power_capacity_mw" in battery_raw
            else None
        ),
        self_discharge_rate_per_hour=_optional_number(
            battery_raw,
            "self_discharge_rate_per_hour",
            float,
            0.0,
        ),
        minimum_charge_mw=_optional_number(battery_raw, "minimum_charge_mw", float, 0.0),
        minimum_discharge_mw=_optional_number(
            battery_raw,
            "minimum_discharge_mw",
            float,
            0.0,
        ),
        charge_ramp_mw_per_hour=(
            _number(battery_raw, "charge_ramp_mw_per_hour", float)
            if "charge_ramp_mw_per_hour" in battery_raw
            else None
        ),
        discharge_ramp_mw_per_hour=(
            _number(battery_raw, "discharge_ramp_mw_per_hour", float)
            if "discharge_ramp_mw_per_hour" in battery_raw
            else None
        ),
    )
    network = NetworkConfig(
        loss_fraction=_number(network_raw, "loss_fraction", float),
        transfer_capacity_mw=_number(network_raw, "transfer_capacity_mw", float),
    )
    imports = ImportConfig(
        maximum_power_mw=_number(import_raw, "maximum_power_mw", float),
        price_eur_per_mwh=_number(import_raw, "price_eur_per_mwh", float),
        emission_factor_tonnes_per_mwh=_number(import_raw, "emission_factor_tonnes_per_mwh", float),
    )
    penalties = PenaltyConfig(
        renewable_curtailment_eur_per_mwh=_number(
            penalties_raw, "renewable_curtailment_eur_per_mwh", float
        ),
        lost_load_eur_per_mwh=_number(penalties_raw, "lost_load_eur_per_mwh", float),
        carbon_price_eur_per_tonne=_number(penalties_raw, "carbon_price_eur_per_tonne", float),
    )

    base = config_path.parent
    input_csv = (base / _string(paths_raw, "input_csv")).resolve()
    output_directory = (base / _string(paths_raw, "output_directory")).resolve()
    paths = PathConfig(input_csv=input_csv, output_directory=output_directory)
    solver = SolverConfig(
        solver_time_limit_seconds=simulation.solver_time_limit_seconds,
        mip_relative_gap=simulation.mip_relative_gap,
        allow_non_optimal_solution=simulation.allow_non_optimal_solution,
    )
    portfolio = _legacy_portfolio(solar, wind, thermal, battery, imports)

    config = ModelConfig(
        schema_version=schema_version,
        solver=solver,
        portfolio=portfolio,
        simulation=simulation,
        solar=solar,
        wind=wind,
        thermal=thermal,
        battery=battery,
        network=network,
        imports=imports,
        penalties=penalties,
        paths=paths,
    )
    validate_config(config)
    return config


def _load_schema_v2(config_path: Path, raw: Mapping[str, Any]) -> ModelConfig:
    _validate_allowed_keys(raw, "", ROOT_KEYS)
    _validate_required_keys(raw, "", REQUIRED_ROOT_KEYS)
    for section_name, allowed_keys in SECTION_KEYS.items():
        section = _section(raw, section_name)
        _validate_allowed_keys(section, section_name, allowed_keys)
        required_keys = allowed_keys - OPTIONAL_SECTION_KEYS.get(section_name, set())
        _validate_required_keys(section, section_name, required_keys)
    for section_name, allowed_keys in LIST_SECTION_KEYS.items():
        items = _list_section(
            raw,
            section_name,
            required=section_name not in {"fuels", "hydro_units"},
        )
        for index, item in enumerate(items):
            path = f"{section_name}[{index}]"
            _validate_allowed_keys(item, path, allowed_keys)
            _validate_required_keys(item, path, LIST_REQUIRED_KEYS[section_name])

    scenario_raw = _section(raw, "scenario")
    simulation_raw = _section(raw, "simulation")
    solver_raw = _section(raw, "solver")
    network_raw = _section(raw, "aggregate_network")
    penalties_raw = _section(raw, "penalties")
    paths_raw = _section(raw, "paths")

    solver = SolverConfig(
        solver_time_limit_seconds=_number_at(
            solver_raw, "solver_time_limit_seconds", float, "solver"
        ),
        mip_relative_gap=_number_at(solver_raw, "mip_relative_gap", float, "solver"),
        allow_non_optimal_solution=_boolean_at(solver_raw, "allow_non_optimal_solution", "solver"),
    )
    simulation = SimulationConfig(
        time_step_hours=_number_at(simulation_raw, "time_step_hours", float, "simulation"),
        solver_time_limit_seconds=solver.solver_time_limit_seconds,
        mip_relative_gap=solver.mip_relative_gap,
        allow_non_optimal_solution=solver.allow_non_optimal_solution,
    )
    zones = tuple(
        ZoneConfig(id=_id_at(item, "id", f"zones[{index}]"))
        for index, item in enumerate(_list_section(raw, "zones"))
    )
    buses = tuple(
        BusConfig(
            id=_id_at(item, "id", f"buses[{index}]"),
            zone_id=_id_at(item, "zone_id", f"buses[{index}]"),
        )
        for index, item in enumerate(_list_section(raw, "buses"))
    )
    lines = tuple(
        TransmissionLineConfig(
            id=_id_at(item, "id", f"lines[{index}]"),
            from_bus_id=_id_at(item, "from_bus_id", f"lines[{index}]"),
            to_bus_id=_id_at(item, "to_bus_id", f"lines[{index}]"),
            susceptance=_number_at(item, "susceptance", float, f"lines[{index}]"),
            capacity_mw=_number_at(item, "capacity_mw", float, f"lines[{index}]"),
            availability_factor=_optional_number_at(
                item,
                "availability_factor",
                float,
                1.0,
                f"lines[{index}]",
            ),
            availability_factor_key=(
                _input_key_at(item, "availability_factor_key", f"lines[{index}]")
                if "availability_factor_key" in item
                else None
            ),
        )
        for index, item in enumerate(_list_section(raw, "lines"))
    )
    fuels = tuple(
        _parse_fuel(item, f"fuels[{index}]")
        for index, item in enumerate(_list_section(raw, "fuels", required=False))
    )
    renewable_generators = tuple(
        _parse_renewable_generator(item, f"renewable_generators[{index}]")
        for index, item in enumerate(_list_section(raw, "renewable_generators"))
    )
    thermal_generators = tuple(
        _parse_thermal_generator(item, f"thermal_generators[{index}]")
        for index, item in enumerate(_list_section(raw, "thermal_generators"))
    )
    storage_units = tuple(
        _parse_storage_unit(item, f"storage_units[{index}]")
        for index, item in enumerate(_list_section(raw, "storage_units"))
    )
    hydro_units = tuple(
        _parse_hydro_unit(item, f"hydro_units[{index}]")
        for index, item in enumerate(_list_section(raw, "hydro_units", required=False))
    )
    imports = tuple(
        ImportResourceConfig(
            id=_id_at(item, "id", f"imports[{index}]"),
            bus_id=_id_at(item, "bus_id", f"imports[{index}]"),
            config=ImportConfig(
                maximum_power_mw=_number_at(item, "maximum_power_mw", float, f"imports[{index}]"),
                price_eur_per_mwh=_number_at(item, "price_eur_per_mwh", float, f"imports[{index}]"),
                emission_factor_tonnes_per_mwh=_number_at(
                    item, "emission_factor_tonnes_per_mwh", float, f"imports[{index}]"
                ),
            ),
        )
        for index, item in enumerate(_list_section(raw, "imports"))
    )
    demand = tuple(
        _parse_demand(item, f"demand[{index}]")
        for index, item in enumerate(_list_section(raw, "demand"))
    )
    portfolio = PortfolioConfig(
        scenario=ScenarioConfig(id=_id_at(scenario_raw, "id", "scenario")),
        fuels=fuels,
        zones=zones,
        buses=buses,
        lines=lines,
        renewable_generators=renewable_generators,
        thermal_generators=thermal_generators,
        storage_units=storage_units,
        hydro_units=hydro_units,
        imports=imports,
        demand=demand,
    )
    portfolio = _portfolio_with_default_fuels(portfolio)
    _validate_portfolio(portfolio)

    solar = _primary_solar(renewable_generators)
    wind = _primary_wind(renewable_generators)
    thermal = thermal_generators[0].config
    battery = storage_units[0].config
    network_mode = _optional_string_at(
        network_raw, "network_mode", "aggregate", "aggregate_network"
    )
    if network_mode not in {"aggregate", "nodal"}:
        raise ConfigurationError("aggregate_network.network_mode must be aggregate or nodal")
    network = NetworkConfig(
        loss_fraction=_number_at(network_raw, "loss_fraction", float, "aggregate_network"),
        transfer_capacity_mw=_number_at(
            network_raw, "transfer_capacity_mw", float, "aggregate_network"
        ),
        network_mode=cast(Literal["aggregate", "nodal"], network_mode),
        slack_bus_id=(
            _id_at(network_raw, "slack_bus_id", "aggregate_network")
            if "slack_bus_id" in network_raw
            else None
        ),
    )
    import_config = imports[0].config
    penalties = PenaltyConfig(
        renewable_curtailment_eur_per_mwh=_number_at(
            penalties_raw, "renewable_curtailment_eur_per_mwh", float, "penalties"
        ),
        lost_load_eur_per_mwh=_number_at(
            penalties_raw, "lost_load_eur_per_mwh", float, "penalties"
        ),
        carbon_price_eur_per_tonne=_number_at(
            penalties_raw, "carbon_price_eur_per_tonne", float, "penalties"
        ),
    )
    base = config_path.parent
    paths = PathConfig(
        input_csv=(base / _string_at(paths_raw, "input_csv", "paths")).resolve(),
        output_directory=(base / _string_at(paths_raw, "output_directory", "paths")).resolve(),
    )
    config = ModelConfig(
        schema_version=CURRENT_SCHEMA_VERSION,
        solver=solver,
        portfolio=portfolio,
        simulation=simulation,
        solar=solar,
        wind=wind,
        thermal=thermal,
        battery=battery,
        network=network,
        imports=import_config,
        penalties=penalties,
        paths=paths,
    )
    _validate_schema_v2_assets(portfolio)
    validate_config(config)
    return config


def _parse_fuel(item: Mapping[str, Any], path: str) -> FuelConfig:
    return FuelConfig(
        id=_id_at(item, "id", path),
        price_eur_per_mwh_thermal=_number_at(
            item,
            "price_eur_per_mwh_thermal",
            float,
            path,
        ),
        co2_factor_tonnes_per_mwh_thermal=_number_at(
            item,
            "co2_factor_tonnes_per_mwh_thermal",
            float,
            path,
        ),
        price_time_series_key=(
            _string_at(item, "price_time_series_key", path)
            if "price_time_series_key" in item
            else None
        ),
        lower_heating_value_mj_per_unit=(
            _number_at(item, "lower_heating_value_mj_per_unit", float, path)
            if "lower_heating_value_mj_per_unit" in item
            else None
        ),
        methane_factor_tonnes_per_mwh_thermal=_optional_number_at(
            item,
            "methane_factor_tonnes_per_mwh_thermal",
            float,
            0.0,
            path,
        ),
        nox_factor_kg_per_mwh_thermal=_optional_number_at(
            item,
            "nox_factor_kg_per_mwh_thermal",
            float,
            0.0,
            path,
        ),
        sox_factor_kg_per_mwh_thermal=_optional_number_at(
            item,
            "sox_factor_kg_per_mwh_thermal",
            float,
            0.0,
            path,
        ),
    )


def _parse_renewable_generator(
    item: Mapping[str, Any],
    path: str,
) -> RenewableGeneratorConfig:
    kind = _string_at(item, "kind", path)
    if kind not in {"solar", "wind"}:
        raise ConfigurationError(f"{path}.kind must be one of: solar, wind")
    default_key = "irradiance_w_m2" if kind == "solar" else "wind_speed_m_s"
    ambient_temperature_key = None
    if kind == "solar":
        ambient_temperature_key = _optional_input_key_at(
            item,
            "ambient_temperature_key",
            "ambient_temperature_c",
            path,
        )
    return RenewableGeneratorConfig(
        id=_id_at(item, "id", path),
        kind=cast(Literal["solar", "wind"], kind),
        bus_id=_id_at(item, "bus_id", path),
        capacity_mw=_number_at(item, "capacity_mw", float, path),
        time_series_key=_optional_input_key_at(item, "time_series_key", default_key, path),
        ambient_temperature_key=ambient_temperature_key,
        performance_ratio=_optional_number_at(item, "performance_ratio", float, 0.86, path),
        reference_irradiance_w_m2=_optional_number_at(
            item,
            "reference_irradiance_w_m2",
            float,
            1000.0,
            path,
        ),
        temperature_coefficient_per_c=_optional_number_at(
            item,
            "temperature_coefficient_per_c",
            float,
            -0.004,
            path,
        ),
        nominal_operating_cell_temperature_c=_optional_number_at(
            item,
            "nominal_operating_cell_temperature_c",
            float,
            45.0,
            path,
        ),
        cut_in_speed_m_s=_optional_number_at(item, "cut_in_speed_m_s", float, 3.0, path),
        rated_speed_m_s=_optional_number_at(item, "rated_speed_m_s", float, 12.0, path),
        cut_out_speed_m_s=_optional_number_at(item, "cut_out_speed_m_s", float, 25.0, path),
    )


def _parse_thermal_generator(item: Mapping[str, Any], path: str) -> ThermalGeneratorConfig:
    maximum_output = _number_at(item, "maximum_output_mw", float, path)
    return ThermalGeneratorConfig(
        id=_id_at(item, "id", path),
        bus_id=_id_at(item, "bus_id", path),
        fuel_id=_id_at(item, "fuel_id", path),
        config=ThermalConfig(
            name=_string_at(item, "name", path),
            minimum_output_mw=_number_at(item, "minimum_output_mw", float, path),
            maximum_output_mw=maximum_output,
            ramp_up_mw_per_hour=_number_at(item, "ramp_up_mw_per_hour", float, path),
            ramp_down_mw_per_hour=_number_at(item, "ramp_down_mw_per_hour", float, path),
            startup_ramp_mw=_optional_number_at(
                item,
                "startup_ramp_mw",
                float,
                maximum_output,
                path,
            ),
            shutdown_ramp_mw=_optional_number_at(
                item,
                "shutdown_ramp_mw",
                float,
                maximum_output,
                path,
            ),
            variable_cost_eur_per_mwh=_number_at(item, "variable_cost_eur_per_mwh", float, path),
            no_load_cost_eur_per_hour=_number_at(item, "no_load_cost_eur_per_hour", float, path),
            startup_cost_eur=_number_at(item, "startup_cost_eur", float, path),
            shutdown_cost_eur=_number_at(item, "shutdown_cost_eur", float, path),
            emission_factor_tonnes_per_mwh=_number_at(
                item,
                "emission_factor_tonnes_per_mwh",
                float,
                path,
            ),
            minimum_up_hours=_number_at(item, "minimum_up_hours", float, path),
            minimum_down_hours=_number_at(item, "minimum_down_hours", float, path),
            initial_on=_boolean_at(item, "initial_on", path),
            initial_output_mw=_number_at(item, "initial_output_mw", float, path),
            initial_up_time_hours=_optional_number_at(
                item,
                "initial_up_time_hours",
                float,
                0.0,
                path,
            ),
            initial_down_time_hours=_optional_number_at(
                item,
                "initial_down_time_hours",
                float,
                0.0,
                path,
            ),
            terminal_commitment_mode=cast(
                TerminalCommitmentMode,
                _optional_string_at(
                    item,
                    "terminal_commitment_mode",
                    "forbid_incomplete_transitions",
                    path,
                ),
            ),
            terminal_on=_optional_nullable_boolean_at(item, "terminal_on", path),
            minimum_fuel_input_mwh_per_hour=_optional_number_at(
                item,
                "minimum_fuel_input_mwh_per_hour",
                float,
                0.0,
                path,
            ),
            heat_rate_segments=_parse_heat_rate_segments(item, path),
            startup_categories=_parse_startup_categories(item, path),
        ),
        must_run=_optional_boolean_at(item, "must_run", False, path),
        availability_factor=_optional_number_at(item, "availability_factor", float, 1.0, path),
        availability_factor_key=(
            _string_at(item, "availability_factor_key", path)
            if "availability_factor_key" in item
            else None
        ),
    )


def _parse_heat_rate_segments(
    item: Mapping[str, Any],
    path: str,
) -> tuple[HeatRateSegmentConfig, ...]:
    if "heat_rate_segments" not in item:
        return ()
    raw = item["heat_rate_segments"]
    if not isinstance(raw, list):
        raise ConfigurationError(f"{path}.heat_rate_segments must be a list")
    allowed = {"id", "capacity_mw", "heat_rate_mwh_thermal_per_mwh"}
    segments: list[HeatRateSegmentConfig] = []
    for index, segment in enumerate(raw):
        segment_path = f"{path}.heat_rate_segments[{index}]"
        if not isinstance(segment, Mapping):
            raise ConfigurationError(f"{segment_path} must be a mapping")
        _validate_allowed_keys(segment, segment_path, allowed)
        _validate_required_keys(segment, segment_path, allowed)
        segments.append(
            HeatRateSegmentConfig(
                id=_id_at(segment, "id", segment_path),
                capacity_mw=_number_at(segment, "capacity_mw", float, segment_path),
                heat_rate_mwh_thermal_per_mwh=_number_at(
                    segment,
                    "heat_rate_mwh_thermal_per_mwh",
                    float,
                    segment_path,
                ),
            )
        )
    return tuple(segments)


def _parse_startup_categories(
    item: Mapping[str, Any],
    path: str,
) -> tuple[StartupCategoryConfig, ...]:
    if "startup_categories" not in item:
        return ()
    raw = item["startup_categories"]
    if not isinstance(raw, list):
        raise ConfigurationError(f"{path}.startup_categories must be a list")
    required = {"id", "minimum_down_time_hours", "startup_cost_eur"}
    allowed = required | {"startup_fuel_input_mwh_thermal"}
    categories: list[StartupCategoryConfig] = []
    for index, category in enumerate(raw):
        category_path = f"{path}.startup_categories[{index}]"
        if not isinstance(category, Mapping):
            raise ConfigurationError(f"{category_path} must be a mapping")
        _validate_allowed_keys(category, category_path, allowed)
        _validate_required_keys(category, category_path, required)
        categories.append(
            StartupCategoryConfig(
                id=_id_at(category, "id", category_path),
                minimum_down_time_hours=_number_at(
                    category,
                    "minimum_down_time_hours",
                    float,
                    category_path,
                ),
                startup_cost_eur=_number_at(category, "startup_cost_eur", float, category_path),
                startup_fuel_input_mwh_thermal=_optional_number_at(
                    category,
                    "startup_fuel_input_mwh_thermal",
                    float,
                    0.0,
                    category_path,
                ),
            )
        )
    return tuple(categories)


def _parse_storage_unit(item: Mapping[str, Any], path: str) -> StorageUnitConfig:
    technology = _optional_string_at(item, "technology", "battery", path)
    if technology not in {"battery", "pumped_storage"}:
        raise ConfigurationError(f"{path}.technology must be one of: battery, pumped_storage")
    return StorageUnitConfig(
        id=_id_at(item, "id", path),
        bus_id=_id_at(item, "bus_id", path),
        config=BatteryConfig(
            energy_capacity_mwh=_number_at(item, "energy_capacity_mwh", float, path),
            power_capacity_mw=_number_at(item, "power_capacity_mw", float, path),
            charge_power_capacity_mw=(
                _number_at(item, "charge_power_capacity_mw", float, path)
                if "charge_power_capacity_mw" in item
                else None
            ),
            discharge_power_capacity_mw=(
                _number_at(item, "discharge_power_capacity_mw", float, path)
                if "discharge_power_capacity_mw" in item
                else None
            ),
            minimum_soc_mwh=_number_at(item, "minimum_soc_mwh", float, path),
            maximum_soc_mwh=_number_at(item, "maximum_soc_mwh", float, path),
            initial_soc_mwh=_number_at(item, "initial_soc_mwh", float, path),
            charge_efficiency=_number_at(item, "charge_efficiency", float, path),
            discharge_efficiency=_number_at(item, "discharge_efficiency", float, path),
            self_discharge_rate_per_hour=_optional_number_at(
                item,
                "self_discharge_rate_per_hour",
                float,
                0.0,
                path,
            ),
            minimum_charge_mw=_optional_number_at(item, "minimum_charge_mw", float, 0.0, path),
            minimum_discharge_mw=_optional_number_at(
                item,
                "minimum_discharge_mw",
                float,
                0.0,
                path,
            ),
            charge_ramp_mw_per_hour=(
                _number_at(item, "charge_ramp_mw_per_hour", float, path)
                if "charge_ramp_mw_per_hour" in item
                else None
            ),
            discharge_ramp_mw_per_hour=(
                _number_at(item, "discharge_ramp_mw_per_hour", float, path)
                if "discharge_ramp_mw_per_hour" in item
                else None
            ),
            throughput_cost_eur_per_mwh=_number_at(
                item,
                "throughput_cost_eur_per_mwh",
                float,
                path,
            ),
            minimum_final_soc_mwh=_number_at(item, "minimum_final_soc_mwh", float, path),
            terminal_soc_mode=cast(
                Literal["minimum", "exact", "cyclic", "free"],
                _optional_string_at(item, "terminal_soc_mode", "minimum", path),
            ),
            technology=cast(Literal["battery", "pumped_storage"], technology),
            availability_factor=_optional_number_at(
                item,
                "availability_factor",
                float,
                1.0,
                path,
            ),
            availability_factor_key=(
                _string_at(item, "availability_factor_key", path)
                if "availability_factor_key" in item
                else None
            ),
            degradation_bands=_parse_degradation_bands(item, path),
        ),
    )


def _parse_degradation_bands(
    item: Mapping[str, Any],
    path: str,
) -> tuple[StorageDegradationBandConfig, ...]:
    if "degradation_bands" not in item:
        return ()
    raw = item["degradation_bands"]
    if not isinstance(raw, list):
        raise ConfigurationError(f"{path}.degradation_bands must be a list")
    allowed = {"id", "capacity_mwh", "cost_eur_per_mwh"}
    bands: list[StorageDegradationBandConfig] = []
    for index, band in enumerate(raw):
        band_path = f"{path}.degradation_bands[{index}]"
        if not isinstance(band, Mapping):
            raise ConfigurationError(f"{band_path} must be a mapping")
        _validate_allowed_keys(band, band_path, allowed)
        _validate_required_keys(band, band_path, allowed)
        bands.append(
            StorageDegradationBandConfig(
                id=_id_at(band, "id", band_path),
                capacity_mwh=_number_at(band, "capacity_mwh", float, band_path),
                cost_eur_per_mwh=_number_at(band, "cost_eur_per_mwh", float, band_path),
            )
        )
    return tuple(bands)


def _parse_hydro_unit(item: Mapping[str, Any], path: str) -> HydroUnitConfig:
    kind = _optional_string_at(item, "kind", "reservoir", path)
    if kind not in {"reservoir", "run_of_river"}:
        raise ConfigurationError(f"{path}.kind must be one of: reservoir, run_of_river")
    spill_capacity = (
        _number_at(item, "spill_capacity_mw", float, path) if "spill_capacity_mw" in item else None
    )
    return HydroUnitConfig(
        id=_id_at(item, "id", path),
        bus_id=_id_at(item, "bus_id", path),
        kind=cast(Literal["reservoir", "run_of_river"], kind),
        inflow_time_series_key=_input_key_at(item, "inflow_time_series_key", path),
        turbine_capacity_mw=_number_at(item, "turbine_capacity_mw", float, path),
        turbine_efficiency=_number_at(item, "turbine_efficiency", float, path),
        minimum_reservoir_mwh=_optional_number_at(
            item,
            "minimum_reservoir_mwh",
            float,
            0.0,
            path,
        ),
        maximum_reservoir_mwh=_optional_number_at(
            item,
            "maximum_reservoir_mwh",
            float,
            0.0,
            path,
        ),
        initial_reservoir_mwh=_optional_number_at(
            item,
            "initial_reservoir_mwh",
            float,
            0.0,
            path,
        ),
        minimum_final_reservoir_mwh=_optional_number_at(
            item,
            "minimum_final_reservoir_mwh",
            float,
            0.0,
            path,
        ),
        terminal_reservoir_mode=cast(
            Literal["minimum", "exact", "cyclic", "free"],
            _optional_string_at(item, "terminal_reservoir_mode", "minimum", path),
        ),
        spill_capacity_mw=spill_capacity,
        minimum_release_mw=_optional_number_at(item, "minimum_release_mw", float, 0.0, path),
        evaporation_rate_per_hour=_optional_number_at(
            item,
            "evaporation_rate_per_hour",
            float,
            0.0,
            path,
        ),
        water_value_eur_per_mwh=_optional_number_at(
            item,
            "water_value_eur_per_mwh",
            float,
            0.0,
            path,
        ),
        upstream_hydro_id=(
            _id_at(item, "upstream_hydro_id", path) if "upstream_hydro_id" in item else None
        ),
        cascade_delay_hours=_optional_number_at(item, "cascade_delay_hours", float, 0.0, path),
    )


def _parse_demand(item: Mapping[str, Any], path: str) -> DemandConfig:
    kind = _optional_string_at(item, "kind", "fixed", path)
    if kind not in {"fixed", "curtailable", "shiftable", "deferrable", "ev_charging"}:
        raise ConfigurationError(
            f"{path}.kind must be one of: fixed, curtailable, shiftable, deferrable, ev_charging"
        )
    task_end_period = (
        _integer_at(item, "task_end_period", path) if "task_end_period" in item else None
    )
    return DemandConfig(
        id=_id_at(item, "id", path),
        bus_id=_id_at(item, "bus_id", path),
        time_series_key=_input_key_at(item, "time_series_key", path),
        kind=cast(
            Literal["fixed", "curtailable", "shiftable", "deferrable", "ev_charging"],
            kind,
        ),
        sector=_string_at(item, "sector", path) if "sector" in item else None,
        value_of_lost_load_eur_per_mwh=(
            _number_at(item, "value_of_lost_load_eur_per_mwh", float, path)
            if "value_of_lost_load_eur_per_mwh" in item
            else None
        ),
        voluntary_curtailment_cost_eur_per_mwh=_optional_number_at(
            item,
            "voluntary_curtailment_cost_eur_per_mwh",
            float,
            0.0,
            path,
        ),
        maximum_curtailment_fraction=_optional_number_at(
            item,
            "maximum_curtailment_fraction",
            float,
            0.0,
            path,
        ),
        maximum_curtailment_mw=(
            _number_at(item, "maximum_curtailment_mw", float, path)
            if "maximum_curtailment_mw" in item
            else None
        ),
        shift_up_capacity_mw=_optional_number_at(item, "shift_up_capacity_mw", float, 0.0, path),
        shift_down_capacity_mw=_optional_number_at(
            item,
            "shift_down_capacity_mw",
            float,
            0.0,
            path,
        ),
        shift_window_hours=_optional_number_at(item, "shift_window_hours", float, 0.0, path),
        rebound_fraction=_optional_number_at(item, "rebound_fraction", float, 0.0, path),
        shift_cost_eur_per_mwh=_optional_number_at(
            item,
            "shift_cost_eur_per_mwh",
            float,
            0.0,
            path,
        ),
        task_power_capacity_mw=_optional_number_at(
            item,
            "task_power_capacity_mw",
            float,
            0.0,
            path,
        ),
        task_required_energy_mwh=_optional_number_at(
            item,
            "task_required_energy_mwh",
            float,
            0.0,
            path,
        ),
        task_start_period=_optional_integer_at(item, "task_start_period", 0, path),
        task_end_period=task_end_period,
        task_unserved_penalty_eur_per_mwh=_optional_number_at(
            item,
            "task_unserved_penalty_eur_per_mwh",
            float,
            0.0,
            path,
        ),
        temperature_time_series_key=(
            _input_key_at(item, "temperature_time_series_key", path)
            if "temperature_time_series_key" in item
            else None
        ),
        heating_base_temperature_c=(
            _number_at(item, "heating_base_temperature_c", float, path)
            if "heating_base_temperature_c" in item
            else None
        ),
        cooling_base_temperature_c=(
            _number_at(item, "cooling_base_temperature_c", float, path)
            if "cooling_base_temperature_c" in item
            else None
        ),
        heating_sensitivity_mw_per_c=_optional_number_at(
            item,
            "heating_sensitivity_mw_per_c",
            float,
            0.0,
            path,
        ),
        cooling_sensitivity_mw_per_c=_optional_number_at(
            item,
            "cooling_sensitivity_mw_per_c",
            float,
            0.0,
            path,
        ),
    )


def _legacy_portfolio(
    solar: SolarConfig,
    wind: WindConfig,
    thermal: ThermalConfig,
    battery: BatteryConfig,
    imports: ImportConfig,
) -> PortfolioConfig:
    return PortfolioConfig(
        scenario=ScenarioConfig(id="legacy_single_system"),
        fuels=(
            FuelConfig(
                id="legacy_thermal_fuel",
                price_eur_per_mwh_thermal=0.0,
                co2_factor_tonnes_per_mwh_thermal=0.0,
            ),
        ),
        zones=(ZoneConfig(id="system"),),
        buses=(BusConfig(id="system", zone_id="system"),),
        lines=(),
        renewable_generators=(
            RenewableGeneratorConfig(
                id="solar_1",
                kind="solar",
                bus_id="system",
                capacity_mw=solar.capacity_mw,
                time_series_key="irradiance_w_m2",
                ambient_temperature_key="ambient_temperature_c",
                performance_ratio=solar.performance_ratio,
                reference_irradiance_w_m2=solar.reference_irradiance_w_m2,
                temperature_coefficient_per_c=solar.temperature_coefficient_per_c,
                nominal_operating_cell_temperature_c=solar.nominal_operating_cell_temperature_c,
            ),
            RenewableGeneratorConfig(
                id="wind_1",
                kind="wind",
                bus_id="system",
                capacity_mw=wind.capacity_mw,
                time_series_key="wind_speed_m_s",
                cut_in_speed_m_s=wind.cut_in_speed_m_s,
                rated_speed_m_s=wind.rated_speed_m_s,
                cut_out_speed_m_s=wind.cut_out_speed_m_s,
            ),
        ),
        thermal_generators=(
            ThermalGeneratorConfig(
                id="thermal_1",
                bus_id="system",
                fuel_id="legacy_thermal_fuel",
                config=thermal,
            ),
        ),
        storage_units=(StorageUnitConfig(id="battery_1", bus_id="system", config=battery),),
        hydro_units=(),
        imports=(ImportResourceConfig(id="imports_1", bus_id="system", config=imports),),
        demand=(DemandConfig(id="demand_1", bus_id="system", time_series_key="demand_mw"),),
    )


def _portfolio_with_default_fuels(portfolio: PortfolioConfig) -> PortfolioConfig:
    if portfolio.fuels:
        return portfolio
    fuels_by_id = {
        generator.fuel_id: FuelConfig(
            id=generator.fuel_id,
            price_eur_per_mwh_thermal=0.0,
            co2_factor_tonnes_per_mwh_thermal=0.0,
        )
        for generator in portfolio.thermal_generators
    }
    return PortfolioConfig(
        scenario=portfolio.scenario,
        fuels=tuple(fuels_by_id[key] for key in sorted(fuels_by_id)),
        zones=portfolio.zones,
        buses=portfolio.buses,
        lines=portfolio.lines,
        renewable_generators=portfolio.renewable_generators,
        thermal_generators=portfolio.thermal_generators,
        storage_units=portfolio.storage_units,
        hydro_units=portfolio.hydro_units,
        imports=portfolio.imports,
        demand=portfolio.demand,
    )


def _validate_portfolio(portfolio: PortfolioConfig) -> None:
    _validate_unique_ids(portfolio.fuels, "fuels")
    _validate_unique_ids(portfolio.zones, "zones")
    _validate_unique_ids(portfolio.buses, "buses")
    _validate_unique_ids(portfolio.lines, "lines")
    _validate_unique_ids(portfolio.renewable_generators, "renewable_generators")
    _validate_unique_ids(portfolio.thermal_generators, "thermal_generators")
    _validate_unique_ids(portfolio.storage_units, "storage_units")
    _validate_unique_ids(portfolio.hydro_units, "hydro_units")
    _validate_unique_ids(portfolio.imports, "imports")
    _validate_unique_ids(portfolio.demand, "demand")
    if not portfolio.renewable_generators:
        raise ConfigurationError("renewable_generators must include at least one item")
    if not portfolio.thermal_generators:
        raise ConfigurationError("thermal_generators must include at least one item")
    if not portfolio.fuels:
        raise ConfigurationError("fuels must include at least one item")
    if not portfolio.storage_units:
        raise ConfigurationError("storage_units must include at least one item")
    if not portfolio.imports:
        raise ConfigurationError("imports must include at least one item")
    if not portfolio.demand:
        raise ConfigurationError("demand must include at least one item")
    zone_ids = {zone.id for zone in portfolio.zones}
    bus_ids = {bus.id for bus in portfolio.buses}
    fuel_ids = {fuel.id for fuel in portfolio.fuels}
    hydro_ids = {hydro.id for hydro in portfolio.hydro_units}
    for index, bus in enumerate(portfolio.buses):
        if bus.zone_id not in zone_ids:
            raise ConfigurationError(
                f"buses[{index}].zone_id references unknown zone: {bus.zone_id}"
            )
    for index, line in enumerate(portfolio.lines):
        if line.from_bus_id not in bus_ids:
            raise ConfigurationError(
                f"lines[{index}].from_bus_id references unknown bus: {line.from_bus_id}"
            )
        if line.to_bus_id not in bus_ids:
            raise ConfigurationError(
                f"lines[{index}].to_bus_id references unknown bus: {line.to_bus_id}"
            )
        if line.from_bus_id == line.to_bus_id:
            raise ConfigurationError(f"lines[{index}] cannot connect a bus to itself")
    for section, items in (
        ("renewable_generators", portfolio.renewable_generators),
        ("thermal_generators", portfolio.thermal_generators),
        ("storage_units", portfolio.storage_units),
        ("hydro_units", portfolio.hydro_units),
        ("imports", portfolio.imports),
        ("demand", portfolio.demand),
    ):
        for index, item in enumerate(items):
            bus_id = cast(Any, item).bus_id
            if bus_id not in bus_ids:
                raise ConfigurationError(
                    f"{section}[{index}].bus_id references unknown bus: {bus_id}"
                )
    for index, generator in enumerate(portfolio.thermal_generators):
        if generator.fuel_id not in fuel_ids:
            raise ConfigurationError(
                f"thermal_generators[{index}].fuel_id references unknown fuel: {generator.fuel_id}"
            )
    for index, hydro in enumerate(portfolio.hydro_units):
        if hydro.upstream_hydro_id is not None and hydro.upstream_hydro_id not in hydro_ids:
            raise ConfigurationError(
                f"hydro_units[{index}].upstream_hydro_id references unknown hydro unit: "
                f"{hydro.upstream_hydro_id}"
            )


def _validate_unique_ids(items: tuple[Any, ...], path: str) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        identifier = cast(str, item.id)
        if identifier in seen:
            raise ConfigurationError(f"Duplicate identifier at {path}[{index}].id: {identifier}")
        seen.add(identifier)


def _validate_schema_v2_assets(portfolio: PortfolioConfig) -> None:
    for index, fuel in enumerate(portfolio.fuels):
        _validate_fuel_at(fuel, f"fuels[{index}]")
    solar_found = False
    wind_found = False
    for index, generator in enumerate(portfolio.renewable_generators):
        path = f"renewable_generators[{index}]"
        _check_nonnegative_at(f"{path}.capacity_mw", generator.capacity_mw)
        if generator.kind == "solar":
            solar_found = True
            _validate_solar_generator_at(generator, path)
        else:
            wind_found = True
            _validate_wind_generator_at(generator, path)
    if not solar_found:
        raise ConfigurationError("renewable_generators must include at least one solar generator")
    if not wind_found:
        raise ConfigurationError("renewable_generators must include at least one wind generator")
    for index, thermal_generator in enumerate(portfolio.thermal_generators):
        _check_fraction_at(
            f"thermal_generators[{index}].availability_factor",
            thermal_generator.availability_factor,
        )
        _validate_thermal_config_at(thermal_generator.config, f"thermal_generators[{index}]")
    for index, unit in enumerate(portfolio.storage_units):
        _validate_storage_config_at(unit.config, f"storage_units[{index}]")
    for index, hydro in enumerate(portfolio.hydro_units):
        _validate_hydro_unit_at(hydro, f"hydro_units[{index}]")
    for index, demand in enumerate(portfolio.demand):
        _validate_demand_at(demand, f"demand[{index}]")
    for index, line in enumerate(portfolio.lines):
        if line.susceptance <= 0.0:
            raise ConfigurationError(f"lines[{index}].susceptance must be positive")
        if line.capacity_mw <= 0.0:
            raise ConfigurationError(f"lines[{index}].capacity_mw must be positive")
        _check_fraction_at(f"lines[{index}].availability_factor", line.availability_factor)
    for index, resource in enumerate(portfolio.imports):
        for name, value in (
            ("maximum_power_mw", resource.config.maximum_power_mw),
            ("price_eur_per_mwh", resource.config.price_eur_per_mwh),
            (
                "emission_factor_tonnes_per_mwh",
                resource.config.emission_factor_tonnes_per_mwh,
            ),
        ):
            _check_nonnegative_at(f"imports[{index}].{name}", value)


def _validate_fuel_at(fuel: FuelConfig, path: str) -> None:
    for name, value in (
        ("price_eur_per_mwh_thermal", fuel.price_eur_per_mwh_thermal),
        ("co2_factor_tonnes_per_mwh_thermal", fuel.co2_factor_tonnes_per_mwh_thermal),
        ("methane_factor_tonnes_per_mwh_thermal", fuel.methane_factor_tonnes_per_mwh_thermal),
        ("nox_factor_kg_per_mwh_thermal", fuel.nox_factor_kg_per_mwh_thermal),
        ("sox_factor_kg_per_mwh_thermal", fuel.sox_factor_kg_per_mwh_thermal),
    ):
        _check_nonnegative_at(f"{path}.{name}", value)
    if (
        fuel.lower_heating_value_mj_per_unit is not None
        and fuel.lower_heating_value_mj_per_unit <= 0.0
    ):
        raise ConfigurationError(f"{path}.lower_heating_value_mj_per_unit must be positive")


def _validate_solar_generator_at(generator: RenewableGeneratorConfig, path: str) -> None:
    _check_fraction_at(
        f"{path}.performance_ratio",
        _required_float(generator.performance_ratio, path),
    )
    reference = _required_float(generator.reference_irradiance_w_m2, path)
    if reference <= 0.0:
        raise ConfigurationError(f"{path}.reference_irradiance_w_m2 must be positive")
    _required_float(generator.temperature_coefficient_per_c, path)
    _required_float(generator.nominal_operating_cell_temperature_c, path)


def _validate_wind_generator_at(generator: RenewableGeneratorConfig, path: str) -> None:
    cut_in = _required_float(generator.cut_in_speed_m_s, path)
    rated = _required_float(generator.rated_speed_m_s, path)
    cut_out = _required_float(generator.cut_out_speed_m_s, path)
    if not 0.0 <= cut_in < rated < cut_out:
        raise ConfigurationError(f"{path} wind speeds must satisfy cut-in < rated < cut-out")


def _validate_thermal_config_at(th: ThermalConfig, path: str) -> None:
    if not 0.0 <= th.minimum_output_mw <= th.maximum_output_mw:
        raise ConfigurationError(f"{path}.minimum_output_mw and maximum_output_mw are inconsistent")
    for name, value in (
        ("ramp_up_mw_per_hour", th.ramp_up_mw_per_hour),
        ("ramp_down_mw_per_hour", th.ramp_down_mw_per_hour),
        ("startup_ramp_mw", th.startup_ramp_mw),
        ("shutdown_ramp_mw", th.shutdown_ramp_mw),
        ("variable_cost_eur_per_mwh", th.variable_cost_eur_per_mwh),
        ("no_load_cost_eur_per_hour", th.no_load_cost_eur_per_hour),
        ("startup_cost_eur", th.startup_cost_eur),
        ("shutdown_cost_eur", th.shutdown_cost_eur),
        ("emission_factor_tonnes_per_mwh", th.emission_factor_tonnes_per_mwh),
        ("minimum_fuel_input_mwh_per_hour", th.minimum_fuel_input_mwh_per_hour),
        ("initial_up_time_hours", th.initial_up_time_hours),
        ("initial_down_time_hours", th.initial_down_time_hours),
    ):
        _check_nonnegative_at(f"{path}.{name}", value)
    if th.heat_rate_segments:
        _validate_heat_rate_segments_at(th, path)
    if th.startup_categories:
        _validate_startup_categories_at(th, path)
    if th.minimum_up_hours <= 0.0:
        raise ConfigurationError(f"{path}.minimum_up_hours must be positive")
    if th.minimum_down_hours <= 0.0:
        raise ConfigurationError(f"{path}.minimum_down_hours must be positive")
    if th.minimum_output_mw > 0.0 and th.startup_ramp_mw < th.minimum_output_mw:
        raise ConfigurationError(f"{path}.startup_ramp_mw must be at least minimum_output_mw")
    if th.minimum_output_mw > 0.0 and th.shutdown_ramp_mw < th.minimum_output_mw:
        raise ConfigurationError(f"{path}.shutdown_ramp_mw must be at least minimum_output_mw")
    if th.terminal_commitment_mode not in {
        "forbid_incomplete_transitions",
        "carry_residual_obligations",
        "fixed_terminal_commitment",
    }:
        raise ConfigurationError(f"{path}.terminal_commitment_mode has an unsupported value")
    if th.terminal_commitment_mode == "fixed_terminal_commitment":
        if th.terminal_on is None:
            raise ConfigurationError(
                f"{path}.terminal_on is required when terminal_commitment_mode is fixed"
            )
    elif th.terminal_on is not None:
        raise ConfigurationError(
            f"{path}.terminal_on is only valid when terminal_commitment_mode is fixed"
        )
    if th.initial_on:
        if not th.minimum_output_mw <= th.initial_output_mw <= th.maximum_output_mw:
            raise ConfigurationError(f"{path}.initial_output_mw is outside operating bounds")
        if th.initial_down_time_hours != 0.0:
            raise ConfigurationError(f"{path}.initial_down_time_hours must be zero when online")
    elif th.initial_output_mw != 0.0:
        raise ConfigurationError(f"{path}.initial_output_mw must be zero when initially off")
    elif th.initial_up_time_hours != 0.0:
        raise ConfigurationError(f"{path}.initial_up_time_hours must be zero when initially off")


def _validate_heat_rate_segments_at(th: ThermalConfig, path: str) -> None:
    seen: set[str] = set()
    total_capacity = 0.0
    previous_heat_rate = -1.0
    for index, segment in enumerate(th.heat_rate_segments):
        segment_path = f"{path}.heat_rate_segments[{index}]"
        if segment.id in seen:
            raise ConfigurationError(f"Duplicate identifier at {segment_path}.id: {segment.id}")
        seen.add(segment.id)
        if segment.capacity_mw <= 0.0:
            raise ConfigurationError(f"{segment_path}.capacity_mw must be positive")
        if segment.heat_rate_mwh_thermal_per_mwh <= 0.0:
            raise ConfigurationError(
                f"{segment_path}.heat_rate_mwh_thermal_per_mwh must be positive"
            )
        if segment.heat_rate_mwh_thermal_per_mwh < previous_heat_rate:
            raise ConfigurationError(
                f"{segment_path}.heat_rate_mwh_thermal_per_mwh must be nondecreasing"
            )
        previous_heat_rate = segment.heat_rate_mwh_thermal_per_mwh
        total_capacity += segment.capacity_mw
    dispatchable_range = th.maximum_output_mw - th.minimum_output_mw
    if abs(total_capacity - dispatchable_range) > 1e-9:
        raise ConfigurationError(
            f"{path}.heat_rate_segments capacity must equal maximum_output_mw minus "
            "minimum_output_mw"
        )


def _validate_startup_categories_at(th: ThermalConfig, path: str) -> None:
    seen: set[str] = set()
    previous_threshold = -1.0
    for index, category in enumerate(th.startup_categories):
        category_path = f"{path}.startup_categories[{index}]"
        if category.id in seen:
            raise ConfigurationError(f"Duplicate identifier at {category_path}.id: {category.id}")
        seen.add(category.id)
        for name, value in (
            ("minimum_down_time_hours", category.minimum_down_time_hours),
            ("startup_cost_eur", category.startup_cost_eur),
            ("startup_fuel_input_mwh_thermal", category.startup_fuel_input_mwh_thermal),
        ):
            _check_nonnegative_at(f"{category_path}.{name}", value)
        if category.minimum_down_time_hours <= previous_threshold:
            raise ConfigurationError(
                f"{category_path}.minimum_down_time_hours must be strictly increasing"
            )
        previous_threshold = category.minimum_down_time_hours
    if th.startup_categories[0].minimum_down_time_hours != 0.0:
        raise ConfigurationError(f"{path}.startup_categories must start at 0 down hours")


def _validate_storage_config_at(bat: BatteryConfig, path: str) -> None:
    for name, value in (
        ("energy_capacity_mwh", bat.energy_capacity_mwh),
        ("power_capacity_mw", bat.power_capacity_mw),
        ("minimum_soc_mwh", bat.minimum_soc_mwh),
        ("maximum_soc_mwh", bat.maximum_soc_mwh),
        ("initial_soc_mwh", bat.initial_soc_mwh),
        ("self_discharge_rate_per_hour", bat.self_discharge_rate_per_hour),
        ("minimum_charge_mw", bat.minimum_charge_mw),
        ("minimum_discharge_mw", bat.minimum_discharge_mw),
        ("throughput_cost_eur_per_mwh", bat.throughput_cost_eur_per_mwh),
        ("minimum_final_soc_mwh", bat.minimum_final_soc_mwh),
    ):
        _check_nonnegative_at(f"{path}.{name}", value)
    for optional_name, optional_value in (
        ("charge_power_capacity_mw", bat.charge_power_capacity_mw),
        ("discharge_power_capacity_mw", bat.discharge_power_capacity_mw),
        ("charge_ramp_mw_per_hour", bat.charge_ramp_mw_per_hour),
        ("discharge_ramp_mw_per_hour", bat.discharge_ramp_mw_per_hour),
    ):
        if optional_value is not None:
            _check_nonnegative_at(f"{path}.{optional_name}", optional_value)
    _check_fraction_at(f"{path}.self_discharge_rate_per_hour", bat.self_discharge_rate_per_hour)
    _check_fraction_at(f"{path}.availability_factor", bat.availability_factor)
    _check_fraction_at(f"{path}.charge_efficiency", bat.charge_efficiency)
    _check_fraction_at(f"{path}.discharge_efficiency", bat.discharge_efficiency)
    if bat.charge_efficiency == 0.0 or bat.discharge_efficiency == 0.0:
        raise ConfigurationError(f"{path} battery efficiencies must be positive")
    charge_capacity = _storage_charge_capacity(bat)
    discharge_capacity = _storage_discharge_capacity(bat)
    if bat.minimum_charge_mw > charge_capacity:
        raise ConfigurationError(f"{path}.minimum_charge_mw exceeds charge power capacity")
    if bat.minimum_discharge_mw > discharge_capacity:
        raise ConfigurationError(f"{path}.minimum_discharge_mw exceeds discharge power capacity")
    if bat.maximum_soc_mwh > bat.energy_capacity_mwh:
        raise ConfigurationError(f"{path}.maximum_soc_mwh exceeds energy capacity")
    if not bat.minimum_soc_mwh <= bat.initial_soc_mwh <= bat.maximum_soc_mwh:
        raise ConfigurationError(f"{path}.initial_soc_mwh is outside bounds")
    if not bat.minimum_soc_mwh <= bat.minimum_final_soc_mwh <= bat.maximum_soc_mwh:
        raise ConfigurationError(f"{path}.minimum_final_soc_mwh is outside bounds")
    if bat.terminal_soc_mode not in {"minimum", "exact", "cyclic", "free"}:
        raise ConfigurationError(f"{path}.terminal_soc_mode has an unsupported value")
    if bat.technology not in {"battery", "pumped_storage"}:
        raise ConfigurationError(f"{path}.technology has an unsupported value")
    _validate_degradation_bands_at(bat, path)


def _storage_charge_capacity(bat: BatteryConfig) -> float:
    return (
        bat.charge_power_capacity_mw
        if bat.charge_power_capacity_mw is not None
        else bat.power_capacity_mw
    )


def _storage_discharge_capacity(bat: BatteryConfig) -> float:
    return (
        bat.discharge_power_capacity_mw
        if bat.discharge_power_capacity_mw is not None
        else bat.power_capacity_mw
    )


def _validate_degradation_bands_at(bat: BatteryConfig, path: str) -> None:
    seen: set[str] = set()
    previous_cost = -1.0
    for index, band in enumerate(bat.degradation_bands):
        band_path = f"{path}.degradation_bands[{index}]"
        if band.id in seen:
            raise ConfigurationError(f"Duplicate identifier at {band_path}.id: {band.id}")
        seen.add(band.id)
        if band.capacity_mwh <= 0.0:
            raise ConfigurationError(f"{band_path}.capacity_mwh must be positive")
        if band.cost_eur_per_mwh < 0.0:
            raise ConfigurationError(f"{band_path}.cost_eur_per_mwh must be non-negative")
        if band.cost_eur_per_mwh < previous_cost:
            raise ConfigurationError(f"{band_path}.cost_eur_per_mwh must be nondecreasing")
        previous_cost = band.cost_eur_per_mwh


def _validate_hydro_unit_at(hydro: HydroUnitConfig, path: str) -> None:
    for name, value in (
        ("turbine_capacity_mw", hydro.turbine_capacity_mw),
        ("minimum_reservoir_mwh", hydro.minimum_reservoir_mwh),
        ("maximum_reservoir_mwh", hydro.maximum_reservoir_mwh),
        ("initial_reservoir_mwh", hydro.initial_reservoir_mwh),
        ("minimum_final_reservoir_mwh", hydro.minimum_final_reservoir_mwh),
        ("minimum_release_mw", hydro.minimum_release_mw),
        ("evaporation_rate_per_hour", hydro.evaporation_rate_per_hour),
        ("water_value_eur_per_mwh", hydro.water_value_eur_per_mwh),
        ("cascade_delay_hours", hydro.cascade_delay_hours),
    ):
        _check_nonnegative_at(f"{path}.{name}", value)
    if hydro.spill_capacity_mw is not None:
        _check_nonnegative_at(f"{path}.spill_capacity_mw", hydro.spill_capacity_mw)
    _check_fraction_at(f"{path}.turbine_efficiency", hydro.turbine_efficiency)
    if hydro.turbine_efficiency == 0.0:
        raise ConfigurationError(f"{path}.turbine_efficiency must be positive")
    _check_fraction_at(f"{path}.evaporation_rate_per_hour", hydro.evaporation_rate_per_hour)
    if hydro.maximum_reservoir_mwh < hydro.minimum_reservoir_mwh:
        raise ConfigurationError(f"{path}.maximum_reservoir_mwh must be at least minimum")
    if not (
        hydro.minimum_reservoir_mwh <= hydro.initial_reservoir_mwh <= hydro.maximum_reservoir_mwh
    ):
        raise ConfigurationError(f"{path}.initial_reservoir_mwh is outside bounds")
    if not (
        hydro.minimum_reservoir_mwh
        <= hydro.minimum_final_reservoir_mwh
        <= hydro.maximum_reservoir_mwh
    ):
        raise ConfigurationError(f"{path}.minimum_final_reservoir_mwh is outside bounds")
    if hydro.terminal_reservoir_mode not in {"minimum", "exact", "cyclic", "free"}:
        raise ConfigurationError(f"{path}.terminal_reservoir_mode has an unsupported value")
    if hydro.kind == "run_of_river":
        for name, value in (
            ("minimum_reservoir_mwh", hydro.minimum_reservoir_mwh),
            ("maximum_reservoir_mwh", hydro.maximum_reservoir_mwh),
            ("initial_reservoir_mwh", hydro.initial_reservoir_mwh),
            ("minimum_final_reservoir_mwh", hydro.minimum_final_reservoir_mwh),
        ):
            if value != 0.0:
                raise ConfigurationError(f"{path}.{name} must be zero for run_of_river")


def _validate_demand_at(demand: DemandConfig, path: str) -> None:
    if demand.kind not in {"fixed", "curtailable", "shiftable", "deferrable", "ev_charging"}:
        raise ConfigurationError(f"{path}.kind has an unsupported value")
    for name, value in (
        ("voluntary_curtailment_cost_eur_per_mwh", demand.voluntary_curtailment_cost_eur_per_mwh),
        ("maximum_curtailment_fraction", demand.maximum_curtailment_fraction),
        ("shift_up_capacity_mw", demand.shift_up_capacity_mw),
        ("shift_down_capacity_mw", demand.shift_down_capacity_mw),
        ("shift_window_hours", demand.shift_window_hours),
        ("rebound_fraction", demand.rebound_fraction),
        ("shift_cost_eur_per_mwh", demand.shift_cost_eur_per_mwh),
        ("task_power_capacity_mw", demand.task_power_capacity_mw),
        ("task_required_energy_mwh", demand.task_required_energy_mwh),
        ("task_unserved_penalty_eur_per_mwh", demand.task_unserved_penalty_eur_per_mwh),
        ("heating_sensitivity_mw_per_c", demand.heating_sensitivity_mw_per_c),
        ("cooling_sensitivity_mw_per_c", demand.cooling_sensitivity_mw_per_c),
    ):
        _check_nonnegative_at(f"{path}.{name}", value)
    if demand.value_of_lost_load_eur_per_mwh is not None:
        _check_nonnegative_at(
            f"{path}.value_of_lost_load_eur_per_mwh",
            demand.value_of_lost_load_eur_per_mwh,
        )
    if demand.maximum_curtailment_mw is not None:
        _check_nonnegative_at(f"{path}.maximum_curtailment_mw", demand.maximum_curtailment_mw)
    _check_fraction_at(f"{path}.maximum_curtailment_fraction", demand.maximum_curtailment_fraction)
    _check_fraction_at(f"{path}.rebound_fraction", demand.rebound_fraction)
    if demand.task_start_period < 0:
        raise ConfigurationError(f"{path}.task_start_period must be non-negative")
    if demand.task_end_period is not None and demand.task_end_period <= demand.task_start_period:
        raise ConfigurationError(f"{path}.task_end_period must exceed task_start_period")
    if demand.kind == "fixed" and any(
        value > 0.0
        for value in (
            demand.maximum_curtailment_fraction,
            demand.shift_up_capacity_mw,
            demand.shift_down_capacity_mw,
            demand.task_power_capacity_mw,
            demand.task_required_energy_mwh,
        )
    ):
        raise ConfigurationError(f"{path} fixed demand cannot define flexibility quantities")
    if demand.kind == "curtailable" and (
        demand.maximum_curtailment_fraction == 0.0 and demand.maximum_curtailment_mw is None
    ):
        raise ConfigurationError(f"{path} curtailable demand requires a curtailment limit")
    if demand.kind == "shiftable" and (
        demand.shift_up_capacity_mw == 0.0 or demand.shift_down_capacity_mw == 0.0
    ):
        raise ConfigurationError(f"{path} shiftable demand requires up and down shift capacity")
    if demand.kind in {"deferrable", "ev_charging"} and (
        demand.task_power_capacity_mw == 0.0 or demand.task_required_energy_mwh == 0.0
    ):
        raise ConfigurationError(f"{path} task demand requires power capacity and required energy")
    if demand.temperature_time_series_key is None and (
        demand.heating_base_temperature_c is not None
        or demand.cooling_base_temperature_c is not None
        or demand.heating_sensitivity_mw_per_c > 0.0
        or demand.cooling_sensitivity_mw_per_c > 0.0
    ):
        raise ConfigurationError(f"{path}.temperature_time_series_key is required")


def _check_nonnegative_at(name: str, value: float) -> None:
    if value < 0.0:
        raise ConfigurationError(f"{name} must be non-negative")


def _check_fraction_at(name: str, value: float, *, allow_one: bool = True) -> None:
    upper_valid = value <= 1.0 if allow_one else value < 1.0
    if value < 0.0 or not upper_valid:
        operator = "[0, 1]" if allow_one else "[0, 1)"
        raise ConfigurationError(f"{name} must be in {operator}")


def _required_float(value: float | None, path: str) -> float:
    if value is None:
        raise ConfigurationError(f"{path} is missing a required numeric renewable parameter")
    return value


def _primary_solar(generators: tuple[RenewableGeneratorConfig, ...]) -> SolarConfig:
    for generator in generators:
        if generator.kind == "solar":
            return SolarConfig(
                capacity_mw=generator.capacity_mw,
                performance_ratio=_required_float(generator.performance_ratio, generator.id),
                reference_irradiance_w_m2=_required_float(
                    generator.reference_irradiance_w_m2,
                    generator.id,
                ),
                temperature_coefficient_per_c=_required_float(
                    generator.temperature_coefficient_per_c,
                    generator.id,
                ),
                nominal_operating_cell_temperature_c=_required_float(
                    generator.nominal_operating_cell_temperature_c,
                    generator.id,
                ),
            )
    raise ConfigurationError("renewable_generators must include at least one solar generator")


def _primary_wind(generators: tuple[RenewableGeneratorConfig, ...]) -> WindConfig:
    for generator in generators:
        if generator.kind == "wind":
            return WindConfig(
                capacity_mw=generator.capacity_mw,
                cut_in_speed_m_s=_required_float(generator.cut_in_speed_m_s, generator.id),
                rated_speed_m_s=_required_float(generator.rated_speed_m_s, generator.id),
                cut_out_speed_m_s=_required_float(generator.cut_out_speed_m_s, generator.id),
            )
    raise ConfigurationError("renewable_generators must include at least one wind generator")


def migrate_legacy_config(path: str | Path) -> dict[str, Any]:
    """Convert a legacy single-system configuration into schema version 2."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    raw = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a mapping")
    schema_version = _optional_number(raw, "schema_version", int, LEGACY_SCHEMA_VERSION)
    if schema_version != LEGACY_SCHEMA_VERSION:
        raise ConfigurationError("migrate-config only accepts legacy schema_version 1 files")
    config = _load_legacy_config(config_path, raw, schema_version)
    paths_raw = _section(raw, "paths")
    return _schema_v2_mapping_from_config(
        config,
        input_csv=_string(paths_raw, "input_csv"),
        output_directory=_string(paths_raw, "output_directory"),
    )


def resolved_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    """Return a deterministic JSON-ready representation of a resolved configuration."""
    return cast(dict[str, Any], _json_ready(asdict(config)))


def migrate_config_file(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for legacy configuration migration."""
    return migrate_legacy_config(path)


def resolved_config_dict(config: ModelConfig) -> dict[str, Any]:
    """Backward-compatible alias for resolved configuration serialization."""
    return resolved_config_to_dict(config)


def resolved_config_yaml(config: ModelConfig) -> str:
    """Return canonical YAML for a resolved configuration."""
    return cast(str, yaml.safe_dump(resolved_config_to_dict(config), sort_keys=True))


def _schema_v2_mapping_from_config(
    config: ModelConfig,
    *,
    input_csv: str,
    output_directory: str,
) -> dict[str, Any]:
    portfolio = config.portfolio
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "scenario": {"id": portfolio.scenario.id},
        "simulation": {"time_step_hours": config.simulation.time_step_hours},
        "solver": {
            "solver_time_limit_seconds": config.solver.solver_time_limit_seconds,
            "mip_relative_gap": config.solver.mip_relative_gap,
            "allow_non_optimal_solution": config.solver.allow_non_optimal_solution,
        },
        "zones": [{"id": zone.id} for zone in portfolio.zones],
        "fuels": [_fuel_mapping(fuel) for fuel in portfolio.fuels],
        "buses": [{"id": bus.id, "zone_id": bus.zone_id} for bus in portfolio.buses],
        "lines": [_line_mapping(line) for line in portfolio.lines],
        "aggregate_network": _network_mapping(config.network),
        "renewable_generators": [
            _renewable_generator_mapping(generator) for generator in portfolio.renewable_generators
        ],
        "thermal_generators": [
            _thermal_generator_mapping(generator) for generator in portfolio.thermal_generators
        ],
        "storage_units": [_storage_unit_mapping(unit) for unit in portfolio.storage_units],
        "hydro_units": [_hydro_unit_mapping(unit) for unit in portfolio.hydro_units],
        "imports": [
            {
                "id": resource.id,
                "bus_id": resource.bus_id,
                "maximum_power_mw": resource.config.maximum_power_mw,
                "price_eur_per_mwh": resource.config.price_eur_per_mwh,
                "emission_factor_tonnes_per_mwh": resource.config.emission_factor_tonnes_per_mwh,
            }
            for resource in portfolio.imports
        ],
        "demand": [_demand_mapping(demand) for demand in portfolio.demand],
        "penalties": {
            "renewable_curtailment_eur_per_mwh": (
                config.penalties.renewable_curtailment_eur_per_mwh
            ),
            "lost_load_eur_per_mwh": config.penalties.lost_load_eur_per_mwh,
            "carbon_price_eur_per_tonne": config.penalties.carbon_price_eur_per_tonne,
        },
        "paths": {"input_csv": input_csv, "output_directory": output_directory},
    }


def _line_mapping(line: TransmissionLineConfig) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": line.id,
        "from_bus_id": line.from_bus_id,
        "to_bus_id": line.to_bus_id,
        "susceptance": line.susceptance,
        "capacity_mw": line.capacity_mw,
    }
    if line.availability_factor != 1.0:
        item["availability_factor"] = line.availability_factor
    if line.availability_factor_key is not None:
        item["availability_factor_key"] = line.availability_factor_key
    return item


def _network_mapping(network: NetworkConfig) -> dict[str, Any]:
    item: dict[str, Any] = {
        "loss_fraction": network.loss_fraction,
        "transfer_capacity_mw": network.transfer_capacity_mw,
        "network_mode": network.network_mode,
    }
    if network.slack_bus_id is not None:
        item["slack_bus_id"] = network.slack_bus_id
    return item


def _fuel_mapping(fuel: FuelConfig) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": fuel.id,
        "price_eur_per_mwh_thermal": fuel.price_eur_per_mwh_thermal,
        "co2_factor_tonnes_per_mwh_thermal": fuel.co2_factor_tonnes_per_mwh_thermal,
    }
    if fuel.price_time_series_key is not None:
        item["price_time_series_key"] = fuel.price_time_series_key
    if fuel.lower_heating_value_mj_per_unit is not None:
        item["lower_heating_value_mj_per_unit"] = fuel.lower_heating_value_mj_per_unit
    if fuel.methane_factor_tonnes_per_mwh_thermal != 0.0:
        item["methane_factor_tonnes_per_mwh_thermal"] = fuel.methane_factor_tonnes_per_mwh_thermal
    if fuel.nox_factor_kg_per_mwh_thermal != 0.0:
        item["nox_factor_kg_per_mwh_thermal"] = fuel.nox_factor_kg_per_mwh_thermal
    if fuel.sox_factor_kg_per_mwh_thermal != 0.0:
        item["sox_factor_kg_per_mwh_thermal"] = fuel.sox_factor_kg_per_mwh_thermal
    return item


def _renewable_generator_mapping(generator: RenewableGeneratorConfig) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": generator.id,
        "kind": generator.kind,
        "bus_id": generator.bus_id,
        "capacity_mw": generator.capacity_mw,
        "time_series_key": generator.time_series_key,
    }
    if generator.kind == "solar":
        item.update(
            {
                "ambient_temperature_key": generator.ambient_temperature_key,
                "performance_ratio": generator.performance_ratio,
                "reference_irradiance_w_m2": generator.reference_irradiance_w_m2,
                "temperature_coefficient_per_c": generator.temperature_coefficient_per_c,
                "nominal_operating_cell_temperature_c": (
                    generator.nominal_operating_cell_temperature_c
                ),
            }
        )
    else:
        item.update(
            {
                "cut_in_speed_m_s": generator.cut_in_speed_m_s,
                "rated_speed_m_s": generator.rated_speed_m_s,
                "cut_out_speed_m_s": generator.cut_out_speed_m_s,
            }
        )
    return item


def _thermal_generator_mapping(generator: ThermalGeneratorConfig) -> dict[str, Any]:
    thermal = generator.config
    item: dict[str, Any] = {
        "id": generator.id,
        "bus_id": generator.bus_id,
        "fuel_id": generator.fuel_id,
        "name": thermal.name,
        "minimum_output_mw": thermal.minimum_output_mw,
        "maximum_output_mw": thermal.maximum_output_mw,
        "ramp_up_mw_per_hour": thermal.ramp_up_mw_per_hour,
        "ramp_down_mw_per_hour": thermal.ramp_down_mw_per_hour,
        "startup_ramp_mw": thermal.startup_ramp_mw,
        "shutdown_ramp_mw": thermal.shutdown_ramp_mw,
        "variable_cost_eur_per_mwh": thermal.variable_cost_eur_per_mwh,
        "no_load_cost_eur_per_hour": thermal.no_load_cost_eur_per_hour,
        "startup_cost_eur": thermal.startup_cost_eur,
        "shutdown_cost_eur": thermal.shutdown_cost_eur,
        "emission_factor_tonnes_per_mwh": thermal.emission_factor_tonnes_per_mwh,
        "minimum_up_hours": thermal.minimum_up_hours,
        "minimum_down_hours": thermal.minimum_down_hours,
        "initial_on": thermal.initial_on,
        "initial_output_mw": thermal.initial_output_mw,
        "initial_up_time_hours": thermal.initial_up_time_hours,
        "initial_down_time_hours": thermal.initial_down_time_hours,
        "terminal_commitment_mode": thermal.terminal_commitment_mode,
    }
    if thermal.minimum_fuel_input_mwh_per_hour != 0.0:
        item["minimum_fuel_input_mwh_per_hour"] = thermal.minimum_fuel_input_mwh_per_hour
    if thermal.heat_rate_segments:
        item["heat_rate_segments"] = [
            {
                "id": segment.id,
                "capacity_mw": segment.capacity_mw,
                "heat_rate_mwh_thermal_per_mwh": segment.heat_rate_mwh_thermal_per_mwh,
            }
            for segment in thermal.heat_rate_segments
        ]
    if thermal.startup_categories:
        item["startup_categories"] = [
            {
                "id": category.id,
                "minimum_down_time_hours": category.minimum_down_time_hours,
                "startup_cost_eur": category.startup_cost_eur,
                "startup_fuel_input_mwh_thermal": category.startup_fuel_input_mwh_thermal,
            }
            for category in thermal.startup_categories
        ]
    if thermal.terminal_on is not None:
        item["terminal_on"] = thermal.terminal_on
    if generator.must_run:
        item["must_run"] = generator.must_run
    if generator.availability_factor != 1.0:
        item["availability_factor"] = generator.availability_factor
    if generator.availability_factor_key is not None:
        item["availability_factor_key"] = generator.availability_factor_key
    return item


def _storage_unit_mapping(unit: StorageUnitConfig) -> dict[str, Any]:
    battery = unit.config
    item: dict[str, Any] = {
        "id": unit.id,
        "bus_id": unit.bus_id,
        "technology": battery.technology,
        "energy_capacity_mwh": battery.energy_capacity_mwh,
        "power_capacity_mw": battery.power_capacity_mw,
        "minimum_soc_mwh": battery.minimum_soc_mwh,
        "maximum_soc_mwh": battery.maximum_soc_mwh,
        "initial_soc_mwh": battery.initial_soc_mwh,
        "charge_efficiency": battery.charge_efficiency,
        "discharge_efficiency": battery.discharge_efficiency,
        "self_discharge_rate_per_hour": battery.self_discharge_rate_per_hour,
        "minimum_charge_mw": battery.minimum_charge_mw,
        "minimum_discharge_mw": battery.minimum_discharge_mw,
        "throughput_cost_eur_per_mwh": battery.throughput_cost_eur_per_mwh,
        "minimum_final_soc_mwh": battery.minimum_final_soc_mwh,
        "terminal_soc_mode": battery.terminal_soc_mode,
    }
    if battery.charge_power_capacity_mw is not None:
        item["charge_power_capacity_mw"] = battery.charge_power_capacity_mw
    if battery.discharge_power_capacity_mw is not None:
        item["discharge_power_capacity_mw"] = battery.discharge_power_capacity_mw
    if battery.charge_ramp_mw_per_hour is not None:
        item["charge_ramp_mw_per_hour"] = battery.charge_ramp_mw_per_hour
    if battery.discharge_ramp_mw_per_hour is not None:
        item["discharge_ramp_mw_per_hour"] = battery.discharge_ramp_mw_per_hour
    if battery.availability_factor != 1.0:
        item["availability_factor"] = battery.availability_factor
    if battery.availability_factor_key is not None:
        item["availability_factor_key"] = battery.availability_factor_key
    if battery.degradation_bands:
        item["degradation_bands"] = [
            {
                "id": band.id,
                "capacity_mwh": band.capacity_mwh,
                "cost_eur_per_mwh": band.cost_eur_per_mwh,
            }
            for band in battery.degradation_bands
        ]
    return item


def _hydro_unit_mapping(unit: HydroUnitConfig) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": unit.id,
        "bus_id": unit.bus_id,
        "kind": unit.kind,
        "inflow_time_series_key": unit.inflow_time_series_key,
        "turbine_capacity_mw": unit.turbine_capacity_mw,
        "turbine_efficiency": unit.turbine_efficiency,
        "minimum_reservoir_mwh": unit.minimum_reservoir_mwh,
        "maximum_reservoir_mwh": unit.maximum_reservoir_mwh,
        "initial_reservoir_mwh": unit.initial_reservoir_mwh,
        "minimum_final_reservoir_mwh": unit.minimum_final_reservoir_mwh,
        "terminal_reservoir_mode": unit.terminal_reservoir_mode,
        "minimum_release_mw": unit.minimum_release_mw,
        "evaporation_rate_per_hour": unit.evaporation_rate_per_hour,
        "water_value_eur_per_mwh": unit.water_value_eur_per_mwh,
    }
    if unit.spill_capacity_mw is not None:
        item["spill_capacity_mw"] = unit.spill_capacity_mw
    if unit.upstream_hydro_id is not None:
        item["upstream_hydro_id"] = unit.upstream_hydro_id
        item["cascade_delay_hours"] = unit.cascade_delay_hours
    return item


def _demand_mapping(demand: DemandConfig) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": demand.id,
        "bus_id": demand.bus_id,
        "time_series_key": demand.time_series_key,
    }
    defaults = DemandConfig(
        id=demand.id, bus_id=demand.bus_id, time_series_key=demand.time_series_key
    )
    for key, value in asdict(demand).items():
        if key in {"id", "bus_id", "time_series_key"}:
            continue
        if value != getattr(defaults, key):
            item[key] = value
    return item


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def validate_config(config: ModelConfig) -> None:
    """Validate cross-field constraints in a model configuration."""
    if config.simulation.time_step_hours <= 0.0:
        raise ConfigurationError("simulation.time_step_hours must be positive")
    if config.simulation.solver_time_limit_seconds <= 0.0:
        raise ConfigurationError("solver_time_limit_seconds must be positive")
    _check_fraction("mip_relative_gap", config.simulation.mip_relative_gap)

    _check_nonnegative("solar.capacity_mw", config.solar.capacity_mw)
    _check_fraction("solar.performance_ratio", config.solar.performance_ratio)
    if config.solar.reference_irradiance_w_m2 <= 0.0:
        raise ConfigurationError("reference_irradiance_w_m2 must be positive")

    _check_nonnegative("wind.capacity_mw", config.wind.capacity_mw)
    if not (
        0.0
        <= config.wind.cut_in_speed_m_s
        < config.wind.rated_speed_m_s
        < config.wind.cut_out_speed_m_s
    ):
        raise ConfigurationError("Wind speeds must satisfy cut-in < rated < cut-out")

    th = config.thermal
    if not 0.0 <= th.minimum_output_mw <= th.maximum_output_mw:
        raise ConfigurationError("Thermal output bounds are inconsistent")
    for name, value in (
        ("ramp_up_mw_per_hour", th.ramp_up_mw_per_hour),
        ("ramp_down_mw_per_hour", th.ramp_down_mw_per_hour),
        ("startup_ramp_mw", th.startup_ramp_mw),
        ("shutdown_ramp_mw", th.shutdown_ramp_mw),
        ("variable_cost_eur_per_mwh", th.variable_cost_eur_per_mwh),
        ("no_load_cost_eur_per_hour", th.no_load_cost_eur_per_hour),
        ("startup_cost_eur", th.startup_cost_eur),
        ("shutdown_cost_eur", th.shutdown_cost_eur),
        ("emission_factor_tonnes_per_mwh", th.emission_factor_tonnes_per_mwh),
    ):
        _check_nonnegative(f"thermal.{name}", value)
    if th.minimum_up_hours <= 0.0 or th.minimum_down_hours <= 0.0:
        raise ConfigurationError("Minimum up/down times must be positive")
    if th.minimum_output_mw > 0.0 and th.startup_ramp_mw < th.minimum_output_mw:
        raise ConfigurationError(
            "thermal.startup_ramp_mw must be at least thermal.minimum_output_mw "
            "because startup_ramp_mw is the maximum output allowed in a startup period"
        )
    if th.minimum_output_mw > 0.0 and th.shutdown_ramp_mw < th.minimum_output_mw:
        raise ConfigurationError(
            "thermal.shutdown_ramp_mw must be at least thermal.minimum_output_mw "
            "because shutdown_ramp_mw is the maximum previous-period output allowed "
            "for a shutdown"
        )
    if th.terminal_commitment_mode not in {
        "forbid_incomplete_transitions",
        "carry_residual_obligations",
        "fixed_terminal_commitment",
    }:
        raise ConfigurationError(
            "thermal.terminal_commitment_mode must be one of "
            "forbid_incomplete_transitions, carry_residual_obligations, "
            "fixed_terminal_commitment"
        )
    if th.terminal_commitment_mode == "fixed_terminal_commitment":
        if th.terminal_on is None:
            raise ConfigurationError(
                "thermal.terminal_on is required when terminal_commitment_mode is "
                "fixed_terminal_commitment"
            )
    elif th.terminal_on is not None:
        raise ConfigurationError(
            "thermal.terminal_on is only valid when terminal_commitment_mode is "
            "fixed_terminal_commitment"
        )
    if th.initial_on:
        if not th.minimum_output_mw <= th.initial_output_mw <= th.maximum_output_mw:
            raise ConfigurationError("Initial thermal output is outside operating bounds")
        if th.initial_down_time_hours != 0.0:
            raise ConfigurationError(
                "initial_down_time_hours must be zero when thermal.initial_on is true"
            )
    elif th.initial_output_mw != 0.0:
        raise ConfigurationError("Initial thermal output must be zero when initially off")
    elif th.initial_up_time_hours != 0.0:
        raise ConfigurationError(
            "initial_up_time_hours must be zero when thermal.initial_on is false"
        )
    _check_nonnegative("thermal.initial_up_time_hours", th.initial_up_time_hours)
    _check_nonnegative("thermal.initial_down_time_hours", th.initial_down_time_hours)

    bat = config.battery
    for name, value in (
        ("energy_capacity_mwh", bat.energy_capacity_mwh),
        ("power_capacity_mw", bat.power_capacity_mw),
        ("minimum_soc_mwh", bat.minimum_soc_mwh),
        ("maximum_soc_mwh", bat.maximum_soc_mwh),
        ("initial_soc_mwh", bat.initial_soc_mwh),
        ("self_discharge_rate_per_hour", bat.self_discharge_rate_per_hour),
        ("minimum_charge_mw", bat.minimum_charge_mw),
        ("minimum_discharge_mw", bat.minimum_discharge_mw),
        ("throughput_cost_eur_per_mwh", bat.throughput_cost_eur_per_mwh),
        ("minimum_final_soc_mwh", bat.minimum_final_soc_mwh),
    ):
        _check_nonnegative(f"battery.{name}", value)
    for optional_name, optional_value in (
        ("charge_power_capacity_mw", bat.charge_power_capacity_mw),
        ("discharge_power_capacity_mw", bat.discharge_power_capacity_mw),
        ("charge_ramp_mw_per_hour", bat.charge_ramp_mw_per_hour),
        ("discharge_ramp_mw_per_hour", bat.discharge_ramp_mw_per_hour),
    ):
        if optional_value is not None:
            _check_nonnegative(f"battery.{optional_name}", optional_value)
    _check_fraction("battery.charge_efficiency", bat.charge_efficiency)
    _check_fraction("battery.discharge_efficiency", bat.discharge_efficiency)
    _check_fraction("battery.self_discharge_rate_per_hour", bat.self_discharge_rate_per_hour)
    _check_fraction("battery.availability_factor", bat.availability_factor)
    if bat.charge_efficiency == 0.0 or bat.discharge_efficiency == 0.0:
        raise ConfigurationError("Battery efficiencies must be positive")
    if bat.minimum_charge_mw > _storage_charge_capacity(bat):
        raise ConfigurationError("battery.minimum_charge_mw exceeds charge power capacity")
    if bat.minimum_discharge_mw > _storage_discharge_capacity(bat):
        raise ConfigurationError("battery.minimum_discharge_mw exceeds discharge power capacity")
    if bat.maximum_soc_mwh > bat.energy_capacity_mwh:
        raise ConfigurationError("maximum_soc_mwh exceeds energy capacity")
    if not bat.minimum_soc_mwh <= bat.initial_soc_mwh <= bat.maximum_soc_mwh:
        raise ConfigurationError("Initial battery SOC is outside bounds")
    if not bat.minimum_soc_mwh <= bat.minimum_final_soc_mwh <= bat.maximum_soc_mwh:
        raise ConfigurationError("Minimum final SOC is outside bounds")
    if bat.terminal_soc_mode not in {"minimum", "exact", "cyclic", "free"}:
        raise ConfigurationError(
            "battery.terminal_soc_mode must be one of minimum, exact, cyclic, free"
        )
    if bat.technology not in {"battery", "pumped_storage"}:
        raise ConfigurationError("battery.technology must be one of battery, pumped_storage")
    _validate_degradation_bands_at(bat, "battery")

    _check_fraction("network.loss_fraction", config.network.loss_fraction, allow_one=False)
    if config.network.transfer_capacity_mw <= 0.0:
        raise ConfigurationError("network.transfer_capacity_mw must be positive")
    if config.network.network_mode not in {"aggregate", "nodal"}:
        raise ConfigurationError("network.network_mode must be one of aggregate, nodal")
    if config.network.slack_bus_id is not None and config.network.slack_bus_id not in {
        bus.id for bus in config.portfolio.buses
    }:
        raise ConfigurationError("network.slack_bus_id references unknown bus")
    if config.network.network_mode == "nodal":
        _validate_nodal_network(config)

    for name, value in (
        ("imports.maximum_power_mw", config.imports.maximum_power_mw),
        ("imports.price_eur_per_mwh", config.imports.price_eur_per_mwh),
        (
            "imports.emission_factor_tonnes_per_mwh",
            config.imports.emission_factor_tonnes_per_mwh,
        ),
        (
            "penalties.renewable_curtailment_eur_per_mwh",
            config.penalties.renewable_curtailment_eur_per_mwh,
        ),
        ("penalties.lost_load_eur_per_mwh", config.penalties.lost_load_eur_per_mwh),
        (
            "penalties.carbon_price_eur_per_tonne",
            config.penalties.carbon_price_eur_per_tonne,
        ),
    ):
        _check_nonnegative(name, value)


def _validate_nodal_network(config: ModelConfig) -> None:
    bus_ids = [bus.id for bus in config.portfolio.buses]
    if len(bus_ids) <= 1:
        return
    if not config.portfolio.lines:
        raise ConfigurationError("nodal network mode requires lines for multiple buses")
    adjacency = {bus_id: set[str]() for bus_id in bus_ids}
    for line in config.portfolio.lines:
        adjacency[line.from_bus_id].add(line.to_bus_id)
        adjacency[line.to_bus_id].add(line.from_bus_id)
    visited: set[str] = set()
    stack = [bus_ids[0]]
    while stack:
        bus_id = stack.pop()
        if bus_id in visited:
            continue
        visited.add(bus_id)
        stack.extend(adjacency[bus_id] - visited)
    if visited != set(bus_ids):
        missing = ", ".join(sorted(set(bus_ids) - visited))
        raise ConfigurationError(f"nodal network is disconnected; unreachable buses: {missing}")
