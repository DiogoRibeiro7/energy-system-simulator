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
        "minimum_soc_mwh",
        "maximum_soc_mwh",
        "initial_soc_mwh",
        "charge_efficiency",
        "discharge_efficiency",
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
    "battery": {"terminal_soc_mode"},
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
REQUIRED_ROOT_KEYS = ROOT_KEYS - {"hydro_units"}
SECTION_KEYS = {
    "scenario": {"id"},
    "simulation": {"time_step_hours"},
    "solver": {
        "solver_time_limit_seconds",
        "mip_relative_gap",
        "allow_non_optimal_solution",
    },
    "aggregate_network": {"loss_fraction", "transfer_capacity_mw"},
    "penalties": LEGACY_SECTION_KEYS["penalties"],
    "paths": LEGACY_SECTION_KEYS["paths"],
}
LIST_SECTION_KEYS = {
    "zones": {"id"},
    "buses": {"id", "zone_id"},
    "lines": {"id", "from_bus_id", "to_bus_id", "susceptance", "capacity_mw"},
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
    | {"id", "bus_id", "fuel_id", "must_run", "availability_factor", "availability_factor_key"},
    "storage_units": LEGACY_SECTION_KEYS["battery"] | {"id", "bus_id"},
    "hydro_units": {"id", "bus_id"},
    "imports": LEGACY_SECTION_KEYS["imports"] | {"id", "bus_id"},
    "demand": {"id", "bus_id", "time_series_key"},
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
    | {"must_run", "availability_factor", "availability_factor_key"},
    "storage_units": OPTIONAL_SECTION_KEYS["battery"],
    "hydro_units": set(),
    "imports": set(),
    "demand": set(),
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


@dataclass(frozen=True)
class StorageUnitConfig:
    id: str
    bus_id: str
    config: BatteryConfig


@dataclass(frozen=True)
class HydroUnitConfig:
    id: str
    bus_id: str


@dataclass(frozen=True)
class NetworkConfig:
    loss_fraction: float
    transfer_capacity_mw: float


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
        _validate_required_keys(section, section_name, allowed_keys)
    for section_name, allowed_keys in LIST_SECTION_KEYS.items():
        items = _list_section(raw, section_name, required=section_name != "hydro_units")
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
        )
        for index, item in enumerate(_list_section(raw, "lines"))
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
        HydroUnitConfig(
            id=_id_at(item, "id", f"hydro_units[{index}]"),
            bus_id=_id_at(item, "bus_id", f"hydro_units[{index}]"),
        )
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
        DemandConfig(
            id=_id_at(item, "id", f"demand[{index}]"),
            bus_id=_id_at(item, "bus_id", f"demand[{index}]"),
            time_series_key=_input_key_at(item, "time_series_key", f"demand[{index}]"),
        )
        for index, item in enumerate(_list_section(raw, "demand"))
    )
    portfolio = PortfolioConfig(
        scenario=ScenarioConfig(id=_id_at(scenario_raw, "id", "scenario")),
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
    _validate_portfolio(portfolio)

    solar = _primary_solar(renewable_generators)
    wind = _primary_wind(renewable_generators)
    thermal = thermal_generators[0].config
    battery = storage_units[0].config
    network = NetworkConfig(
        loss_fraction=_number_at(network_raw, "loss_fraction", float, "aggregate_network"),
        transfer_capacity_mw=_number_at(
            network_raw, "transfer_capacity_mw", float, "aggregate_network"
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
        ),
        must_run=_optional_boolean_at(item, "must_run", False, path),
        availability_factor=_optional_number_at(item, "availability_factor", float, 1.0, path),
        availability_factor_key=(
            _string_at(item, "availability_factor_key", path)
            if "availability_factor_key" in item
            else None
        ),
    )


def _parse_storage_unit(item: Mapping[str, Any], path: str) -> StorageUnitConfig:
    return StorageUnitConfig(
        id=_id_at(item, "id", path),
        bus_id=_id_at(item, "bus_id", path),
        config=BatteryConfig(
            energy_capacity_mwh=_number_at(item, "energy_capacity_mwh", float, path),
            power_capacity_mw=_number_at(item, "power_capacity_mw", float, path),
            minimum_soc_mwh=_number_at(item, "minimum_soc_mwh", float, path),
            maximum_soc_mwh=_number_at(item, "maximum_soc_mwh", float, path),
            initial_soc_mwh=_number_at(item, "initial_soc_mwh", float, path),
            charge_efficiency=_number_at(item, "charge_efficiency", float, path),
            discharge_efficiency=_number_at(item, "discharge_efficiency", float, path),
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


def _validate_portfolio(portfolio: PortfolioConfig) -> None:
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
    if not portfolio.storage_units:
        raise ConfigurationError("storage_units must include at least one item")
    if not portfolio.imports:
        raise ConfigurationError("imports must include at least one item")
    if not portfolio.demand:
        raise ConfigurationError("demand must include at least one item")
    zone_ids = {zone.id for zone in portfolio.zones}
    bus_ids = {bus.id for bus in portfolio.buses}
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


def _validate_unique_ids(items: tuple[Any, ...], path: str) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        identifier = cast(str, item.id)
        if identifier in seen:
            raise ConfigurationError(f"Duplicate identifier at {path}[{index}].id: {identifier}")
        seen.add(identifier)


def _validate_schema_v2_assets(portfolio: PortfolioConfig) -> None:
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
    for index, line in enumerate(portfolio.lines):
        _check_nonnegative_at(f"lines[{index}].capacity_mw", line.capacity_mw)
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
        ("initial_up_time_hours", th.initial_up_time_hours),
        ("initial_down_time_hours", th.initial_down_time_hours),
    ):
        _check_nonnegative_at(f"{path}.{name}", value)
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


def _validate_storage_config_at(bat: BatteryConfig, path: str) -> None:
    for name, value in (
        ("energy_capacity_mwh", bat.energy_capacity_mwh),
        ("power_capacity_mw", bat.power_capacity_mw),
        ("minimum_soc_mwh", bat.minimum_soc_mwh),
        ("maximum_soc_mwh", bat.maximum_soc_mwh),
        ("initial_soc_mwh", bat.initial_soc_mwh),
        ("throughput_cost_eur_per_mwh", bat.throughput_cost_eur_per_mwh),
        ("minimum_final_soc_mwh", bat.minimum_final_soc_mwh),
    ):
        _check_nonnegative_at(f"{path}.{name}", value)
    _check_fraction_at(f"{path}.charge_efficiency", bat.charge_efficiency)
    _check_fraction_at(f"{path}.discharge_efficiency", bat.discharge_efficiency)
    if bat.charge_efficiency == 0.0 or bat.discharge_efficiency == 0.0:
        raise ConfigurationError(f"{path} battery efficiencies must be positive")
    if bat.maximum_soc_mwh > bat.energy_capacity_mwh:
        raise ConfigurationError(f"{path}.maximum_soc_mwh exceeds energy capacity")
    if not bat.minimum_soc_mwh <= bat.initial_soc_mwh <= bat.maximum_soc_mwh:
        raise ConfigurationError(f"{path}.initial_soc_mwh is outside bounds")
    if not bat.minimum_soc_mwh <= bat.minimum_final_soc_mwh <= bat.maximum_soc_mwh:
        raise ConfigurationError(f"{path}.minimum_final_soc_mwh is outside bounds")
    if bat.terminal_soc_mode not in {"minimum", "exact", "cyclic", "free"}:
        raise ConfigurationError(f"{path}.terminal_soc_mode has an unsupported value")


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
        "buses": [{"id": bus.id, "zone_id": bus.zone_id} for bus in portfolio.buses],
        "lines": [
            {
                "id": line.id,
                "from_bus_id": line.from_bus_id,
                "to_bus_id": line.to_bus_id,
                "susceptance": line.susceptance,
                "capacity_mw": line.capacity_mw,
            }
            for line in portfolio.lines
        ],
        "aggregate_network": {
            "loss_fraction": config.network.loss_fraction,
            "transfer_capacity_mw": config.network.transfer_capacity_mw,
        },
        "renewable_generators": [
            _renewable_generator_mapping(generator) for generator in portfolio.renewable_generators
        ],
        "thermal_generators": [
            _thermal_generator_mapping(generator) for generator in portfolio.thermal_generators
        ],
        "storage_units": [_storage_unit_mapping(unit) for unit in portfolio.storage_units],
        "hydro_units": [{"id": unit.id, "bus_id": unit.bus_id} for unit in portfolio.hydro_units],
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
        "demand": [
            {
                "id": demand.id,
                "bus_id": demand.bus_id,
                "time_series_key": demand.time_series_key,
            }
            for demand in portfolio.demand
        ],
        "penalties": {
            "renewable_curtailment_eur_per_mwh": (
                config.penalties.renewable_curtailment_eur_per_mwh
            ),
            "lost_load_eur_per_mwh": config.penalties.lost_load_eur_per_mwh,
            "carbon_price_eur_per_tonne": config.penalties.carbon_price_eur_per_tonne,
        },
        "paths": {"input_csv": input_csv, "output_directory": output_directory},
    }


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
    return {
        "id": unit.id,
        "bus_id": unit.bus_id,
        "energy_capacity_mwh": battery.energy_capacity_mwh,
        "power_capacity_mw": battery.power_capacity_mw,
        "minimum_soc_mwh": battery.minimum_soc_mwh,
        "maximum_soc_mwh": battery.maximum_soc_mwh,
        "initial_soc_mwh": battery.initial_soc_mwh,
        "charge_efficiency": battery.charge_efficiency,
        "discharge_efficiency": battery.discharge_efficiency,
        "throughput_cost_eur_per_mwh": battery.throughput_cost_eur_per_mwh,
        "minimum_final_soc_mwh": battery.minimum_final_soc_mwh,
        "terminal_soc_mode": battery.terminal_soc_mode,
    }


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
        ("throughput_cost_eur_per_mwh", bat.throughput_cost_eur_per_mwh),
        ("minimum_final_soc_mwh", bat.minimum_final_soc_mwh),
    ):
        _check_nonnegative(f"battery.{name}", value)
    _check_fraction("battery.charge_efficiency", bat.charge_efficiency)
    _check_fraction("battery.discharge_efficiency", bat.discharge_efficiency)
    if bat.charge_efficiency == 0.0 or bat.discharge_efficiency == 0.0:
        raise ConfigurationError("Battery efficiencies must be positive")
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

    _check_fraction("network.loss_fraction", config.network.loss_fraction, allow_one=False)
    if config.network.transfer_capacity_mw <= 0.0:
        raise ConfigurationError("network.transfer_capacity_mw must be positive")

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
