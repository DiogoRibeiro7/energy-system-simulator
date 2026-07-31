from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml

from energy_system_simulator.exceptions import ConfigurationError

T = TypeVar("T", int, float)


def _number(section: Mapping[str, Any], key: str, expected: type[T]) -> T:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key!r} must be numeric")
    return expected(value)


def _integer(section: Mapping[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{key!r} must be an integer")
    return value


def _boolean(section: Mapping[str, Any], key: str) -> bool:
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
    return _boolean(section, key)


def _optional_string(section: Mapping[str, Any], key: str, default: str) -> str:
    if key not in section:
        return default
    return _string(section, key)


def _string(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key!r} must be a non-empty string")
    return value


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Missing or invalid section: {name}")
    return value


def _check_nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise ConfigurationError(f"{name} must be non-negative")


def _check_fraction(name: str, value: float, *, allow_one: bool = True) -> None:
    upper_valid = value <= 1.0 if allow_one else value < 1.0
    if value < 0.0 or not upper_valid:
        operator = "[0, 1]" if allow_one else "[0, 1)"
        raise ConfigurationError(f"{name} must be in {operator}")


@dataclass(frozen=True)
class SimulationConfig:
    time_step_hours: float
    solver_time_limit_seconds: float
    mip_relative_gap: float
    allow_non_optimal_solution: bool


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
class NetworkConfig:
    loss_fraction: float
    transfer_capacity_mw: float


@dataclass(frozen=True)
class ImportConfig:
    maximum_power_mw: float
    price_eur_per_mwh: float
    emission_factor_tonnes_per_mwh: float


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
class ModelConfig:
    simulation: SimulationConfig
    solar: SolarConfig
    wind: WindConfig
    thermal: ThermalConfig
    battery: BatteryConfig
    network: NetworkConfig
    imports: ImportConfig
    penalties: PenaltyConfig
    paths: PathConfig


def load_config(path: str | Path) -> ModelConfig:
    """Load and validate a YAML model configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Configuration root must be a mapping")

    simulation_raw = _section(raw, "simulation")
    solar_raw = _section(raw, "solar")
    wind_raw = _section(raw, "wind")
    thermal_raw = _section(raw, "thermal")
    battery_raw = _section(raw, "battery")
    network_raw = _section(raw, "network")
    import_raw = _section(raw, "imports")
    penalties_raw = _section(raw, "penalties")
    paths_raw = _section(raw, "paths")

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

    config = ModelConfig(
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
