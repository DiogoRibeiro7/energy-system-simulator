from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from energy_system_simulator.config import ModelConfig, StorageUnitConfig
from energy_system_simulator.exceptions import EnergySystemError


class FrequencyAdequacyError(EnergySystemError):
    """Invalid frequency-adequacy request or unsupported dispatch output."""


class FrequencyDispatchResult(Protocol):
    """Solved dispatch output required by the frequency evaluator."""

    @property
    def timeseries(self) -> pd.DataFrame:
        """Period-indexed solved dispatch table."""
        ...

    @property
    def objective_eur(self) -> float:
        """Base-case dispatch objective."""
        ...


@dataclass(frozen=True)
class FrequencyEvaluation:
    """Frequency proxy outputs separate from base dispatch accounting."""

    records: pd.DataFrame
    summary: dict[str, object]

    @property
    def adequate(self) -> bool:
        """Whether every period passes all configured frequency proxy checks."""
        return bool(self.summary["adequate"])

    def write(self, directory: str | Path) -> None:
        """Write frequency diagnostics to a directory."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.records.to_csv(output / "frequency_adequacy.csv", index=False)
        (output / "frequency_summary.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def evaluate_frequency_adequacy(
    config: ModelConfig,
    result: FrequencyDispatchResult,
) -> FrequencyEvaluation:
    """Evaluate transparent frequency-security proxies for a solved dispatch."""
    frame = result.timeseries
    _validate_required_columns(config, frame)
    records = [_evaluate_period(config, frame, period) for period in range(len(frame))]
    records_frame = pd.DataFrame.from_records(records)
    rocof_values = records_frame["rocof_hz_per_s"].to_numpy(dtype=np.float64)
    finite_rocof = rocof_values[np.isfinite(rocof_values)]
    max_rocof = float(finite_rocof.max()) if len(finite_rocof) else None
    scarcity = records_frame[~records_frame["adequate"]]
    binding = _binding_record(records_frame)
    summary: dict[str, object] = {
        "schema_version": 1,
        "formulation": "post_dispatch_frequency_adequacy_proxy",
        "dynamic_frequency_simulation": False,
        "assumptions": (
            "Linear planning proxy only; it does not model electromagnetic transients, "
            "governor dynamics, protection systems, or spatial frequency modes."
        ),
        "periods_checked": len(records_frame),
        "adequate": bool(records_frame["adequate"].all()),
        "scarcity_periods": len(scarcity),
        "minimum_inertia_mw_s": float(records_frame["total_inertia_mw_s"].min()),
        "maximum_rocof_hz_per_s": max_rocof,
        "maximum_response_shortfall_mw": float(records_frame["response_shortfall_mw"].max()),
        "maximum_inertia_shortfall_mw_s": float(records_frame["inertia_shortfall_mw_s"].max()),
        "binding_period": int(binding["period"]),
        "binding_limitation": str(binding["limitation"]),
        "base_objective_eur": float(result.objective_eur),
        "base_costs_are_separate": True,
    }
    return FrequencyEvaluation(records=records_frame, summary=summary)


def _validate_required_columns(config: ModelConfig, frame: pd.DataFrame) -> None:
    for thermal_unit in config.portfolio.thermal_generators:
        _require_column(frame, f"thermal_on__{thermal_unit.id}")
        _require_column(frame, f"thermal_output_mw__{thermal_unit.id}")
        _require_column(frame, f"thermal_capacity_available_mw__{thermal_unit.id}")
    for storage_unit in config.portfolio.storage_units:
        _require_column(frame, f"storage_soc_mwh__{storage_unit.id}")
        _require_column(frame, f"storage_discharge_mw__{storage_unit.id}")
    for hydro_unit in config.portfolio.hydro_units:
        _require_column(frame, f"hydro_generation_mw__{hydro_unit.id}")


def _evaluate_period(
    config: ModelConfig,
    frame: pd.DataFrame,
    period: int,
) -> dict[str, object]:
    frequency = config.frequency
    thermal = _thermal_contribution(config, frame, period)
    hydro = _hydro_contribution(config, frame, period)
    storage = _storage_contribution(config, frame, period)
    largest_infeed = max(thermal["largest"], hydro["largest"], _import_infeed(frame, period))
    largest_loss = max(
        frequency.credible_loss_mw,
        frequency.credible_loss_fraction_of_largest_online_infeed * largest_infeed,
    )
    synchronous_inertia = thermal["inertia"] + hydro["inertia"]
    total_inertia = synchronous_inertia + storage["synthetic_inertia"]
    rocof = _rocof_hz_per_s(
        nominal_frequency_hz=frequency.nominal_frequency_hz,
        largest_loss_mw=largest_loss,
        inertia_mw_s=total_inertia,
    )
    inertia_shortfall = max(0.0, frequency.minimum_inertia_mw_s - total_inertia)
    damping_credit = (
        frequency.demand_damping_mw_per_hz * frequency.quasi_steady_state_frequency_deviation_hz
    )
    response_requirement = max(0.0, largest_loss - damping_credit)
    sustained_response = thermal["primary_response"] + hydro["primary_response"]
    fast_response = storage["fast_response"]
    response_shortfall = max(0.0, response_requirement - sustained_response - fast_response)
    rocof_violation = bool(rocof > frequency.maximum_rocof_hz_per_s)
    inertia_violation = inertia_shortfall > 1e-9
    response_violation = response_shortfall > 1e-9
    adequate = not (rocof_violation or inertia_violation or response_violation)
    return {
        "schema_version": 1,
        "period": period,
        "adequate": adequate,
        "limitation": _limitation(inertia_violation, rocof_violation, response_violation),
        "synchronous_inertia_mw_s": synchronous_inertia,
        "synthetic_inertia_mw_s": storage["synthetic_inertia"],
        "total_inertia_mw_s": total_inertia,
        "minimum_inertia_mw_s": frequency.minimum_inertia_mw_s,
        "inertia_shortfall_mw_s": inertia_shortfall,
        "largest_online_infeed_mw": largest_infeed,
        "largest_credible_loss_mw": largest_loss,
        "rocof_hz_per_s": rocof,
        "maximum_rocof_hz_per_s": frequency.maximum_rocof_hz_per_s,
        "response_requirement_mw": response_requirement,
        "sustained_primary_response_mw": sustained_response,
        "thermal_primary_response_mw": thermal["primary_response"],
        "hydro_primary_response_mw": hydro["primary_response"],
        "fast_frequency_response_mw": fast_response,
        "response_shortfall_mw": response_shortfall,
        "demand_damping_credit_mw": damping_credit,
        "thermal_inertia_mw_s": thermal["inertia"],
        "hydro_inertia_mw_s": hydro["inertia"],
    }


def _thermal_contribution(
    config: ModelConfig,
    frame: pd.DataFrame,
    period: int,
) -> dict[str, float]:
    inertia = 0.0
    primary_response = 0.0
    largest = 0.0
    for unit in config.portfolio.thermal_generators:
        output = _value(frame, f"thermal_output_mw__{unit.id}", period)
        online = _value(frame, f"thermal_on__{unit.id}", period) >= 0.5
        if not online:
            continue
        inertia += unit.synchronous_inertia_mw_s
        largest = max(largest, output)
        if (
            unit.primary_response_time_seconds
            <= config.frequency.maximum_primary_response_time_seconds
        ):
            reserve_column = f"thermal_upward_reserve_mw__{unit.id}"
            headroom = max(
                0.0,
                _value(frame, f"thermal_capacity_available_mw__{unit.id}", period) - output,
            )
            reserve = _value(frame, reserve_column, period) if reserve_column in frame else headroom
            primary_response += min(unit.primary_response_mw, reserve, headroom)
    return {"inertia": inertia, "primary_response": primary_response, "largest": largest}


def _hydro_contribution(
    config: ModelConfig,
    frame: pd.DataFrame,
    period: int,
) -> dict[str, float]:
    inertia = 0.0
    primary_response = 0.0
    largest = 0.0
    for unit in config.portfolio.hydro_units:
        output = _value(frame, f"hydro_generation_mw__{unit.id}", period)
        if output <= 1e-9:
            continue
        inertia += unit.synchronous_inertia_mw_s
        largest = max(largest, output)
        if (
            unit.primary_response_time_seconds
            <= config.frequency.maximum_primary_response_time_seconds
        ):
            headroom = max(0.0, unit.turbine_capacity_mw - output)
            primary_response += min(unit.primary_response_mw, headroom)
    return {"inertia": inertia, "primary_response": primary_response, "largest": largest}


def _storage_contribution(
    config: ModelConfig,
    frame: pd.DataFrame,
    period: int,
) -> dict[str, float]:
    fast_response = 0.0
    synthetic_inertia = 0.0
    for unit in config.portfolio.storage_units:
        available_power = _storage_available_response_power(unit, frame, period)
        available_energy_mwh = max(
            0.0,
            _value(frame, f"storage_soc_mwh__{unit.id}", period) - unit.config.minimum_soc_mwh,
        )
        if unit.fast_frequency_response_time_seconds <= (
            config.frequency.maximum_fast_response_time_seconds
        ):
            energy_limited_mw = (
                available_energy_mwh * 3600.0 / unit.fast_frequency_response_duration_seconds
            )
            fast_response += min(
                unit.fast_frequency_response_mw,
                available_power,
                energy_limited_mw,
            )
        power_limited_inertia = (
            available_power
            * config.frequency.nominal_frequency_hz
            / (2.0 * config.frequency.maximum_rocof_hz_per_s)
        )
        energy_limited_inertia = available_energy_mwh * 3600.0
        synthetic_inertia += min(
            unit.synthetic_inertia_mw_s,
            power_limited_inertia,
            energy_limited_inertia,
        )
    return {"fast_response": fast_response, "synthetic_inertia": synthetic_inertia}


def _storage_available_response_power(
    unit: StorageUnitConfig,
    frame: pd.DataFrame,
    period: int,
) -> float:
    configured_power = (
        unit.config.discharge_power_capacity_mw
        if unit.config.discharge_power_capacity_mw is not None
        else unit.config.power_capacity_mw
    )
    discharge = _value(frame, f"storage_discharge_mw__{unit.id}", period)
    return max(0.0, configured_power * unit.config.availability_factor - discharge)


def _import_infeed(frame: pd.DataFrame, period: int) -> float:
    return max(0.0, _value(frame, "imports_mw", period)) if "imports_mw" in frame else 0.0


def _rocof_hz_per_s(
    *,
    nominal_frequency_hz: float,
    largest_loss_mw: float,
    inertia_mw_s: float,
) -> float:
    if largest_loss_mw <= 0.0:
        return 0.0
    if inertia_mw_s <= 0.0:
        return float("inf")
    return nominal_frequency_hz * largest_loss_mw / (2.0 * inertia_mw_s)


def _limitation(
    inertia_violation: bool,
    rocof_violation: bool,
    response_violation: bool,
) -> str:
    if inertia_violation:
        return "inertia"
    if rocof_violation:
        return "rocof"
    if response_violation:
        return "response"
    return ""


def _binding_record(records: pd.DataFrame) -> pd.Series:
    ranking = records.assign(
        binding_score=(
            records["inertia_shortfall_mw_s"]
            + records["response_shortfall_mw"] * 1_000.0
            + records["rocof_hz_per_s"].replace(np.inf, 1_000_000.0)
        )
    )
    return ranking.sort_values("binding_score", ascending=False).iloc[0]


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame:
        raise FrequencyAdequacyError(
            f"Frequency adequacy evaluation requires output column: {column}"
        )


def _value(frame: pd.DataFrame, column: str, period: int) -> float:
    return float(frame[column].iloc[period])
