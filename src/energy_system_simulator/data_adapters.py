from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
import yaml

from energy_system_simulator.exceptions import DataValidationError

CanonicalColumn = Literal[
    "demand_mw",
    "irradiance_w_m2",
    "ambient_temperature_c",
    "wind_speed_m_s",
    "import_price_eur_per_mwh",
    "import_capacity_mw",
    "outage_availability",
    "hydro_inflow_mw",
]
MissingMethod = Literal["reject", "interpolate", "forward_fill", "mark"]
ResamplingRule = Literal["power_average", "energy_sum"]

SIMULATION_REQUIRED_COLUMNS = (
    "demand_mw",
    "irradiance_w_m2",
    "ambient_temperature_c",
    "wind_speed_m_s",
)


@dataclass(frozen=True)
class MissingDataPolicy:
    """Explicit missing-data treatment for adapter output."""

    method: MissingMethod = "reject"
    limit: int = 0


@dataclass(frozen=True)
class DataProvenance:
    """Reproducibility metadata for a local immutable data snapshot."""

    provider: str
    source: str
    retrieved_at_utc: str
    licence: str
    original_timezone: str
    transformation_steps: tuple[str, ...]
    checksum_sha256: str
    missing_data_treatment: str
    temporal_aggregation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataValidationReport:
    """Validation metrics for canonical data before or after transformation."""

    rows: int
    start_utc: str | None
    end_utc: str | None
    duplicate_timestamps: int
    missing_intervals: int
    gap_starts_utc: tuple[str, ...]
    missing_values_by_column: dict[str, int]
    min_by_column: dict[str, float]
    max_by_column: dict[str, float]
    energy_totals_mwh: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterResult:
    """Canonical data frame plus adapter-specific provenance and validation."""

    frame: pd.DataFrame
    provenance: DataProvenance
    validation_report: DataValidationReport


@dataclass(frozen=True)
class SnapshotResult:
    """Written canonical snapshot and manifest paths."""

    output_csv: Path
    manifest_json: Path
    validation_report: DataValidationReport
    provenance: tuple[DataProvenance, ...]
    output_checksum_sha256: str


class PublicDataAdapter(Protocol):
    """Provider-neutral adapter interface."""

    def transform(self) -> AdapterResult:
        """Transform provider-specific local data into canonical UTC time series."""


@dataclass(frozen=True)
class EuropeanDemandCsvAdapter:
    """File adapter for European demand CSV extracts."""

    path: Path
    provider: str
    source: str
    licence: str
    retrieved_at_utc: str
    timezone: str
    timestamp_column: str = "timestamp"
    demand_column: str = "load"
    demand_unit: Literal["MW", "GW", "MWh", "kWh"] = "MW"
    missing_policy: MissingDataPolicy = field(default_factory=MissingDataPolicy)
    temporal_aggregation: str = "native"

    def transform(self) -> AdapterResult:
        raw = pd.read_csv(self.path)
        self._require_columns(raw, (self.timestamp_column, self.demand_column))
        timestamps = local_timestamps_to_utc(raw[self.timestamp_column], self.timezone)
        demand = pd.to_numeric(raw[self.demand_column], errors="coerce")
        frame = pd.DataFrame({"timestamp": timestamps, "demand_mw": demand})
        frame = _sort_and_reject_duplicate_timestamps(frame)
        frame["demand_mw"] = _convert_demand_to_mw(
            frame["demand_mw"],
            self.demand_unit,
            _interval_hours(frame["timestamp"]),
        )
        frame = apply_missing_policy(frame, self.missing_policy)
        provenance = DataProvenance(
            provider=self.provider,
            source=self.source,
            retrieved_at_utc=self.retrieved_at_utc,
            licence=self.licence,
            original_timezone=self.timezone,
            transformation_steps=(
                "parsed local timestamps",
                "converted timestamps to UTC",
                f"converted demand from {self.demand_unit} to MW",
                f"applied missing-data policy {self.missing_policy.method}",
            ),
            checksum_sha256=file_sha256(self.path),
            missing_data_treatment=_missing_policy_label(self.missing_policy),
            temporal_aggregation=self.temporal_aggregation,
        )
        return AdapterResult(
            frame=frame,
            provenance=provenance,
            validation_report=validate_canonical_frame(frame, time_step_hours=None),
        )

    def _require_columns(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
        missing = [column for column in columns if column not in frame]
        if missing:
            raise DataValidationError(f"{self.path} missing required columns: {missing}")


@dataclass(frozen=True)
class WeatherCsvAdapter:
    """File adapter for provider-specific weather driver CSV extracts."""

    path: Path
    provider: str
    source: str
    licence: str
    retrieved_at_utc: str
    timezone: str
    timestamp_column: str = "timestamp"
    column_map: dict[CanonicalColumn, str] = field(default_factory=dict)
    missing_policy: MissingDataPolicy = field(default_factory=MissingDataPolicy)
    temporal_aggregation: str = "native"

    def transform(self) -> AdapterResult:
        raw = pd.read_csv(self.path)
        required = (self.timestamp_column, *self.column_map.values())
        missing = [column for column in required if column not in raw]
        if missing:
            raise DataValidationError(f"{self.path} missing required columns: {missing}")
        data: dict[str, Any] = {
            "timestamp": local_timestamps_to_utc(raw[self.timestamp_column], self.timezone)
        }
        for canonical, source in self.column_map.items():
            data[canonical] = pd.to_numeric(raw[source], errors="coerce")
        frame = _sort_and_reject_duplicate_timestamps(pd.DataFrame(data))
        frame = apply_missing_policy(frame, self.missing_policy)
        provenance = DataProvenance(
            provider=self.provider,
            source=self.source,
            retrieved_at_utc=self.retrieved_at_utc,
            licence=self.licence,
            original_timezone=self.timezone,
            transformation_steps=(
                "parsed local timestamps",
                "converted timestamps to UTC",
                "renamed provider columns to canonical weather drivers",
                f"applied missing-data policy {self.missing_policy.method}",
            ),
            checksum_sha256=file_sha256(self.path),
            missing_data_treatment=_missing_policy_label(self.missing_policy),
            temporal_aggregation=self.temporal_aggregation,
        )
        return AdapterResult(
            frame=frame,
            provenance=provenance,
            validation_report=validate_canonical_frame(frame, time_step_hours=None),
        )


def build_canonical_snapshot(
    adapters: tuple[PublicDataAdapter, ...],
    *,
    output_csv: str | Path,
    manifest_json: str | Path,
    time_step_hours: float,
    missing_policy: MissingDataPolicy | None = None,
    resample_frequency: str | None = None,
) -> SnapshotResult:
    """Merge local adapter outputs, validate them, and write a reproducible snapshot."""
    if not adapters:
        raise DataValidationError("At least one data adapter is required")
    policy = missing_policy or MissingDataPolicy()
    results = [adapter.transform() for adapter in adapters]
    frame = _merge_frames([result.frame for result in results])
    before = validate_canonical_frame(frame, time_step_hours=None)
    frame = apply_missing_policy(frame, policy)
    if resample_frequency is not None:
        frame = resample_canonical(
            frame,
            resample_frequency,
            _default_resampling_rules(frame),
        )
    report = validate_canonical_frame(frame, time_step_hours=time_step_hours)
    for column in SIMULATION_REQUIRED_COLUMNS:
        if column not in frame:
            raise DataValidationError(f"Canonical snapshot missing required column {column!r}")
    output_path = Path(output_csv)
    manifest_path = Path(manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, lineterminator="\n")
    output_checksum = file_sha256(output_path)
    payload = {
        "schema_version": 1,
        "canonical_schema": list(SIMULATION_REQUIRED_COLUMNS),
        "output_csv": str(output_path),
        "output_checksum_sha256": output_checksum,
        "time_step_hours": time_step_hours,
        "missing_data_treatment": _missing_policy_label(policy),
        "temporal_aggregation": resample_frequency or "native",
        "validation_before": before.to_dict(),
        "validation_after": report.to_dict(),
        "sources": [result.provenance.to_dict() for result in results],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return SnapshotResult(
        output_csv=output_path,
        manifest_json=manifest_path,
        validation_report=report,
        provenance=tuple(result.provenance for result in results),
        output_checksum_sha256=output_checksum,
    )


def run_data_preparation_spec(path: str | Path) -> SnapshotResult:
    """Run a local data-preparation YAML spec."""
    spec_path = Path(path)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError("Data preparation spec must be a mapping")
    base_dir = spec_path.parent
    adapters = tuple(_adapter_from_spec(item, base_dir) for item in payload.get("adapters", []))
    missing_policy = _missing_policy_from_mapping(payload.get("missing_data", {}))
    output_csv = _resolve_path(base_dir, str(payload["output_csv"]))
    manifest_json = _resolve_path(base_dir, str(payload["manifest_json"]))
    return build_canonical_snapshot(
        adapters,
        output_csv=output_csv,
        manifest_json=manifest_json,
        time_step_hours=float(payload["time_step_hours"]),
        missing_policy=missing_policy,
        resample_frequency=payload.get("resample_frequency"),
    )


def local_timestamps_to_utc(values: pd.Series, timezone: str) -> pd.Series:
    """Parse local timestamps, handle DST, and return UTC timestamps."""
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise DataValidationError("One or more provider timestamps cannot be parsed")
    if parsed.dt.tz is None:
        localized = parsed.dt.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    else:
        localized = parsed
    return localized.dt.tz_convert("UTC")


def apply_missing_policy(frame: pd.DataFrame, policy: MissingDataPolicy) -> pd.DataFrame:
    """Apply explicit missing-data policy to non-timestamp columns."""
    result = frame.copy()
    value_columns = [column for column in result.columns if column != "timestamp"]
    if not value_columns:
        return result
    if policy.method == "reject":
        missing = result[value_columns].isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            raise DataValidationError(f"Missing values rejected: {missing.to_dict()}")
        return result
    if policy.method == "interpolate":
        result[value_columns] = result[value_columns].interpolate(
            method="linear",
            limit=policy.limit or None,
            limit_direction="both",
        )
    elif policy.method == "forward_fill":
        result[value_columns] = result[value_columns].ffill(limit=policy.limit or None)
    elif policy.method == "mark":
        for column in value_columns:
            result[f"{column}_is_missing"] = result[column].isna()
    else:
        raise DataValidationError(f"Unknown missing-data policy: {policy.method}")
    remaining = result[value_columns].isna().sum()
    remaining = remaining[remaining > 0]
    if policy.method != "mark" and not remaining.empty:
        raise DataValidationError(
            f"Missing-data policy left untreated values: {remaining.to_dict()}"
        )
    return result


def resample_canonical(
    frame: pd.DataFrame,
    frequency: str,
    rules: dict[str, ResamplingRule],
) -> pd.DataFrame:
    """Resample canonical data with unit-aware deterministic rules."""
    if "timestamp" not in frame:
        raise DataValidationError("Canonical frame must include timestamp")
    indexed = frame.set_index(pd.DatetimeIndex(frame["timestamp"])).drop(columns=["timestamp"])
    pieces: list[pd.DataFrame] = []
    for column in indexed.columns:
        if column.endswith("_is_missing"):
            values = indexed[column].resample(frequency).max()
        elif rules.get(column, "power_average") == "energy_sum":
            values = indexed[column].resample(frequency).sum()
        else:
            values = indexed[column].resample(frequency).mean()
        pieces.append(values.to_frame(column))
    result = pd.concat(pieces, axis=1).reset_index(names="timestamp")
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result


def validate_canonical_frame(
    frame: pd.DataFrame,
    *,
    time_step_hours: float | None,
) -> DataValidationReport:
    """Build a validation report and reject duplicate timestamps."""
    if "timestamp" not in frame:
        raise DataValidationError("Canonical frame must include timestamp")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise DataValidationError("Canonical frame contains unparseable timestamps")
    duplicate_count = int(timestamps.duplicated().sum())
    if duplicate_count:
        raise DataValidationError(
            f"Canonical frame contains {duplicate_count} duplicate timestamps"
        )
    ordered = frame.copy()
    ordered["timestamp"] = timestamps
    ordered = ordered.sort_values("timestamp")
    missing_intervals = 0
    gap_starts: list[str] = []
    if len(ordered) > 1:
        timestamp_values = pd.DatetimeIndex(ordered["timestamp"]).to_numpy(dtype="datetime64[ns]")
        difference_seconds = np.diff(timestamp_values) / np.timedelta64(1, "s")
        if time_step_hours is None:
            expected_seconds = float(pd.Series(difference_seconds).mode().iloc[0])
        else:
            expected_seconds = time_step_hours * 3600.0
        gaps = difference_seconds[difference_seconds > expected_seconds]
        missing_intervals = int(
            sum(max(0, round(float(delta) / expected_seconds) - 1) for delta in gaps)
        )
        gap_starts = [
            ordered["timestamp"].iloc[index].isoformat()
            for index, delta in enumerate(difference_seconds, start=1)
            if delta > expected_seconds
        ]
    numeric = ordered.drop(columns=["timestamp"]).select_dtypes(include=[np.number])
    dt = time_step_hours if time_step_hours is not None else _interval_hours(ordered["timestamp"])
    return DataValidationReport(
        rows=len(ordered),
        start_utc=ordered["timestamp"].iloc[0].isoformat() if len(ordered) else None,
        end_utc=ordered["timestamp"].iloc[-1].isoformat() if len(ordered) else None,
        duplicate_timestamps=duplicate_count,
        missing_intervals=missing_intervals,
        gap_starts_utc=tuple(gap_starts),
        missing_values_by_column={
            column: int(ordered[column].isna().sum())
            for column in ordered.columns
            if column != "timestamp"
        },
        min_by_column={column: float(numeric[column].min()) for column in numeric.columns},
        max_by_column={column: float(numeric[column].max()) for column in numeric.columns},
        energy_totals_mwh={
            column: float(numeric[column].sum() * dt)
            for column in numeric.columns
            if column.endswith("_mw")
        },
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0].copy()
    for frame in frames[1:]:
        result = result.merge(frame, on="timestamp", how="outer", validate="one_to_one")
    return result.sort_values("timestamp").reset_index(drop=True)


def _sort_and_reject_duplicate_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("timestamp").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise DataValidationError("Provider data produced duplicate UTC timestamps")
    return result


def _convert_demand_to_mw(values: pd.Series, unit: str, interval_hours: float) -> pd.Series:
    if unit == "MW":
        return values
    if unit == "GW":
        return values * 1000.0
    if unit == "MWh":
        return values / interval_hours
    if unit == "kWh":
        return values / 1000.0 / interval_hours
    raise DataValidationError(f"Unsupported demand unit: {unit}")


def _interval_hours(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 1.0
    deltas = pd.to_datetime(timestamps, utc=True).diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        return 1.0
    return float(positive.mode().iloc[0] / pd.Timedelta(hours=1))


def _default_resampling_rules(frame: pd.DataFrame) -> dict[str, ResamplingRule]:
    return {
        column: "energy_sum" if column.endswith("_mwh") else "power_average"
        for column in frame.columns
        if column != "timestamp"
    }


def _adapter_from_spec(payload: Any, base_dir: Path) -> PublicDataAdapter:
    if not isinstance(payload, dict):
        raise DataValidationError("Adapter spec must be a mapping")
    kind = payload.get("kind")
    source_path = _resolve_path(base_dir, str(payload["path"]))
    provider = str(payload["provider"])
    source = str(payload["source"])
    licence = str(payload["licence"])
    retrieved_at = str(payload["retrieved_at_utc"])
    timezone = str(payload["timezone"])
    timestamp_column = str(payload.get("timestamp_column", "timestamp"))
    missing_policy = _missing_policy_from_mapping(payload.get("missing_data", {}))
    temporal_aggregation = str(payload.get("temporal_aggregation", "native"))
    if kind == "european_demand_csv":
        return EuropeanDemandCsvAdapter(
            path=source_path,
            provider=provider,
            source=source,
            licence=licence,
            retrieved_at_utc=retrieved_at,
            timezone=timezone,
            timestamp_column=timestamp_column,
            missing_policy=missing_policy,
            temporal_aggregation=temporal_aggregation,
            demand_column=str(payload.get("demand_column", "load")),
            demand_unit=cast(Literal["MW", "GW", "MWh", "kWh"], payload.get("demand_unit", "MW")),
        )
    if kind == "weather_csv":
        return WeatherCsvAdapter(
            path=source_path,
            provider=provider,
            source=source,
            licence=licence,
            retrieved_at_utc=retrieved_at,
            timezone=timezone,
            timestamp_column=timestamp_column,
            missing_policy=missing_policy,
            temporal_aggregation=temporal_aggregation,
            column_map=cast(dict[CanonicalColumn, str], dict(payload["column_map"])),
        )
    raise DataValidationError(f"Unsupported adapter kind: {kind}")


def _missing_policy_from_mapping(payload: Any) -> MissingDataPolicy:
    if not payload:
        return MissingDataPolicy()
    if not isinstance(payload, dict):
        raise DataValidationError("missing_data must be a mapping")
    return MissingDataPolicy(
        method=payload.get("method", "reject"),
        limit=int(payload.get("limit", 0)),
    )


def _resolve_path(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _missing_policy_label(policy: MissingDataPolicy) -> str:
    return f"{policy.method}(limit={policy.limit})"
