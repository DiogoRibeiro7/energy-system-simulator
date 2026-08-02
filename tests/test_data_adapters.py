from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from energy_system_simulator.cli import main
from energy_system_simulator.data import load_input_data
from energy_system_simulator.data_adapters import (
    EuropeanDemandCsvAdapter,
    MissingDataPolicy,
    WeatherCsvAdapter,
    build_canonical_snapshot,
    file_sha256,
    resample_canonical,
    run_data_preparation_spec,
    validate_canonical_frame,
)
from energy_system_simulator.exceptions import DataValidationError

FIXTURES = Path(__file__).parent / "fixtures" / "public_data"


def _demand_adapter(path: Path, *, unit: str = "GW") -> EuropeanDemandCsvAdapter:
    return EuropeanDemandCsvAdapter(
        path=path,
        provider="ENTSO-E Transparency Platform",
        source="fixture-demand",
        licence="fixture",
        retrieved_at_utc="2026-08-02T00:00:00Z",
        timezone="Europe/Paris",
        timestamp_column="local_time",
        demand_column="load_gw" if unit == "GW" else "load_mw",
        demand_unit=unit,
    )


def _weather_adapter(path: Path, *, policy: MissingDataPolicy | None = None) -> WeatherCsvAdapter:
    return WeatherCsvAdapter(
        path=path,
        provider="ERA5 fixture",
        source="fixture-weather",
        licence="fixture",
        retrieved_at_utc="2026-08-02T00:00:00Z",
        timezone="Europe/Paris",
        timestamp_column="local_time",
        column_map={
            "irradiance_w_m2": "ghi",
            "ambient_temperature_c": "temp_c",
            "wind_speed_m_s": "wind_ms",
        },
        missing_policy=policy or MissingDataPolicy(),
    )


def test_dst_spring_forward_local_times_are_canonical_utc() -> None:
    result = _demand_adapter(FIXTURES / "european_demand_dst.csv").transform()

    assert result.frame["timestamp"].dt.tz is not None
    assert result.frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2026-03-28T23:00:00Z",
        "2026-03-29T00:00:00Z",
        "2026-03-29T01:00:00Z",
    ]
    assert result.frame["demand_mw"].tolist() == pytest.approx([1000.0, 1100.0, 1200.0])
    assert result.provenance.original_timezone == "Europe/Paris"


def test_duplicate_utc_timestamps_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="duplicate"):
        _demand_adapter(FIXTURES / "duplicate_demand.csv", unit="MW").transform()


def test_missing_data_policies_are_explicit_and_limited() -> None:
    with pytest.raises(DataValidationError, match="Missing values rejected"):
        _weather_adapter(FIXTURES / "weather_missing.csv").transform()

    interpolated = _weather_adapter(
        FIXTURES / "weather_missing.csv",
        policy=MissingDataPolicy(method="interpolate", limit=1),
    ).transform()

    assert interpolated.frame["irradiance_w_m2"].tolist() == pytest.approx([0.0, 50.0, 100.0])
    assert "interpolate" in interpolated.provenance.missing_data_treatment


def test_unit_conversion_from_interval_energy_to_power(tmp_path: Path) -> None:
    path = tmp_path / "energy.csv"
    pd.DataFrame(
        {
            "local_time": ["2026-01-01 00:00", "2026-01-01 00:30"],
            "load_mw": [5.0, 10.0],
        }
    ).to_csv(path, index=False)
    adapter = EuropeanDemandCsvAdapter(
        path=path,
        provider="fixture",
        source="energy-intervals",
        licence="fixture",
        retrieved_at_utc="2026-08-02T00:00:00Z",
        timezone="UTC",
        timestamp_column="local_time",
        demand_column="load_mw",
        demand_unit="MWh",
    )

    result = adapter.transform()

    assert result.frame["demand_mw"].tolist() == pytest.approx([10.0, 20.0])


def test_power_average_and_energy_sum_resampling() -> None:
    frame = pd.read_csv(FIXTURES / "quarter_hour.csv")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    result = resample_canonical(
        frame,
        "1h",
        {"power_mw": "power_average", "energy_mwh": "energy_sum"},
    )

    assert result["power_mw"].iloc[0] == pytest.approx(25.0)
    assert result["energy_mwh"].iloc[0] == pytest.approx(25.0)


def test_validation_report_detects_missing_intervals_and_energy_totals() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"],
                utc=True,
            ),
            "demand_mw": [10.0, 20.0],
        }
    )

    report = validate_canonical_frame(frame, time_step_hours=1.0)

    assert report.missing_intervals == 1
    assert report.energy_totals_mwh["demand_mw"] == pytest.approx(30.0)


def test_snapshot_manifest_contains_checksums_and_is_simulation_ready(tmp_path: Path) -> None:
    output_csv = tmp_path / "snapshot.csv"
    manifest_json = tmp_path / "snapshot.manifest.json"

    result = build_canonical_snapshot(
        (
            _demand_adapter(FIXTURES / "european_demand_dst.csv"),
            _weather_adapter(FIXTURES / "weather_local.csv"),
        ),
        output_csv=output_csv,
        manifest_json=manifest_json,
        time_step_hours=1.0,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert result.output_checksum_sha256 == file_sha256(output_csv)
    assert manifest["sources"][0]["provider"] == "ENTSO-E Transparency Platform"
    assert manifest["output_checksum_sha256"] == file_sha256(output_csv)
    assert load_input_data(output_csv, time_step_hours=1.0).shape[0] == 3


def test_prepare_data_cli_spec_uses_local_files_only(tmp_path: Path) -> None:
    spec = tmp_path / "prepare.yaml"
    spec.write_text(
        f"""
output_csv: {tmp_path / "prepared.csv"}
manifest_json: {tmp_path / "prepared.manifest.json"}
time_step_hours: 1.0
adapters:
  - kind: european_demand_csv
    path: {FIXTURES / "european_demand_dst.csv"}
    provider: ENTSO-E Transparency Platform
    source: fixture-demand
    licence: fixture
    retrieved_at_utc: "2026-08-02T00:00:00Z"
    timezone: Europe/Paris
    timestamp_column: local_time
    demand_column: load_gw
    demand_unit: GW
  - kind: weather_csv
    path: {FIXTURES / "weather_local.csv"}
    provider: ERA5 fixture
    source: fixture-weather
    licence: fixture
    retrieved_at_utc: "2026-08-02T00:00:00Z"
    timezone: Europe/Paris
    timestamp_column: local_time
    column_map:
      irradiance_w_m2: ghi
      ambient_temperature_c: temp_c
      wind_speed_m_s: wind_ms
""",
        encoding="utf-8",
    )

    result = run_data_preparation_spec(spec)

    assert result.output_csv.is_file()
    assert result.manifest_json.is_file()


def test_cli_prepare_data_writes_snapshot(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "prepare.yaml"
    spec.write_text(
        f"""
output_csv: {tmp_path / "prepared.csv"}
manifest_json: {tmp_path / "prepared.manifest.json"}
time_step_hours: 1.0
adapters:
  - kind: european_demand_csv
    path: {FIXTURES / "european_demand_dst.csv"}
    provider: ENTSO-E Transparency Platform
    source: fixture-demand
    licence: fixture
    retrieved_at_utc: "2026-08-02T00:00:00Z"
    timezone: Europe/Paris
    timestamp_column: local_time
    demand_column: load_gw
    demand_unit: GW
  - kind: weather_csv
    path: {FIXTURES / "weather_local.csv"}
    provider: ERA5 fixture
    source: fixture-weather
    licence: fixture
    retrieved_at_utc: "2026-08-02T00:00:00Z"
    timezone: Europe/Paris
    timestamp_column: local_time
    column_map:
      irradiance_w_m2: ghi
      ambient_temperature_c: temp_c
      wind_speed_m_s: wind_ms
""",
        encoding="utf-8",
    )

    main(["prepare-data", "--spec", str(spec)])

    captured = capsys.readouterr()
    assert "Canonical data written" in captured.out
    assert (tmp_path / "prepared.csv").is_file()
