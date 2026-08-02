# Public Data Adapters

Public-data ingestion is separate from optimisation. Simulation runs continue to
read local CSV snapshots through `paths.input_csv`; no provider-specific code or
network access is imported by dispatch or planning modules.

## Canonical Schema

Adapter outputs use UTC timestamps and provider-neutral column names:

- `timestamp`
- `demand_mw`
- `irradiance_w_m2`
- `ambient_temperature_c`
- `wind_speed_m_s`
- `import_price_eur_per_mwh`
- `import_capacity_mw`
- `outage_availability`
- `hydro_inflow_mw`

The normal simulator input snapshot must include the existing required columns:
`demand_mw`, `irradiance_w_m2`, `ambient_temperature_c`, and `wind_speed_m_s`.
Additional canonical columns can be used by portfolio configurations that
reference them.

## Local Snapshot Workflow

Prepare data from local provider extracts with:

```bash
poetry run energy-sim prepare-data --spec examples/data/prepare_public_fixture.yaml
```

The command writes a canonical CSV plus a JSON manifest. Downloads are expected
to happen outside the simulation run, in explicit user scripts or data
collection commands. Tests use small committed fixture files and never contact
live services.

## Provenance

Each source records:

- provider
- source URL or dataset identifier
- retrieval timestamp
- licence
- original timezone
- transformation steps
- input checksum
- missing-data treatment
- temporal aggregation

The snapshot manifest also records the output checksum and validation reports
before and after merging/resampling.

## Timezones And DST

Adapters convert local timestamps to UTC. Naive local timestamps are localized
with the configured IANA timezone. Daylight-saving spring-forward days with 23
local hours and fall-back days with 25 local hours are handled during conversion.
Duplicate UTC timestamps are rejected.

## Missing Data

Missing data policy is explicit:

- `reject`: fail if any values are missing.
- `interpolate`: bounded linear interpolation with `limit`.
- `forward_fill`: bounded forward fill with `limit`.
- `mark`: preserve missing values and add `*_is_missing` marker columns.

Long gaps are not silently filled; if the configured bounded policy cannot treat
all missing values, transformation fails.

## Resampling

`resample_canonical` uses unit-aware rules:

- `power_average`: average MW or weather-driver values over the target interval.
- `energy_sum`: sum MWh values over the target interval.

This avoids mixing up power averages and energy totals during aggregation.
