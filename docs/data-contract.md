# Input data contract

The simulator reads a CSV table with one row per interval.

## Required columns

- `timestamp`: ISO-8601 timestamp. Values must be unique and strictly increasing after parsing.
- `demand_mw`: finite and non-negative.
- `irradiance_w_m2`: finite and non-negative.
- `ambient_temperature_c`: finite.
- `wind_speed_m_s`: finite and non-negative.

Portfolio configurations may reference additional numeric weather, demand,
availability, fuel-price, or hydro-inflow columns through asset
`time_series_key`, `availability_factor_key`, `price_time_series_key`, and
`inflow_time_series_key` fields. Extra columns are preserved by the loader and
validated when a configured asset reads them.

## Temporal rules

- Intervals must be equally spaced.
- The spacing must match `simulation.time_step_hours` within numerical tolerance.
- Missing intervals must be inserted or otherwise resolved before simulation.
- Time-zone-aware timestamps are recommended. The example data use UTC.

## Units

The code does not infer or convert units. Supplying kW where MW are expected will produce numerically valid but physically incorrect results.

Hydro inflow columns referenced by `hydro_units[].inflow_time_series_key` are
MW-water. Reservoir state and water-balance residuals are reported in
MWh-water. The first hydro model uses a constant turbine efficiency to convert
MW-water release into electrical MW generation.

## Configuration rules

The YAML configuration supports `schema_version: 1` for legacy aggregate-system
files and `schema_version: 2` for typed portfolio files. The field is optional
only for backward-compatible 0.1.x files; unknown future versions are rejected so
that new schema shapes cannot be misread by older code.

Configuration parsing is strict. Unknown root or section fields fail validation,
and misspelled fields include one deterministic suggestion when there is a close
match. Duplicate YAML keys are rejected before values are parsed. Validation
errors distinguish missing required fields, unknown fields, wrong scalar types,
invalid numeric ranges, and cross-field inconsistencies.

Legacy single-resource fields remain supported through deterministic migration
to the portfolio schema. Deprecated fields must be documented in the changelog
before removal, and removal requires a new schema version.
