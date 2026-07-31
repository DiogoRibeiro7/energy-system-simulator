# Configuration

Energy System Simulator supports two configuration schemas.

## Schema Versions

`schema_version: 1` is the legacy aggregate-system schema used by
`configs/example.yaml`. It remains loadable for backward compatibility and is
resolved into the same typed portfolio model used by newer configurations.

`schema_version: 2` is the portfolio schema. It replaces single `solar`, `wind`,
`thermal`, and `battery` sections with lists:

- `fuels`
- `renewable_generators`
- `thermal_generators`
- `storage_units`
- `hydro_units`
- `imports`
- `demand`

The current optimisation engine evaluates multiple renewable assets by ID and
optimizes all configured thermal generators as a generator-indexed unit
commitment fleet. Storage and import resources are still projected to the first
compatible aggregate resource until their indexed formulations are introduced.

## Validation Approach

The project stays with frozen dataclasses plus explicit parsing and validation.
This avoids adding a runtime validation dependency while keeping every supported
field typed at the domain boundary. Parsing, key validation, and semantic
cross-validation are separate code paths:

- parsing rejects malformed scalar values and non-mapping sections;
- schema validation rejects unknown or missing fields;
- semantic validation checks identifiers, bus and zone references, time-series
  keys, physical bounds, efficiencies, and initial states.

Errors include the exact configuration path, for example
`thermal_generators[1].initial_output_mw`.

## Identifiers

Identifiers must be non-empty strings without whitespace and must be unique
inside their section. The schema validates IDs for scenarios, zones, buses,
lines, renewable generators, thermal generators, storage units, hydro units,
imports, and demand entries.

All asset `bus_id` values must reference a declared bus. Bus `zone_id` values
must reference a declared zone. Line endpoints must reference declared buses.

## Time-Series Keys

Time-series references are non-empty column names. The default example uses the
base input data contract columns:

- `demand_mw`
- `irradiance_w_m2`
- `ambient_temperature_c`
- `wind_speed_m_s`

Solar generators default to `time_series_key: irradiance_w_m2` and
`ambient_temperature_key: ambient_temperature_c`, but portfolio files may point
different solar assets at different irradiance and ambient-temperature columns.
Wind generators default to `time_series_key: wind_speed_m_s`. Demand entries
default to `time_series_key: demand_mw`. Referenced columns are validated after
the input CSV is loaded.

Thermal generators may set:

- `must_run`: optional boolean. When true, the unit is committed in every period.
- `availability_factor`: optional static capacity multiplier in `[0, 1]`.
- `availability_factor_key`: optional input CSV column containing period-specific
  multipliers in `[0, 1]`. It is multiplied by `availability_factor`.

Fuel and heat-rate fields are available in schema v2:

- `fuels[].price_eur_per_mwh_thermal` is the scalar fuel price. A fuel may
  instead set `price_time_series_key` to read period-specific prices from the
  input CSV, with the scalar price retained as metadata and fallback for direct
  solver calls.
- `fuels[].co2_factor_tonnes_per_mwh_thermal` defines direct CO2 per thermal
  MWh. Optional methane, NOx, and SOx factors are reported quantities.
- `thermal_generators[].minimum_fuel_input_mwh_per_hour` defines fuel input for
  the online minimum-output block.
- `thermal_generators[].heat_rate_segments` defines incremental output blocks
  above minimum output. Segment capacities must sum to
  `maximum_output_mw - minimum_output_mw`, and heat rates must be nondecreasing.
- `thermal_generators[].startup_categories` defines hot, warm, and cold startup
  behaviour using strictly increasing `minimum_down_time_hours` thresholds.
  Startup fuel is included in reported fuel input, emissions, and startup cost.

When `heat_rate_segments` is omitted, the generator remains in compatibility
mode and uses `variable_cost_eur_per_mwh` plus
`emission_factor_tonnes_per_mwh`.

Storage units may set:

- `technology`: `battery` or `pumped_storage`; this is metadata for reporting
  while both use the same linear energy-storage equations.
- `charge_power_capacity_mw` and `discharge_power_capacity_mw`: optional
  independent power limits. When omitted, `power_capacity_mw` is used for both.
- `self_discharge_rate_per_hour`: hourly standing energy loss fraction.
- `minimum_charge_mw` and `minimum_discharge_mw`: optional minimum power when
  the corresponding operating mode is active.
- `charge_ramp_mw_per_hour` and `discharge_ramp_mw_per_hour`: optional ramp
  limits on charge and discharge power.
- `availability_factor` and `availability_factor_key`: static and optional
  time-series multipliers in `[0, 1]`.
- `degradation_bands`: optional throughput bands with nondecreasing
  `cost_eur_per_mwh`. The approximation assigns charge-plus-discharge
  throughput to the cheapest available bands each period.

Each storage asset has its own terminal SOC mode: `minimum`, `exact`, `cyclic`,
or `free`.

## Migration

Convert a legacy configuration with:

```bash
poetry run energy-sim migrate-config --config configs/example.yaml --output configs/example.v2.yaml
```

Without `--output`, the migrated YAML is written to standard output. Migration is
deterministic and preserves the aggregate simulation meaning by creating a
single system zone and bus, one solar generator, one wind generator, one thermal
generator, one storage unit, one import resource, and one demand entry.

## Examples

- `configs/example.yaml` is the backward-compatible schema v1 example.
- `configs/portfolio_two_thermal.yaml` is a schema v2 example with two thermal
  generators and two renewable generators.
- `tests/fixtures/invalid_portfolio_missing_bus.yaml` is intentionally invalid
  and is used by the validation tests.

Resolved configurations are serialized into run manifests using a canonical
JSON/YAML-ready representation with absolute paths.
