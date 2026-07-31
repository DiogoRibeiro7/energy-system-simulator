# Configuration

Energy System Simulator supports two configuration schemas.

## Schema Versions

`schema_version: 1` is the legacy aggregate-system schema used by
`configs/example.yaml`. It remains loadable for backward compatibility and is
resolved into the same typed portfolio model used by newer configurations.

`schema_version: 2` is the portfolio schema. It replaces single `solar`, `wind`,
`thermal`, and `battery` sections with lists:

- `renewable_generators`
- `thermal_generators`
- `storage_units`
- `hydro_units`
- `imports`
- `demand`

The current optimisation engine evaluates multiple renewable assets by ID and
reports their availability, used output, and curtailment. Thermal, storage, and
import resources are still projected to the first compatible aggregate resource
until the multi-unit formulation is introduced.

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
