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

The current optimisation engine evaluates multiple renewable assets by ID,
optimizes all configured thermal generators as a generator-indexed unit
commitment fleet, and dispatches indexed storage, hydro, and demand-response
portfolios. Imports are still represented as a single aggregate resource.

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

Hydro units may set:

- `kind`: `reservoir` or `run_of_river`. Run-of-river units must have zero
  reservoir storage fields and cannot shift inflow across periods.
- `inflow_time_series_key`: required input CSV column for natural inflow in
  MW-water, the energy-equivalent water power entering the reservoir.
- `turbine_capacity_mw`: maximum electrical hydro output.
- `turbine_efficiency`: constant conversion from MW-water release to electrical
  MW generation.
- `minimum_reservoir_mwh`, `maximum_reservoir_mwh`, and
  `initial_reservoir_mwh`: reservoir state bounds and initial state in
  MWh-water.
- `minimum_final_reservoir_mwh` and `terminal_reservoir_mode`: terminal storage
  policy, using `minimum`, `exact`, `cyclic`, or `free`.
- `spill_capacity_mw`: optional finite spill limit in MW-water. Omit it for
  unlimited spill.
- `minimum_release_mw`: optional environmental minimum release, met by turbine
  release plus spill in MW-water.
- `evaporation_rate_per_hour`: hourly standing water loss fraction.
- `water_value_eur_per_mwh`: optional terminal value for retained MWh-water.
- `upstream_hydro_id` and `cascade_delay_hours`: typed cascade metadata. The
  current optimisation does not yet add upstream releases to downstream inflows.

Demand entities may set:

- `kind`: `fixed`, `curtailable`, `shiftable`, `deferrable`, or `ev_charging`.
  Existing fixed demand remains the default.
- `sector`: optional reporting label, for example residential, industrial, or
  transport. The schema does not hard-code sector names.
- `value_of_lost_load_eur_per_mwh`: optional entity-specific involuntary
  shedding cost. When omitted, `penalties.lost_load_eur_per_mwh` is used.
- `maximum_curtailment_fraction`, `maximum_curtailment_mw`, and
  `voluntary_curtailment_cost_eur_per_mwh`: voluntary curtailment limits and
  utility-loss cost for curtailable or shiftable demand.
- `shift_up_capacity_mw`, `shift_down_capacity_mw`, `shift_window_hours`,
  `rebound_fraction`, and `shift_cost_eur_per_mwh`: energy-conserving load
  shifting. A non-positive shift window conserves energy over the full horizon;
  otherwise conservation is enforced over non-overlapping windows.
- `task_power_capacity_mw`, `task_required_energy_mwh`, `task_start_period`,
  `task_end_period`, and `task_unserved_penalty_eur_per_mwh`: deferrable task
  or EV-charging requirements. `task_end_period` is exclusive.
- `temperature_time_series_key`, `heating_base_temperature_c`,
  `cooling_base_temperature_c`, `heating_sensitivity_mw_per_c`, and
  `cooling_sensitivity_mw_per_c`: deterministic heating- and cooling-degree
  demand adjustments applied before dispatch.

Network dispatch may run in two modes:

- `aggregate_network.network_mode`: `aggregate` preserves the historical
  single-balance formulation. `nodal` enables an integrated lossless DC power
  flow with bus voltage angles, directed line flows, symmetric thermal limits,
  and per-bus balances.
- `aggregate_network.slack_bus_id`: optional slack bus for nodal mode. When
  omitted, the first configured bus is used.
- `lines[].availability_factor` and `lines[].availability_factor_key`: static
  and time-varying line outage multipliers. The effective rating is
  `capacity_mw * availability_factor * availability_factor_key`.

Operating reserves are configured in the optional `reserves` section. When the
section is omitted, reserve requirements are zero and the dispatch formulation
is unchanged. Reserve requirement terms are additive:

- `upward_fixed_mw` and `downward_fixed_mw`: fixed reserve capacity in MW.
- `upward_demand_fraction` and `downward_demand_fraction`: reserve as a
  fraction of source-side demand.
- `upward_renewable_fraction` and `downward_renewable_fraction`: reserve as a
  fraction of renewable availability.
- `largest_online_contingency_fraction`: upward reserve adder based on the
  largest committed thermal capacity.
- `response_duration_hours`: reserve delivery duration used for ramp and
  storage-energy limits.
- `*_shortfall_penalty_eur_per_mw_hour`: explicit shortage penalties for unmet
  reserve requirements.
- `thermal_*_cost_eur_per_mw_hour`, `storage_*_cost_eur_per_mw_hour`,
  `demand_response_upward_cost_eur_per_mw_hour`, and `import_*_cost_eur_per_mw_hour`:
  reserve procurement costs by provider class and product.
- `demand_response_upward_fraction`: maximum upward reserve from curtailable or
  shiftable demand as a fraction of baseline demand, also capped by configured
  curtailment capability.
- `allow_import_reserves`: enables import upward/downward reserve. Imports are
  otherwise excluded from reserve provision.

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
- `configs/portfolio_hydro.yaml` is a schema v2 example with a reservoir unit,
  a run-of-river unit, and synthetic seasonal hydro inflows.
- `configs/portfolio_demand_response.yaml` is a schema v2 example with
  residential fixed demand, industrial shiftable demand, and an EV fleet.
- `configs/portfolio_nodal_three_bus.yaml` is a schema v2 example with a
  three-bus DC network, line-flow exports, and a time-varying line outage.
- `tests/fixtures/invalid_portfolio_missing_bus.yaml` is intentionally invalid
  and is used by the validation tests.

Resolved configurations are serialized into run manifests using a canonical
JSON/YAML-ready representation with absolute paths.
