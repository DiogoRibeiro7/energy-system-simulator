# Input data contract

The simulator reads a CSV table with one row per interval.

## Required columns

- `timestamp`: ISO-8601 timestamp. Values must be unique and strictly increasing after parsing.
- `demand_mw`: finite and non-negative.
- `irradiance_w_m2`: finite and non-negative.
- `ambient_temperature_c`: finite.
- `wind_speed_m_s`: finite and non-negative.

## Temporal rules

- Intervals must be equally spaced.
- The spacing must match `simulation.time_step_hours` within numerical tolerance.
- Missing intervals must be inserted or otherwise resolved before simulation.
- Time-zone-aware timestamps are recommended. The example data use UTC.

## Units

The code does not infer or convert units. Supplying kW where MW are expected will produce numerically valid but physically incorrect results.
