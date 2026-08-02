# Renewable Model Extensions

Renewable assets keep the original deterministic models as
`availability_model: simple`. Detailed modes add explicit physics and loss
accounting without changing the dispatch interface: the optimiser still receives
one final available-MW series per asset, while `asset_timeseries.csv` reports
the intermediate quantities used to reconcile gross potential to final
availability.

## Solar Models

Use `availability_model: simple` for teaching examples and fast scenario
screening. It preserves the legacy irradiance, ambient-temperature,
performance-ratio, temperature-derating, and AC-capacity clipping calculation.

Use `availability_model: detailed` when separate DC and AC sizing, inverter
efficiency, degradation, maintenance, snow, soiling, and availability schedules
matter. The detailed solar model reports:

- `dc_potential_mw`
- `ac_potential_mw`
- `clipping_loss_mw`
- `temperature_loss_mw`
- `other_derating_loss_mw`
- `available_mw`

`time_series_key` is interpreted as plane-of-array irradiance by default. If the
input is global horizontal irradiance, set:

```yaml
irradiance_basis: global_horizontal
transposition_model: isotropic_fixed_tilt
tilt_degrees: 30.0
albedo: 0.2
```

The transposition is deterministic and intentionally simple. It does not invent
missing meteorological data. Snow and soiling are either scalar configured loss
fractions or explicit input columns containing period loss fractions in `[0, 1]`.

## Wind Models

Use `availability_model: simple` for classroom and planning cases where a cubic
partial-load curve is enough.

Use `availability_model: detailed` when hub-height adjustment, air-density
correction, wake losses, electrical losses, and maintenance schedules should be
tracked separately while keeping the simple cubic curve.

Use `availability_model: power_curve` when a tabulated manufacturer curve is
available. Curve points are provided inline:

```yaml
availability_model: power_curve
turbine_count: 50
turbine_rated_capacity_mw: 3.0
power_curve:
  - wind_speed_m_s: 0.0
    power_mw: 0.0
  - wind_speed_m_s: 4.0
    power_mw: 0.0
  - wind_speed_m_s: 12.0
    power_mw: 3.0
  - wind_speed_m_s: 25.0
    power_mw: 3.0
```

Wind-curve speeds must be strictly increasing and powers must be non-negative
and no larger than the rated capacity. Runtime wind-speed values outside the
curve range are rejected instead of extrapolated.

The wind profile reports:

- `hub_height_wind_speed_m_s`
- `gross_potential_mw`
- `high_wind_shutdown_loss_mw`
- `wake_loss_mw`
- `electrical_loss_mw`
- `availability_loss_mw`
- `available_mw`

## Model Selection

For teaching, prefer `simple` because every step can be inspected by hand and
the minimum required weather data stays small.

For planning studies, prefer `detailed` when asset availability, maintenance,
and explicit loss schedules are important to the scenario outcome.

For engineering approximation, prefer `power_curve` for wind if a validated
curve is available, and prefer detailed solar with plane-of-array irradiance
when the irradiance dataset already supplies it.

## Example Figures

Regenerate the renewable model figures with:

```bash
poetry run python scripts/plot_renewable_models.py
```

The script writes:

- `docs/figures/solar_derating_decomposition.png`
- `docs/figures/wind_power_curve.png`
