# Iberian Approximation Case Study

This case study is a two-zone Portugal-Spain approximation for one summer week.
It is designed for reproducible model exercises, not for operational replay.

## Study Design

- Period: 2024-06-24T00:00:00Z for 168 hourly periods.
- Topology: two buses, `pt` and `es`, connected by line `pt-es`.
- Flow sign convention: positive `line_flow_mw__pt-es` is Portugal to Spain.
- Demand and installed capacities are calibrated to Ember 2024 country aggregates.
- Hourly demand, weather, hydro inflow, and prices are deterministic approximations.

## Aggregate Validation

| Area | Ember demand TWh | Annualized snapshot TWh | Difference % |
|---|---:|---:|---:|
| Portugal | 57.74 | 57.74 | -0.000 |
| Spain | 270.71 | 270.71 | 0.000 |

The demand snapshot is normalized to reproduce annual demand if the modeled week is
repeated across a year. Generation and flow totals are model outcomes and
are compared against Ember annual totals only as a reasonableness check.

## Baseline Results

- Solver status: `optimal`
- Objective: EUR 165,873,543
- Unserved energy: 0.000 MWh
- Renewable share: 51.76%
- CO2 emissions: 547,909 tonnes

| Quantity | Annualized model TWh | Ember 2024 TWh | Difference % |
|---|---:|---:|---:|
| Demand | 328.45 | 328.45 | 0.0 |
| Solar | 102.91 | 65.38 | 57.4 |
| Wind | 70.02 | 76.54 | -8.5 |
| Hydro | 34.33 | 49.30 | -30.4 |
| Gas | 75.18 | 57.97 | 29.7 |
| Nuclear | 51.63 | 54.53 | -5.3 |

The summer-week weather approximation intentionally overstates annualized solar
generation when repeated across a full year; this is a limitation of using one
compact representative week rather than a weather-year sample.

## Cross-Border Flow

- Maximum Portugal-to-Spain flow: 1,362.6 MW
- Maximum Spain-to-Portugal flow: 3,500.0 MW

## Sensitivities

| Scenario | Objective EUR | Emissions tonnes | Unserved MWh | Renewable share |
|---|---:|---:|---:|---:|
| solar-plus-25 | 151,104,118 | 501,011 | 0.000 | 56.24% |
| interconnection-plus-50 | 165,873,094 | 547,907 | 0.000 | 51.76% |
| gas-retirement-30 | 165,874,227 | 547,911 | 0.000 | 51.76% |
| hydro-drought-proxy | 192,874,292 | 643,794 | 0.000 | 51.72% |
| battery-plus-50 | 162,132,402 | 532,396 | 0.000 | 51.84% |
| carbon-price-120 | 189,256,261 | 412,767 | 0.000 | 51.72% |

## Limitations

- This is not a full nodal representation of the Portuguese or Spanish grids.
- Thermal fleets are grouped by technology and represented as a few aggregate units.
- Hydro inflows are stylized weekly profiles calibrated only qualitatively.
- External trade is represented as a single import resource at Spain plus a PT-ES line.
- Weather and price profiles are deterministic approximations, not historical reanalysis.
