# Reporting

Simulation outputs include the legacy files `timeseries.csv`,
`asset_timeseries.csv`, `summary.json`, and `manifest.json`, plus versioned
auditable tables:

For the 1.0 line, versioned reporting tables use output schema version 1. New
columns may be added only through explicit versioned-table policy; existing
columns retain their meaning, units, and aggregation semantics.

| Table | Purpose |
|---|---|
| `system_timeseries_v1.csv` | System-level dispatch, demand, residuals, emissions, reserves, and network quantities by period. |
| `asset_timeseries_v1.csv` | Tidy asset-level time series with `asset_id`, `asset_type`, `variable`, unit, and value. |
| `bus_timeseries_v1.csv` | Nodal bus angles and balance residuals when nodal dispatch is enabled. Empty with stable columns otherwise. |
| `line_timeseries_v1.csv` | Line flows, utilisation, and overload residuals when lines exist. Empty with stable columns otherwise. |
| `cost_components_v1.csv` | Objective cost components in EUR. The sum reproduces `summary.objective_eur`. |
| `emissions_v1.csv` | Thermal, import, and total emissions by period. The total reproduces summary emissions. |
| `reliability_events_v1.csv` | Periods with unserved energy events. Empty with stable columns when no events occur. |
| `solver_diagnostics_v1.csv` | Solver status, backend status, gaps, runtime, node count, and message. |
| `summary_metrics_v1.csv` | Flattened scalar summary metrics with units and aggregation rules. |

`data_dictionary.csv` contains one row for every generated table column, with
unit, sign convention, aggregation rule, and a short description.

`diagnostics.json` records automatic findings for balance residuals, bound
violations, simultaneous incompatible modes, load shedding, terminal-state
violations, solver status, missing timestamps, and asset availability.

`report.md` is a self-contained Markdown index that highlights solver status,
objective value, unserved energy, balance diagnostics, table files, diagnostics,
and plot references.

`energy-sim security-check` writes separate security diagnostics instead of
changing the standard simulation tables. Its `security_contingencies.csv` and
`security_summary.json` files report post-contingency feasibility, emergency
actions, redispatch, binding overloaded elements, and security cost while
preserving the base-case objective and cost accounting.

`energy-sim frequency-check` also writes separate diagnostics. Its
`frequency_adequacy.csv` and `frequency_summary.json` files report inertia,
largest credible loss, RoCoF, primary response, fast frequency response,
synthetic inertia, shortfalls, and scarcity periods without changing standard
simulation cost tables.

When plots are enabled, the report directory includes dispatch, storage,
thermal, hydro, network, reserve, unserved-energy, cost, emissions, and duration
curve figures. Optional components are represented by empty plots or empty stable
tables instead of failing the report.

Compare two or more output directories with:

```bash
poetry run python scripts/compare_outputs.py outputs/base outputs/scenario --output outputs/comparison.md
```
