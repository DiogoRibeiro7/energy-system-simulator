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

`dashboard/index.html` is a structured local dashboard app generated from the
stable output tables and `summary.json`. It writes separate HTML, CSS,
JavaScript, and data files under `dashboard/` and includes interactive dispatch,
storage, emissions, cost, and diagnostic views. A portable single-file
`dashboard.html` is also written for audit bundles.

Regenerate the app for an existing output directory with:

```bash
poetry run energy-sim dashboard --output-dir outputs/example --app --overwrite
```

Serve it locally with:

```bash
poetry run energy-sim dashboard --output-dir outputs/example --app --serve --overwrite
```

Flexible-electrification demand assets add EV and heat-pump columns to standard
simulation outputs. EV outputs include vehicle energy, delivered mobility energy,
V2G discharge, unmet task/departure energy, and V2G degradation cost. Heat-pump
outputs include COP, thermal heat demand, electrical input, thermal storage,
backup heat, comfort violations, backup heat cost, and backup heat emissions.

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

`energy-sim ac-validate` writes `ac_validation.csv` and
`ac_validation_summary.json` for selected periods only. The tables report
Newton-Raphson convergence, voltage-limit violations, reactive-limit violations,
branch MVA overloads, active losses, and DC-versus-AC active-flow mismatch.

`energy-sim distribution-study` writes distribution-specific files:
`distribution_timeseries.csv`, `distribution_hosting_capacity.csv`, and
`distribution_summary.json`. Distribution columns use the `dist_` prefix and
report radial feeder voltages, active/reactive branch flows, approximate branch
loading, post-solve losses, PV curtailment, substation import/export, reverse
flow, flexible-load reduction, and customer-side versus grid-side battery
throughput.

`energy-sim hydrogen-study` writes `hydrogen_timeseries.csv` and
`hydrogen_summary.json`. Hydrogen outputs use `MWh_LHV` as the canonical
carrier unit and report electrolyser electricity consumption, hydrogen
production, inventory, storage charge/discharge, curtailment, delivered demand,
shortage, reconversion, conversion losses, process emissions, balance residuals,
round-trip efficiency, and a marginal-value proxy.

`energy-sim heat-study` writes `heat_timeseries.csv` and `heat_summary.json`.
Heat outputs report heat by source, CHP electricity, electricity purchases and
exports, electric-boiler and heat-pump consumption, fuel use, emissions, heat
dumping, unmet heat, thermal-storage operation, aggregate heat-network losses,
and independent heat, electricity, and storage balance residuals.

When plots are enabled, the report directory includes dispatch, storage,
thermal, hydro, network, reserve, unserved-energy, cost, emissions, and duration
curve figures. Optional components are represented by empty plots or empty stable
tables instead of failing the report.

Compare two or more output directories with:

```bash
poetry run python scripts/compare_outputs.py outputs/base outputs/scenario --output outputs/comparison.md
```
