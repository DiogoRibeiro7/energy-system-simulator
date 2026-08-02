# Scenario Experiments

Scenario experiments make policy and engineering studies reproducible without
editing configuration files by hand. They are exposed through
`energy-sim run-scenarios` and the Python helper `run_experiment_file`.

## Experiment YAML

An experiment has a base configuration, an output directory, and one or more
scenario definitions:

```yaml
base_config: ../configs/example.yaml
output_directory: ../outputs/carbon-price-experiment
workers: 2
resume: false

scenarios:
  - id: reference
    overrides: {}

sweeps:
  - parameter: penalties.carbon_price_eur_per_tonne
    values: [0.0, 50.0, 100.0]

grid:
  parameters:
    solar.capacity_mw: [80.0, 120.0]
    battery.energy_capacity_mwh: [120.0, 240.0]
```

`scenarios` are explicit named cases. `sweeps` are one-factor experiments.
`grid.parameters` is expanded as a Cartesian product.

Override paths are validated against the typed configuration dataclasses. Tuple
assets can be addressed with numeric selectors, for example
`portfolio.thermal_generators[0].config.variable_cost_eur_per_mwh` in schema v2
files.

## Command Line

```bash
poetry run energy-sim run-scenarios \
  --experiment examples/experiments/policy_sweeps.yaml \
  --workers 2 \
  --no-plots
```

The runner refuses to write into a non-empty experiment output directory unless
`--resume` is supplied or the YAML has `resume: true`. Resume mode skips only
scenarios whose `scenario_manifest.json` exists and whose override checksum
matches the requested scenario.

## Outputs

Each scenario gets a normal simulation output directory containing:

- `config.yaml`
- `timeseries.csv`
- `asset_timeseries.csv`
- `summary.json`
- `manifest.json`
- `scenario_manifest.json`

The experiment directory also contains:

- `experiment_manifest.json`
- `summary.csv`

The aggregate `summary.csv` includes scenario identifiers, labels, parameter
columns, solver status, objective, cost components, emissions, renewable share,
curtailment, unserved energy, thermal starts, storage cycles, and network or
reserve metrics when those are available in the simulation summary. Failed
scenarios are recorded as `ok = false`; the run then raises an error instead of
silently omitting them.

Scenario IDs are generated from canonical parameter values and a short SHA-256
digest. This keeps IDs stable across machines and worker counts.

## Sensitivities And Plots

`finite_difference_sensitivity` runs a central finite difference for one scalar
configuration value. It is labelled as a numerical sensitivity of a non-smooth
MILP value function, not as an analytic derivative.

Plotting helpers operate only on aggregate tables:

- `plot_response_curve`
- `plot_heatmap`
- `plot_tradeoff_frontier`

They do not read hidden simulator state.
