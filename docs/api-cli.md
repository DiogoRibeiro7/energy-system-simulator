# CLI and Python API

The supported command-line entry point is `energy-sim`.

## Commands

- `validate --config CONFIG`: validate configuration and referenced input data.
- `validate-config --config CONFIG`: validate configuration only.
- `validate-data --config CONFIG`: validate referenced input data.
- `migrate-config --config CONFIG [--output PATH]`: migrate legacy configs.
- `simulate --config CONFIG`: run deterministic or configured rolling simulation.
- `rolling-horizon --config CONFIG`: run with `rolling_horizon.enabled=true`.
- `run-scenarios --experiment EXPERIMENT`: run a scenario experiment.
- `scenario-experiment --experiment EXPERIMENT`: alias for scenario experiments.
- `run-experiment --study STUDY`: run a registered research experiment study.
- `reproduce-experiment --manifest MANIFEST`: verify hashes and rerun a recorded
  research experiment.
- `analyze-experiment --study STUDY`: regenerate research tables, figures, and report.
- `reliability-study --config CONFIG`: run a sequential Monte Carlo reliability study.
- `security-check --config CONFIG --output DIR`: evaluate explicit N-1
  post-contingency security for nodal dispatch.
- `frequency-check --config CONFIG --output DIR`: evaluate post-dispatch
  frequency adequacy proxies.
- `ac-validate --config CONFIG --output DIR`: validate selected nodal dispatch
  periods with AC power flow.
- `distribution-study --problem PROBLEM --output DIR`: run a standalone radial
  distribution-feeder dispatch or hosting-capacity study.
- `hydrogen-study --problem PROBLEM --output DIR`: run a standalone hydrogen
  production, storage, demand, and reconversion study.
- `capacity-planning --problem PROBLEM`: run a capacity-expansion problem YAML.
- `compare-outputs OUT1 OUT2 --output REPORT.md`: compare output directories.
- `export-model --config CONFIG --output MODEL.lp`: export formulation LP.
- `export-formulation --config CONFIG --output MODEL.lp`: alias for formulation export.
- `prepare-data --spec SPEC`: build a canonical input snapshot.
- `capabilities`: show version, commands, exit codes, and solver backend.

Use `--dry-run` with `simulate` or `rolling-horizon` to validate inputs and
report formulation dimensions without calling the solver. Use `--set PATH=VALUE`
for command-line overrides; paths are validated against the typed configuration
and recorded in `manifest.json`.

Output directories and files are not overwritten unless `--overwrite` or
`--resume` is supplied.

## Exit Codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 2 | Invalid configuration |
| 3 | Invalid data |
| 4 | Infeasible model |
| 5 | Solver or execution failure |
| 6 | Partial feasible result accepted by policy |

## Python API

The package root exports the supported library lifecycle:

```python
import pandas as pd
import energy_system_simulator as ess

config = ess.load_model_config("configs/example.yaml")
data = pd.read_csv("data/example_hourly.csv").head(24)

ess.validate_model_config(config)
validated = ess.validate_data(data, time_step_hours=config.simulation.time_step_hours)
problem = ess.build_model(config, validated)
result = ess.solve(config, validated)
```

For a file-based run with standard outputs:

```python
import energy_system_simulator as ess

result = ess.run_simulation(
    "configs/example.yaml",
    create_plots=False,
    overwrite=True,
)
```

Public exceptions inherit from `EnergySystemError`; configuration, data, and
optimisation failures use `ConfigurationError`, `DataValidationError`, and
`OptimisationError`.

Nodal security diagnostics are available through `evaluate_security`:

```python
import energy_system_simulator as ess

config = ess.load_model_config("configs/portfolio_nodal_three_bus.yaml")
result = ess.solve(config)
security = ess.evaluate_security(config, result)
security.write("outputs/security")
```

Frequency adequacy diagnostics are similarly available:

```python
import energy_system_simulator as ess

config = ess.load_model_config("configs/frequency_low_inertia.yaml")
result = ess.solve(config)
frequency = ess.evaluate_frequency_adequacy(config, result)
frequency.write("outputs/frequency")
```

AC validation is available for selected nodal dispatch periods:

```python
import energy_system_simulator as ess

config = ess.load_model_config("configs/portfolio_nodal_three_bus.yaml")
result = ess.solve(config)
validation = ess.validate_ac_power_flow(config, result)
validation.write("outputs/ac-validation")
```

Radial distribution-feeder studies use a separate problem object:

```python
import energy_system_simulator as ess

problem = ess.load_distribution_problem("configs/distribution_radial_feeder.yaml")
result = ess.run_distribution_study(problem, mode="hosting_capacity")
result.write("outputs/hosting")
```

Hydrogen subsystem studies also use a separate problem object:

```python
import energy_system_simulator as ess

problem = ess.load_hydrogen_problem("configs/hydrogen_system.yaml")
result = ess.run_hydrogen_study(problem)
result.write("outputs/hydrogen")
```

## 1.0 Compatibility Notes

`scenario-experiment` and `export-formulation` are compatibility aliases. New
automation should prefer `run-scenarios` and `export-model`. Legacy aggregate
output files remain supported, but the versioned `*_v1.csv` files are the stable
audit surface for the 1.0 line.
