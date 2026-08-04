# Energy System Simulator

A transparent, optimisation-based simulator for an electrical energy system with renewable generation, dispatchable thermal plants, storage, reservoir hydro, demand response, imports, distribution constraints, and hourly end-user consumption.

The model is designed for research, teaching, and policy experiments. It uses explicit physical and economic constraints rather than machine-learning methods.

## Main capabilities

- Solar generation with simple and detailed DC/AC derating models.
- Wind generation with simple cubic curves or validated tabulated power curves.
- Thermal unit commitment with minimum and maximum output, ramp limits,
  start-up and shutdown decisions, minimum up/down times, terminal commitment
  policy, fuel definitions, piecewise heat-rate segments, startup categories,
  fuel cost, and emissions.
- Storage portfolio dispatch for batteries and pumped storage with exact
  charge/discharge modes, state-of-charge limits, efficiency losses,
  self-discharge, terminal policies, throughput cost, and degradation metrics.
- Reservoir and run-of-river hydro dispatch with energy-equivalent water
  balances, inflows, release, spill, evaporation, terminal storage policies,
  environmental releases, and optional terminal water value.
- Sector demand portfolios with fixed, curtailable, shiftable, deferrable, and
  EV-charging demand, including sector-specific lost-load costs and explicit
  demand-response accounting.
- Aggregated distribution losses and transfer-capacity constraints.
- Optional electricity imports.
- Renewable curtailment and involuntary load shedding.
- Mixed-integer optimisation using `scipy.optimize.milp`.
- Rolling-horizon simulation with deterministic checkpoint resume.
- Sequential Monte Carlo reliability studies with seeded outage trajectories.
- Explicit post-contingency N-1 security checks for nodal dispatch.
- Frequency adequacy proxies for inertia, RoCoF, and response scarcity.
- Optional AC power-flow validation for selected nodal dispatch periods.
- Standalone radial distribution-feeder studies with rooftop PV, behind-the-meter
  batteries, flexible load, voltage limits, branch ratings, and hosting capacity.
- Scenario-based stochastic dispatch with value-of-information benchmarks.
- Optional post-dispatch market prices and settlements from fixed-commitment LP duals.
- Single-year capacity-expansion planning with representative-period weights.
- Scenario experiments with declarative overrides, sweeps, grids, and resume manifests.
- Research experiment directories with pre-specified metrics, reproducibility
  manifests, generated tables, figure metadata, and limitations templates.
- Public-data adapters that produce local canonical snapshots with provenance manifests.
- CSV results, JSON summary metrics, and diagnostic plots.
- A small DC power-flow module for network experiments.

## Mathematical core

For each period \(t\), the source-side balance is

\[
R_t + P_t + H_t + D_t^{\mathrm{bat}} + I_t + L_t
= G_t + C_t^{\mathrm{bat}},
\]

where \(G_t\) is demand adjusted for distribution losses, \(R_t\) is renewable generation used, \(P_t\) is thermal output, \(H_t\) is hydro generation, \(I_t\) is imported power, and \(L_t\) is source-equivalent load shedding.

The optimiser minimizes operating cost, start-up and shutdown cost, import cost, battery degradation, renewable curtailment, emissions cost, and the value of lost load.

## Repository structure

```text
energy-system-simulator/
├── configs/                  Example YAML configuration
├── data/                     Example hourly input data
├── docs/                     Mathematical and architecture documentation
├── scripts/                  Reproducible data-generation scripts
├── src/energy_system_simulator/
│   ├── dispatch/             Mixed-integer unit commitment
│   ├── generation/           Solar and wind models
│   ├── network/              Distribution and DC power flow
│   ├── reporting/            Metrics and figures
│   ├── simulation/           End-to-end simulation engine
│   └── storage/              Battery model helpers
└── tests/                    Unit and integration tests
```

The full documentation index is in [`docs/index.md`](docs/index.md).
Renewable model details, validation rules, and example figure generation are
documented in `docs/renewable-models.md`.
Rolling-horizon configuration and state-transfer behaviour are documented in
`docs/rolling-horizon.md`. Probabilistic adequacy studies are documented in
`docs/reliability.md`. Scenario-based stochastic dispatch is documented in
`docs/stochastic-dispatch.md`. Market pricing and settlement outputs are
documented in `docs/market-model.md`. Capacity-expansion planning is documented
in `docs/capacity-expansion.md`. Scenario experiments are documented in
`docs/scenario-experiments.md`. Research experiment structure and reproduction
commands are documented in `docs/research-experiments.md`. Public-data adapters
and provenance manifests are documented in `docs/public-data-adapters.md`.
N-1 post-contingency security checks are documented in
`docs/security-constrained-dispatch.md`. Frequency adequacy proxy checks are
documented in `docs/frequency-adequacy.md`. AC validation is documented in
`docs/ac-validation.md`. Distribution feeder studies are documented in
`docs/distribution-feeder.md`.

## Requirements

- Python 3.11 or later
- Poetry 1.8 or later

## Installation

> Commercial production use is not granted by the public licence.

```bash
poetry install
```

## Run the example

Quick start:

```bash
poetry run energy-sim validate --config configs/example.yaml
poetry run energy-sim simulate --config configs/example.yaml --overwrite
```

Run the schema v2 hydro portfolio example:

```bash
poetry run energy-sim validate --config configs/portfolio_hydro.yaml
poetry run energy-sim simulate --config configs/portfolio_hydro.yaml --overwrite
```

Run the demand-response portfolio example:

```bash
poetry run energy-sim validate --config configs/portfolio_demand_response.yaml
poetry run energy-sim simulate --config configs/portfolio_demand_response.yaml --overwrite
```

Run the nodal DC network example:

```bash
poetry run energy-sim validate --config configs/portfolio_nodal_three_bus.yaml
poetry run energy-sim simulate --config configs/portfolio_nodal_three_bus.yaml --overwrite
```

Check explicit N-1 security for the same nodal example:

```bash
poetry run energy-sim security-check --config configs/portfolio_nodal_three_bus.yaml --output outputs/security --overwrite
```

Run the low-inertia frequency adequacy example:

```bash
poetry run energy-sim frequency-check --config configs/frequency_low_inertia.yaml --output outputs/frequency --overwrite
```

Validate selected nodal dispatch periods against AC power flow:

```bash
poetry run energy-sim ac-validate --config configs/portfolio_nodal_three_bus.yaml --output outputs/ac-validation --overwrite
```

Run the radial distribution feeder example and a hosting-capacity study:

```bash
poetry run energy-sim distribution-study --problem configs/distribution_radial_feeder.yaml --output outputs/distribution --overwrite
poetry run energy-sim distribution-study --problem configs/distribution_radial_feeder.yaml --output outputs/hosting --mode hosting-capacity --overwrite
```

Equivalent module invocation:

```bash
poetry run python -m energy_system_simulator simulate --config configs/example.yaml --overwrite
```

Run the complete storage-value research experiment:

```bash
poetry run energy-sim run-experiment --study experiments/storage_value --overwrite --no-plots
```

Results are written to `outputs/example/`:

- `timeseries.csv`
- `asset_timeseries.csv`
- `summary.json`
- `manifest.json`
- `dispatch.png`
- `battery_soc.png`

## Model selection

Use the deterministic dispatch model for operational scenario analysis,
teaching examples, and transparent policy experiments. Use rolling horizon for
longer horizons where full-horizon MILPs are too large. Use reliability studies
when the question concerns outage-driven adequacy risk. Use stochastic dispatch
when the question concerns forecast uncertainty or value of information. Use
capacity expansion for fixed representative-period planning studies.

## Input data contract

The hourly CSV must contain:

| Column | Unit | Meaning |
|---|---:|---|
| `timestamp` | ISO-8601 | Start of the interval |
| `demand_mw` | MW | End-user electrical demand |
| `irradiance_w_m2` | W/m² | Global horizontal irradiance |
| `ambient_temperature_c` | °C | Ambient temperature |
| `wind_speed_m_s` | m/s | Hub-height wind speed |

See [`docs/data-contract.md`](docs/data-contract.md) for validation rules.

Configuration supports the legacy aggregate schema and the newer typed portfolio
schema. See [`docs/configuration.md`](docs/configuration.md) for schema versions,
validation rules, examples, and `energy-sim migrate-config`.

## Quality checks

```bash
make verify
```

Individual checks:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
poetry run python scripts/validate_licensing.py
poetry run python scripts/validate_version.py
poetry run python scripts/validate_release_readiness.py
poetry run python scripts/check_example_data.py
poetry run python scripts/validate_examples.py
poetry run python scripts/run_verification_benchmarks.py
poetry run python scripts/benchmark_example.py
poetry run python scripts/benchmark_scaling.py
poetry run python scripts/compare_baseline.py
```

Mathematical verification cases and CI benchmark budgets are documented in
[`docs/verification.md`](docs/verification.md).
Solver backend capabilities, LP export, and scaling benchmarks are documented in
[`docs/solver-backends.md`](docs/solver-backends.md).
Versioned output tables, diagnostics, reports, and output comparison are documented in
[`docs/reporting.md`](docs/reporting.md).
CLI commands, exit codes, dry-run behavior, and the public Python API are documented in
[`docs/api-cli.md`](docs/api-cli.md).
Python and solver compatibility are documented in
[`docs/compatibility.md`](docs/compatibility.md).
The reproducible Portugal-Spain approximation case study lives in
[`case_studies/iberia`](case_studies/iberia).

See [docs/development.md](docs/development.md) for clean-checkout, editable
install, and wheel-install smoke workflows.

Run the committed stress suite:

```bash
make stress
```

## Capability Matrix

| Area | Current support |
|---|---|
| Dispatch | Multi-unit thermal commitment with startup, shutdown, ramp, minimum-duration, terminal policy, must-run, availability-factor, fuel, heat-rate segment, and startup-category constraints |
| Renewables | Multiple configured solar and wind assets with asset-level availability, used output, and curtailment reporting |
| Storage | Multiple batteries and pumped-storage assets with exact charge/discharge modes, independent power limits, self-discharge, terminal SOC policies, and degradation accounting |
| Hydro | Reservoir and run-of-river assets with inflows, release, spill, energy-equivalent reservoir state, evaporation losses, environmental releases, terminal policies, and water-value accounting |
| Demand | Demand entities with fixed, curtailable, shiftable, deferrable, and EV-charging modes, temperature-sensitive preprocessing, sector-specific lost-load values, and demand-response cost accounting |
| Network | Aggregate delivery losses and transfer-capacity shedding in dispatch |
| Markets | Optional dual-based energy prices, nodal LMPs, congestion rent, consumer payments, generator revenue, and make-whole uplift |
| Planning | Continuous single-year capacity expansion for generation, storage, interconnectors, and transmission with weighted representative periods |
| Experiments | Declarative scenario sweeps and grids plus registered research studies with hashed manifests, pre-specified comparisons, generated tables, and figure metadata |
| Data | Local public-data adapters for demand and weather snapshots with UTC timestamps, validation reports, checksums, and provenance |
| DC flow | Standalone fixed-injection DC power flow with overload diagnostics |
| Outputs | Aggregate CSV time series, tidy renewable, thermal, storage, and hydro asset time series, fuel and emissions accounting, JSON summary, manifest, plots, benchmark, baseline, and stress comparison table |

## Modelling scope

The default simulation uses a single aggregated distribution network. This is appropriate for system planning and policy analysis. The repository also contains a DC power-flow solver that can be integrated into zonal or nodal studies. Detailed low-voltage AC analysis, frequency dynamics, protection systems, and transient stability are outside the current scope.

See [docs/model-status.md](docs/model-status.md) for current release status and
limitations.

## Citation

If you use this software in research, cite Energy System Simulator using the
metadata in [CITATION.cff](CITATION.cff).

## License

This project is source-available under the Business Source License 1.1
(`BUSL-1.1`).

Personal, educational, academic, research, evaluation, testing, development,
and other non-production uses are permitted under the terms described in
[LICENSE](LICENSE) and [LICENSING.md](LICENSING.md).

Private commercial production use requires a separate paid commercial licence.
See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

Each released version converts to the Apache License 2.0 on its applicable
Change Date.
