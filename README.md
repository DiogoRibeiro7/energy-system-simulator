# Energy System Simulator

[![CI](https://github.com/DiogoRibeiro7/energy-system-simulator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/energy-system-simulator/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/DiogoRibeiro7/energy-system-simulator)](https://github.com/DiogoRibeiro7/energy-system-simulator/releases/latest)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](docs/compatibility.md)
[![License](https://img.shields.io/badge/license-BUSL--1.1-orange)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-blue)](https://mypy-lang.org/)

A transparent, optimisation-based simulator for an electrical energy system with
renewable generation, dispatchable thermal plants, storage, reservoir hydro,
demand response, imports, distribution constraints, and hourly end-user
consumption.

The model is designed for research, teaching, and policy experiments. It uses
explicit physical and economic constraints rather than machine-learning methods,
so every result can be traced back to a documented equation and a versioned
input.

**Contents:** [Quick start](#quick-start) ·
[Python API](#python-api) ·
[Capabilities](#capabilities) ·
[Example studies](#example-studies) ·
[Model selection](#model-selection) ·
[Mathematical core](#mathematical-core) ·
[Input data](#input-data-contract) ·
[Documentation](#documentation) ·
[Development](#development) ·
[Scope and limitations](#modelling-scope-and-limitations) ·
[Citation](#citation) ·
[License](#license)

## Quick start

Requirements:

- Python 3.11, 3.12, or 3.13
- Poetry 1.8 or later

> Commercial production use is not granted by the public licence. See
> [License](#license).

```bash
git clone https://github.com/DiogoRibeiro7/energy-system-simulator.git
cd energy-system-simulator
poetry install
```

Validate and run the example system:

```bash
poetry run energy-sim validate --config configs/example.yaml
poetry run energy-sim simulate --config configs/example.yaml --overwrite
```

Equivalent module invocation:

```bash
poetry run python -m energy_system_simulator simulate --config configs/example.yaml --overwrite
```

Results are written to `outputs/example/`, including:

| File | Content |
| --- | --- |
| `timeseries.csv` | Aggregate hourly dispatch |
| `asset_timeseries.csv` | Tidy per-asset renewable, thermal, storage, and hydro series |
| `summary.json` | Cost, emissions, curtailment, and adequacy metrics |
| `manifest.json` | Inputs, versions, and hashes for reproducibility |
| `dispatch.png`, `battery_soc.png` | Diagnostic plots |

Versioned output tables, diagnostics, and reports are described in
[`docs/reporting.md`](docs/reporting.md).

### Dashboard

Generate and serve the structured dashboard for those outputs:

```bash
poetry run energy-sim dashboard --output-dir outputs/example --app --serve --overwrite
```

Open the local URL printed by the command, usually `http://127.0.0.1:8765`. If
that port is already in use, choose another one with `--port 8010`.

Each simulation has its own output directory. For the flexible-electrification
example:

```bash
poetry run energy-sim simulate --config configs/flexible_electrification.yaml --overwrite
poetry run energy-sim dashboard --output-dir outputs/flexible-electrification --app --serve --overwrite
```

## Python API

The package root exports the supported library lifecycle:

```python
import energy_system_simulator as ess

result = ess.run_simulation(
    "configs/example.yaml",
    create_plots=False,
    overwrite=True,
)
```

Lower-level steps (`load_model_config`, `validate_data`, `build_model`, `solve`),
public exceptions, and CLI exit codes are documented in
[`docs/api-cli.md`](docs/api-cli.md).

## Capabilities

### Generation and dispatch

- **Thermal unit commitment** — multi-unit commitment with minimum and maximum
  output, ramp limits, start-up and shutdown decisions, minimum up/down times,
  must-run and availability factors, terminal commitment policy, fuel
  definitions, piecewise heat-rate segments, startup categories, fuel cost, and
  emissions.
- **Renewables** — multiple solar and wind assets with asset-level availability,
  used output, and curtailment reporting. Solar offers simple and detailed DC/AC
  derating models; wind offers simple cubic curves or validated tabulated power
  curves.
- **Storage** — batteries and pumped storage with exact charge/discharge modes,
  independent power limits, state-of-charge limits, efficiency losses,
  self-discharge, terminal policies, throughput cost, and degradation metrics.
- **Hydro** — reservoir and run-of-river dispatch with energy-equivalent water
  balances, inflows, release, spill, evaporation, environmental releases,
  terminal storage policies, and optional terminal water value.
- **Imports, curtailment, and load shedding** — optional electricity imports,
  renewable curtailment, and involuntary load shedding.
- **Solver** — mixed-integer optimisation using `scipy.optimize.milp`, with LP
  export for external solvers.

### Demand and electrification

- **Sector demand portfolios** — fixed, curtailable, shiftable, deferrable, and
  EV-charging demand with temperature-sensitive preprocessing, sector-specific
  lost-load costs, and explicit demand-response accounting.
- **Flexible electrification** — EV fleets with availability and V2G, plus heat
  pumps with COP, thermal storage, backup heat, and comfort bounds.

### Network, security, and adequacy

- Aggregated distribution losses and transfer-capacity constraints.
- Nodal DC dispatch and a standalone fixed-injection DC power-flow module with
  overload diagnostics.
- Explicit post-contingency N-1 security checks for nodal dispatch.
- Frequency adequacy proxies for inertia, RoCoF, and response scarcity.
- Optional AC power-flow validation for selected nodal dispatch periods.
- Standalone radial distribution-feeder studies with rooftop PV,
  behind-the-meter batteries, flexible load, voltage limits, branch ratings, and
  hosting capacity.

### Sector coupling

- **Hydrogen** — standalone subsystem studies with electrolyser production,
  `MWh_LHV` storage, exogenous demand, reconversion, losses, and emissions
  assumptions.
- **District heat and CHP** — heat-only boilers, electric boilers, heat pumps,
  thermal storage, heat-network losses, and convex CHP operating regions.

### Uncertainty, markets, and planning

- Rolling-horizon simulation with deterministic checkpoint resume.
- Sequential Monte Carlo reliability studies with seeded outage trajectories.
- Scenario-based stochastic dispatch with value-of-information benchmarks.
- Optional post-dispatch market prices and settlements from fixed-commitment LP
  duals: energy prices, nodal LMPs, congestion rent, consumer payments,
  generator revenue, and make-whole uplift.
- Continuous single-year capacity expansion for generation, storage,
  interconnectors, and transmission with weighted representative periods.

### Experiments, data, and outputs

- Scenario experiments with declarative overrides, sweeps, grids, and resume
  manifests.
- Research experiment directories with pre-specified metrics, reproducibility
  manifests, generated tables, figure metadata, and limitations templates.
- Public-data adapters that produce local canonical snapshots with UTC
  timestamps, validation reports, checksums, and provenance manifests.
- CSV time series, fuel and emissions accounting, JSON summary metrics,
  manifests, diagnostic plots, a dashboard, and benchmark, baseline, and stress
  comparison tables.

## Example studies

Every configuration under [`configs/`](configs/) can be checked first with
`energy-sim validate --config <file>`.

```bash
# Schema v2 hydro portfolio
poetry run energy-sim simulate --config configs/portfolio_hydro.yaml --overwrite

# Demand-response portfolio
poetry run energy-sim simulate --config configs/portfolio_demand_response.yaml --overwrite

# Flexible electrification (EV fleets and heat pumps)
poetry run energy-sim simulate --config configs/flexible_electrification.yaml --overwrite

# Nodal DC network
poetry run energy-sim simulate --config configs/portfolio_nodal_three_bus.yaml --overwrite
```

Security, frequency, and AC checks on nodal dispatch:

```bash
# Explicit N-1 security check
poetry run energy-sim security-check --config configs/portfolio_nodal_three_bus.yaml --output outputs/security --overwrite

# Low-inertia frequency adequacy
poetry run energy-sim frequency-check --config configs/frequency_low_inertia.yaml --output outputs/frequency --overwrite

# AC power-flow validation of selected periods
poetry run energy-sim ac-validate --config configs/portfolio_nodal_three_bus.yaml --output outputs/ac-validation --overwrite
```

Standalone subsystem studies:

```bash
# Radial distribution feeder, then a hosting-capacity study
poetry run energy-sim distribution-study --problem configs/distribution_radial_feeder.yaml --output outputs/distribution --overwrite
poetry run energy-sim distribution-study --problem configs/distribution_radial_feeder.yaml --output outputs/hosting --mode hosting-capacity --overwrite

# Hydrogen subsystem
poetry run energy-sim hydrogen-study --problem configs/hydrogen_system.yaml --output outputs/hydrogen --overwrite

# District heat and CHP
poetry run energy-sim heat-study --problem configs/chp_heat_system.yaml --output outputs/heat --overwrite
```

Research experiments and case studies:

```bash
# Complete storage-value research experiment
poetry run energy-sim run-experiment --study experiments/storage_value --overwrite --no-plots
```

The reproducible Portugal-Spain approximation case study lives in
[`case_studies/iberia`](case_studies/iberia). Run `energy-sim capabilities` to
list every command and exit code.

## Model selection

| Question | Approach | Documentation |
| --- | --- | --- |
| Operational scenario analysis, teaching examples, transparent policy experiments | Deterministic dispatch | [`docs/model.md`](docs/model.md) |
| Long horizons where a full-horizon MILP is too large | Rolling horizon | [`docs/rolling-horizon.md`](docs/rolling-horizon.md) |
| Outage-driven adequacy risk | Reliability study | [`docs/reliability.md`](docs/reliability.md) |
| Forecast uncertainty or value of information | Stochastic dispatch | [`docs/stochastic-dispatch.md`](docs/stochastic-dispatch.md) |
| Fixed representative-period investment planning | Capacity expansion | [`docs/capacity-expansion.md`](docs/capacity-expansion.md) |

## Mathematical core

For each period $t$, the source-side balance is

$$
R_t + P_t + H_t + D_t^{\mathrm{bat}} + I_t + L_t
= G_t + C_t^{\mathrm{bat}},
$$

where $G_t$ is demand adjusted for distribution losses, $R_t$ is renewable
generation used, $P_t$ is thermal output, $H_t$ is hydro generation,
$D_t^{\mathrm{bat}}$ and $C_t^{\mathrm{bat}}$ are storage discharge and charge,
$I_t$ is imported power, and $L_t$ is source-equivalent load shedding.

The optimiser minimizes operating cost, start-up and shutdown cost, import cost,
battery degradation, renewable curtailment, emissions cost, and the value of
lost load. The full formulation is in [`docs/model.md`](docs/model.md).

## Input data contract

The hourly CSV must contain:

| Column | Unit | Meaning |
| --- | ---: | --- |
| `timestamp` | ISO-8601 | Start of the interval |
| `demand_mw` | MW | End-user electrical demand |
| `irradiance_w_m2` | W/m² | Global horizontal irradiance |
| `ambient_temperature_c` | °C | Ambient temperature |
| `wind_speed_m_s` | m/s | Hub-height wind speed |

See [`docs/data-contract.md`](docs/data-contract.md) for validation rules.

Configuration supports the legacy aggregate schema and the newer typed portfolio
schema. See [`docs/configuration.md`](docs/configuration.md) for schema versions,
validation rules, examples, and `energy-sim migrate-config`.

## Documentation

The full documentation index is in [`docs/index.md`](docs/index.md).

| Topic | Documents |
| --- | --- |
| Core model | [Architecture](docs/architecture.md) · [Mathematical model](docs/model.md) · [Configuration](docs/configuration.md) · [Data contract](docs/data-contract.md) |
| Interfaces | [CLI and Python API](docs/api-cli.md) · [Reporting and outputs](docs/reporting.md) · [Solver backends and LP export](docs/solver-backends.md) |
| Generation and operations | [Renewable models](docs/renewable-models.md) · [Rolling horizon](docs/rolling-horizon.md) · [Reliability](docs/reliability.md) · [Stochastic dispatch](docs/stochastic-dispatch.md) · [Market model](docs/market-model.md) · [Capacity expansion](docs/capacity-expansion.md) |
| Network and security | [N-1 security checks](docs/security-constrained-dispatch.md) · [Frequency adequacy](docs/frequency-adequacy.md) · [AC validation](docs/ac-validation.md) · [Distribution feeder](docs/distribution-feeder.md) |
| Sector coupling | [Hydrogen](docs/hydrogen.md) · [District heat and CHP](docs/heat.md) |
| Studies and data | [Scenario experiments](docs/scenario-experiments.md) · [Research experiments](docs/research-experiments.md) · [Public-data adapters](docs/public-data-adapters.md) · [Data provenance](docs/data-provenance-inventory.md) · [Iberian case study](case_studies/iberia) |
| Quality and releases | [Verification](docs/verification.md) · [Compatibility](docs/compatibility.md) · [Development](docs/development.md) · [Model status](docs/model-status.md) · [Changelog](CHANGELOG.md) |

## Repository structure

```text
energy-system-simulator/
├── case_studies/             Reproducible case studies (Iberia)
├── configs/                  Example YAML configurations
├── data/                     Example hourly input data
├── docs/                     Mathematical and architecture documentation
├── examples/                 Example scripts, experiments, and stress cases
├── experiments/              Registered research experiments
├── licensing/                Machine-readable release and licence metadata
├── scripts/                  Validation, benchmark, and data-generation scripts
├── src/energy_system_simulator/
│   ├── dispatch/             Mixed-integer unit commitment
│   ├── generation/           Solar and wind models
│   ├── network/              Distribution and DC power flow
│   ├── reporting/            Metrics, figures, and dashboard
│   ├── simulation/           End-to-end simulation engine
│   ├── storage/              Battery model helpers
│   ├── api.py, cli.py        Public Python API and `energy-sim` CLI
│   └── *.py                  Study modules (planning, market, security,
│                             frequency, hydrogen, heat, feeder, scenarios, ...)
└── tests/                    Unit and integration tests
```

## Development

Run the complete verification suite:

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

Run the committed stress suite:

```bash
make stress
```

Mathematical verification cases and CI benchmark budgets are documented in
[`docs/verification.md`](docs/verification.md). See
[`docs/development.md`](docs/development.md) for clean-checkout, editable
install, and wheel-install smoke workflows.

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For help, see [SUPPORT.md](SUPPORT.md);
to report a vulnerability, see [SECURITY.md](SECURITY.md).

## Modelling scope and limitations

The default simulation uses a single aggregated distribution network. This is
appropriate for system planning and policy analysis. Nodal studies use a linear
lossless DC approximation; the AC validation bridge and the frequency adequacy
proxies are post-dispatch checks, not dynamic simulations. Detailed low-voltage
AC analysis, frequency dynamics, protection systems, and transient stability are
outside the current scope.

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
