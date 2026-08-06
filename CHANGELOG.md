# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows semantic versioning.

## [Unreleased]

## [1.1.1] - 2026-08-06

### Fixed

- Consolidated dispatch result frames between accounting phases to avoid Pandas
  DataFrame fragmentation warnings during wide result-table generation.

## [1.1.0] - 2026-08-06

### Added

- Structured local dashboard app output under `dashboard/`, with separate
  `index.html`, `styles.css`, `app.js`, `data.json`, and `data.js` files.
- `energy-sim dashboard --app` to regenerate the structured dashboard app for
  an existing output directory.
- `energy-sim dashboard --serve` to serve the structured dashboard locally with
  the Python standard library.
- Self-contained `dashboard.html` output for portable audit bundles.

### Changed

- Standard simulation reports now link to both the structured dashboard app and
  the portable single-file dashboard.
- Reporting documentation now describes dashboard app generation and serving.

## [1.0.0] - 2026-08-04

### Added

- Explicit post-contingency N-1 security checks for nodal dispatch, with
  separate contingency diagnostics and LODF validation helpers.
- Post-dispatch frequency adequacy proxies for inertia, RoCoF, primary
  response, fast frequency response, and synthetic inertia diagnostics.
- Optional AC power-flow validation bridge for selected nodal dispatch periods.
- Standalone radial distribution-feeder studies with DER dispatch, voltage and
  thermal constraints, battery-side accounting, and hosting-capacity outputs.
- Flexible electrification demand services for EV fleets with V2G and heat
  pumps with thermal storage, COP profiles, backup heat, comfort penalties, and
  service-level reporting.
- Standalone hydrogen subsystem studies with electrolyser production, lossy
  storage inventory, exogenous demand, reconversion, explicit emissions
  assumptions, and `MWh_LHV` balance reporting.
- Standalone district-heat and CHP subsystem studies with heat-only boilers,
  electric boilers, heat pumps, thermal storage, convex CHP operating regions,
  heat-network losses, fuel/emissions reconciliation, and coupling outputs.
- Typed portfolio configuration schema, legacy config migration, canonical
  resolved configuration serialization, and portfolio validation fixtures.
- Multi-asset renewable availability orchestration with tidy asset time-series
  output and asset-level renewable reconciliation.
- Generator-indexed thermal unit commitment with per-unit output, commitment,
  startup, shutdown, availability, cost, and emissions reporting.
- Typed fuel definitions, piecewise heat-rate segment variables, fuel-price
  time series, startup categories, fuel-input accounting, and non-CO2 thermal
  emissions diagnostics.
- Indexed storage portfolio dispatch for batteries and pumped storage,
  including exact operating modes, self-discharge, per-asset terminal SOC,
  availability factors, ramp limits, and degradation accounting.
- Reservoir and run-of-river hydro dispatch with natural inflows, turbine
  release, spill, environmental releases, evaporation, terminal reservoir
  policies, water-value accounting, and a synthetic hydro portfolio example.
- Sector demand and demand-response dispatch with fixed, curtailable,
  shiftable, deferrable, and EV-charging entities, entity-specific lost-load
  costs, temperature-sensitive demand preprocessing, task completion penalties,
  and a demand-response portfolio example.
- Integrated nodal DC dispatch with bus balances, line-flow limits, slack-bus
  handling, and overload diagnostics.
- Operating reserve requirements, reserve shortfall accounting, and reserve
  diagnostics in versioned output tables.
- Detailed renewable model extensions for solar derating, wind power curves,
  availability factors, soiling, snow, wake, and electrical losses.
- Rolling-horizon simulation with deterministic state transfer, checkpointing,
  resume support, and full-horizon comparisons for small cases.
- Sequential Monte Carlo reliability studies with seeded outage trajectories,
  adequacy metrics, and confidence intervals.
- Scenario-based uncertainty and stochastic dispatch utilities with
  value-of-information benchmarks.
- Fixed-commitment market pricing and settlement outputs.
- Single-year capacity-expansion planning with representative-period weights.
- Scenario experiments with declarative overrides, sweeps, grids, manifests,
  resumable runs, and aggregate sensitivity tables.
- Public-data adapters that produce canonical local input snapshots with
  checksums and provenance manifests.
- Mathematical verification benchmarks, scaling benchmarks, solver backend
  abstraction, LP export, and benchmark reporting.
- Versioned reporting tables, data dictionaries, diagnostics, comparison
  reports, and generated plots.
- Hardened CLI and public Python API with capabilities reporting, validated
  overrides, dry-run support, and package install smoke tests.
- Reproducible Iberian approximation case study.
- Research experiment framework with pre-specified metrics, reproducibility
  manifests, generated Markdown and LaTeX tables, figure metadata, and an
  example storage-value study.
- 1.0 release documentation index, compatibility matrix, release checklist,
  and release-validation report.

### Changed

- The public release surface is now versioned as `1.0.0`.
- Configuration parsing is strict: unknown fields, duplicate YAML keys,
  unsupported schema versions, and invalid references fail clearly.
- Supported configuration schemas are frozen at aggregate schema 1 and typed
  portfolio schema 2 for the 1.0 line. Output tables remain frozen at output
  schema version 1.

### Deprecated

- `scenario-experiment` remains as an alias for `run-scenarios`.
- `export-formulation` remains as an alias for `export-model`.
- Legacy aggregate output files remain for compatibility; versioned `*_v1.csv`
  tables are the preferred audit interface.

### Breaking Changes From 0.1.0

- Misspelled and unknown YAML keys are rejected instead of silently ignored.
- Output directories and report files are not overwritten unless `--overwrite`
  or `--resume` is supplied.
- Solver failures, infeasible models, and partial feasible incumbents are mapped
  to explicit domain statuses and CLI exit codes.
- Legacy aggregate configurations are still supported, but resolved runs are
  internally represented as typed portfolios.

### Model Scope

- The simulator is intended for transparent research, teaching, policy
  experiments, and reproducible optimisation case studies.
- It is not a production grid-operations tool and does not model AC power flow,
  frequency dynamics, protection systems, transient stability, or detailed
  distribution-network operation.

### Validation

- `make verify` passed locally on Windows with Python 3.13.5.
- Package source distribution and wheel build and install cleanly in a fresh
  smoke environment.

## [0.1.1] - 2026-07-31

### Added

- Repository governance and maintenance files.
- Business Source License 1.1 licensing documentation and validation.
- Baseline audit, reproducibility checks, formulation benchmark, run manifest,
  and coverage enforcement.
- Numerical tolerances, duration-aware unit commitment constraints, exact
  battery operating-mode exclusivity, detailed objective accounting, solver
  diagnostics, and energy reconciliation metrics.
- Audited baseline fixture, comparison command, and remediation roadmap.
- Release owner, citation, and commercial-contact metadata validation.
- Authoritative version helper, CLI version output, and package install smoke
  checks.
- Explicit thermal terminal commitment modes, residual terminal-state reporting,
  and strict minimum up/down enforcement at the horizon boundary.
- Regression coverage for 15-minute, 30-minute, and hourly time-step scaling,
  residual initial-state obligations, ramp rates, costs, emissions, and battery
  terminal modes.
- Typed solver-result interpretation with explicit domain statuses, raw backend
  diagnostics, and safe objective-gap reporting.
- Explicit startup and shutdown ramp semantics, validation for impossible
  transition limits, and hand-computable transition-ramp regressions.
- Strict YAML configuration parsing with schema versioning, duplicate-key
  detection, unknown-field suggestions, and JSON validation output.
- Immutable numerical policy for feasibility, integrality, objective, cleanup,
  reporting, timestamp, and DC power-balance tolerances with residual summaries.
- DC power-flow overload diagnostics and clearer aggregate-versus-nodal network
  semantics.

## [0.1.0] - 2026-07-31

### Added

- Initial energy system simulator package.
- Example configuration and hourly input data.
- Documentation for architecture, model formulation, and data contracts.
- CI workflow for linting, type checking, and tests.
