# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows semantic versioning.

## [Unreleased]

### Added

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
