# Baseline Audit

> Historical snapshot: this audit records the pre-remediation 0.1.0 baseline.
> See `docs/model-status.md` for the current corrected baseline status.

This audit records the 0.1.0 baseline before broadening the model scope. It
does not change the mathematical formulation.

## Module Boundaries and Data Flow

The current package is organized around explicit stages:

- `config.py` parses YAML into typed dataclasses and validates cross-field
  constraints.
- `data.py` loads the hourly CSV, parses UTC timestamps, validates required
  columns, rejects non-finite values, and checks constant spacing.
- `generation/solar.py` and `generation/wind.py` convert weather drivers into
  renewable availability.
- `network/distribution.py` converts end-user demand into source-side gross
  demand using an aggregate loss and transfer-capacity approximation.
- `dispatch/unit_commitment.py` builds and solves the single-thermal-unit MILP
  with SciPy HiGHS.
- `simulation/engine.py` coordinates loading, generation, network preparation,
  dispatch, and summary accounting.
- `reporting/report.py` writes CSV, JSON, plots, and a machine-readable run
  manifest.
- `network/dc_power_flow.py` provides a standalone DC power-flow solver that is
  not integrated into dispatch.
- `cli.py` exposes `validate` and `simulate` commands.

The end-to-end flow is:

```text
YAML config
  -> typed ModelConfig
  -> CSV input data
  -> solar and wind availability
  -> aggregate network demand preparation
  -> single-unit commitment dispatch
  -> simulation accounting
  -> timeseries.csv, summary.json, manifest.json, optional plots
```

## Current MILP Decision Variables

For each period `t`, the dispatch formulation contains:

| Variable | Type | Unit | Meaning |
|---|---|---:|---|
| `renewable_used_mw[t]` | continuous | MW | Source-side renewable generation used |
| `thermal_output_mw[t]` | continuous | MW | Thermal generator output |
| `thermal_on[t]` | binary | 1 | Thermal commitment state |
| `thermal_startup[t]` | binary | 1 | Startup indicator |
| `thermal_shutdown[t]` | binary | 1 | Shutdown indicator |
| `battery_charge_mw[t]` | continuous | MW | Battery charging power |
| `battery_discharge_mw[t]` | continuous | MW | Battery discharging power |
| `battery_soc_mwh[t]` | continuous | MWh | Battery state of charge |
| `imports_mw[t]` | continuous | MW | Source-side imports |
| `source_load_shed_mw[t]` | continuous | MW | Source-equivalent dispatch shedding |

For the committed 336-period example, this gives 2,352 continuous variables
and 1,008 binary variables.

## Objective Terms

The optimization objective minimizes:

- thermal variable generation cost;
- thermal no-load cost;
- startup cost;
- shutdown cost;
- import energy cost;
- battery throughput cost on charge and discharge energy;
- carbon cost for thermal generation and imports;
- renewable curtailment penalty;
- source-equivalent dispatch load-shedding penalty.

The simulation engine adds fixed network-capacity shedding cost after the
dispatch solve because aggregate network-capacity shedding is calculated before
the MILP.

## Constraints

The current MILP includes:

- source-side power balance for every period;
- renewable-use upper bounds;
- thermal minimum and maximum output linked to commitment;
- startup and shutdown transition identity;
- startup and shutdown mutual-exclusion inequality;
- ramp-up and ramp-down constraints with startup and shutdown relaxations;
- battery state-of-charge transition;
- battery power and energy bounds;
- import bounds;
- source-load-shedding bounds;
- rolling minimum up-time and down-time constraints;
- minimum final battery state of charge.

The example formulation has 3,361 linear constraints and 11,081 sparse matrix
non-zero entries.

## Accounting Conventions

Dispatch variables are source-side. End-user demand is first capped by aggregate
network transfer capacity, converted to source-side gross demand by dividing by
network efficiency, and then passed to the dispatch model.

Network-capacity shedding is end-user-side and occurs before optimization.
Dispatch load shedding is optimized as source-side `source_load_shed_mw` and is
converted back to delivered load shedding with network efficiency.

Reported `served_demand_mw` is end-user demand less both network-capacity
shedding and delivered dispatch shedding. Reported network losses are calculated
on source power sent to load after dispatch shedding.

## Assumptions and Simplifications

- One aggregate solar plant, one wind farm, one thermal generator, and one
  battery are modeled.
- The network in dispatch is an aggregate transfer-capacity and loss model.
- DC power flow is available only as a standalone experiment helper.
- Input data are hourly in the committed example, although the loader accepts a
  configurable constant time step.
- Thermal minimum up/down values are currently integer configuration fields and
  behave as periods, despite being named as hours.
- Battery charging and discharging are not mutually exclusive in the MILP.
- Solver success is currently required; feasible non-optimal incumbents are not
  represented as a separate public result state.
- Cost accounting is summarized, but not yet decomposed into a reconciled
  component table.

## Known Correctness Risks and Technical Debt

- Time-step conversion risks remain for duration semantics, especially thermal
  minimum up/down fields that are named as hours but used as period counts.
- Initial-condition handling only covers `initial_on` and `initial_output_mw`;
  it does not track residual minimum up/down obligations.
- Battery charge/discharge exclusivity is not enforced exactly.
- Solver termination handling does not distinguish all relevant states such as
  time-limited feasible incumbent, infeasible, unbounded, and solver error.
- Objective accounting lacks an automated reconciliation between solver
  objective and reported components.
- Energy reconciliation metrics are not yet reported per period.
- Startup and shutdown ramp relaxations use maximum output as a broad relaxation
  rather than explicit startup and shutdown ramp limits.
- DC power flow is not integrated into dispatch.
- Public API boundaries are narrow but not formally versioned.

## Baseline Measurements

Measured locally on Windows with Python 3.13 and SciPy HiGHS:

| Metric | Value |
|---|---:|
| Periods | 336 |
| Continuous variables | 2,352 |
| Integer variables | 1,008 |
| Linear constraints | 3,361 |
| Matrix non-zero count | 11,081 |
| Build time | about 0.006 s |
| Solve time | about 0.13-0.16 s |
| Solver termination | Optimal |
| Objective | EUR 6,326,658.917 |
| MIP gap | 0.0 |
| Test coverage | 84.28% |

Timing is machine-dependent. Formulation size and objective are expected to
remain deterministic for the committed example unless model behavior changes.

## Dependency Graph for Next Milestones

```text
Numerical tolerances and invariant checks
  -> duration semantics and initial-state corrections
  -> objective and energy reconciliation
  -> typed configuration schema
  -> asset identifiers and portfolio registry
  -> multi-unit thermal commitment
  -> fuel, heat-rate, and emissions detail
  -> generalized storage
  -> nodal network integration
  -> reserves, reliability, scenarios, and market accounting
```

The safest next step is correctness and accounting hardening before adding
portfolio or network scope.
