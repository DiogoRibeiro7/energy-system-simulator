# Model Status

Current corrected baseline: `0.1.1`.

The `0.1.1` release is a correction release for the single-system simulator. It
keeps the current aggregate optimisation formulation while fixing packaging
metadata, finite-horizon commitment semantics, solver-status reporting,
transition-ramp validation, configuration strictness, numerical policy, and DC
power-flow diagnostics. The `Unreleased` line adds typed portfolio
configuration, asset-level renewable availability reporting,
generator-indexed thermal unit commitment, typed fuels, piecewise heat-rate
segments, startup categories, richer thermal emissions accounting, and indexed
storage portfolio, hydro dispatch, and demand-response dispatch.

## Release Decision

The release remains in the `0.1.x` line because the core single-system modelling
surface is unchanged and the `schema_version: 1` field remains optional for
existing valid configurations. Strict rejection of unknown configuration fields
is treated as a correctness fix because previously misspelled fields could be
silently ignored.

## Stress Coverage

Run all committed stress cases with:

```bash
make stress
```

The stress suite writes `outputs/stress/summary.csv` and checks:

- renewable surplus and curtailment penalties;
- import-constrained peak demand with battery discharge;
- aggregate transfer-capacity shedding;
- dispatch-side load shedding;
- thermal startup and shutdown with non-zero transition costs;
- battery charge/discharge cycling with exact exclusivity;
- carbon-price-sensitive import use;
- invalid transition-ramp configuration rejection;
- carry-forward terminal commitment residual obligations.

## Current Limitations

- Thermal dispatch supports multiple committed generators, and storage dispatch
  supports multiple batteries or pumped-storage assets. Hydro dispatch supports
  reservoir and run-of-river units. Demand dispatch supports fixed,
  curtailable, shiftable, deferrable, and EV-charging entities. Imports are
  still represented as a single aggregate resource.
- Solar and wind availability can be evaluated for multiple configured assets,
  with aggregate renewable dispatch allocated back to assets for reporting.
- The dispatch network is an aggregate transfer-and-loss representation, not a
  nodal OPF.
- The standalone DC power-flow utility solves fixed injections and reports
  overloads; it does not redispatch or enforce line limits.
- Multi-period startup/shutdown trajectories beyond category-specific startup
  accounting, reserves, indexed imports, hydro cascade coupling, and stochastic scenarios are
  reserved for later roadmap work.
