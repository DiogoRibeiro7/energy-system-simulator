# Model Status

Current corrected baseline: `0.1.1`.

The `0.1.1` release is a correction release for the single-system simulator. It
keeps the current aggregate optimisation formulation while fixing packaging
metadata, finite-horizon commitment semantics, solver-status reporting,
transition-ramp validation, configuration strictness, numerical policy, and DC
power-flow diagnostics. The `Unreleased` line adds typed portfolio
configuration that resolves to this aggregate formulation until later
multi-asset formulation work is implemented.

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

- One thermal unit, one battery, one aggregate solar plant, and one aggregate wind
  farm are supported in the dispatch model.
- The dispatch network is an aggregate transfer-and-loss representation, not a
  nodal OPF.
- The standalone DC power-flow utility solves fixed injections and reports
  overloads; it does not redispatch or enforce line limits.
- Multi-period startup/shutdown trajectories, reserves, hydro, stochastic
  scenarios, and multi-asset portfolio schemas are reserved for later roadmap
  work.
