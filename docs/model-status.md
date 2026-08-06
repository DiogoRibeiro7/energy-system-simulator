# Model Status

Current stable release: `1.1.1`.

The `1.1.1` release is the stable research and teaching release for the current
roadmap scope. It includes dashboard visualization outputs, dispatch result-frame
consolidation for wide reports, typed portfolio configuration, asset-level renewable
availability reporting, generator-indexed thermal unit commitment, typed fuels,
piecewise heat-rate segments, startup categories, richer emissions accounting,
indexed storage, hydro dispatch, demand response, nodal DC dispatch, reserves,
rolling horizon, reliability, stochastic dispatch, market settlement, capacity
expansion, scenario experiments, public-data provenance, reporting diagnostics,
the Iberian case study, and research experiment manifests.

## Release Decision

The 1.0 line freezes supported configuration schemas 1 and 2 and output schema
version 1. Strict rejection of unknown configuration fields is part of the
public contract because misspelled fields otherwise create non-reproducible
studies.

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

- Imports are still represented as a single aggregate resource in the main
  dispatch path.
- Nodal dispatch uses a linear lossless DC approximation, not AC power flow.
- The standalone DC power-flow utility solves fixed injections and reports
  overloads; redispatch belongs to the integrated nodal dispatch mode.
- Multi-period startup/shutdown trajectories beyond category-specific startup
  accounting, security-constrained unit commitment, hydro cascade coupling, and
  detailed distribution-network modelling are reserved for later roadmap work.
