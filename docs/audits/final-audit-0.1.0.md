# Final Audit 0.1.0

This document is a pre-remediation baseline snapshot for the current repository
state used by the remediation roadmap. It records deterministic model outputs
that should change only after a documented semantic change.

## Example Results

The committed example uses `configs/example.yaml` and
`data/example_hourly.csv`.

| Metric | Value |
|---|---:|
| Periods | 336 |
| Objective | EUR 6,326,658.917369775 |
| Renewable share of primary generation | 0.2609934691826571 |
| Total emissions | 21,170.872695455033 tonnes |
| Imports | 707.2247487293002 MWh |
| Unserved energy | 0.0 MWh |

## Formulation Size

| Metric | Value |
|---|---:|
| Continuous variables | 2,352 |
| Integer variables | 1,344 |
| Binary variables | 1,344 |
| Linear constraints | 4,036 |
| Sparse matrix non-zeros | 13,099 |

## Reconciliation Residuals

| Residual | Value |
|---|---:|
| Maximum source balance residual | 5.684341886080802e-14 MW |
| Maximum delivered-demand residual | 0.0 MW |
| Maximum battery-state residual | 2.842170943040401e-14 MWh |
| Objective reconciliation residual | 0.0 EUR |

## Verification Snapshot

Local verification on Python 3.13 after the terminal commitment update reported:

- `54 passed`
- branch-aware coverage: 86.12%
- solver status: optimal
- MIP gap: 0.0

## Regression Fixture

The machine-readable expected values are stored in
`tests/fixtures/baseline_0_1_0.json`. The baseline comparison checks deterministic
model outputs with numeric tolerances and excludes intentionally non-deterministic
manifest fields such as generated timestamps, local paths, and Git commit hashes.

The FIX-03 terminal commitment change added three strict horizon-boundary
constraints to the example formulation. The example objective, emissions,
imports, unserved energy, and renewable share are unchanged.

## Known Defects and Required Follow-up

The following remediation items remain tracked:

- FIX-01: legal-owner and licensing-contact metadata has been resolved in the
  remediation history.
- FIX-02: source-tree and installed-package version authority have been unified
  across source, editable-install, and wheel-install execution modes.
- FIX-03: terminal commitment obligations after the final period are represented
  as explicit strict, carry-forward, and fixed terminal policies.
- FIX-04: additional time-step and initial-state regression coverage is required.
- FIX-05: solver termination mapping still needs an isolated regression suite.
- FIX-06: transition ramp semantics are explicit and covered by a dedicated
  regression matrix.
- FIX-07: configuration parsing does not yet reject every unknown field.
- FIX-08: numerical tolerance policy exists but still needs broader enforcement
  coverage.
- FIX-09: aggregate network and standalone DC power-flow semantics need clearer
  validation and documentation.

## Baseline Update Policy

The baseline fixture may be updated only after a documented semantic change that
intentionally alters deterministic model outputs. The update must include:

- the reason for the changed result;
- the equations or accounting semantics affected;
- before-and-after benchmark values;
- reviewer approval recorded in the change history.

Do not update the fixture merely to make a failing comparison pass.
