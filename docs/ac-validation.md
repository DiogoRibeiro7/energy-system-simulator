# AC Validation Bridge

`ac-validate` checks selected nodal DC-dispatch periods with a nonlinear AC
power-flow solve. It is a validation layer only: it never changes dispatch,
does not run AC optimal power flow, and makes no AC-feasibility claim for
periods that were not selected.

The implementation uses a narrow internal Newton-Raphson solver instead of
adding a heavy AC-OPF dependency. That dependency decision keeps packaging and
auditability stable while covering the bridge requirement: fixed active-power
dispatch is mapped into an AC network and diagnostics are returned explicitly.

Supported AC metadata:

- `buses[]`: voltage magnitude limits, initial voltage magnitude, initial
  voltage angle, and shunt reactive demand.
- `lines[]`: resistance, reactance, line charging, MVA rating, and transformer
  tap ratio. If `ac_reactance_pu` is omitted, validation falls back to
  `1 / susceptance` for continuity with the DC network definition.
- `aggregate_network.ac_base_mva`: per-unit base for AC impedance and power.
- `demand[]`: reactive demand in Mvar per MW of served active demand.
- `thermal_generators[]`, `hydro_units[]`, and `renewable_generators[]`:
  optional reactive-power min/max limits.

Run validation on the default selected periods:

```bash
poetry run energy-sim ac-validate \
  --config configs/portfolio_nodal_three_bus.yaml \
  --output outputs/ac-validation \
  --overwrite
```

Select explicit periods or timestamps:

```bash
poetry run energy-sim ac-validate \
  --config configs/portfolio_nodal_three_bus.yaml \
  --output outputs/ac-validation \
  --period 0 \
  --timestamp 2026-01-01T12:00:00 \
  --overwrite
```

Default period selection includes peak demand, peak renewable availability, and
peak line utilisation. Supplying `--policy` overrides that default; repeat the
flag to combine policies.

Outputs:

- `ac_validation.csv`: convergence status, voltage violations, branch MVA
  overloads, reactive-limit violations, active losses, and mismatch versus DC
  active branch flow for each selected period.
- `ac_validation_summary.json`: selected periods, aggregate validity,
  non-convergence count, maximum violations, and the explicit statement that
  unvalidated periods have no AC-feasibility claim.

The hand-verifiable two-bus reference fixture is stored in
`tests/fixtures/ac_two_bus_reference.json`.
