# Frequency Adequacy Proxies

`frequency-check` evaluates planning-grade frequency adequacy proxies for a
solved dispatch. It does not run electromagnetic transient simulation and does
not model governor dynamics, protection systems, or spatial frequency modes.

The checker uses configured asset metadata:

- Thermal and hydro units can provide synchronous inertia in MW*s and sustained
  primary response in MW when online.
- Storage units can provide fast frequency response in MW and optional
  synthetic inertia in MW*s, capped by discharge power and state of charge.
- The largest credible infeed loss is the maximum of `frequency.credible_loss_mw`
  and the configured fraction of the largest online thermal, hydro, or import
  infeed.

For each period, total inertia is:

\[
I_t = I_t^{sync} + I_t^{synthetic}.
\]

The RoCoF proxy is:

\[
\mathrm{RoCoF}_t = \frac{f_0 L_t}{2 I_t},
\]

where \(f_0\) is nominal frequency and \(L_t\) is the largest credible loss. If
loss is positive and inertia is zero, RoCoF is infinite.

The quasi-steady response requirement is:

\[
Q_t = \max(0, L_t - D \Delta f),
\]

where \(D\) is demand damping in MW/Hz and \(\Delta f\) is the allowed
quasi-steady frequency deviation. Sustained primary response and fast frequency
response are reported separately, then summed for the response shortfall check.

Run the low-inertia example:

```bash
poetry run energy-sim frequency-check \
  --config configs/frequency_low_inertia.yaml \
  --output outputs/frequency-low-inertia-check \
  --overwrite
```

The command writes:

- `frequency_adequacy.csv`: one row per period with inertia, largest loss,
  RoCoF, response provision by class, shortfalls, and binding limitation.
- `frequency_summary.json`: aggregate adequacy status, scarcity-period count,
  binding period, and a statement that this is not dynamic frequency simulation.

These diagnostics are separate from base-case energy and reserve costs. They are
intended for screening commitment and planning studies, not for operational
stability certification.
