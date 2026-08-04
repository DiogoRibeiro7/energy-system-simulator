# Hydrogen Subsystem

`hydrogen-study` runs a standalone multi-period hydrogen subsystem model for
sector-coupling and long-duration storage studies. It is separate from the core
electricity dispatch formulation and uses renewable surplus and electricity
deficit profiles as exogenous interfaces.

The canonical internal hydrogen unit is `MWh_LHV`. Electricity remains in MW or
MWh. Efficiencies must be positive and no greater than one, so the model cannot
create free energy through inconsistent heating-value conversions.
The Python API exposes `hydrogen_kg_to_mwh_lhv` and
`hydrogen_mwh_lhv_to_kg` helpers for boundary conversion.

## Assets

The YAML problem schema contains four asset groups:

- `electrolyser`: electrical input capacity, `MWh_LHV` output efficiency,
  optional always-on minimum load, optional ramp limit, and variable cost.
- `storage`: `MWh_LHV` inventory capacity, charge/discharge rates, standing
  losses, initial inventory, and minimum terminal inventory.
- `demand`: exogenous hydrogen demand in `MWh_LHV` and shortage penalty.
- `reconverter`: fuel-cell or hydrogen-turbine capacity, conversion efficiency,
  variable cost, and direct process-emission assumptions.

Run the committed example with:

```bash
poetry run energy-sim hydrogen-study --problem configs/hydrogen_system.yaml --output outputs/hydrogen --overwrite
```

## Equations

For each period \(t\), electrolyser output is tied to electrical input:

\[
h^{prod}_t = \eta^{el} p^{el}_t \Delta t
\]

Hydrogen carrier balance allows either current production or storage discharge
to serve demand, reconversion, storage charging, or curtailment:

\[
h^{prod}_t+h^{dis}_t
=h^{store}_t+h^{del}_t+h^{rec}_t+h^{curt}_t
\]

Inventory is lossy:

\[
s_t=(1-\lambda)^{\Delta t}s_{t-1}+h^{store}_t-h^{dis}_t
\]

Demand shortage and electricity-deficit shortage are explicit slacks:

\[
h^{del}_t+h^{short}_t=d^H_t
\]

\[
p^{rec}_t+p^{unserved}_t=d^E_t
\]

Reconversion uses configured efficiency:

\[
p^{rec}_t = \eta^{rec}h^{rec}_t / \Delta t
\]

## Outputs

The output directory contains:

- `hydrogen_timeseries.csv`: period-level production, storage, demand,
  reconversion, losses, emissions, and balance residuals.
- `hydrogen_summary.json`: objective value, electricity consumed, hydrogen
  produced, delivered, curtailed, shortage, ending inventory, reconverted
  electricity, round-trip efficiency, losses, emissions, and a marginal-value
  proxy.

Hydrogen is not reported as emission-free by default. The summary includes an
explicit emissions statement, and reconversion emissions are calculated from the
configured `emission_tonnes_per_mwh_h2` value.
