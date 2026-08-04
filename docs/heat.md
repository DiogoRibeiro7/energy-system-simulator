# District Heat and CHP

`heat-study` runs a standalone multi-energy district-heat model with coupled
electricity accounting. It is intended for heat-sector studies where the heat
balance, thermal storage, and CHP operating region need to be audited separately
from the core electricity dispatch formulation.

## Assets

The schema supports:

- Heat-only boilers with heat capacity, fuel efficiency, fuel cost, variable
  heat cost, and fuel emissions.
- Electric boilers with electrical input and heat output efficiency.
- Heat pumps with period-specific COP profiles.
- Thermal storage with inventory, charge/discharge rates, standing losses,
  initial inventory, and terminal inventory.
- CHP units represented by convex operating-region vertices. Each vertex defines
  electric output, heat output, and fuel input, and the model chooses convex
  combinations of those vertices.

The committed example is:

```bash
poetry run energy-sim heat-study --problem configs/chp_heat_system.yaml --output outputs/heat --overwrite
```

## Balances

Heat demand is served on the end-use side while the model accounts for aggregate
network delivery efficiency \(\eta_n\):

\[
q^{boiler}_t+q^{eb}_t+q^{hp}_t+q^{chp}_t+q^{dis}_t+q^{unmet}_t/\eta_n
=d^H_t/\eta_n+q^{charge}_t+q^{dump}_t.
\]

The electricity balance accounts for native electricity demand, electric
boilers, heat pumps, CHP power, imports, shortage, and exports:

\[
p^{chp}_t+p^{buy}_t+p^{short}_t
=d^E_t+p^{eb}_t+p^{hp}_t+p^{export}_t.
\]

Exports are limited to CHP production, so purchased electricity cannot be
arbitraged through the export variable.

Thermal storage evolves independently:

\[
s_t=(1-\lambda)^{\Delta t}s_{t-1}+q^{charge}_t\Delta t-q^{dis}_t\Delta t.
\]

For each CHP unit \(u\), operating vertices \(v\) define a convex hull:

\[
p^{chp}_{u,t}=\sum_v \lambda_{u,v,t}p_{u,v},
\quad
q^{chp}_{u,t}=\sum_v \lambda_{u,v,t}q_{u,v},
\quad
f^{chp}_{u,t}=\sum_v \lambda_{u,v,t}f_{u,v},
\quad
\sum_v \lambda_{u,v,t}\le 1.
\]

This prevents the unit from independently selecting impossible maximum heat and
maximum electric output.

## Outputs

The output directory contains:

- `heat_timeseries.csv`: heat and electricity balances, source outputs,
  electric-heating consumption, CHP fuel, storage operation, network losses,
  emissions, and residuals.
- `heat_summary.json`: heat by source, electricity from CHP, purchased/exported
  electricity, unmet heat, heat dumping, storage throughput, fuel by source,
  total emissions, balance residuals, and a coupling statement.

Fuel costs and emissions are assigned to fuel input once and are not split or
double-counted across CHP heat and electricity products.
