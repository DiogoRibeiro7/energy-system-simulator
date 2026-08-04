# Distribution Feeder Studies

`distribution-study` runs a standalone radial feeder model. It does not reuse the
transmission DC network schema or output column names.

## Model

The feeder model uses a balanced single-phase equivalent and linearised DistFlow
constraints. Active and reactive branch flows are represented explicitly, voltage
is tracked as squared per-unit magnitude, and branch limits use a linear diamond
approximation of apparent-power loading:

- `P_ij - sum(P_jk) = net active demand at bus j`
- `Q_ij - sum(Q_jk) = fixed reactive demand at bus j`
- `V_j^2 = V_i^2 - 2 * (r_ij * P_ij + x_ij * Q_ij) / base_MVA`
- `abs(P_ij) + abs(Q_ij) <= rating_MVA`, implemented as four linear inequalities.

Losses are reported after the solve as a quadratic diagnostic. They are not fed
back into the linear optimisation objective or balances.

## Inputs

Use a separate YAML file such as `configs/distribution_radial_feeder.yaml` with:

- `buses`: radial bus IDs, fixed active/reactive load profiles, and voltage bounds.
- `branches`: parent-child feeder branches with per-unit resistance/reactance and
  MVA rating.
- `rooftop_pv`: installed capacity, availability profile, curtailment cost, and
  optional `hosting_capacity_max_mw`.
- `batteries`: behind-the-meter batteries with `side: customer_side` or
  `side: grid_side`; reports keep those categories separate.
- `flexible_loads`: controllable demand reduction limits and cost.

## CLI

```bash
energy-sim distribution-study --problem configs/distribution_radial_feeder.yaml --output outputs/distribution --overwrite
energy-sim distribution-study --problem configs/distribution_radial_feeder.yaml --output outputs/hosting --mode hosting-capacity --overwrite
```

## Outputs

The command writes:

- `distribution_timeseries.csv`: `dist_`-prefixed voltage, branch flow, DER,
  battery, flexible-load, and substation columns.
- `distribution_hosting_capacity.csv`: incremental hosting capacity by PV site.
- `distribution_summary.json`: model family, approximation notes, min/max voltage,
  branch loading, losses, curtailment, reverse flow, and battery throughput split
  between customer-side and grid-side assets.

Hosting-capacity mode maximises incremental rooftop PV while enforcing voltage
and thermal limits. By default it allows no curtailment of incremental DER output;
set `hosting.max_curtailment_fraction` to document an explicit curtailment policy.
