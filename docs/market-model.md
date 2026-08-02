# Market Pricing And Settlement

Market analysis is an optional post-dispatch calculation. It does not alter the physical dispatch
unless a future market-specific bidding model explicitly changes the optimisation inputs.

The simulator keeps the following concepts separate:

- **Production cost**: fuel, variable operating cost, no-load cost, startup/shutdown cost, carbon
  cost, import cost, storage throughput cost, and other physical dispatch cost components.
- **Objective penalties**: penalty terms used to make a dispatch feasible or encode preferences,
  including lost load, reserve shortfall, curtailment, and flexible-demand penalties.
- **Energy price**: the marginal price from the relevant power-balance constraint in the pricing
  LP, reported in EUR/MWh.
- **Nodal marginal price or LMP**: a bus-level marginal energy price from nodal balance constraints.
- **Zonal marginal price**: not currently implemented as a separate market aggregation.
- **Uplift or make-whole payment**: out-of-market compensation when market revenue does not recover
  committed production costs.
- **Consumer payment**: metered demand multiplied by the relevant energy price.
- **Generator revenue**: metered output multiplied by the relevant energy price, plus reserve revenue
  where reserve procurement prices are configured.
- **Congestion rent**: transmission shadow value implied by nodal price separation and line flows.
- **Scarcity rent**: value associated with served or unserved energy priced at scarcity levels.

## Pricing Method

Mixed-integer unit commitment dispatch has binary commitment, startup, shutdown, storage-mode, and
startup-category variables. Raw MILP duals are not used or labelled as prices.

For pricing, `MarketAnalyzer` fixes all integer decisions at the accepted dispatch solution and
re-solves the continuous LP. Prices are then read only from equality duals on balance constraints.
The implementation uses SciPy HiGHS equality marginals for an `Ax = b` minimisation problem. Because
the balance rows are written as generation plus unserved energy minus load, the marginal price is:

```text
price_eur_per_mwh = balance_dual / time_step_hours
```

This sign convention is covered by tests for marginal thermal generation, scarcity pricing, fixed
commitment uplift, and congested nodal dispatch.

## Nodal Prices

In nodal mode, each bus balance constraint produces an LMP. The slack-bus LMP is reported as the
energy component, and the congestion component is reported as:

```text
congestion_component = bus_lmp - slack_bus_lmp
```

This decomposition is a DC dispatch approximation. It does not attempt to model losses, ancillary
service co-optimisation, financial transmission rights, or zonal redispatch rules.

## Scarcity Pricing

Load shedding and reserve shortage penalties can set scarcity prices when they are marginal. The
lost-load penalty acts as an explicit cap for energy scarcity prices in the current implementation.
Reserve shortage penalties remain dispatch objective penalties unless a market-specific reserve
pricing rule is configured.

## Settlement

Thermal generator settlement reports:

- energy revenue
- reserve revenue
- variable cost
- startup cost
- no-load cost
- gross margin
- committed cost
- make-whole payment

Consumer payments, generator energy revenue, import energy revenue, congestion rent, uplift, and
the residual are reconciled under the documented accounting rule:

```text
residual = consumer_payment
         - generator_energy_revenue
         - import_energy_revenue
         - congestion_rent
         - uplift
```

Residuals can remain when policy charges, scarcity treatment, reserve settlement, losses, or demand
adjustment rules are not represented as a closed market ledger.

## Market Rule Warning

Electricity market rules vary materially by jurisdiction and product design. These outputs are
economically meaningful diagnostics for the simulator's dispatch formulation, not a claim to
replicate any specific ISO, RTO, pool, bilateral, balancing, or retail settlement rulebook.
