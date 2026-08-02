# Stochastic Dispatch

Stochastic dispatch is exposed as a separate model family through
`StochasticDispatch`. Deterministic `SimulationEngine` and `UnitCommitment`
behavior is unchanged.

Each `StochasticScenario` has an ID, probability, renewable availability, gross
demand, and optional scenario-specific import prices, hydro inflows, line
availability, import availability, thermal availability, and storage converter
availability. Probabilities must be non-negative and sum to one within the
configured tolerance.

The implementation uses a two-stage scenario workflow:

- First-stage thermal commitment decisions are shared across scenarios for
  `commitment_horizon_periods`.
- Second-stage dispatch, storage operation, curtailment, imports, reserves, and
  shedding are re-solved for each scenario with that first-stage commitment
  fixed.
- Candidate first-stage commitments are generated from the expected-value
  solution and the wait-and-see scenario solutions, then evaluated by expected
  cost. This keeps stochastic dispatch separate from the deterministic
  formulation while enforcing non-anticipativity for the shared commitment
  prefix.

The objective is expected operating cost. With CVaR enabled, candidate selection
minimizes:

```text
expected_cost + cvar_weight * CVaR_alpha(scenario_cost)
```

The result reports scenario costs, expected objective, optional CVaR, shared
first-stage commitments, and value-of-information benchmarks:

- Expected-value solution objective: deterministic solve on probability-weighted
  inputs.
- Expected result of using the expected-value solution: expected scenario cost
  after fixing the expected-value first-stage commitment.
- Wait-and-see expected objective: probability-weighted cost when every scenario
  can choose its own commitment.
- Value of the stochastic solution: expected-value evaluation minus stochastic
  evaluation.
- Expected value of perfect information: stochastic evaluation minus
  wait-and-see evaluation.

Synthetic scenario generation is deterministic from an explicit seed and only
perturbs supplied base arrays. Scenario generation stays outside the optimiser.

Model size grows roughly linearly with the number of evaluated scenario
subproblems and candidate first-stage commitments. For a deterministic problem
with `N` variables and `M` constraints, evaluating `S` scenarios and `C`
candidate commitments solves about `S * (C + 1)` deterministic-sized MILPs plus
one expected-value MILP. In practice, start with 3-10 scenarios for interactive
analysis, use 10-50 for batch sensitivity studies, and reserve larger scenario
sets for simplified portfolios or stronger solver infrastructure.
