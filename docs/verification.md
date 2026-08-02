# Verification and Benchmark Suite

This project keeps mathematical verification separate from ordinary unit tests in
`tests/benchmarks/`. These tests use tiny systems with closed-form or
independently enumerated optima, so failures identify formulation regressions
rather than scenario calibration changes.

## Numerical Tolerances

Verification cases use `energy_system_simulator.constants.DEFAULT_NUMERICAL_POLICY`.
Power-flow, dispatch, reserve, and demand-balance quantities are checked with
absolute tolerances around the primal feasibility policy. Objective and
reconciliation checks use the objective reconciliation policy. Runtime values are
never snapshotted; they are only compared with broad performance budgets.

## Equation Mapping

| Component | Verification case | Mathematical check |
|---|---|---|
| Economic dispatch merit order | `test_single_period_economic_dispatch_matches_hand_merit_order` | Least-cost units are dispatched in ascending variable cost until demand is met. |
| Single-unit commitment | `test_single_unit_commitment_matches_independent_enumeration` | Optimiser objective and binary state match brute-force enumeration over all on/off schedules. |
| Multi-unit startup costs | `test_two_unit_startup_cost_case_matches_independent_enumeration` | Startup and production tradeoffs match an independent enumeration of both units. |
| Storage energy balance | `test_battery_arbitrage_matches_known_price_solution` | A lossless battery charges in the low-price period and discharges in the high-price period. |
| Reservoir water balance and terminal value | `test_reservoir_allocation_matches_two_period_water_value_solution` | Stored water is retained when terminal water value exceeds avoided thermal cost. |
| Two-bus DC OPF | `test_two_bus_dc_opf_matches_known_congestion_result` | Line capacity binds at 40 MW and the sink bus sheds the remaining 60 MW. |
| Three-bus DC loop flow | `test_three_bus_loop_flow_matches_independent_matrix_calculation` | Independent susceptance-matrix solution gives 45 MW on each radial path and 0 MW on the cross line. |
| Reserve procurement | `test_reserve_procurement_matches_headroom_limit` | Upward reserve equals the known thermal headroom requirement with zero shortfall. |
| Demand shifting | `test_demand_shifting_preserves_exact_energy` | Shift-up and shift-down energy are equal over the horizon. |
| Reliability metrics | `test_reliability_metrics_match_small_enumerated_outage_model` | Expected unserved energy and LOLP match exact two-state outage enumeration. |

Regression-focused cases for time-step scaling, terminal commitment obligations,
stress-case coverage, baselines, and reconciliation remain in the main test
suite. The fixture manifest at `tests/benchmarks/regression_fixtures.json`
catalogs the known regression cases and the tests that cover them. The benchmark
package complements those tests by tying representative constraints directly to
small mathematical examples.

## Benchmark Outputs

Run:

```bash
poetry run python scripts/run_verification_benchmarks.py
```

The command writes:

- `outputs/verification/summary.json`
- `outputs/verification/summary.csv`

The JSON and CSV schemas are tested in `tests/benchmarks/test_verification_outputs.py`.
The same test module applies generous CI budgets for build time, solve time, and
formulation size to catch order-of-magnitude regressions without depending on a
specific machine.

## Known Limits

The verification cases are intentionally small. They prove the implementation on
analytically tractable branches and do not validate production-scale planning
assumptions, stochastic calibration, or market-rule policy choices. Those topics
remain covered by scenario, stress, baseline, and domain-specific tests.
