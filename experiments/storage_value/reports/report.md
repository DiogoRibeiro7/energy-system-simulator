# Value of Storage Under Increasing Renewable Penetration

## Research Question

How does adding a four-hour battery change operating cost, emissions, renewable curtailment, and renewable share when renewable capacity increases?


## Hypotheses

- Battery storage reduces total operating cost by shifting renewable output into higher net-load hours.
- Battery storage reduces renewable curtailment more in the high-renewable case than in the low-renewable case.
- Battery storage lowers emissions when it displaces thermal generation rather than only reducing curtailment.

## Assumptions

- Model version: dispatch-milp-v1
- Data version: synthetic-storage-value-24h-v1
- Seed: 20260803
- Scenario metrics and comparisons were specified before execution in `study.yaml`.

## Model Outputs

| Scenario | Total system cost [EUR] | Total emissions [tonnes] | Renewable share [fraction] | Renewable curtailment [MWh] | Battery equivalent cycles [cycles] |
| --- | --- | --- | --- | --- | --- |
| high-renewables-no-storage | 125265.97 | 413.157 | 0.571 | 648.961 | 0.0 |
| high-renewables-storage | 109306.763 | 358.693 | 0.631 | 475.048 | 1.003 |
| low-renewables-no-storage | 237315.663 | 786.799 | 0.182 | 0.0 | 0.0 |
| low-renewables-storage | 237315.663 | 786.799 | 0.182 | 0.0 | 0.0 |

## Descriptive Comparisons

| Comparison | Baseline | Scenario | Metric | Unit | Baseline value | Scenario value | Change | Change [%] | Paired seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_renewables_storage_value | low-renewables-no-storage | low-renewables-storage | Total system cost | EUR | 237315.663 | 237315.663 | 0.0 | 0.0 | True |
| high_renewables_storage_value | high-renewables-no-storage | high-renewables-storage | Total system cost | EUR | 125265.97 | 109306.763 | -15959.207 | -12.74 | True |
| high_renewables_curtailment_reduction | high-renewables-no-storage | high-renewables-storage | Renewable curtailment | MWh | 648.961 | 475.048 | -173.913 | -26.799 | True |

## Uncertainty and Sensitivity

| Type | Scope | Metric | Unit | Lower | Upper |
| --- | --- | --- | --- | --- | --- |
| Deterministic sensitivity range | renewable_penetration | Total system cost | EUR | 109306.763 | 237315.663 |
| Deterministic sensitivity range | renewable_penetration | Renewable curtailment | MWh | 0.0 | 648.961 |

## Interpretation

These are model-conditioned descriptive results. Differences report the outcomes of fixed scenario definitions under identical input data and, where specified, paired stochastic seeds.

## What the Model Can Identify

- Dispatch cost, emissions, curtailment, and storage operation under the stated assumptions.
- Within-model differences between paired scenario definitions that use the same input profile and seed.

## What the Model Cannot Identify

- Causal effects in historical power systems without a separate identification strategy.
- Investment-optimal storage sizing, since capacity is fixed in this experiment.
- Reliability impacts outside the represented one-day deterministic operating problem.

## Figures

- `figures/storage_value_objective.png`: Total operating cost from the scenario summary, generated from result files.
- `figures/storage_value_curtailment.png`: Renewable curtailment by scenario, generated from result files.

## Limitations

- One synthetic 24-hour profile is illustrative and not a planning-grade weather year.
- The experiment omits unit forced outages and long-duration hydro constraints.
- The single-node network approximation cannot identify location-specific congestion value.
