# Reliability Simulation

Reliability studies wrap the deterministic dispatch engine with sequential Monte
Carlo outage trajectories. Each replication preserves chronology, storage state,
unit commitment, demand response, and network constraints while applying
time-varying availability multipliers to assets.

```python
from energy_system_simulator.config import load_config
from energy_system_simulator.simulation import (
    OutageModel,
    ReliabilityStudy,
    ReliabilityStudyConfig,
)

config = load_config("configs/portfolio_two_thermal.yaml")
study = ReliabilityStudy(
    config,
    ReliabilityStudyConfig(
        replications=100,
        seed=20260802,
        outage_models=(
            OutageModel("north-ccgt", "thermal", 0.06, 12.0),
            OutageModel("south-peaker", "thermal", 0.10, 6.0),
            OutageModel("south-battery", "storage", 0.04, 8.0),
            OutageModel("north-south", "line", 0.02, 4.0),
            OutageModel("market-imports", "import", 0.03, 5.0),
        ),
    ),
)
result = study.run()
```

Outage models are typed as `thermal`, `renewable`, `storage`, `line`, or
`import`. A forced-outage rate and mean time to repair define a two-state
availability process. Common-cause groups can be supplied separately and linked
from individual `OutageModel.group_id` values; an asset is unavailable whenever
its independent path or linked group path is unavailable. The same study seed and
replication index produce identical paths and metrics.

The result reports:

- `loss_of_load_probability`: fraction of simulated periods with load shedding.
- `loss_of_load_expectation_hours_per_year`: annualized scarcity hours.
- `expected_unserved_energy_mwh`: unserved energy over the simulated horizon.
- `expected_demand_not_served_mw`: average unserved demand over the simulated
  horizon.
- `scarcity_event_frequency_per_year`: annualized count of contiguous scarcity
  events.
- `scarcity_event_mean_duration_hours`: average duration of scarcity events.
- `metric_distribution`: per-replication values for the reported metrics.
- `confidence_intervals`: normal-approximation intervals across successful
  replications.
- `attributed_unserved_energy_mwh_by_type`: mean unserved energy attributed to
  active outage types when a scarcity period has an attributable outage.

`SimulationEngine(...).run()` may already contain deterministic load shedding if
the configured deterministic system cannot serve load. That is an economic and
physical outcome for one availability profile. Reliability simulation is
different: it estimates adequacy risk by repeatedly sampling probabilistic
outage paths around the deterministic dispatch formulation.

For a compact comparison of a base portfolio with an additional peaker and
battery, run:

```bash
poetry run python scripts/run_reliability_example.py
```
