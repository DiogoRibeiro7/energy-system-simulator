# Architecture

## Data flow

```text
CSV weather and demand
        │
        ▼
Input validation
        │
        ▼
AssetRegistry resolves portfolio assets
        │
        ├── renewable availability by asset
        ├── thermal generator fleet and fuel prices
        ├── storage portfolio availability by asset
        ├── hydro inflows by reservoir and run-of-river asset
        ├── demand profiles and deterministic demand adjustments
        └── aggregate distribution preprocessing
                  │
                  ▼
          MILP unit commitment
                  │
                  ▼
          Aggregate and asset-level results
```

## Design principles

1. **Explicit units**: public names include units where practical.
2. **Deterministic execution**: the same inputs and configuration produce the same solution.
3. **Model transparency**: equations are documented and implemented directly.
4. **Separation of concerns**: generation, network, storage, dispatch, and reporting are independent modules.
5. **Validation at boundaries**: configuration and input tables are checked before optimisation.

## Extension points

- Additional renewable assets are resolved through the asset registry without
  changing the dispatch formulation.
- Thermal generators use typed fuels, heat-rate segments, and startup
  categories inside the generator-indexed MILP.
- Storage assets share indexed energy-balance equations for batteries and
  pumped storage.
- Hydro assets use indexed water-balance equations for reservoirs and
  run-of-river plants. Cascade metadata is typed, while coupled downstream
  inflow optimisation is reserved for a later extension.
- Demand assets use indexed curtailment, shifting, task-charging, and
  entity-specific shedding variables when configured.
- Zonal or nodal networks can replace the aggregated distribution representation.
- Reserve requirements can be added as capacity constraints.
- Demand response is represented through explicit demand-entity variables in the
  aggregate dispatch formulation.
- Stochastic scenarios can wrap the deterministic model without changing its internal equations.
