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
        ├── demand by configured column
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
- Multiple thermal units can be added by introducing a generator index to the MILP.
- Hydro reservoirs can use intertemporal water-balance constraints.
- Zonal or nodal networks can replace the aggregated distribution representation.
- Reserve requirements can be added as capacity constraints.
- Demand response can be represented through shiftable and curtailable load variables.
- Stochastic scenarios can wrap the deterministic model without changing its internal equations.
