# Architecture

## Data flow

```text
CSV weather and demand
        │
        ▼
Input validation
        │
        ├── Solar model
        ├── Wind model
        └── Distribution preprocessing
                  │
                  ▼
          MILP unit commitment
                  │
                  ▼
          Results and diagnostics
```

## Design principles

1. **Explicit units**: public names include units where practical.
2. **Deterministic execution**: the same inputs and configuration produce the same solution.
3. **Model transparency**: equations are documented and implemented directly.
4. **Separation of concerns**: generation, network, storage, dispatch, and reporting are independent modules.
5. **Validation at boundaries**: configuration and input tables are checked before optimisation.

## Extension points

- Multiple thermal units can be added by introducing a generator index to the MILP.
- Hydro reservoirs can use intertemporal water-balance constraints.
- Zonal or nodal networks can replace the aggregated distribution representation.
- Reserve requirements can be added as capacity constraints.
- Demand response can be represented through shiftable and curtailable load variables.
- Stochastic scenarios can wrap the deterministic model without changing its internal equations.
