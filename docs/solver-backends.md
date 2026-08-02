# Solver Backends

The dispatch formulation builds a backend-neutral `SolverProblem` containing:

- objective coefficients;
- variable lower and upper bounds;
- integrality flags;
- sparse linear constraints with row names;
- stable variable metadata and names.

`UnitCommitment` no longer imports `scipy.optimize` directly. The default
backend remains SciPy/HiGHS, with conversion to SciPy `Bounds` and
`LinearConstraint` isolated in `energy_system_simulator.dispatch.solver`.

## Capability Matrix

| Backend | MILP | LP duals | Warm starts | Time limits | MIP gaps | Node counts | Infeasibility diagnostics | Solution pools | LP export | MPS export |
|---|---|---|---|---|---|---|---|---|---|---|
| `scipy` | yes | yes for fixed-commitment LP pricing | no | yes | yes | yes when reported by HiGHS | no | no | yes | no |

SciPy is the only installed backend. Requests for any other backend fail with a
clear installation message rather than silently changing solver semantics.

## Model Export

Export a stable LP debug file without solving:

```bash
poetry run energy-sim export-model --config configs/example.yaml --output outputs/debug/model.lp
```

The LP writer is intentionally simple and deterministic. It is meant for
inspection and backend debugging, not as a replacement for the default solve
path.

## Scaling Benchmarks

Run:

```bash
poetry run python scripts/benchmark_scaling.py
```

The script writes:

- `outputs/benchmarks/scaling.csv`
- `outputs/benchmarks/scaling.md`

Rows isolate growth in periods, thermal units, storage assets, buses, and
repeated scenarios. Runtime thresholds are intentionally not asserted in this
script because local solver and CPU variance can dominate small examples.
