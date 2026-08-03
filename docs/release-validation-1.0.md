# Release Validation 1.0.0

Validation date: 2026-08-03.

## Environment

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.13.5 |
| Package manager | Poetry |
| Package version | 1.0.0 |
| Default solver | `scipy.optimize.milp` |

## Commands Run

```bash
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy src
poetry run python scripts/check_example_data.py
poetry run python scripts/validate_examples.py
poetry run pytest --cov=energy_system_simulator --cov-report=term-missing
poetry run coverage report --include="src/energy_system_simulator/dispatch/*" --fail-under=90
poetry run energy-sim validate --config configs/example.yaml
poetry run energy-sim simulate --config configs/example.yaml --overwrite
poetry run python scripts/compare_outputs.py outputs/example outputs/example --output outputs/comparison.md
poetry run python scripts/run_stress_cases.py
poetry run python scripts/run_verification_benchmarks.py
poetry run python scripts/benchmark_example.py
poetry run python scripts/compare_baseline.py
poetry run python scripts/validate_licensing.py
poetry run python scripts/validate_version.py
poetry run python scripts/validate_release_readiness.py
poetry run python scripts/smoke_editable_install.py
poetry build
poetry run python scripts/smoke_wheel_install.py
poetry run energy-sim run-experiment --study experiments/storage_value --overwrite --no-plots
```

## Results

| Check | Result |
|---|---|
| Formatting, lint, type checking | Passed |
| Test suite | Passed: 275 tests |
| Core dispatch coverage gate | Passed: at least 90% for `src/energy_system_simulator/dispatch/*` |
| Example validation and simulation | Passed |
| Example dry-run validation | Passed |
| Stress cases | Passed |
| Verification benchmarks | Passed |
| Benchmark and baseline comparison | Passed |
| Licensing and version metadata | Passed |
| Release readiness metadata/docs check | Passed |
| Committed data provenance inventory | Passed |
| Editable install smoke | Passed |
| Source distribution and wheel build | Passed |
| Wheel install smoke | Passed |
| Storage-value research experiment | Passed |

## Public Interfaces and Schemas

- CLI commands are documented in `docs/api-cli.md`.
- Public Python exports are listed from `energy_system_simulator.__all__`.
- Committed CSV data sources are documented in
  `docs/data-provenance-inventory.md`.
- Aggregate configuration schema 1 and typed portfolio schema 2 are supported.
- Unknown future configuration versions fail clearly.
- Versioned output tables use schema version 1 and are documented in
  `docs/reporting.md`.

## Known Limitations

- The default dispatch model is deterministic unless a reliability or stochastic
  study is explicitly configured.
- The default network model is aggregate; nodal mode uses a linear lossless DC
  approximation, not AC power flow.
- Security-constrained unit commitment, N-1 contingency enforcement, transient
  stability, protection, frequency dynamics, and detailed distribution-network
  modelling are outside the 1.0 scope.
- Hydro cascade metadata is parsed, but delayed upstream release coupling is not
  yet enforced in the dispatch formulation.
- Imports remain represented as an aggregate resource in the main dispatch path.

## Unresolved Risks

- Large MILP instances may require external solver workflows beyond the default
  SciPy/HiGHS backend.
- Public case-study data are simplified approximations intended for reproducible
  teaching and research examples, not official system-operation records.
- Performance warnings from Pandas fragmented DataFrames are known in some
  reporting paths; they do not affect numerical results but may matter for very
  large studies.
