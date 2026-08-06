# Release Validation 1.1.0

Validation date: 2026-08-06.

## Environment

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.13.5 |
| Package manager | Poetry |
| Package version | 1.1.0 |
| Default solver | `scipy.optimize.milp` |

## Commands Run

```bash
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy src
poetry run pytest --cov=energy_system_simulator --cov-report=term-missing
poetry run energy-sim simulate --config configs/example.yaml --overwrite
node --check outputs/example/dashboard/app.js
node --check outputs/example/dashboard/data.js
```

## Results

| Check | Result |
|---|---|
| Formatting, lint, type checking | Passed |
| Test suite | Passed: 309 tests |
| Coverage gate | Passed: 83.10%, above the 80% threshold |
| Example simulation | Passed |
| Structured dashboard app generation | Passed |
| Dashboard JavaScript syntax checks | Passed |

## Release Scope

Version 1.1.0 adds local dashboard visualization outputs for simulation result
directories. The release preserves configuration schemas 1 and 2 and output
schema version 1.

## Known Limitations

- The dashboard is a local diagnostic and reporting view; it is not a
  multi-user hosted monitoring service.
- The dashboard charts render from generated output tables and do not rerun or
  mutate simulations.
- Browser screenshot validation was not run because Playwright is not installed
  in the validation environment.
