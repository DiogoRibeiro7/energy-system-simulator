# Release Validation 1.1.2

Validation date: 2026-08-06.

## Environment

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.13.5 |
| Package manager | Poetry |
| Package version | 1.1.2 |
| Default solver | `scipy.optimize.milp` |

## Commands Run

```bash
poetry run python scripts/validate_version.py
poetry run python scripts/validate_licensing.py
poetry run python scripts/validate_release_readiness.py
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy src
poetry run pytest --cov=energy_system_simulator --cov-report=term-missing
poetry build
```

## Results

| Check | Result |
|---|---|
| Version metadata | Passed |
| Licensing metadata | Passed |
| Release readiness | Passed |
| Formatting, lint, type checking | Passed |
| Test suite | Passed: 310 tests |
| Coverage gate | Passed: 83.26%, above the 80% threshold |
| Package build | Passed |

## Release Scope

Version 1.1.2 is a documentation and CI-maintenance patch release. It adds
direct README and reference documentation for running the local dashboard and
makes CI matrix reruns independent. The release preserves configuration schemas
1 and 2 and output schema version 1.

## Known Limitations

- The dashboard is a local diagnostic and reporting view; it is not a
  multi-user hosted monitoring service.
- Browser screenshot validation was not run because Playwright is not installed
  in the validation environment.
