# Release Validation 1.1.3

Validation date: 2026-08-06.

## Environment

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.13.5 |
| Package manager | Poetry |
| Package version | 1.1.3 |
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
| Version metadata, including `.zenodo.json` | Passed |
| Licensing metadata | Passed |
| Release readiness | Passed |
| Formatting, lint, type checking | Passed |
| Test suite | Passed: 311 tests |
| Coverage gate | Passed: 83.26%, above the 80% threshold |
| Package build | Passed |

## Release Scope

Version 1.1.3 is a metadata correction release. It synchronizes Zenodo metadata
with the package and citation version and extends validation so future releases
fail if `.zenodo.json` drifts from `pyproject.toml`.

## Known Limitations

- The dashboard is a local diagnostic and reporting view; it is not a
  multi-user hosted monitoring service.
- Browser screenshot validation was not run because Playwright is not installed
  in the validation environment.
