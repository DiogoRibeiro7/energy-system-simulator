# Release Validation 1.1.4

Validation date: 2026-09-18.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows |
| Python | 3.13.5 |
| Package manager | Poetry |
| Package version | 1.1.4 |
| Default solver | `scipy.optimize.milp` |

CI additionally runs the same checks on Linux with Python 3.11, 3.12, and 3.13.

## Commands Run

```bash
poetry run python scripts/validate_version.py
poetry run python scripts/validate_licensing.py
poetry run python scripts/validate_release_readiness.py
poetry run python scripts/check_example_data.py
poetry run python scripts/validate_examples.py
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy src
poetry run pytest --cov=energy_system_simulator --cov-report=term-missing
poetry run coverage report --include="src/energy_system_simulator/dispatch/*" --fail-under=90
poetry run python scripts/run_stress_cases.py
poetry run python scripts/run_verification_benchmarks.py
poetry run python scripts/benchmark_example.py
poetry run python scripts/compare_baseline.py
poetry build
poetry run python scripts/smoke_wheel_install.py
poetry run python scripts/smoke_editable_install.py
```

## Results

| Check | Result |
| --- | --- |
| Version metadata, including `.zenodo.json` | Passed |
| Licensing metadata, including the citation DOI check | Passed |
| Release readiness | Passed |
| Example data reproducibility and example dry runs | Passed |
| Formatting, lint, type checking (mypy 2.3.1, pandas-stubs 3.0.5) | Passed |
| Test suite | Passed: 316 tests |
| Coverage gate | Passed: 83.20%, above the 80% threshold |
| Dispatch core coverage gate | Passed: 92%, above the 90% threshold |
| Stress suite, verification benchmarks, example benchmark | Passed |
| Baseline comparison | Passed |
| Package build | Passed |
| Wheel and editable install smoke tests | Passed |
| Wheel contents | `py.typed` present; metadata carries version 1.1.4 and the DOI URL |

## Release Scope

Version 1.1.4 is a packaging, tooling, and documentation release with one
defensive fix. Simulator results, configuration schemas, and output schemas are
unchanged; the baseline comparison confirms identical example outputs.

- The package now ships a `py.typed` marker and complete project metadata.
- Releases are citable through the Zenodo concept DOI
  `10.5281/zenodo.21797556`, and validation fails if the README and
  `CITATION.cff` disagree on it.
- Residual diagnostics and AC validation period selection now use row positions
  instead of index labels. Behaviour is unchanged for the default-indexed frames
  the simulator produces.
- CI is hardened, CodeQL analysis is enabled, and a tag-triggered workflow
  builds, smoke-tests, and attaches release artifacts.

## Known Limitations

- The dashboard is a local diagnostic and reporting view; it is not a
  multi-user hosted monitoring service.
- Browser screenshot validation was not run because Playwright is not installed
  in the validation environment.
- Model limitations are unchanged from 1.1.3; see `docs/model-status.md`.

## Unresolved Risks

- The tag-triggered `Release` workflow runs for the first time with this
  release. If it fails, build locally with `poetry build` and attach the
  artifacts to the GitHub release by hand, as was done for earlier versions.
- Zenodo holds no records for 1.1.1 and 1.1.2, and three records for 1.1.0.
  This does not affect the concept DOI. Confirm after publishing that Zenodo
  archived 1.1.4 and issued a version DOI.
