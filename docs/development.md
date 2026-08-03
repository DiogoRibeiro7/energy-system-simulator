# Development Workflow

This repository uses Poetry with the `src/` package layout. The authoritative
project version is `tool.poetry.version` in `pyproject.toml`.

## Clean Checkout

From a clean checkout:

```bash
poetry install
make verify
```

`make verify` runs formatting, linting, strict type checking, deterministic data
validation, example dry-run validation, tests with coverage, the core
formulation coverage gate, example validation, example simulation, stress cases,
mathematical verification benchmarks, example benchmarking, baseline comparison,
output comparison, release metadata validation, version metadata validation,
release-readiness validation, editable-install smoke checks, package build, and
wheel-install smoke checks.

## Focused Commands

```bash
make install
make test
make core-coverage
make validate-examples
make validate
make simulate
poetry run energy-sim simulate --config configs/example.yaml --dry-run
make verification
make benchmark
poetry run python scripts/benchmark_scaling.py
make baseline
make compare-outputs
make research-experiment
make validate-research-experiment
make iberia-case-study
make validate-iberia-case-study
make release-metadata
make version-metadata
make release-readiness
make editable-smoke
make wheel-smoke
```

## Version Authority

`pyproject.toml` is the single version authority. Runtime version reporting uses
`energy_system_simulator.metadata.get_package_version()`, which uses the
repository's `pyproject.toml` in a valid source checkout and falls back to
installed package metadata when no source checkout is present.

The fallback intentionally returns `0.0.0+unknown` only when no installed package
metadata and no project metadata are available.
