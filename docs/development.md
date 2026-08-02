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
validation, tests with coverage, example validation, example simulation, stress
cases, mathematical verification benchmarks, example benchmarking, baseline
comparison, release metadata validation, version metadata validation,
editable-install smoke checks, package build, and wheel-install smoke checks.

## Focused Commands

```bash
make install
make test
make validate
make simulate
make verification
make benchmark
make baseline
make release-metadata
make version-metadata
make editable-smoke
make wheel-smoke
```

## Version Authority

`pyproject.toml` is the single version authority. Runtime version reporting uses
`energy_system_simulator.metadata.get_package_version()`, which prefers installed
package metadata and falls back to the repository's `pyproject.toml` in a valid
source checkout.

The fallback intentionally returns `0.0.0+unknown` only when no installed package
metadata and no project metadata are available.
