# Release Checklist

Use this checklist before tagging a release.

## Citation

- `CITATION.cff` version matches `pyproject.toml`.
- `CITATION.cff` release date matches `licensing/metadata.json`.
- Repository URL and author metadata are current.
- `README.md` links to citation guidance.

## Reproducibility

- `poetry install` succeeds from a clean checkout.
- `make verify` passes.
- `make research-experiment` reproduces the storage-value study.
- `make iberia-case-study` and `make validate-iberia-case-study` pass when the
  case-study outputs need regeneration.
- `poetry build` creates source and wheel artifacts.
- `poetry run python scripts/smoke_wheel_install.py` validates the wheel.

## Data and Licensing

- Committed data files are synthetic, derived from documented public sources, or
  accompanied by local provenance documentation.
- `docs/data-provenance-inventory.md` covers every committed CSV class.
- `scripts/validate_licensing.py` passes.
- `licensing/releases.json` includes the release version, release date, Change
  Date, and change license.
- No GPL, AGPL, SSPL, Commons Clause, or incompatible assets were added without
  explicit approval.

## Public Surface

- CLI commands are listed in `docs/api-cli.md`.
- Configuration schemas are documented in `docs/configuration.md`.
- Output tables are documented in `docs/reporting.md`.
- Breaking changes are listed in `CHANGELOG.md`.
- Known limitations and unresolved risks are listed in the current
  `docs/release-validation-*.md` report.
