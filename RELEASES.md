# Release Licensing

Every released version of Energy System Simulator has its own release date and
Change Date. The Change Date is exactly four calendar years after the release
date for that version.

The machine-readable release licensing manifest is stored in
`licensing/releases.json`.

Before publishing a release:

1. Add or update the release entry in `licensing/releases.json`.
2. Set `release_date` to the public release date.
3. Set `change_date` to the same month and day exactly four years later.
4. Set `change_license` to `Apache-2.0`.
5. Run `poetry run python scripts/validate_licensing.py`.

Tagged releases should preserve the licence parameters and manifest entry that
apply to that version.

## Released Versions

| Version | Release Date | Change Date | Change License |
|---|---:|---:|---|
| 1.1.0 | 2026-08-06 | 2030-08-06 | Apache-2.0 |
| 1.0.0 | 2026-08-04 | 2030-08-04 | Apache-2.0 |
| 0.1.1 | 2026-07-31 | 2030-07-31 | Apache-2.0 |
| 0.1.0 | 2026-07-30 | 2030-07-30 | Apache-2.0 |
