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
