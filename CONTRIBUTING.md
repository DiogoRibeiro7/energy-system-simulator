# Contributing

Thanks for improving Energy System Simulator.

## Development Setup

This project uses Python 3.11 or later and Poetry.

```bash
poetry install
```

Run the simulator locally with:

```bash
poetry run energy-sim validate --config configs/example.yaml
poetry run energy-sim simulate --config configs/example.yaml
```

## Quality Checks

Run these before opening a pull request:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
poetry run python scripts/validate_licensing.py
```

## Pull Requests

- Keep changes focused and explain the motivation.
- Add or update tests when behavior changes.
- Update documentation when user-facing behavior, configuration, or data
  contracts change.
- Sign off commits using Developer Certificate of Origin style sign-off:
  `git commit -s`.
- Prefer small pull requests that can be reviewed independently.

## Contribution Terms

Contributions are submitted under the repository's current licence. The
maintainer may offer the project under separate commercial terms.

By contributing, you confirm that:

- You have the right to submit the work.
- Your contribution may be used under the repository's current licence and the
  applicable future Change License.
- Third-party code includes compatible licensing and required attribution.
- You will not add GPL, AGPL, SSPL, Commons Clause, or other incompatible code
  without prior maintainer approval.

No Contributor Licence Agreement is currently provided. Legal review is
recommended before introducing a CLA for dual or commercial licensing.

## Style

Follow the existing project structure and naming. Use typed Python and keep
domain logic explicit rather than hidden behind broad abstractions.
