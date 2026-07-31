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
```

## Pull Requests

- Keep changes focused and explain the motivation.
- Add or update tests when behavior changes.
- Update documentation when user-facing behavior, configuration, or data
  contracts change.
- Prefer small pull requests that can be reviewed independently.

## Style

Follow the existing project structure and naming. Use typed Python and keep
domain logic explicit rather than hidden behind broad abstractions.
