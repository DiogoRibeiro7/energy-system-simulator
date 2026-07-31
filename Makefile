.PHONY: install test lint typecheck validate simulate clean

install:
	poetry install

test:
	poetry run pytest --cov=energy_system_simulator --cov-report=term-missing

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

validate:
	poetry run energy-sim validate --config configs/example.yaml

simulate:
	poetry run energy-sim simulate --config configs/example.yaml

clean:
	rm -rf outputs .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
