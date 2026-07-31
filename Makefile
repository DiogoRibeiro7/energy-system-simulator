.PHONY: install format lint typecheck test validate simulate check-example-data benchmark baseline release-metadata verify clean

install:
	poetry install

test:
	poetry run pytest --cov=energy_system_simulator --cov-report=term-missing

format:
	poetry run ruff format --check .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

validate:
	poetry run energy-sim validate --config configs/example.yaml

simulate:
	poetry run energy-sim simulate --config configs/example.yaml

check-example-data:
	poetry run python scripts/check_example_data.py

benchmark:
	poetry run python scripts/benchmark_example.py

baseline:
	poetry run python scripts/compare_baseline.py

release-metadata:
	poetry run python scripts/validate_licensing.py

verify: format lint typecheck check-example-data test validate simulate benchmark baseline release-metadata

clean:
	rm -rf outputs .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
