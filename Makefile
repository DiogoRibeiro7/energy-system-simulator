.PHONY: install format lint typecheck test validate simulate stress verification check-example-data benchmark baseline release-metadata version-metadata package editable-smoke wheel-smoke verify clean

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

stress:
	poetry run python scripts/run_stress_cases.py

verification:
	poetry run python scripts/run_verification_benchmarks.py

check-example-data:
	poetry run python scripts/check_example_data.py

benchmark:
	poetry run python scripts/benchmark_example.py

baseline:
	poetry run python scripts/compare_baseline.py

release-metadata:
	poetry run python scripts/validate_licensing.py

version-metadata:
	poetry run python scripts/validate_version.py

package:
	poetry build

editable-smoke:
	poetry run python scripts/smoke_editable_install.py

wheel-smoke: package
	poetry run python scripts/smoke_wheel_install.py

verify: format lint typecheck check-example-data test validate simulate stress verification benchmark baseline release-metadata version-metadata editable-smoke wheel-smoke

clean:
	rm -rf outputs .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
