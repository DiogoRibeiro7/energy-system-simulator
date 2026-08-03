.PHONY: install format lint typecheck test core-coverage validate simulate stress verification check-example-data validate-examples benchmark scaling-benchmark baseline compare-outputs research-experiment validate-research-experiment iberia-case-study validate-iberia-case-study release-metadata version-metadata release-readiness package editable-smoke wheel-smoke verify clean

install:
	poetry install

test:
	poetry run pytest --cov=energy_system_simulator --cov-report=term-missing

core-coverage:
	poetry run coverage report --include="src/energy_system_simulator/dispatch/*" --fail-under=90

format:
	poetry run ruff format --check .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

validate:
	poetry run energy-sim validate --config configs/example.yaml

simulate:
	poetry run energy-sim simulate --config configs/example.yaml --overwrite

stress:
	poetry run python scripts/run_stress_cases.py

verification:
	poetry run python scripts/run_verification_benchmarks.py

check-example-data:
	poetry run python scripts/check_example_data.py

validate-examples:
	poetry run python scripts/validate_examples.py

benchmark:
	poetry run python scripts/benchmark_example.py

scaling-benchmark:
	poetry run python scripts/benchmark_scaling.py

baseline:
	poetry run python scripts/compare_baseline.py

compare-outputs:
	poetry run python scripts/compare_outputs.py outputs/example outputs/example --output outputs/comparison.md

research-experiment:
	poetry run energy-sim run-experiment --study experiments/storage_value --overwrite --no-plots

validate-research-experiment:
	poetry run energy-sim analyze-experiment --study experiments/storage_value

iberia-case-study:
	poetry run python case_studies/iberia/scripts/build_case_study.py --run --overwrite

validate-iberia-case-study:
	poetry run python case_studies/iberia/scripts/validate_case_study.py

release-metadata:
	poetry run python scripts/validate_licensing.py

version-metadata:
	poetry run python scripts/validate_version.py

release-readiness:
	poetry run python scripts/validate_release_readiness.py

package:
	poetry build

editable-smoke:
	poetry run python scripts/smoke_editable_install.py

wheel-smoke: package
	poetry run python scripts/smoke_wheel_install.py

verify: format lint typecheck check-example-data validate-examples test core-coverage validate simulate compare-outputs stress verification benchmark baseline release-metadata version-metadata release-readiness editable-smoke wheel-smoke

clean:
	rm -rf outputs .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
