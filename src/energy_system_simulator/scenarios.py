from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, cast

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_system_simulator.config import ModelConfig, load_config, resolved_config_to_dict
from energy_system_simulator.exceptions import EnergySystemError, OptimisationError
from energy_system_simulator.reporting import write_outputs
from energy_system_simulator.simulation import SimulationEngine

OverrideMap = dict[str, Any]


class ScenarioExperimentError(EnergySystemError):
    """Invalid scenario experiment definition or execution."""


def run_experiment_file(
    path: str | Path,
    *,
    workers: int | None = None,
    resume: bool | None = None,
    create_plots: bool = True,
) -> pd.DataFrame:
    """Load and run a scenario experiment YAML file."""
    experiment_path = Path(path)
    payload = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScenarioExperimentError("Experiment YAML must contain a mapping")
    experiment = ScenarioExperiment.from_mapping(payload, experiment_path.parent)
    return ScenarioRunner(
        experiment,
        workers=workers if workers is not None else experiment.workers,
        resume=resume if resume is not None else experiment.resume,
        create_plots=create_plots,
    ).run()


class ScenarioExperiment:
    """Resolved base configuration and declarative scenario expansion."""

    def __init__(
        self,
        *,
        base_config_path: Path,
        output_directory: Path,
        scenarios: tuple[ScenarioDefinition, ...],
        workers: int = 1,
        resume: bool = False,
    ) -> None:
        self.base_config_path = base_config_path
        self.output_directory = output_directory
        self.scenarios = scenarios
        self.workers = workers
        self.resume = resume

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], base_dir: Path) -> ScenarioExperiment:
        base_config = _required_path(payload, "base_config", base_dir)
        output_directory = _required_path(payload, "output_directory", base_dir)
        workers = int(payload.get("workers", 1))
        if workers <= 0:
            raise ScenarioExperimentError("workers must be a positive integer")
        scenarios = resolve_scenarios(payload)
        if not scenarios:
            raise ScenarioExperimentError("Experiment must define at least one scenario")
        return cls(
            base_config_path=base_config,
            output_directory=output_directory,
            scenarios=tuple(scenarios),
            workers=workers,
            resume=bool(payload.get("resume", False)),
        )


class ScenarioDefinition:
    """One reproducible scenario after explicit, sweep, or grid expansion."""

    def __init__(
        self,
        *,
        label: str,
        overrides: OverrideMap,
        source: str,
        order: int,
    ) -> None:
        self.label = label
        self.overrides = dict(overrides)
        self.source = source
        self.order = order
        self.id = stable_scenario_id(self.overrides)


class ScenarioRunner:
    """Run scenario experiments using normal simulation outputs per scenario."""

    def __init__(
        self,
        experiment: ScenarioExperiment,
        *,
        workers: int,
        resume: bool,
        create_plots: bool,
    ) -> None:
        self.experiment = experiment
        self.workers = workers
        self.resume = resume
        self.create_plots = create_plots

    def run(self) -> pd.DataFrame:
        output = self.experiment.output_directory
        self._prepare_output_directory(output)
        base_config = load_config(self.experiment.base_config_path)
        self._write_experiment_manifest(base_config)
        records: list[dict[str, Any]] = []
        if self.workers == 1:
            for scenario in self.experiment.scenarios:
                try:
                    records.append(
                        _run_scenario_once(
                            base_config,
                            self.experiment.output_directory,
                            scenario,
                            self.resume,
                            self.create_plots,
                        )
                    )
                except Exception as error:
                    records.append(_failure_record(scenario, error))
        else:
            self._run_parallel(records)
        records.sort(key=lambda item: int(item["scenario_order"]))
        aggregate = pd.DataFrame.from_records(records)
        aggregate_path = output / "summary.csv"
        aggregate.to_csv(aggregate_path, index=False)
        if bool((~aggregate["ok"]).any()):
            raise OptimisationError(f"One or more scenarios failed; see {aggregate_path}")
        return aggregate

    def _run_parallel(self, records: list[dict[str, Any]]) -> None:
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    _run_scenario_process,
                    str(self.experiment.base_config_path),
                    str(self.experiment.output_directory),
                    scenario,
                    self.resume,
                    self.create_plots,
                ): scenario
                for scenario in self.experiment.scenarios
            }
            for future in as_completed(futures):
                scenario = futures[future]
                try:
                    records.append(future.result())
                except Exception as error:
                    records.append(_failure_record(scenario, error))

    def _prepare_output_directory(self, output: Path) -> None:
        if output.exists() and any(output.iterdir()) and not self.resume:
            raise ScenarioExperimentError(
                f"Experiment output directory already exists and is not empty: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)

    def _write_experiment_manifest(self, base_config: ModelConfig) -> None:
        payload = {
            "schema_version": 1,
            "base_config_path": str(self.experiment.base_config_path.resolve()),
            "base_config_sha256": _file_sha256(self.experiment.base_config_path),
            "input_file": str(base_config.paths.input_csv),
            "input_file_sha256": _file_sha256(base_config.paths.input_csv),
            "scenario_count": len(self.experiment.scenarios),
            "workers": self.workers,
            "resume": self.resume,
            "scenario_ids": [scenario.id for scenario in self.experiment.scenarios],
        }
        (self.experiment.output_directory / "experiment_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def resolve_scenarios(payload: dict[str, Any]) -> list[ScenarioDefinition]:
    """Expand explicit scenarios, one-factor sweeps, and Cartesian grids."""
    definitions: list[ScenarioDefinition] = []
    order = 0
    for item in payload.get("scenarios", []) or []:
        if not isinstance(item, dict):
            raise ScenarioExperimentError("Each scenario must be a mapping")
        overrides = _override_mapping(item.get("overrides", {}))
        definitions.append(
            ScenarioDefinition(
                label=str(item.get("id", f"scenario-{order}")),
                overrides=overrides,
                source="explicit",
                order=order,
            )
        )
        order += 1
    for item in payload.get("sweeps", []) or []:
        if not isinstance(item, dict):
            raise ScenarioExperimentError("Each sweep must be a mapping")
        parameter = str(item["parameter"])
        values = item.get("values")
        if not isinstance(values, list):
            raise ScenarioExperimentError("Sweep values must be a list")
        for value in values:
            definitions.append(
                ScenarioDefinition(
                    label=f"{parameter}={_canonical_value(value)}",
                    overrides={parameter: value},
                    source="sweep",
                    order=order,
                )
            )
            order += 1
    grid = payload.get("grid")
    if grid is not None:
        parameters = grid.get("parameters") if isinstance(grid, dict) else None
        if not isinstance(parameters, dict):
            raise ScenarioExperimentError("grid.parameters must be a mapping")
        for overrides in _cartesian_product(parameters):
            definitions.append(
                ScenarioDefinition(
                    label="grid-" + str(order),
                    overrides=overrides,
                    source="grid",
                    order=order,
                )
            )
            order += 1
    return definitions


def _run_scenario_process(
    base_config_path: str,
    output_directory: str,
    scenario: ScenarioDefinition,
    resume: bool,
    create_plots: bool,
) -> dict[str, Any]:
    base_config = load_config(Path(base_config_path))
    return _run_scenario_once(
        base_config,
        Path(output_directory),
        scenario,
        resume,
        create_plots,
    )


def _run_scenario_once(
    base_config: ModelConfig,
    output_directory: Path,
    scenario: ScenarioDefinition,
    resume: bool,
    create_plots: bool,
) -> dict[str, Any]:
    scenario_output = output_directory / scenario.id
    scenario_manifest = scenario_output / "scenario_manifest.json"
    summary_path = scenario_output / "summary.json"
    if resume and _completed_scenario_matches(scenario_manifest, scenario):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return _success_record(scenario, summary, resumed=True)

    scenario_output.mkdir(parents=True, exist_ok=True)
    config = apply_overrides(base_config, scenario.overrides)
    config = apply_overrides(config, {"paths.output_directory": str(scenario_output)})
    config_path = scenario_output / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(resolved_config_to_dict(config), sort_keys=False),
        encoding="utf-8",
    )
    result = SimulationEngine(config).run()
    result.summary["solver_status"] = result.solver_status
    result.summary["backend_solver_status"] = result.backend_solver_status
    write_outputs(
        result,
        scenario_output,
        config=config,
        config_path=config_path,
        create_plots=create_plots,
    )
    _write_scenario_manifest(output_directory, scenario, config_path, result.summary)
    return _success_record(scenario, result.summary, resumed=False)


def _completed_scenario_matches(
    manifest_path: Path,
    scenario: ScenarioDefinition,
) -> bool:
    summary_path = manifest_path.parent / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        payload.get("scenario_id") == scenario.id
        and payload.get("overrides_sha256") == _json_sha256(scenario.overrides)
    )


def _write_scenario_manifest(
    output_directory: Path,
    scenario: ScenarioDefinition,
    config_path: Path,
    summary: dict[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "scenario_id": scenario.id,
        "scenario_label": scenario.label,
        "scenario_source": scenario.source,
        "scenario_order": scenario.order,
        "overrides": scenario.overrides,
        "overrides_sha256": _json_sha256(scenario.overrides),
        "scenario_config": str(config_path.resolve()),
        "scenario_config_sha256": _file_sha256(config_path),
        "solver_status": summary.get("solver_status"),
        "objective_eur": summary.get("objective_eur"),
    }
    (output_directory / scenario.id / "scenario_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _success_record(
    scenario: ScenarioDefinition,
    summary: dict[str, Any],
    *,
    resumed: bool,
) -> dict[str, Any]:
    record = _aggregate_summary(summary)
    record.update(_parameter_columns(scenario.overrides))
    record.update(
        {
            "scenario_id": scenario.id,
            "scenario_label": scenario.label,
            "scenario_source": scenario.source,
            "scenario_order": scenario.order,
            "ok": True,
            "resumed": resumed,
            "error_type": None,
            "error_message": None,
        }
    )
    return record


def _failure_record(
    scenario: ScenarioDefinition,
    error: Exception,
) -> dict[str, Any]:
    record = _parameter_columns(scenario.overrides)
    record.update(
        {
            "scenario_id": scenario.id,
            "scenario_label": scenario.label,
            "scenario_source": scenario.source,
            "scenario_order": scenario.order,
            "ok": False,
            "resumed": False,
            "solver_status": "failed",
            "error_type": error.__class__.__name__,
            "error_message": str(error),
        }
    )
    return record


def apply_overrides(config: ModelConfig, overrides: OverrideMap) -> ModelConfig:
    """Return a copied configuration with validated dotted-path overrides applied."""
    result: Any = config
    for path, value in sorted(overrides.items()):
        result = _apply_one_override(result, path, value)
    return cast(ModelConfig, result)


def stable_scenario_id(overrides: OverrideMap) -> str:
    """Generate a stable ID from canonical override parameter values."""
    digest = _json_sha256(overrides)[:12]
    if not overrides:
        return "scenario-base-" + digest
    stem = "__".join(
        _slug(f"{path}-{_canonical_value(value)}") for path, value in sorted(overrides.items())
    )
    return f"{stem[:64]}-{digest}"


def finite_difference_sensitivity(
    base_config: ModelConfig,
    parameter_path: str,
    *,
    step: float,
) -> dict[str, float | str]:
    """Numerical local sensitivity for one scalar continuous parameter."""
    if step <= 0.0:
        raise ValueError("step must be positive")
    current = _get_path(base_config, parameter_path)
    if not isinstance(current, int | float):
        raise ValueError("finite-difference sensitivity requires a scalar numeric parameter")
    low = apply_overrides(base_config, {parameter_path: float(current) - step})
    high = apply_overrides(base_config, {parameter_path: float(current) + step})
    low_result = SimulationEngine(low).run()
    high_result = SimulationEngine(high).run()
    return {
        "parameter": parameter_path,
        "base_value": float(current),
        "step": step,
        "low_objective_eur": low_result.objective_eur,
        "high_objective_eur": high_result.objective_eur,
        "objective_eur_per_unit": (high_result.objective_eur - low_result.objective_eur)
        / (2.0 * step),
        "label": "central finite difference of a non-smooth MILP value function",
    }


def plot_response_curve(
    aggregate: pd.DataFrame,
    *,
    parameter_column: str,
    metric_column: str,
    output_path: str | Path,
) -> None:
    frame = aggregate.sort_values(parameter_column)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame[parameter_column], frame[metric_column], marker="o")
    axis.set_xlabel(parameter_column)
    axis.set_ylabel(metric_column)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_heatmap(
    aggregate: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    metric_column: str,
    output_path: str | Path,
) -> None:
    pivot = aggregate.pivot(index=y_column, columns=x_column, values=metric_column)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    axis.set_xticks(range(len(pivot.columns)), labels=[str(value) for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), labels=[str(value) for value in pivot.index])
    axis.set_xlabel(x_column)
    axis.set_ylabel(y_column)
    figure.colorbar(image, ax=axis, label=metric_column)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_tradeoff_frontier(
    aggregate: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    output_path: str | Path,
) -> None:
    frame = aggregate.sort_values([x_metric, y_metric])
    frontier = []
    best_y = float("inf")
    for _, row in frame.iterrows():
        value = float(row[y_metric])
        if value < best_y:
            frontier.append(row)
            best_y = value
    frontier_frame = pd.DataFrame(frontier)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(frame[x_metric], frame[y_metric], alpha=0.35)
    if not frontier_frame.empty:
        axis.plot(frontier_frame[x_metric], frontier_frame[y_metric], marker="o")
    axis.set_xlabel(x_metric)
    axis.set_ylabel(y_metric)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _apply_one_override(config: Any, path: str, value: Any) -> Any:
    tokens = _path_tokens(path)
    return _replace_path(config, tokens, value)


def _replace_path(current: Any, tokens: list[str | int], value: Any) -> Any:
    if not tokens:
        return _coerce_value(current, value)
    token = tokens[0]
    if isinstance(token, int):
        if not isinstance(current, tuple):
            raise ScenarioExperimentError("Numeric path selectors can only index tuples")
        if token < 0 or token >= len(current):
            raise ScenarioExperimentError("Override tuple index is out of range")
        items = list(current)
        items[token] = _replace_path(items[token], tokens[1:], value)
        return tuple(items)
    if not is_dataclass(current) or not hasattr(current, token):
        raise ScenarioExperimentError(f"Unknown override path: {'.'.join(map(str, tokens))}")
    child = getattr(current, token)
    dataclass_current = cast(Any, current)
    return replace(dataclass_current, **{token: _replace_path(child, tokens[1:], value)})


def _get_path(config: Any, path: str) -> Any:
    current = config
    for token in _path_tokens(path):
        current = current[token] if isinstance(token, int) else getattr(current, token)
    return current


def _path_tokens(path: str) -> list[str | int]:
    if not path or ".." in path:
        raise ScenarioExperimentError("Override paths must be non-empty dotted paths")
    tokens: list[str | int] = []
    for part in path.split("."):
        while "[" in part:
            prefix, _, rest = part.partition("[")
            if prefix:
                tokens.append(prefix)
            index_text, separator, suffix = rest.partition("]")
            if separator != "]" or not index_text.isdigit():
                raise ScenarioExperimentError(f"Invalid override path selector: {path}")
            tokens.append(int(index_text))
            part = suffix
        if part:
            tokens.append(part)
    return tokens


def _coerce_value(existing: Any, value: Any) -> Any:
    if isinstance(existing, Path):
        return Path(str(value))
    if isinstance(existing, bool):
        return bool(value)
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    if existing is None:
        return value
    if isinstance(existing, str):
        return str(value)
    return value


def _aggregate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "solver_status": summary.get("solver_status", "unknown"),
        "backend_solver_status": summary.get("backend_solver_status"),
        "objective_eur": summary.get("objective_eur"),
        "total_emissions_tonnes": summary.get("total_emissions_tonnes"),
        "renewable_share": summary.get("renewable_share_of_primary_generation"),
        "renewable_curtailed_mwh": summary.get("renewable_curtailed_mwh"),
        "unserved_energy_mwh": summary.get("unserved_energy_mwh"),
        "thermal_starts": summary.get("thermal_starts"),
        "storage_cycles": _storage_cycles(summary),
        "network_max_abs_line_utilisation": _nested_get(
            summary,
            ("network", "max_abs_line_utilisation"),
        ),
        "reserve_upward_shortfall_mwh": _nested_get(
            summary,
            ("reserves", "upward_shortfall_mwh"),
        ),
    }
    costs = summary.get("cost_components_eur", {})
    if isinstance(costs, dict):
        for key, value in costs.items():
            record[f"cost_{key}"] = value
    return record


def _storage_cycles(summary: dict[str, Any]) -> float | None:
    storage = summary.get("storage_assets")
    if not isinstance(storage, dict):
        return None
    total = 0.0
    found = False
    for metrics in storage.values():
        if isinstance(metrics, dict) and "equivalent_full_cycles" in metrics:
            total += float(metrics["equivalent_full_cycles"])
            found = True
    return total if found else None


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for token in path:
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _parameter_columns(overrides: OverrideMap) -> dict[str, Any]:
    return {f"param_{path}": value for path, value in overrides.items()}


def _override_mapping(value: Any) -> OverrideMap:
    if not isinstance(value, dict):
        raise ScenarioExperimentError("overrides must be a mapping")
    return {str(key): item for key, item in value.items()}


def _cartesian_product(parameters: dict[str, Any]) -> list[OverrideMap]:
    items = [(str(path), values) for path, values in parameters.items()]
    for path, values in items:
        if not isinstance(values, list):
            raise ScenarioExperimentError(f"Grid values for {path} must be a list")
    result: list[OverrideMap] = [{}]
    for path, values in items:
        result = [dict(existing, **{path: value}) for existing in result for value in values]
    return result


def _required_path(payload: dict[str, Any], key: str, base_dir: Path) -> Path:
    raw = payload.get(key)
    if not isinstance(raw, str):
        raise ScenarioExperimentError(f"{key} is required")
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _canonical_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_value(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
