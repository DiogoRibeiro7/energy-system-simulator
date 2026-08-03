from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from energy_system_simulator.config import load_config
from energy_system_simulator.data_adapters import file_sha256
from energy_system_simulator.exceptions import EnergySystemError
from energy_system_simulator.metadata import get_package_version
from energy_system_simulator.scenarios import run_experiment_file


class ExperimentError(EnergySystemError):
    """Invalid research experiment definition, analysis, or reproduction request."""


@dataclass(frozen=True)
class ExperimentSpec:
    """Validated research experiment metadata and resolved file locations."""

    study_dir: Path
    id: str
    title: str
    research_question: str
    hypotheses: tuple[str, ...]
    model_version: str
    data_version: str
    seed: int
    base_config_path: Path
    scenario_file_path: Path
    output_directory: Path
    metrics: tuple[dict[str, Any], ...]
    comparisons: tuple[dict[str, Any], ...]
    uncertainty_intervals: tuple[dict[str, Any], ...]
    sensitivity_ranges: tuple[dict[str, Any], ...]
    figures: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    model_can_identify: tuple[str, ...]
    model_cannot_identify: tuple[str, ...]
    raw: dict[str, Any]


def load_experiment_spec(study_dir: str | Path) -> ExperimentSpec:
    """Load and validate a research experiment study definition."""
    root = Path(study_dir).resolve()
    path = root / "study.yaml"
    if not path.is_file():
        raise ExperimentError(f"Study definition not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentError("study.yaml must contain a mapping")
    mapping = cast(dict[str, Any], payload)
    metrics = _required_dict_list(mapping, "metrics")
    comparisons = _required_dict_list(mapping, "comparisons")
    if not metrics:
        raise ExperimentError("study.yaml must pre-specify at least one metric")
    if not comparisons:
        raise ExperimentError("study.yaml must pre-specify at least one comparison")
    return ExperimentSpec(
        study_dir=root,
        id=_required_str(mapping, "id"),
        title=_required_str(mapping, "title"),
        research_question=_required_str(mapping, "research_question"),
        hypotheses=tuple(_required_str_list(mapping, "hypotheses")),
        model_version=_required_str(mapping, "model_version"),
        data_version=_required_str(mapping, "data_version"),
        seed=int(mapping.get("seed", 1)),
        base_config_path=(root / _required_str(mapping, "base_config")).resolve(),
        scenario_file_path=(root / _required_str(mapping, "scenario_file")).resolve(),
        output_directory=(root / _required_str(mapping, "output_directory")).resolve(),
        metrics=tuple(metrics),
        comparisons=tuple(comparisons),
        uncertainty_intervals=tuple(
            _dict_list(mapping.get("uncertainty_intervals", []), "uncertainty_intervals")
        ),
        sensitivity_ranges=tuple(
            _dict_list(mapping.get("sensitivity_ranges", []), "sensitivity_ranges")
        ),
        figures=tuple(_dict_list(mapping.get("figures", []), "figures")),
        limitations=tuple(_str_list(mapping.get("limitations", []), "limitations")),
        model_can_identify=tuple(
            _str_list(mapping.get("model_can_identify", []), "model_can_identify")
        ),
        model_cannot_identify=tuple(
            _str_list(mapping.get("model_cannot_identify", []), "model_cannot_identify")
        ),
        raw=mapping,
    )


def run_research_experiment(
    study_dir: str | Path,
    *,
    overwrite: bool = False,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Execute a pre-registered research experiment and generate analysis artifacts."""
    spec = load_experiment_spec(study_dir)
    _validate_spec_files(spec)
    if spec.output_directory.exists() and any(spec.output_directory.iterdir()):
        if not overwrite:
            raise ExperimentError(
                f"Experiment output directory already exists: {spec.output_directory}. "
                "Use --overwrite."
            )
        _remove_within_study(spec.output_directory, spec.study_dir)
    _validate_pre_registered_design(spec)

    summary = run_experiment_file(spec.scenario_file_path, create_plots=create_plots)
    manifest = build_experiment_manifest(spec, summary)
    manifest_path = spec.output_directory / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    analysis = analyze_research_experiment(spec.study_dir)
    return {
        "manifest": manifest_path,
        "summary": spec.output_directory / "summary.csv",
        "report": analysis["report"],
    }


def reproduce_research_experiment(
    manifest_path: str | Path,
    *,
    overwrite: bool = False,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Verify a research manifest and rerun the recorded experiment definition."""
    manifest = verify_experiment_manifest(manifest_path)
    study_dir = Path(str(manifest["study_dir"]))
    return run_research_experiment(
        study_dir,
        overwrite=overwrite,
        create_plots=create_plots,
    )


def verify_experiment_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify that current experiment inputs match a saved research manifest."""
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentError("Research manifest must contain a JSON object")
    manifest = cast(dict[str, Any], payload)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ExperimentError("Research manifest is missing file checksums")
    mismatches: list[str] = []
    for file_name, expected in cast(Mapping[str, str], files).items():
        current_path = Path(file_name)
        if not current_path.is_file():
            mismatches.append(f"{file_name}: missing")
            continue
        observed = file_sha256(current_path)
        if observed != expected:
            mismatches.append(f"{file_name}: expected {expected}, observed {observed}")
    if mismatches:
        raise ExperimentError("Research manifest checksum mismatch: " + "; ".join(mismatches))
    return manifest


def build_experiment_manifest(spec: ExperimentSpec, summary: pd.DataFrame) -> dict[str, Any]:
    """Build a reproducibility manifest for a completed research experiment."""
    config = load_config(spec.base_config_path)
    files = {
        str((spec.study_dir / "study.yaml").resolve()): file_sha256(spec.study_dir / "study.yaml"),
        str(spec.base_config_path): file_sha256(spec.base_config_path),
        str(spec.scenario_file_path): file_sha256(spec.scenario_file_path),
        str(config.paths.input_csv): file_sha256(config.paths.input_csv),
    }
    summary_path = spec.output_directory / "summary.csv"
    if summary_path.is_file():
        files[str(summary_path.resolve())] = file_sha256(summary_path)
    return {
        "schema_version": 1,
        "study_id": spec.id,
        "title": spec.title,
        "study_dir": str(spec.study_dir),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "package_version": get_package_version(),
        "git_commit": _git_commit(spec.study_dir),
        "model_version": spec.model_version,
        "data_version": spec.data_version,
        "seed": spec.seed,
        "solver": {
            "backend": "scipy.optimize.milp",
            "time_limit_seconds": config.simulation.solver_time_limit_seconds,
            "mip_relative_gap": config.simulation.mip_relative_gap,
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "scenario_count": len(summary),
        "metrics": list(spec.metrics),
        "comparisons": list(spec.comparisons),
        "uncertainty_intervals": list(spec.uncertainty_intervals),
        "sensitivity_ranges": list(spec.sensitivity_ranges),
        "paired_seed_groups": _paired_seed_groups(spec),
        "files": files,
    }


def analyze_research_experiment(study_dir: str | Path) -> dict[str, Path]:
    """Generate Markdown, LaTeX, report, and figure metadata from result files."""
    spec = load_experiment_spec(study_dir)
    summary_path = spec.output_directory / "summary.csv"
    if not summary_path.is_file():
        raise ExperimentError(f"Experiment summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    _validate_summary_columns(spec, summary)

    tables_dir = spec.study_dir / "tables"
    figures_dir = spec.study_dir / "figures"
    reports_dir = spec.study_dir / "reports"
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_table = _metrics_table(spec, summary)
    comparisons_table = _comparisons_table(spec, summary)
    uncertainty_table = _uncertainty_table(spec, summary)
    _write_markdown_table(metrics_table, tables_dir / "metrics.md")
    _write_markdown_table(comparisons_table, tables_dir / "comparisons.md")
    _write_markdown_table(uncertainty_table, tables_dir / "uncertainty.md")
    _write_latex_table(metrics_table, tables_dir / "metrics.tex")
    _write_latex_table(comparisons_table, tables_dir / "comparisons.tex")
    _write_latex_table(uncertainty_table, tables_dir / "uncertainty.tex")

    figure_metadata = _write_figures(spec, summary, figures_dir)
    figure_metadata_path = figures_dir / "figure_metadata.json"
    figure_metadata_path.write_text(
        json.dumps(figure_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = reports_dir / "report.md"
    _write_report(
        spec,
        metrics_table,
        comparisons_table,
        uncertainty_table,
        figure_metadata,
        report_path,
    )
    return {
        "metrics_markdown": tables_dir / "metrics.md",
        "metrics_latex": tables_dir / "metrics.tex",
        "comparisons_markdown": tables_dir / "comparisons.md",
        "comparisons_latex": tables_dir / "comparisons.tex",
        "uncertainty_markdown": tables_dir / "uncertainty.md",
        "uncertainty_latex": tables_dir / "uncertainty.tex",
        "figure_metadata": figure_metadata_path,
        "report": report_path,
    }


def _validate_spec_files(spec: ExperimentSpec) -> None:
    for path in (spec.base_config_path, spec.scenario_file_path):
        if not path.is_file():
            raise ExperimentError(f"Experiment file not found: {path}")
    config = load_config(spec.base_config_path)
    if not config.paths.input_csv.is_file():
        raise ExperimentError(f"Experiment input data not found: {config.paths.input_csv}")


def _validate_pre_registered_design(spec: ExperimentSpec) -> None:
    for metric in spec.metrics:
        for key in ("name", "source_column", "unit"):
            if key not in metric:
                raise ExperimentError(f"Metric is missing required field '{key}'")
    for comparison in spec.comparisons:
        for key in ("id", "baseline", "scenario", "metric"):
            if key not in comparison:
                raise ExperimentError(f"Comparison is missing required field '{key}'")


def _validate_summary_columns(spec: ExperimentSpec, summary: pd.DataFrame) -> None:
    required = {"scenario_label"}
    required.update(str(metric["source_column"]) for metric in spec.metrics)
    missing = sorted(column for column in required if column not in summary.columns)
    if missing:
        raise ExperimentError("Experiment summary is missing columns: " + ", ".join(missing))
    labels = {str(item) for item in summary["scenario_label"].tolist()}
    for comparison in spec.comparisons:
        for key in ("baseline", "scenario"):
            label = str(comparison[key])
            if label not in labels:
                raise ExperimentError(f"Comparison references unknown scenario '{label}'")


def _metrics_table(spec: ExperimentSpec, summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in summary.sort_values("scenario_label").iterrows():
        record: dict[str, Any] = {"Scenario": row["scenario_label"]}
        for metric in spec.metrics:
            header = _metric_header(metric)
            record[header] = _round_value(row[str(metric["source_column"])])
        records.append(record)
    return pd.DataFrame.from_records(records)


def _comparisons_table(spec: ExperimentSpec, summary: pd.DataFrame) -> pd.DataFrame:
    by_label = {str(label): row for label, row in summary.set_index("scenario_label").iterrows()}
    records: list[dict[str, Any]] = []
    for comparison in spec.comparisons:
        metric = _metric_by_name(spec, str(comparison["metric"]))
        source = str(metric["source_column"])
        unit = str(metric["unit"])
        baseline = by_label[str(comparison["baseline"])]
        scenario = by_label[str(comparison["scenario"])]
        baseline_value = float(baseline[source])
        scenario_value = float(scenario[source])
        change = scenario_value - baseline_value
        percent = 0.0 if baseline_value == 0.0 else change / baseline_value * 100.0
        records.append(
            {
                "Comparison": comparison["id"],
                "Baseline": comparison["baseline"],
                "Scenario": comparison["scenario"],
                "Metric": metric.get("display_name", metric["name"]),
                "Unit": unit,
                "Baseline value": _round_value(baseline_value),
                "Scenario value": _round_value(scenario_value),
                "Change": _round_value(change),
                "Change [%]": _round_value(percent),
                "Paired seed": bool(comparison.get("paired_seed", False)),
            }
        )
    return pd.DataFrame.from_records(records)


def _uncertainty_table(spec: ExperimentSpec, summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if {"replication", "scenario_label"}.issubset(summary.columns):
        for metric in spec.metrics:
            source = str(metric["source_column"])
            unit = str(metric["unit"])
            grouped = summary.groupby("scenario_label")[source]
            for scenario, values in grouped:
                mean = float(values.mean())
                half_width = 1.96 * float(values.std(ddof=1)) / (len(values) ** 0.5)
                records.append(
                    {
                        "Type": "Monte Carlo 95% interval",
                        "Scope": scenario,
                        "Metric": metric.get("display_name", metric["name"]),
                        "Unit": unit,
                        "Lower": _round_value(mean - half_width),
                        "Upper": _round_value(mean + half_width),
                    }
                )
    for item in spec.sensitivity_ranges:
        metric = _metric_by_name(spec, str(item["metric"]))
        source = str(metric["source_column"])
        unit = str(metric["unit"])
        labels = [str(label) for label in item.get("scenarios", [])]
        scoped = summary[summary["scenario_label"].isin(labels)] if labels else summary
        records.append(
            {
                "Type": "Deterministic sensitivity range",
                "Scope": item.get("parameter", "pre-specified scenarios"),
                "Metric": metric.get("display_name", metric["name"]),
                "Unit": unit,
                "Lower": _round_value(float(scoped[source].min())),
                "Upper": _round_value(float(scoped[source].max())),
            }
        )
    if not records:
        records.append(
            {
                "Type": "Not specified",
                "Scope": "No Monte Carlo replications or deterministic ranges were configured",
                "Metric": "n/a",
                "Unit": "n/a",
                "Lower": "n/a",
                "Upper": "n/a",
            }
        )
    return pd.DataFrame.from_records(records)


def _write_figures(
    spec: ExperimentSpec,
    summary: pd.DataFrame,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for figure in spec.figures:
        metric = _metric_by_name(spec, str(figure["metric"]))
        source = str(metric["source_column"])
        title = str(figure.get("title", metric.get("display_name", metric["name"])))
        filename = str(figure.get("filename", f"{figure['id']}.png"))
        path = figures_dir / filename
        ordered = summary.sort_values("scenario_label")
        plt.figure(figsize=(8.0, 4.8))
        plt.bar(ordered["scenario_label"], ordered[source], color="#2f6f73")
        plt.ylabel(_metric_header(metric))
        plt.xlabel("Scenario")
        plt.title(title)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        metadata.append(
            {
                "id": figure["id"],
                "path": path.relative_to(spec.study_dir).as_posix(),
                "caption": figure.get("caption", title),
                "source": (spec.output_directory / "summary.csv")
                .relative_to(spec.study_dir)
                .as_posix(),
                "metric": metric["name"],
            }
        )
    return metadata


def _write_report(
    spec: ExperimentSpec,
    metrics_table: pd.DataFrame,
    comparisons_table: pd.DataFrame,
    uncertainty_table: pd.DataFrame,
    figure_metadata: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        f"# {spec.title}",
        "",
        "## Research Question",
        "",
        spec.research_question,
        "",
        "## Hypotheses",
        "",
        *_bullet_lines(spec.hypotheses),
        "",
        "## Assumptions",
        "",
        f"- Model version: {spec.model_version}",
        f"- Data version: {spec.data_version}",
        f"- Seed: {spec.seed}",
        "- Scenario metrics and comparisons were specified before execution in `study.yaml`.",
        "",
        "## Model Outputs",
        "",
        _markdown_table_text(metrics_table),
        "",
        "## Descriptive Comparisons",
        "",
        _markdown_table_text(comparisons_table),
        "",
        "## Uncertainty and Sensitivity",
        "",
        _markdown_table_text(uncertainty_table),
        "",
        "## Interpretation",
        "",
        "These are model-conditioned descriptive results. Differences report the outcomes "
        "of fixed scenario definitions under identical input data and, where specified, "
        "paired stochastic seeds.",
        "",
        "## What the Model Can Identify",
        "",
        *_bullet_lines(spec.model_can_identify),
        "",
        "## What the Model Cannot Identify",
        "",
        *_bullet_lines(spec.model_cannot_identify),
        "",
        "## Figures",
        "",
    ]
    for figure in figure_metadata:
        lines.append(f"- `{figure['path']}`: {figure['caption']}")
    lines.extend(["", "## Limitations", "", *_bullet_lines(spec.limitations), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(_markdown_table_text(frame) + "\n", encoding="utf-8")


def _write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(_latex_table_text(frame) + "\n", encoding="utf-8")


def _markdown_table_text(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[_markdown_cell(value) for value in row] for row in frame.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")


def _latex_table_text(frame: pd.DataFrame) -> str:
    columns = [_latex_cell(column) for column in frame.columns]
    rows = [[_latex_cell(value) for value in row] for row in frame.to_numpy().tolist()]
    alignment = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(columns) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _metric_by_name(spec: ExperimentSpec, name: str) -> dict[str, Any]:
    for metric in spec.metrics:
        if str(metric["name"]) == name:
            return metric
    raise ExperimentError(f"Unknown metric referenced by comparison or figure: {name}")


def _metric_header(metric: Mapping[str, Any]) -> str:
    label = str(metric.get("display_name", metric["name"]))
    return f"{label} [{metric['unit']}]"


def _paired_seed_groups(spec: ExperimentSpec) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for comparison in spec.comparisons:
        if comparison.get("paired_seed", False):
            groups.append(
                {
                    "comparison": comparison["id"],
                    "seed": spec.seed,
                    "scenarios": [comparison["baseline"], comparison["scenario"]],
                }
            )
    return groups


def _round_value(value: Any) -> Any:
    if pd.isna(value):
        return value
    if isinstance(value, bool | str):
        return value
    return round(float(value), 3)


def _bullet_lines(items: Sequence[str]) -> list[str]:
    if not items:
        return ["- Not specified."]
    return [f"- {item}" for item in items]


def _remove_within_study(path: Path, study_dir: Path) -> None:
    resolved_path = path.resolve()
    resolved_study = study_dir.resolve()
    try:
        resolved_path.relative_to(resolved_study)
    except ValueError as error:
        raise ExperimentError(f"Refusing to remove path outside study directory: {path}") from error
    shutil.rmtree(resolved_path)


def _git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"study.yaml must define non-empty string field '{key}'")
    return value


def _required_str_list(mapping: Mapping[str, Any], key: str) -> list[str]:
    return _str_list(mapping.get(key), key)


def _str_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ExperimentError(f"study.yaml field '{key}' must be a list of strings")
    return list(cast(list[str], value))


def _required_dict_list(mapping: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    return _dict_list(mapping.get(key), key)


def _dict_list(value: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ExperimentError(f"study.yaml field '{key}' must be a list of mappings")
    return [dict(cast(dict[str, Any], item)) for item in value]
