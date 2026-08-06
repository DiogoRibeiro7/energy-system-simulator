from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

import yaml

from energy_system_simulator.ac_validation import ACValidationOptions, validate_ac_power_flow
from energy_system_simulator.api import (
    build_model,
    ensure_writable_output_directory,
    load_data,
    load_model_config,
    solve,
    validate_model_config,
)
from energy_system_simulator.config import ModelConfig, migrate_legacy_config, validate_config
from energy_system_simulator.dispatch.solver import export_problem_lp
from energy_system_simulator.distribution_feeder import (
    DistributionMode,
    load_distribution_problem,
    run_distribution_study,
)
from energy_system_simulator.exceptions import (
    ConfigurationError,
    DataValidationError,
    EnergySystemError,
    OptimisationError,
)
from energy_system_simulator.experiments import (
    analyze_research_experiment,
    reproduce_research_experiment,
    run_research_experiment,
)
from energy_system_simulator.frequency import evaluate_frequency_adequacy
from energy_system_simulator.heat import load_heat_problem, run_heat_study
from energy_system_simulator.hydrogen import load_hydrogen_problem, run_hydrogen_study
from energy_system_simulator.metadata import get_package_version
from energy_system_simulator.planning import (
    CapacityExpansionPlanner,
    CapacityExpansionProblem,
    GenerationCandidate,
    InterconnectorCandidate,
    PlanningBlock,
    PlanningPolicy,
    StorageCandidate,
    TransmissionCandidate,
)
from energy_system_simulator.reporting import (
    compare_output_directories,
    write_dashboard,
    write_outputs,
)
from energy_system_simulator.scenarios import apply_overrides, run_experiment_file
from energy_system_simulator.security import (
    SecurityOptions,
    default_contingencies,
    evaluate_security,
)
from energy_system_simulator.simulation import OutageModel, ReliabilityStudy, ReliabilityStudyConfig
from energy_system_simulator.simulation.reliability import OutageAssetType

LOGGER = logging.getLogger(__name__)


class ExitCode(IntEnum):
    SUCCESS = 0
    INVALID_CONFIGURATION = 2
    INVALID_DATA = 3
    INFEASIBLE_MODEL = 4
    SOLVER_FAILURE = 5
    PARTIAL_FEASIBLE_RESULT = 6


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="energy-sim",
        description="Simulate and analyze hybrid electricity systems.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_package_version()}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and requested machine-readable output",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate configuration and input data")
    _add_config_argument(validate)
    _add_json_argument(validate)

    validate_config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate configuration only",
    )
    _add_config_argument(validate_config_parser)
    _add_json_argument(validate_config_parser)

    validate_data_parser = subparsers.add_parser(
        "validate-data",
        help="Validate input data referenced by a configuration",
    )
    _add_config_argument(validate_data_parser)
    _add_json_argument(validate_data_parser)

    migrate = subparsers.add_parser(
        "migrate-config",
        help="Migrate a schema_version 1 config to the portfolio schema",
    )
    _add_config_argument(migrate)
    migrate.add_argument("--output", type=Path)
    migrate.add_argument("--overwrite", action="store_true")

    simulate = subparsers.add_parser("simulate", help="Run the configured simulation")
    _add_config_argument(simulate)
    _add_run_arguments(simulate)

    rolling = subparsers.add_parser(
        "rolling-horizon",
        help="Run simulation with rolling-horizon mode enabled",
    )
    _add_config_argument(rolling)
    _add_run_arguments(rolling)

    scenarios = subparsers.add_parser("run-scenarios", help="Run a scenario experiment YAML")
    _add_scenario_arguments(scenarios)
    scenario_experiment = subparsers.add_parser(
        "scenario-experiment",
        help="Run a scenario experiment YAML",
    )
    _add_scenario_arguments(scenario_experiment)

    research_experiment = subparsers.add_parser(
        "run-experiment",
        help="Run a registered research experiment study",
    )
    _add_research_experiment_arguments(research_experiment)

    reproduce_experiment = subparsers.add_parser(
        "reproduce-experiment",
        help="Verify a research manifest and rerun the recorded experiment",
    )
    reproduce_experiment.add_argument("--manifest", type=Path, required=True)
    reproduce_experiment.add_argument("--overwrite", action="store_true")
    reproduce_experiment.add_argument(
        "--no-plots", action="store_true", help="Skip generated experiment PNG plots"
    )

    analyze_experiment = subparsers.add_parser(
        "analyze-experiment",
        help="Regenerate research experiment tables, figures, and report",
    )
    analyze_experiment.add_argument("--study", type=Path, required=True)

    reliability = subparsers.add_parser("reliability-study", help="Run a reliability study")
    _add_config_argument(reliability)
    reliability.add_argument("--replications", type=int, default=1)
    reliability.add_argument("--seed", type=int, default=1)
    reliability.add_argument("--workers", type=int, default=1)
    reliability.add_argument(
        "--outage",
        action="append",
        default=[],
        metavar="ASSET:TYPE:FOR:MTTR",
        help="Outage model, for example unit1:thermal:0.05:8",
    )
    reliability.add_argument("--output", type=Path)
    reliability.add_argument("--overwrite", action="store_true")

    security = subparsers.add_parser(
        "security-check",
        help="Evaluate explicit post-contingency N-1 security for nodal dispatch",
    )
    _add_config_argument(security)
    security.add_argument("--output", type=Path, required=True)
    security.add_argument("--overwrite", action="store_true")
    security.add_argument(
        "--contingencies",
        default="lines,generators",
        help="Comma-separated outage classes: lines,generators,imports",
    )
    security.add_argument(
        "--hard",
        action="store_true",
        help="Disable emergency load shedding and overload slack in contingency checks",
    )
    security.add_argument(
        "--no-committed-reserve-limit",
        action="store_true",
        help="Use ramp/headroom limits instead of procured reserve columns for redispatch bounds",
    )

    frequency = subparsers.add_parser(
        "frequency-check",
        help="Evaluate post-dispatch frequency adequacy proxies",
    )
    _add_config_argument(frequency)
    frequency.add_argument("--output", type=Path, required=True)
    frequency.add_argument("--overwrite", action="store_true")

    ac = subparsers.add_parser(
        "ac-validate",
        help="Validate selected nodal dispatch periods with AC power flow",
    )
    _add_config_argument(ac)
    ac.add_argument("--output", type=Path, required=True)
    ac.add_argument("--overwrite", action="store_true")
    ac.add_argument(
        "--policy",
        action="append",
        choices=("peak_demand", "peak_renewable", "congestion"),
        help="Period-selection policy; may be supplied multiple times",
    )
    ac.add_argument("--period", action="append", type=int, default=[])
    ac.add_argument("--timestamp", action="append", default=[])

    distribution = subparsers.add_parser(
        "distribution-study",
        help="Run a standalone radial distribution-feeder study",
    )
    distribution.add_argument("--problem", type=Path, required=True)
    distribution.add_argument("--output", type=Path, required=True)
    distribution.add_argument("--overwrite", action="store_true")
    distribution.add_argument(
        "--mode",
        choices=("operational", "hosting-capacity"),
        default="operational",
        help="Run ordinary DER dispatch or maximise incremental DER hosting capacity",
    )

    hydrogen = subparsers.add_parser(
        "hydrogen-study",
        help="Run a standalone hydrogen subsystem study",
    )
    hydrogen.add_argument("--problem", type=Path, required=True)
    hydrogen.add_argument("--output", type=Path, required=True)
    hydrogen.add_argument("--overwrite", action="store_true")

    heat = subparsers.add_parser(
        "heat-study",
        help="Run a standalone district heat and CHP subsystem study",
    )
    heat.add_argument("--problem", type=Path, required=True)
    heat.add_argument("--output", type=Path, required=True)
    heat.add_argument("--overwrite", action="store_true")

    planning = subparsers.add_parser("capacity-planning", help="Run a capacity-planning YAML")
    planning.add_argument("--problem", type=Path, required=True)
    planning.add_argument("--output", type=Path)
    planning.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare-outputs", help="Compare output directories")
    compare.add_argument("outputs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--overwrite", action="store_true")

    dashboard = subparsers.add_parser(
        "dashboard",
        help="Write a self-contained HTML dashboard for an output directory",
    )
    dashboard.add_argument("--output-dir", type=Path, required=True)
    dashboard.add_argument("--output", type=Path)
    dashboard.add_argument("--overwrite", action="store_true")

    export = subparsers.add_parser("export-model", help="Export the dispatch formulation")
    _add_export_arguments(export)
    export_formulation = subparsers.add_parser(
        "export-formulation",
        help="Export the dispatch formulation",
    )
    _add_export_arguments(export_formulation)

    prepare_data = subparsers.add_parser(
        "prepare-data",
        help="Transform local public-data files into a canonical input snapshot",
    )
    prepare_data.add_argument("--spec", type=Path, required=True)

    subparsers.add_parser("capabilities", help="Show version, commands, and exit codes")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose, quiet=args.quiet)
    json_output = bool(getattr(args, "json_output", False))
    try:
        exit_code = _dispatch(args)
    except ConfigurationError as error:
        _print_error(error, json_output=json_output, label="Configuration failed")
        raise SystemExit(ExitCode.INVALID_CONFIGURATION) from error
    except DataValidationError as error:
        _print_error(error, json_output=json_output, label="Data validation failed")
        raise SystemExit(ExitCode.INVALID_DATA) from error
    except OptimisationError as error:
        code = (
            ExitCode.INFEASIBLE_MODEL
            if "infeasible" in str(error).lower()
            else ExitCode.SOLVER_FAILURE
        )
        _print_error(error, json_output=json_output, label="Optimisation failed")
        raise SystemExit(code) from error
    except EnergySystemError as error:
        _print_error(error, json_output=json_output, label="Execution failed")
        raise SystemExit(ExitCode.SOLVER_FAILURE) from error
    if exit_code != ExitCode.SUCCESS:
        raise SystemExit(exit_code)


def _dispatch(args: argparse.Namespace) -> ExitCode:
    if args.command == "capabilities":
        print(json.dumps(_capabilities(), indent=2, sort_keys=True))
        return ExitCode.SUCCESS

    if args.command == "migrate-config":
        return _migrate_config(args)

    if args.command in {"run-scenarios", "scenario-experiment"}:
        aggregate = run_experiment_file(
            args.experiment,
            workers=args.workers,
            resume=args.resume,
            create_plots=not args.no_plots,
        )
        print(f"Scenario experiment complete: {len(aggregate)} scenarios")
        return ExitCode.SUCCESS

    if args.command == "run-experiment":
        research_result = run_research_experiment(
            args.study,
            overwrite=args.overwrite,
            create_plots=not args.no_plots,
        )
        print(f"Research experiment complete: {research_result['summary']}")
        print(f"Research manifest written: {research_result['manifest']}")
        print(f"Research report written: {research_result['report']}")
        return ExitCode.SUCCESS

    if args.command == "reproduce-experiment":
        research_result = reproduce_research_experiment(
            args.manifest,
            overwrite=args.overwrite,
            create_plots=not args.no_plots,
        )
        print(f"Research experiment reproduced: {research_result['summary']}")
        print(f"Research manifest written: {research_result['manifest']}")
        print(f"Research report written: {research_result['report']}")
        return ExitCode.SUCCESS

    if args.command == "analyze-experiment":
        analysis_result = analyze_research_experiment(args.study)
        print(f"Research report written: {analysis_result['report']}")
        return ExitCode.SUCCESS

    if args.command == "prepare-data":
        from energy_system_simulator.data_adapters import run_data_preparation_spec

        snapshot = run_data_preparation_spec(args.spec)
        print(f"Canonical data written: {snapshot.output_csv}")
        print(f"Manifest written: {snapshot.manifest_json}")
        return ExitCode.SUCCESS

    if args.command == "compare-outputs":
        _ensure_writable_file(args.output, overwrite=args.overwrite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table = compare_output_directories(args.outputs, args.output)
        print(f"Comparison report written: {args.output}")
        print(f"Compared metrics: {table['metric'].nunique() if not table.empty else 0}")
        return ExitCode.SUCCESS

    if args.command == "dashboard":
        path = args.output if args.output is not None else args.output_dir / "dashboard.html"
        _ensure_writable_file(path, overwrite=args.overwrite)
        try:
            dashboard_path = write_dashboard(args.output_dir, path)
        except (FileNotFoundError, ValueError) as error:
            raise ConfigurationError(str(error)) from error
        print(f"Dashboard written: {dashboard_path}")
        return ExitCode.SUCCESS

    if args.command == "capacity-planning":
        return _run_capacity_planning(args)

    if args.command == "reliability-study":
        return _run_reliability_study(args)

    if args.command == "security-check":
        return _run_security_check(args)

    if args.command == "frequency-check":
        return _run_frequency_check(args)

    if args.command == "ac-validate":
        return _run_ac_validation(args)

    if args.command == "distribution-study":
        return _run_distribution_study(args)

    if args.command == "hydrogen-study":
        return _run_hydrogen_study(args)

    if args.command == "heat-study":
        return _run_heat_study(args)

    if args.command in {"validate", "validate-config", "validate-data"}:
        return _validate(args)

    config, overrides = _load_config_with_overrides(args)

    if args.command == "rolling-horizon":
        overrides = {
            **overrides,
            "rolling_horizon.enabled": True,
        }
        config = apply_overrides(config, {"rolling_horizon.enabled": True})
        validate_config(config)

    if args.command in {"export-model", "export-formulation"}:
        _ensure_writable_file(args.output, overwrite=args.overwrite)
        problem = build_model(config)
        if args.format == "lp":
            export_problem_lp(problem.solver_problem(), args.output)
            print(f"Model exported: {args.output}")
        return ExitCode.SUCCESS

    if args.command in {"simulate", "rolling-horizon"}:
        if args.dry_run:
            problem = build_model(config)
            _print_model_size(problem.statistics)
            return ExitCode.SUCCESS
        ensure_writable_output_directory(
            config.paths.output_directory,
            overwrite=args.overwrite,
            resume=args.resume,
        )
        result = solve(config)
        write_outputs(
            result,
            config.paths.output_directory,
            config=config,
            config_path=args.config,
            create_plots=not args.no_plots,
            command_line_overrides=overrides,
        )
        print(f"Simulation complete: {config.paths.output_directory}")
        print(f"Objective: EUR {result.objective_eur:,.2f}")
        print(f"Unserved energy: {result.summary['unserved_energy_mwh']:.3f} MWh")
        print(f"Renewable share: {result.summary['renewable_share_of_primary_generation']:.2%}")
        if result.solver_status != "optimal":
            return ExitCode.PARTIAL_FEASIBLE_RESULT
        return ExitCode.SUCCESS

    raise RuntimeError(f"Unsupported command: {args.command}")


def _validate(args: argparse.Namespace) -> ExitCode:
    config = load_model_config(args.config)
    periods: int | None = None
    if args.command in {"validate", "validate-data"}:
        periods = len(load_data(config))
    else:
        validate_model_config(config)
    if args.json_output:
        payload: dict[str, object] = {"ok": True}
        if periods is not None:
            payload["periods"] = periods
        print(json.dumps(payload, sort_keys=True))
    elif periods is None:
        print("Configuration valid.")
    else:
        print(f"Configuration valid. Input contains {periods} periods.")
    return ExitCode.SUCCESS


def _migrate_config(args: argparse.Namespace) -> ExitCode:
    migrated = migrate_legacy_config(args.config)
    text = yaml.safe_dump(migrated, sort_keys=False)
    if args.output is not None:
        _ensure_writable_file(args.output, overwrite=args.overwrite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Migrated configuration written: {args.output}")
    else:
        print(text, end="")
    return ExitCode.SUCCESS


def _run_reliability_study(args: argparse.Namespace) -> ExitCode:
    config = load_model_config(args.config)
    study = ReliabilityStudy(
        config,
        ReliabilityStudyConfig(
            replications=args.replications,
            seed=args.seed,
            outage_models=tuple(_parse_outage_model(item) for item in args.outage),
            parallel_workers=args.workers,
        ),
    )
    result = study.run()
    payload = {
        "metrics": result.metrics,
        "confidence_intervals": result.confidence_intervals,
        "convergence": result.convergence,
    }
    if args.output is not None:
        _ensure_writable_file(args.output, overwrite=args.overwrite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Reliability study written: {args.output}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return ExitCode.SUCCESS


def _run_security_check(args: argparse.Namespace) -> ExitCode:
    config = load_model_config(args.config)
    include_lines, include_generators, include_imports = _parse_security_contingencies(
        args.contingencies
    )
    output = ensure_writable_output_directory(
        args.output,
        overwrite=args.overwrite,
        resume=False,
    )
    result = solve(config)
    evaluation = evaluate_security(
        config,
        result,
        contingencies=default_contingencies(
            config,
            include_lines=include_lines,
            include_generators=include_generators,
            include_imports=include_imports,
        ),
        options=SecurityOptions(
            allow_emergency_actions=not args.hard,
            require_committed_reserves=not args.no_committed_reserve_limit,
        ),
    )
    evaluation.write(output)
    print(f"Security diagnostics written: {output}")
    print(f"Secure: {evaluation.secure}")
    security_cost = cast(float | None, evaluation.summary["total_security_cost_eur"])
    if security_cost is None:
        print("Security cost: undefined for infeasible hard checks")
    else:
        print(f"Security cost: EUR {float(security_cost):,.2f}")
    return ExitCode.SUCCESS


def _run_frequency_check(args: argparse.Namespace) -> ExitCode:
    config = load_model_config(args.config)
    output = ensure_writable_output_directory(
        args.output,
        overwrite=args.overwrite,
        resume=False,
    )
    result = solve(config)
    evaluation = evaluate_frequency_adequacy(config, result)
    evaluation.write(output)
    print(f"Frequency diagnostics written: {output}")
    print(f"Adequate: {evaluation.adequate}")
    print(f"Scarcity periods: {int(cast(int, evaluation.summary['scarcity_periods']))}")
    return ExitCode.SUCCESS


def _run_ac_validation(args: argparse.Namespace) -> ExitCode:
    config = load_model_config(args.config)
    output = ensure_writable_output_directory(
        args.output,
        overwrite=args.overwrite,
        resume=False,
    )
    result = solve(config)
    validation = validate_ac_power_flow(
        config,
        result,
        options=ACValidationOptions(
            policies=tuple(args.policy)
            if args.policy
            else ("peak_demand", "peak_renewable", "congestion"),
            periods=tuple(args.period),
            timestamps=tuple(args.timestamp),
        ),
    )
    validation.write(output)
    print(f"AC validation diagnostics written: {output}")
    print(f"Valid: {validation.valid}")
    print(f"Validated periods: {validation.summary['validated_periods']}")
    return ExitCode.SUCCESS


def _run_distribution_study(args: argparse.Namespace) -> ExitCode:
    output = ensure_writable_output_directory(
        args.output,
        overwrite=args.overwrite,
        resume=False,
    )
    problem = load_distribution_problem(args.problem)
    mode: DistributionMode = (
        "hosting_capacity" if args.mode == "hosting-capacity" else "operational"
    )
    result = run_distribution_study(problem, mode=mode)
    result.write(output)
    print(f"Distribution study written: {output}")
    print(f"Mode: {result.mode}")
    print(f"Total hosting capacity: {result.summary['total_hosting_capacity_mw']:.3f} MW")
    return ExitCode.SUCCESS


def _run_hydrogen_study(args: argparse.Namespace) -> ExitCode:
    output = ensure_writable_output_directory(
        args.output,
        overwrite=args.overwrite,
        resume=False,
    )
    problem = load_hydrogen_problem(args.problem)
    result = run_hydrogen_study(problem)
    result.write(output)
    print(f"Hydrogen study written: {output}")
    print(f"Hydrogen produced: {result.summary['hydrogen_produced_mwh_lhv']:.3f} MWh-LHV")
    print(f"Reconverted electricity: {result.summary['reconverted_electricity_mwh']:.3f} MWh")
    return ExitCode.SUCCESS


def _run_heat_study(args: argparse.Namespace) -> ExitCode:
    output = ensure_writable_output_directory(
        args.output,
        overwrite=args.overwrite,
        resume=False,
    )
    problem = load_heat_problem(args.problem)
    result = run_heat_study(problem)
    result.write(output)
    print(f"Heat study written: {output}")
    print(f"Heat unmet: {result.summary['unmet_heat_mwh_th']:.3f} MWh-th")
    print(f"Electricity from CHP: {result.summary['electricity_from_chp_mwh']:.3f} MWh")
    return ExitCode.SUCCESS


def _parse_security_contingencies(raw: str) -> tuple[bool, bool, bool]:
    selected = {item.strip().lower() for item in raw.split(",") if item.strip()}
    supported = {"lines", "generators", "imports"}
    unknown = selected - supported
    if unknown:
        raise ConfigurationError(
            "Unsupported security contingency classes: " + ", ".join(sorted(unknown))
        )
    if not selected:
        raise ConfigurationError("At least one security contingency class is required")
    return "lines" in selected, "generators" in selected, "imports" in selected


def _run_capacity_planning(args: argparse.Namespace) -> ExitCode:
    problem = _capacity_problem_from_yaml(args.problem)
    result = CapacityExpansionPlanner().solve(problem)
    payload = {
        "selected_generation_capacity_mw": result.selected_generation_capacity_mw,
        "selected_storage_power_mw": result.selected_storage_power_mw,
        "selected_storage_energy_mwh": result.selected_storage_energy_mwh,
        "selected_interconnector_capacity_mw": result.selected_interconnector_capacity_mw,
        "selected_transmission_capacity_mw": result.selected_transmission_capacity_mw,
        "annual_costs_eur": result.annual_costs_eur,
        "generation_mix_mwh": result.generation_mix_mwh,
        "curtailment_mwh": result.curtailment_mwh,
        "emissions_tonnes": result.emissions_tonnes,
        "unserved_energy_mwh": result.unserved_energy_mwh,
        "planning_reserve_margin": result.planning_reserve_margin,
        "policy_shadow_prices": result.policy_shadow_prices,
        "objective_eur": result.objective_eur,
        "solver_message": result.solver_message,
    }
    if args.output is not None:
        _ensure_writable_file(args.output, overwrite=args.overwrite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        dispatch_path = args.output.with_suffix(".dispatch.csv")
        result.dispatch.to_csv(dispatch_path, index=False)
        print(f"Capacity plan written: {args.output}")
        print(f"Dispatch written: {dispatch_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return ExitCode.SUCCESS


def _load_config_with_overrides(args: argparse.Namespace) -> tuple[ModelConfig, dict[str, Any]]:
    config = load_model_config(args.config)
    overrides = _parse_overrides(getattr(args, "overrides", []) or [])
    if overrides:
        try:
            config = apply_overrides(config, overrides)
        except EnergySystemError as error:
            raise ConfigurationError(str(error)) from error
        validate_config(config)
    return config, overrides


def _parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        path, separator, raw_value = item.partition("=")
        if not separator or not path:
            raise ConfigurationError("Overrides must use PATH=VALUE syntax")
        overrides[path] = yaml.safe_load(raw_value)
    return overrides


def _parse_outage_model(raw: str) -> OutageModel:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ConfigurationError("Outage models must use ASSET:TYPE:FOR:MTTR syntax")
    asset_id, asset_type, forced_outage_rate, mean_time_to_repair_hours = parts
    if asset_type not in {"thermal", "renewable", "storage", "line", "import"}:
        raise ConfigurationError(f"Unsupported outage asset type: {asset_type}")
    return OutageModel(
        asset_id,
        cast(OutageAssetType, asset_type),
        float(forced_outage_rate),
        float(mean_time_to_repair_hours),
    )


def _capacity_problem_from_yaml(path: Path) -> CapacityExpansionProblem:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Capacity-planning problem must be a mapping")
    return CapacityExpansionProblem(
        demand_mw={str(key): value for key, value in _mapping(payload, "demand_mw").items()},
        representative_weights_hours=payload["representative_weights_hours"],
        annual_hours=float(payload.get("annual_hours", 8760.0)),
        generation_candidates=tuple(
            GenerationCandidate(**item) for item in payload.get("generation_candidates", []) or []
        ),
        storage_candidates=tuple(
            StorageCandidate(**item) for item in payload.get("storage_candidates", []) or []
        ),
        interconnector_candidates=tuple(
            InterconnectorCandidate(**item)
            for item in payload.get("interconnector_candidates", []) or []
        ),
        transmission_candidates=tuple(
            TransmissionCandidate(**item)
            for item in payload.get("transmission_candidates", []) or []
        ),
        policy=PlanningPolicy(**(payload.get("policy") or {})),
        reliability_penalty_eur_per_mwh=float(
            payload.get("reliability_penalty_eur_per_mwh", 100_000.0)
        ),
        blocks=tuple(PlanningBlock(**item) for item in payload.get("blocks", []) or []),
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _print_model_size(statistics: Any) -> None:
    print("Dry run complete. Model was built but not solved.")
    print(f"Continuous variables: {statistics.continuous_variables}")
    print(f"Integer variables: {statistics.integer_variables}")
    print(f"Binary variables: {statistics.binary_variables}")
    print(f"Linear constraints: {statistics.linear_constraints}")
    print(f"Matrix nonzeros: {statistics.matrix_nonzeros}")


def _print_error(error: Exception, *, json_output: bool, label: str) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": error.__class__.__name__,
                        "message": str(error),
                    },
                },
                sort_keys=True,
            )
        )
    else:
        print(f"{label}: {error}", file=sys.stderr)


def _ensure_writable_file(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ConfigurationError(f"Output file already exists: {path}. Use --overwrite.")


def _configure_logging(verbosity: int, *, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    LOGGER.debug("Logging configured at level %s", logging.getLevelName(level))


def _capabilities() -> dict[str, Any]:
    return {
        "version": get_package_version(),
        "commands": [
            "validate",
            "validate-config",
            "validate-data",
            "migrate-config",
            "simulate",
            "rolling-horizon",
            "run-scenarios",
            "scenario-experiment",
            "run-experiment",
            "reproduce-experiment",
            "analyze-experiment",
            "reliability-study",
            "security-check",
            "frequency-check",
            "ac-validate",
            "distribution-study",
            "hydrogen-study",
            "heat-study",
            "capacity-planning",
            "compare-outputs",
            "dashboard",
            "export-model",
            "export-formulation",
            "prepare-data",
            "capabilities",
        ],
        "exit_codes": {code.name.lower(): int(code) for code in ExitCode},
        "solver_backends": ["scipy.optimize.milp"],
        "output_schema_version": 1,
    }


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit result or validation errors as JSON",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation")
    parser.add_argument("--dry-run", action="store_true", help="Build and size the model only")
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow writing into existing outputs"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Allow resuming or reusing output state"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="PATH=VALUE",
        help="Validated dotted configuration override",
    )


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-plots", action="store_true", help="Skip per-scenario PNG plots")


def _add_research_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generated experiment PNG plots"
    )


def _add_export_arguments(parser: argparse.ArgumentParser) -> None:
    _add_config_argument(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("lp",), default="lp")
    parser.add_argument("--overwrite", action="store_true")
