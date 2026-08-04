from __future__ import annotations

import json
from pathlib import Path

import pytest

from energy_system_simulator.cli import main
from energy_system_simulator.distribution_feeder import (
    BehindMeterBattery,
    DistributionBranch,
    DistributionBus,
    DistributionFeederProblem,
    HostingCapacityOptions,
    RooftopPV,
    load_distribution_problem,
    nonlinear_radial_power_flow,
    run_distribution_study,
)


def test_hosting_capacity_is_limited_by_voltage_bound() -> None:
    problem = DistributionFeederProblem(
        base_power_mva=10.0,
        periods=("sunny",),
        substation_bus_id="sub",
        buses=(
            DistributionBus("sub", voltage_min_pu=0.98, voltage_max_pu=1.02),
            DistributionBus("leaf", voltage_min_pu=0.98, voltage_max_pu=1.02),
        ),
        branches=(
            DistributionBranch(
                "sub_leaf",
                from_bus_id="sub",
                to_bus_id="leaf",
                resistance_pu=0.02,
                reactance_pu=0.04,
                rating_mva=20.0,
            ),
        ),
        rooftop_pv=(
            RooftopPV(
                "leaf_pv",
                bus_id="leaf",
                capacity_mw=0.0,
                availability_profile=(1.0,),
                hosting_capacity_max_mw=20.0,
            ),
        ),
        hosting=HostingCapacityOptions(max_curtailment_fraction=0.0),
    )

    result = run_distribution_study(problem, mode="hosting_capacity")

    expected = ((1.02**2) - 1.0) * problem.base_power_mva / (2.0 * 0.02)
    assert result.hosting_capacity_mw["leaf_pv"] == pytest.approx(expected)
    assert result.summary["total_hosting_capacity_mw"] == pytest.approx(expected)
    assert result.timeseries["dist_voltage_pu__leaf"].iloc[0] == pytest.approx(1.02)
    assert result.timeseries["dist_branch_loading_fraction__sub_leaf"].iloc[0] < 1.0


def test_operational_branch_rating_can_bind() -> None:
    problem = DistributionFeederProblem(
        base_power_mva=10.0,
        periods=("peak",),
        substation_bus_id="sub",
        buses=(
            DistributionBus("sub", voltage_min_pu=0.90, voltage_max_pu=1.05),
            DistributionBus("leaf", fixed_load_mw=(1.0,), voltage_min_pu=0.90, voltage_max_pu=1.05),
        ),
        branches=(
            DistributionBranch(
                "sub_leaf",
                from_bus_id="sub",
                to_bus_id="leaf",
                resistance_pu=0.01,
                reactance_pu=0.01,
                rating_mva=1.0,
            ),
        ),
    )

    result = run_distribution_study(problem)

    assert result.timeseries["dist_branch_active_flow_mw__sub_leaf"].iloc[0] == pytest.approx(1.0)
    assert result.timeseries["dist_branch_loading_fraction__sub_leaf"].iloc[0] == pytest.approx(1.0)
    assert result.summary["max_branch_loading_fraction"] == pytest.approx(1.0)


def test_customer_and_grid_side_batteries_have_separate_output_columns() -> None:
    problem = DistributionFeederProblem(
        periods=("idle",),
        substation_bus_id="sub",
        buses=(DistributionBus("sub"), DistributionBus("leaf")),
        branches=(
            DistributionBranch(
                "sub_leaf",
                from_bus_id="sub",
                to_bus_id="leaf",
                resistance_pu=0.01,
                reactance_pu=0.02,
                rating_mva=5.0,
            ),
        ),
        batteries=(
            BehindMeterBattery(
                "customer",
                bus_id="leaf",
                side="customer_side",
                power_capacity_mw=1.0,
                energy_capacity_mwh=2.0,
                initial_soc_mwh=1.0,
            ),
            BehindMeterBattery(
                "grid",
                bus_id="leaf",
                side="grid_side",
                power_capacity_mw=1.0,
                energy_capacity_mwh=2.0,
                initial_soc_mwh=1.0,
            ),
        ),
    )

    result = run_distribution_study(problem)

    assert "dist_customer_side_battery_charge_mw__customer" in result.timeseries
    assert "dist_grid_side_battery_charge_mw__grid" in result.timeseries
    assert result.summary["customer_side_battery_throughput_mwh"] == pytest.approx(0.0)
    assert result.summary["grid_side_battery_throughput_mwh"] == pytest.approx(0.0)


def test_linear_distflow_matches_independent_nonlinear_check_for_small_load() -> None:
    problem = DistributionFeederProblem(
        base_power_mva=10.0,
        periods=("light_load",),
        substation_bus_id="sub",
        buses=(
            DistributionBus("sub"),
            DistributionBus("leaf", fixed_load_mw=(0.2,), fixed_reactive_load_mvar=(0.05,)),
        ),
        branches=(
            DistributionBranch(
                "sub_leaf",
                from_bus_id="sub",
                to_bus_id="leaf",
                resistance_pu=0.02,
                reactance_pu=0.04,
                rating_mva=5.0,
            ),
        ),
    )
    result = run_distribution_study(problem)

    nonlinear_voltage = nonlinear_radial_power_flow(problem, result.timeseries, period=0)

    assert result.timeseries["dist_voltage_pu__leaf"].iloc[0] == pytest.approx(
        nonlinear_voltage["leaf"],
        abs=2e-4,
    )


def test_distribution_cli_writes_namespaced_outputs(tmp_path: Path, capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "distribution"

    main(
        [
            "distribution-study",
            "--problem",
            str(root / "configs" / "distribution_radial_feeder.yaml"),
            "--output",
            str(output),
            "--mode",
            "hosting-capacity",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads((output / "distribution_summary.json").read_text(encoding="utf-8"))
    assert "Distribution study written" in captured.out
    assert (output / "distribution_timeseries.csv").is_file()
    assert (output / "distribution_hosting_capacity.csv").is_file()
    assert summary["model_family"] == "distribution_radial_distflow"
    assert summary["total_hosting_capacity_mw"] > 0.0


def test_distribution_example_loads() -> None:
    root = Path(__file__).resolve().parents[1]

    problem = load_distribution_problem(root / "configs" / "distribution_radial_feeder.yaml")

    assert problem.substation_bus_id == "substation"
    assert {battery.side for battery in problem.batteries} == {"customer_side", "grid_side"}
