from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_system_simulator.config import load_config
from energy_system_simulator.constants import DEFAULT_NUMERICAL_POLICY
from energy_system_simulator.dispatch import UnitCommitment
from energy_system_simulator.dispatch.solver import relative_gap
from energy_system_simulator.exceptions import OptimisationError
from energy_system_simulator.simulation import SimulationEngine


def _unit_commitment() -> UnitCommitment:
    root = Path(__file__).resolve().parents[1]
    return UnitCommitment(load_config(root / "configs" / "example.yaml"))


def test_numerical_policy_is_immutable_and_names_distinct_semantics() -> None:
    policy = DEFAULT_NUMERICAL_POLICY
    assert policy.primal_feasibility_mw < policy.report_rounding
    assert policy.integrality < policy.objective_reconciliation_eur
    assert policy.nonnegative_cleanup < policy.primal_feasibility_mw
    with pytest.raises(FrozenInstanceError):
        policy.integrality = 1.0  # type: ignore[misc]


def test_relative_gap_uses_objective_tolerance_boundary() -> None:
    tolerance = DEFAULT_NUMERICAL_POLICY.objective_reconciliation_eur
    assert relative_gap(0.0, tolerance / 2.0) == 0.0
    assert relative_gap(0.0, tolerance * 2.0) is None
    assert relative_gap(-100.0, -101.0) == pytest.approx(0.01)


def test_integrality_cleanup_accepts_values_inside_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "thermal_on": [1.0 - DEFAULT_NUMERICAL_POLICY.integrality / 2.0],
            "thermal_startup": [DEFAULT_NUMERICAL_POLICY.integrality / 2.0],
            "thermal_shutdown": [0.0],
            "battery_charge_mode": [1.0],
        }
    )

    deviation = _unit_commitment()._coerce_binary_columns(frame)

    assert deviation <= DEFAULT_NUMERICAL_POLICY.integrality
    assert frame["thermal_on"].tolist() == [1]
    assert frame["thermal_startup"].tolist() == [0]


def test_integrality_cleanup_rejects_values_above_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "thermal_on": [1.0 - DEFAULT_NUMERICAL_POLICY.integrality * 2.0],
            "thermal_startup": [0.0],
            "thermal_shutdown": [0.0],
            "battery_charge_mode": [1.0],
        }
    )

    with pytest.raises(OptimisationError, match="Integrality residual"):
        _unit_commitment()._coerce_binary_columns(frame)


def test_nonnegative_cleanup_clips_only_solver_noise() -> None:
    frame = pd.DataFrame(
        {
            "renewable_used_mw": [-DEFAULT_NUMERICAL_POLICY.nonnegative_cleanup / 2.0],
            "thermal_output_mw": [0.0],
            "battery_charge_mw": [0.0],
            "battery_discharge_mw": [0.0],
            "battery_soc_mwh": [0.0],
            "imports_mw": [0.0],
            "source_load_shed_mw": [0.0],
            "renewable_curtailed_mw": [0.0],
        }
    )

    clipped = _unit_commitment()._clip_nonnegative_solver_noise(frame)

    assert clipped == pytest.approx(DEFAULT_NUMERICAL_POLICY.nonnegative_cleanup / 2.0)
    assert frame["renewable_used_mw"].iloc[0] == 0.0


def test_nonnegative_cleanup_rejects_material_negative_values() -> None:
    frame = pd.DataFrame(
        {
            "renewable_used_mw": [-DEFAULT_NUMERICAL_POLICY.nonnegative_cleanup * 2.0],
            "thermal_output_mw": [0.0],
            "battery_charge_mw": [0.0],
            "battery_discharge_mw": [0.0],
            "battery_soc_mwh": [0.0],
            "imports_mw": [0.0],
            "source_load_shed_mw": [0.0],
            "renewable_curtailed_mw": [0.0],
        }
    )

    with pytest.raises(OptimisationError, match="Negative solver value"):
        _unit_commitment()._clip_nonnegative_solver_noise(frame)


def test_residual_summary_identifies_family_worst_period_and_scale() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "source_balance_residual_mw": np.array([0.0, -0.5, 0.25]),
            "left_mw": np.array([10.0, 20.0, 30.0]),
            "right_mw": np.array([10.0, 25.0, 30.0]),
        }
    )

    summary = SimulationEngine._residual_summary(
        frame,
        "source_balance",
        "source_balance_residual_mw",
        ("left_mw", "right_mw"),
    )

    assert summary["equation_family"] == "source_balance"
    assert summary["period_index"] == 1
    assert summary["max_abs_residual"] == pytest.approx(0.5)
    assert summary["scale"] == pytest.approx(45.0)
    assert summary["scale_normalized_residual"] == pytest.approx(0.5 / 45.0)
