"""Numerical policy used across optimisation, validation, and reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalPolicy:
    """Typed tolerances with one semantic purpose per field."""

    primal_feasibility_mw: float = 1e-6
    energy_reconciliation_mwh: float = 1e-6
    integrality: float = 1e-7
    objective_reconciliation_eur: float = 1e-4
    nonnegative_cleanup: float = 1e-9
    report_rounding: float = 1e-5
    time_axis_seconds: float = 1e-6
    dc_power_balance_mw: float = 1e-8


DEFAULT_NUMERICAL_POLICY = NumericalPolicy()
