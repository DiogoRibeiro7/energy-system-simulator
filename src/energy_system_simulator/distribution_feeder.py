from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from energy_system_simulator.exceptions import ConfigurationError, OptimisationError

FloatArray = npt.NDArray[np.float64]
BatterySide = Literal["customer_side", "grid_side"]
DistributionMode = Literal["operational", "hosting_capacity"]


@dataclass(frozen=True)
class DistributionBus:
    """Single-phase equivalent feeder bus."""

    id: str
    fixed_load_mw: npt.ArrayLike = 0.0
    fixed_reactive_load_mvar: npt.ArrayLike = 0.0
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05


@dataclass(frozen=True)
class DistributionBranch:
    """Radial feeder branch. Positive flow is from from_bus_id to to_bus_id."""

    id: str
    from_bus_id: str
    to_bus_id: str
    resistance_pu: float
    reactance_pu: float
    rating_mva: float


@dataclass(frozen=True)
class RooftopPV:
    """Rooftop PV connected behind a distribution bus meter."""

    id: str
    bus_id: str
    capacity_mw: float
    availability_profile: npt.ArrayLike
    curtailment_cost_eur_per_mwh: float = 1_000.0
    hosting_capacity_max_mw: float = 0.0


@dataclass(frozen=True)
class BehindMeterBattery:
    """Distribution battery with explicit customer-side or grid-side accounting."""

    id: str
    bus_id: str
    side: BatterySide
    power_capacity_mw: float
    energy_capacity_mwh: float
    initial_soc_mwh: float
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    variable_cost_eur_per_mwh: float = 0.0


@dataclass(frozen=True)
class ControllableDemand:
    """Curtailable flexible load on a distribution bus."""

    id: str
    bus_id: str
    max_reduction_mw: npt.ArrayLike
    cost_eur_per_mwh: float


@dataclass(frozen=True)
class HostingCapacityOptions:
    """Policy for incremental DER hosting-capacity studies."""

    max_curtailment_fraction: float = 0.0


@dataclass(frozen=True)
class DistributionFeederProblem:
    """Standalone radial distribution-feeder optimisation problem."""

    buses: tuple[DistributionBus, ...]
    branches: tuple[DistributionBranch, ...]
    rooftop_pv: tuple[RooftopPV, ...] = ()
    batteries: tuple[BehindMeterBattery, ...] = ()
    flexible_loads: tuple[ControllableDemand, ...] = ()
    substation_bus_id: str = "substation"
    base_power_mva: float = 10.0
    time_step_hours: float = 1.0
    periods: tuple[str, ...] = ("0",)
    substation_import_limit_mw: float = 1_000.0
    substation_export_limit_mw: float = 1_000.0
    import_cost_eur_per_mwh: float = 0.0
    export_value_eur_per_mwh: float = 0.0
    hosting: HostingCapacityOptions = field(default_factory=HostingCapacityOptions)


@dataclass(frozen=True)
class DistributionStudyResult:
    """Distribution-feeder dispatch, hosting capacity, and diagnostics."""

    mode: DistributionMode
    timeseries: pd.DataFrame
    hosting_capacity_mw: dict[str, float]
    summary: dict[str, Any]
    solver_message: str

    def write(self, output_directory: Path) -> None:
        """Write distribution-specific outputs to a directory."""
        output_directory.mkdir(parents=True, exist_ok=True)
        self.timeseries.to_csv(output_directory / "distribution_timeseries.csv", index=False)
        hosting = pd.DataFrame(
            [
                {"pv_id": pv_id, "hosting_capacity_mw": capacity}
                for pv_id, capacity in sorted(self.hosting_capacity_mw.items())
            ]
        )
        hosting.to_csv(output_directory / "distribution_hosting_capacity.csv", index=False)
        (output_directory / "distribution_summary.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class DistributionFeederModel:
    """Linearised single-phase radial DistFlow model."""

    def solve(
        self,
        problem: DistributionFeederProblem,
        *,
        mode: DistributionMode = "operational",
    ) -> DistributionStudyResult:
        data = _PreparedDistribution.from_problem(problem)
        registry = _Registry()
        _register_variables(data, registry, mode)
        objective = _objective(data, registry, mode)
        lower, upper = _bounds(data, registry, mode)
        eq_a, eq_b, ub_a, ub_b = _constraints(data, registry, mode)

        result = linprog(
            c=objective,
            A_ub=ub_a,
            b_ub=ub_b,
            A_eq=eq_a,
            b_eq=eq_b,
            bounds=list(zip(lower, upper, strict=True)),
            method="highs",
        )
        if result.status != 0 or result.x is None:
            raise OptimisationError(f"Distribution feeder optimisation failed: {result.message}")
        solution = np.asarray(result.x, dtype=np.float64)
        return _build_result(
            data,
            registry,
            solution,
            mode=mode,
            objective_eur=float(result.fun),
            solver_message=str(result.message),
        )


def load_distribution_problem(path: Path | str) -> DistributionFeederProblem:
    """Load a standalone distribution-feeder study YAML."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ConfigurationError("Distribution study YAML must contain a mapping")
    if int(payload.get("schema_version", 1)) != 1:
        raise ConfigurationError("Distribution study schema_version must be 1")
    periods = tuple(str(item) for item in payload.get("periods", ["0"]))
    if not periods:
        raise ConfigurationError("Distribution study must define at least one period")
    buses = tuple(
        _bus(item, len(periods), index) for index, item in enumerate(_items(payload, "buses"))
    )
    branches = tuple(_branch(item, index) for index, item in enumerate(_items(payload, "branches")))
    pv = tuple(
        _pv(item, len(periods), index)
        for index, item in enumerate(payload.get("rooftop_pv", []) or [])
    )
    batteries = tuple(
        _battery(item, index) for index, item in enumerate(payload.get("batteries", []) or [])
    )
    flexible = tuple(
        _flexible_load(item, len(periods), index)
        for index, item in enumerate(payload.get("flexible_loads", []) or [])
    )
    hosting_raw = payload.get("hosting") or {}
    if not isinstance(hosting_raw, Mapping):
        raise ConfigurationError("hosting must be a mapping")
    problem = DistributionFeederProblem(
        buses=buses,
        branches=branches,
        rooftop_pv=pv,
        batteries=batteries,
        flexible_loads=flexible,
        substation_bus_id=str(payload.get("substation_bus_id", buses[0].id)),
        base_power_mva=_positive(payload, "base_power_mva", 10.0),
        time_step_hours=_positive(payload, "time_step_hours", 1.0),
        periods=periods,
        substation_import_limit_mw=_nonnegative(payload, "substation_import_limit_mw", 1_000.0),
        substation_export_limit_mw=_nonnegative(payload, "substation_export_limit_mw", 1_000.0),
        import_cost_eur_per_mwh=float(payload.get("import_cost_eur_per_mwh", 0.0)),
        export_value_eur_per_mwh=float(payload.get("export_value_eur_per_mwh", 0.0)),
        hosting=HostingCapacityOptions(
            max_curtailment_fraction=_fraction(hosting_raw, "max_curtailment_fraction", 0.0)
        ),
    )
    _validate_problem(problem)
    return problem


def run_distribution_study(
    problem: DistributionFeederProblem,
    *,
    mode: DistributionMode = "operational",
) -> DistributionStudyResult:
    """Solve a standalone distribution feeder study."""
    return DistributionFeederModel().solve(problem, mode=mode)


def nonlinear_radial_power_flow(
    problem: DistributionFeederProblem,
    dispatch: pd.DataFrame,
    *,
    period: int = 0,
    iterations: int = 20,
) -> dict[str, float]:
    """Independent backward-forward radial power-flow check for one solved period.

    The calculation uses the same single-phase equivalent and branch ordering but
    includes approximate I-squared losses in the backward sweep. It is intended
    as a fixture-scale validation bridge, not as a full unbalanced feeder solver.
    """
    data = _PreparedDistribution.from_problem(problem)
    row = dispatch.iloc[period]
    voltage = {bus.id: 1.0 for bus in data.buses}
    children = data.children_by_bus
    branch_by_child = {branch.to_bus_id: branch for branch in data.branches}
    active_net = {
        bus.id: float(data.fixed_load_mw[bus.id][period])
        - sum(float(row[f"dist_pv_used_mw__{pv.id}"]) for pv in data.pv_by_bus.get(bus.id, ()))
        - sum(
            float(row[f"dist_{battery.side}_battery_discharge_mw__{battery.id}"])
            for battery in data.battery_by_bus.get(bus.id, ())
        )
        + sum(
            float(row[f"dist_{battery.side}_battery_charge_mw__{battery.id}"])
            for battery in data.battery_by_bus.get(bus.id, ())
        )
        - sum(
            float(row[f"dist_flexible_load_reduction_mw__{load.id}"])
            for load in data.flex_by_bus.get(bus.id, ())
        )
        for bus in data.buses
    }
    reactive_net = {
        bus.id: float(data.fixed_reactive_load_mvar[bus.id][period]) for bus in data.buses
    }
    branch_p = {branch.id: 0.0 for branch in data.branches}
    branch_q = {branch.id: 0.0 for branch in data.branches}

    def backward(bus_id: str) -> tuple[float, float]:
        active = active_net[bus_id]
        reactive = reactive_net[bus_id]
        for child_id in children.get(bus_id, ()):
            child_p, child_q = backward(child_id)
            branch = branch_by_child[child_id]
            v_from = max(voltage[branch.from_bus_id] ** 2, 0.5)
            loss_basis = (child_p**2 + child_q**2) / data.base_power_mva / v_from
            loss_p = branch.resistance_pu * loss_basis
            loss_q = branch.reactance_pu * loss_basis
            branch_p[branch.id] = child_p + loss_p
            branch_q[branch.id] = child_q + loss_q
            active += branch_p[branch.id]
            reactive += branch_q[branch.id]
        return active, reactive

    def forward(bus_id: str) -> None:
        for child_id in children.get(bus_id, ()):
            branch = branch_by_child[child_id]
            voltage_sq = voltage[branch.from_bus_id] ** 2 - 2.0 * (
                branch.resistance_pu * branch_p[branch.id] / data.base_power_mva
                + branch.reactance_pu * branch_q[branch.id] / data.base_power_mva
            )
            voltage[child_id] = float(np.sqrt(max(voltage_sq, 0.0)))
            forward(child_id)

    for _ in range(iterations):
        backward(data.substation_bus_id)
        forward(data.substation_bus_id)
    return voltage


class _Registry:
    def __init__(self) -> None:
        self._blocks: dict[tuple[str, str | None], tuple[int, int]] = {}
        self.size = 0

    def add(self, name: str, size: int, *, asset_id: str | None = None) -> None:
        key = (name, asset_id)
        if key in self._blocks:
            raise ValueError(f"Duplicate distribution variable block: {name}, {asset_id}")
        self._blocks[key] = (self.size, size)
        self.size += size

    def at(self, name: str, period: int = 0, *, asset_id: str | None = None) -> int:
        offset, size = self._blocks[(name, asset_id)]
        if period < 0 or period >= size:
            raise IndexError(f"Period {period} outside block {name}")
        return offset + period

    def values(self, solution: FloatArray, name: str, *, asset_id: str | None = None) -> FloatArray:
        offset, size = self._blocks[(name, asset_id)]
        return solution[offset : offset + size]


@dataclass(frozen=True)
class _PreparedDistribution:
    buses: tuple[DistributionBus, ...]
    branches: tuple[DistributionBranch, ...]
    rooftop_pv: tuple[RooftopPV, ...]
    batteries: tuple[BehindMeterBattery, ...]
    flexible_loads: tuple[ControllableDemand, ...]
    substation_bus_id: str
    base_power_mva: float
    time_step_hours: float
    periods: tuple[str, ...]
    substation_import_limit_mw: float
    substation_export_limit_mw: float
    import_cost_eur_per_mwh: float
    export_value_eur_per_mwh: float
    hosting: HostingCapacityOptions
    fixed_load_mw: dict[str, FloatArray]
    fixed_reactive_load_mvar: dict[str, FloatArray]
    pv_availability: dict[str, FloatArray]
    flexible_reduction_limit_mw: dict[str, FloatArray]
    children_by_bus: dict[str, tuple[str, ...]]
    parent_branch_by_bus: dict[str, DistributionBranch]
    child_branches_by_bus: dict[str, tuple[DistributionBranch, ...]]
    pv_by_bus: dict[str, tuple[RooftopPV, ...]]
    battery_by_bus: dict[str, tuple[BehindMeterBattery, ...]]
    flex_by_bus: dict[str, tuple[ControllableDemand, ...]]

    @classmethod
    def from_problem(cls, problem: DistributionFeederProblem) -> _PreparedDistribution:
        periods = len(problem.periods)
        fixed_load = {
            bus.id: _profile(bus.fixed_load_mw, periods, f"buses[{bus.id}].fixed_load_mw")
            for bus in problem.buses
        }
        fixed_reactive = {
            bus.id: _profile(
                bus.fixed_reactive_load_mvar,
                periods,
                f"buses[{bus.id}].fixed_reactive_load_mvar",
            )
            for bus in problem.buses
        }
        pv_availability = {
            pv.id: _profile(
                pv.availability_profile, periods, f"rooftop_pv[{pv.id}].availability_profile"
            )
            for pv in problem.rooftop_pv
        }
        flexible_limits = {
            load.id: _profile(
                load.max_reduction_mw, periods, f"flexible_loads[{load.id}].max_reduction_mw"
            )
            for load in problem.flexible_loads
        }
        children: dict[str, list[str]] = defaultdict(list)
        parent_branch: dict[str, DistributionBranch] = {}
        child_branches: dict[str, list[DistributionBranch]] = defaultdict(list)
        for branch in problem.branches:
            children[branch.from_bus_id].append(branch.to_bus_id)
            parent_branch[branch.to_bus_id] = branch
            child_branches[branch.from_bus_id].append(branch)
        pv_by_bus: dict[str, list[RooftopPV]] = defaultdict(list)
        battery_by_bus: dict[str, list[BehindMeterBattery]] = defaultdict(list)
        flex_by_bus: dict[str, list[ControllableDemand]] = defaultdict(list)
        for pv in problem.rooftop_pv:
            pv_by_bus[pv.bus_id].append(pv)
        for battery in problem.batteries:
            battery_by_bus[battery.bus_id].append(battery)
        for load in problem.flexible_loads:
            flex_by_bus[load.bus_id].append(load)
        return cls(
            buses=problem.buses,
            branches=problem.branches,
            rooftop_pv=problem.rooftop_pv,
            batteries=problem.batteries,
            flexible_loads=problem.flexible_loads,
            substation_bus_id=problem.substation_bus_id,
            base_power_mva=problem.base_power_mva,
            time_step_hours=problem.time_step_hours,
            periods=problem.periods,
            substation_import_limit_mw=problem.substation_import_limit_mw,
            substation_export_limit_mw=problem.substation_export_limit_mw,
            import_cost_eur_per_mwh=problem.import_cost_eur_per_mwh,
            export_value_eur_per_mwh=problem.export_value_eur_per_mwh,
            hosting=problem.hosting,
            fixed_load_mw=fixed_load,
            fixed_reactive_load_mvar=fixed_reactive,
            pv_availability=pv_availability,
            flexible_reduction_limit_mw=flexible_limits,
            children_by_bus={key: tuple(value) for key, value in children.items()},
            parent_branch_by_bus=parent_branch,
            child_branches_by_bus={key: tuple(value) for key, value in child_branches.items()},
            pv_by_bus={key: tuple(value) for key, value in pv_by_bus.items()},
            battery_by_bus={key: tuple(value) for key, value in battery_by_bus.items()},
            flex_by_bus={key: tuple(value) for key, value in flex_by_bus.items()},
        )


def _register_variables(
    data: _PreparedDistribution,
    registry: _Registry,
    mode: DistributionMode,
) -> None:
    periods = len(data.periods)
    registry.add("substation_import_mw", periods)
    registry.add("substation_export_mw", periods)
    registry.add("substation_reactive_mvar", periods)
    for bus in data.buses:
        registry.add("voltage_sq_pu", periods, asset_id=bus.id)
    for branch in data.branches:
        registry.add("branch_active_mw", periods, asset_id=branch.id)
        registry.add("branch_reactive_mvar", periods, asset_id=branch.id)
    for pv in data.rooftop_pv:
        registry.add("pv_used_mw", periods, asset_id=pv.id)
        registry.add("pv_curtail_mw", periods, asset_id=pv.id)
        if mode == "hosting_capacity":
            registry.add("hosting_capacity_mw", 1, asset_id=pv.id)
    for battery in data.batteries:
        registry.add("battery_charge_mw", periods, asset_id=battery.id)
        registry.add("battery_discharge_mw", periods, asset_id=battery.id)
        registry.add("battery_soc_mwh", periods, asset_id=battery.id)
    for load in data.flexible_loads:
        registry.add("flexible_reduction_mw", periods, asset_id=load.id)


def _objective(
    data: _PreparedDistribution,
    registry: _Registry,
    mode: DistributionMode,
) -> FloatArray:
    objective = np.zeros(registry.size, dtype=np.float64)
    dt = data.time_step_hours
    for t in range(len(data.periods)):
        objective[registry.at("substation_import_mw", t)] = data.import_cost_eur_per_mwh * dt
        objective[registry.at("substation_export_mw", t)] = -data.export_value_eur_per_mwh * dt
    for pv in data.rooftop_pv:
        for t in range(len(data.periods)):
            objective[registry.at("pv_curtail_mw", t, asset_id=pv.id)] = (
                pv.curtailment_cost_eur_per_mwh * dt
            )
        if mode == "hosting_capacity":
            objective[registry.at("hosting_capacity_mw", asset_id=pv.id)] = -1_000_000.0
    for battery in data.batteries:
        for t in range(len(data.periods)):
            objective[registry.at("battery_charge_mw", t, asset_id=battery.id)] = (
                battery.variable_cost_eur_per_mwh + 1e-6
            ) * dt
            objective[registry.at("battery_discharge_mw", t, asset_id=battery.id)] = (
                battery.variable_cost_eur_per_mwh + 1e-6
            ) * dt
    for load in data.flexible_loads:
        for t in range(len(data.periods)):
            objective[registry.at("flexible_reduction_mw", t, asset_id=load.id)] = (
                load.cost_eur_per_mwh * dt
            )
    return objective


def _bounds(
    data: _PreparedDistribution,
    registry: _Registry,
    mode: DistributionMode,
) -> tuple[FloatArray, FloatArray]:
    lower = np.full(registry.size, -np.inf, dtype=np.float64)
    upper = np.full(registry.size, np.inf, dtype=np.float64)
    periods = len(data.periods)
    for t in range(periods):
        _set_bounds(
            registry, lower, upper, "substation_import_mw", t, 0.0, data.substation_import_limit_mw
        )
        _set_bounds(
            registry, lower, upper, "substation_export_mw", t, 0.0, data.substation_export_limit_mw
        )
        _set_bounds(
            registry,
            lower,
            upper,
            "substation_reactive_mvar",
            t,
            -data.substation_import_limit_mw,
            data.substation_import_limit_mw,
        )
    for bus in data.buses:
        for t in range(periods):
            _set_bounds(
                registry,
                lower,
                upper,
                "voltage_sq_pu",
                t,
                bus.voltage_min_pu**2,
                bus.voltage_max_pu**2,
                asset_id=bus.id,
            )
    for branch in data.branches:
        for t in range(periods):
            _set_bounds(
                registry,
                lower,
                upper,
                "branch_active_mw",
                t,
                -branch.rating_mva,
                branch.rating_mva,
                asset_id=branch.id,
            )
            _set_bounds(
                registry,
                lower,
                upper,
                "branch_reactive_mvar",
                t,
                -branch.rating_mva,
                branch.rating_mva,
                asset_id=branch.id,
            )
    for pv in data.rooftop_pv:
        for t in range(periods):
            lower[registry.at("pv_used_mw", t, asset_id=pv.id)] = 0.0
            lower[registry.at("pv_curtail_mw", t, asset_id=pv.id)] = 0.0
        if mode == "hosting_capacity":
            _set_bounds(
                registry,
                lower,
                upper,
                "hosting_capacity_mw",
                0,
                0.0,
                pv.hosting_capacity_max_mw,
                asset_id=pv.id,
            )
    for battery in data.batteries:
        for t in range(periods):
            _set_bounds(
                registry,
                lower,
                upper,
                "battery_charge_mw",
                t,
                0.0,
                battery.power_capacity_mw,
                asset_id=battery.id,
            )
            _set_bounds(
                registry,
                lower,
                upper,
                "battery_discharge_mw",
                t,
                0.0,
                battery.power_capacity_mw,
                asset_id=battery.id,
            )
            _set_bounds(
                registry,
                lower,
                upper,
                "battery_soc_mwh",
                t,
                0.0,
                battery.energy_capacity_mwh,
                asset_id=battery.id,
            )
    for load in data.flexible_loads:
        for t in range(periods):
            _set_bounds(
                registry,
                lower,
                upper,
                "flexible_reduction_mw",
                t,
                0.0,
                float(data.flexible_reduction_limit_mw[load.id][t]),
                asset_id=load.id,
            )
    return lower, upper


def _constraints(
    data: _PreparedDistribution,
    registry: _Registry,
    mode: DistributionMode,
) -> tuple[coo_matrix, FloatArray, coo_matrix, FloatArray]:
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_vals: list[float] = []
    eq_rhs: list[float] = []
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_vals: list[float] = []
    ub_rhs: list[float] = []

    def add_eq(terms: Mapping[int, float], rhs: float) -> None:
        row = len(eq_rhs)
        for col, value in terms.items():
            eq_rows.append(row)
            eq_cols.append(col)
            eq_vals.append(value)
        eq_rhs.append(rhs)

    def add_ub(terms: Mapping[int, float], rhs: float) -> None:
        row = len(ub_rhs)
        for col, value in terms.items():
            ub_rows.append(row)
            ub_cols.append(col)
            ub_vals.append(value)
        ub_rhs.append(rhs)

    periods = len(data.periods)
    for t in range(periods):
        for bus in data.buses:
            active_terms: dict[int, float] = {}
            reactive_terms: dict[int, float] = {}
            if bus.id == data.substation_bus_id:
                active_terms[registry.at("substation_import_mw", t)] = 1.0
                active_terms[registry.at("substation_export_mw", t)] = -1.0
                reactive_terms[registry.at("substation_reactive_mvar", t)] = 1.0
            else:
                branch = data.parent_branch_by_bus[bus.id]
                active_terms[registry.at("branch_active_mw", t, asset_id=branch.id)] = 1.0
                reactive_terms[registry.at("branch_reactive_mvar", t, asset_id=branch.id)] = 1.0
            for branch in data.child_branches_by_bus.get(bus.id, ()):
                active_terms[registry.at("branch_active_mw", t, asset_id=branch.id)] = -1.0
                reactive_terms[registry.at("branch_reactive_mvar", t, asset_id=branch.id)] = -1.0
            for pv in data.pv_by_bus.get(bus.id, ()):
                active_terms[registry.at("pv_used_mw", t, asset_id=pv.id)] = 1.0
            for battery in data.battery_by_bus.get(bus.id, ()):
                active_terms[registry.at("battery_discharge_mw", t, asset_id=battery.id)] = 1.0
                active_terms[registry.at("battery_charge_mw", t, asset_id=battery.id)] = -1.0
            for load in data.flex_by_bus.get(bus.id, ()):
                active_terms[registry.at("flexible_reduction_mw", t, asset_id=load.id)] = 1.0
            add_eq(active_terms, float(data.fixed_load_mw[bus.id][t]))
            add_eq(reactive_terms, float(data.fixed_reactive_load_mvar[bus.id][t]))

        add_eq(
            {registry.at("voltage_sq_pu", t, asset_id=data.substation_bus_id): 1.0},
            1.0,
        )
        for branch in data.branches:
            add_eq(
                {
                    registry.at("voltage_sq_pu", t, asset_id=branch.to_bus_id): 1.0,
                    registry.at("voltage_sq_pu", t, asset_id=branch.from_bus_id): -1.0,
                    registry.at("branch_active_mw", t, asset_id=branch.id): (
                        2.0 * branch.resistance_pu / data.base_power_mva
                    ),
                    registry.at("branch_reactive_mvar", t, asset_id=branch.id): (
                        2.0 * branch.reactance_pu / data.base_power_mva
                    ),
                },
                0.0,
            )
            p_idx = registry.at("branch_active_mw", t, asset_id=branch.id)
            q_idx = registry.at("branch_reactive_mvar", t, asset_id=branch.id)
            add_ub({p_idx: 1.0, q_idx: 1.0}, branch.rating_mva)
            add_ub({p_idx: 1.0, q_idx: -1.0}, branch.rating_mva)
            add_ub({p_idx: -1.0, q_idx: 1.0}, branch.rating_mva)
            add_ub({p_idx: -1.0, q_idx: -1.0}, branch.rating_mva)

        for pv in data.rooftop_pv:
            terms = {
                registry.at("pv_used_mw", t, asset_id=pv.id): 1.0,
                registry.at("pv_curtail_mw", t, asset_id=pv.id): 1.0,
            }
            if mode == "hosting_capacity":
                terms[registry.at("hosting_capacity_mw", asset_id=pv.id)] = -float(
                    data.pv_availability[pv.id][t]
                )
            add_eq(terms, pv.capacity_mw * float(data.pv_availability[pv.id][t]))

        for battery in data.batteries:
            previous_soc = (
                battery.initial_soc_mwh
                if t == 0
                else registry.at("battery_soc_mwh", t - 1, asset_id=battery.id)
            )
            terms = {
                registry.at("battery_soc_mwh", t, asset_id=battery.id): 1.0,
                registry.at("battery_charge_mw", t, asset_id=battery.id): (
                    -battery.charge_efficiency * data.time_step_hours
                ),
                registry.at("battery_discharge_mw", t, asset_id=battery.id): (
                    data.time_step_hours / battery.discharge_efficiency
                ),
            }
            if isinstance(previous_soc, int):
                terms[previous_soc] = -1.0
                rhs = 0.0
            else:
                rhs = previous_soc
            add_eq(terms, rhs)

    if mode == "hosting_capacity" and data.rooftop_pv:
        terms = {}
        rhs = 0.0
        for pv in data.rooftop_pv:
            for t in range(periods):
                terms[registry.at("pv_curtail_mw", t, asset_id=pv.id)] = data.time_step_hours
                rhs += (
                    data.hosting.max_curtailment_fraction
                    * pv.capacity_mw
                    * float(data.pv_availability[pv.id][t])
                    * data.time_step_hours
                )
            terms[registry.at("hosting_capacity_mw", asset_id=pv.id)] = (
                -data.hosting.max_curtailment_fraction
                * float(data.pv_availability[pv.id].sum())
                * data.time_step_hours
            )
        add_ub(terms, rhs)

    eq_matrix = coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(len(eq_rhs), registry.size))
    ub_matrix = coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(len(ub_rhs), registry.size))
    return (
        eq_matrix,
        np.asarray(eq_rhs, dtype=np.float64),
        ub_matrix,
        np.asarray(ub_rhs, dtype=np.float64),
    )


def _build_result(
    data: _PreparedDistribution,
    registry: _Registry,
    solution: FloatArray,
    *,
    mode: DistributionMode,
    objective_eur: float,
    solver_message: str,
) -> DistributionStudyResult:
    frame: dict[str, Sequence[object] | FloatArray] = {"period": data.periods}
    for bus in data.buses:
        voltage_sq = registry.values(solution, "voltage_sq_pu", asset_id=bus.id)
        frame[f"dist_voltage_pu__{bus.id}"] = np.sqrt(np.maximum(voltage_sq, 0.0))
        frame[f"dist_voltage_sq_pu__{bus.id}"] = voltage_sq
        frame[f"dist_fixed_load_mw__{bus.id}"] = data.fixed_load_mw[bus.id]
        frame[f"dist_fixed_reactive_load_mvar__{bus.id}"] = data.fixed_reactive_load_mvar[bus.id]
    for branch in data.branches:
        active = registry.values(solution, "branch_active_mw", asset_id=branch.id)
        reactive = registry.values(solution, "branch_reactive_mvar", asset_id=branch.id)
        frame[f"dist_branch_active_flow_mw__{branch.id}"] = active
        frame[f"dist_branch_reactive_flow_mvar__{branch.id}"] = reactive
        frame[f"dist_branch_loading_fraction__{branch.id}"] = (
            np.abs(active) + np.abs(reactive)
        ) / branch.rating_mva
        frame[f"dist_branch_loss_mw__{branch.id}"] = (
            branch.resistance_pu * (active**2 + reactive**2) / data.base_power_mva
        )
    for pv in data.rooftop_pv:
        frame[f"dist_pv_used_mw__{pv.id}"] = registry.values(solution, "pv_used_mw", asset_id=pv.id)
        frame[f"dist_pv_curtail_mw__{pv.id}"] = registry.values(
            solution, "pv_curtail_mw", asset_id=pv.id
        )
    for battery in data.batteries:
        prefix = f"dist_{battery.side}_battery"
        frame[f"{prefix}_charge_mw__{battery.id}"] = registry.values(
            solution, "battery_charge_mw", asset_id=battery.id
        )
        frame[f"{prefix}_discharge_mw__{battery.id}"] = registry.values(
            solution, "battery_discharge_mw", asset_id=battery.id
        )
        frame[f"{prefix}_soc_mwh__{battery.id}"] = registry.values(
            solution, "battery_soc_mwh", asset_id=battery.id
        )
    for load in data.flexible_loads:
        frame[f"dist_flexible_load_reduction_mw__{load.id}"] = registry.values(
            solution, "flexible_reduction_mw", asset_id=load.id
        )
    frame["dist_substation_import_mw"] = registry.values(solution, "substation_import_mw")
    frame["dist_substation_export_mw"] = registry.values(solution, "substation_export_mw")
    frame["dist_substation_reactive_mvar"] = registry.values(solution, "substation_reactive_mvar")

    timeseries = pd.DataFrame(frame)
    hosting_capacity = {
        pv.id: (
            float(registry.values(solution, "hosting_capacity_mw", asset_id=pv.id)[0])
            if mode == "hosting_capacity"
            else 0.0
        )
        for pv in data.rooftop_pv
    }
    summary = _summary(data, timeseries, hosting_capacity, mode, objective_eur)
    return DistributionStudyResult(
        mode=mode,
        timeseries=timeseries,
        hosting_capacity_mw=hosting_capacity,
        summary=summary,
        solver_message=solver_message,
    )


def _summary(
    data: _PreparedDistribution,
    timeseries: pd.DataFrame,
    hosting_capacity: Mapping[str, float],
    mode: DistributionMode,
    objective_eur: float,
) -> dict[str, Any]:
    voltage_columns = [f"dist_voltage_pu__{bus.id}" for bus in data.buses]
    loading_columns = [f"dist_branch_loading_fraction__{branch.id}" for branch in data.branches]
    loss_columns = [f"dist_branch_loss_mw__{branch.id}" for branch in data.branches]
    curtail_columns = [f"dist_pv_curtail_mw__{pv.id}" for pv in data.rooftop_pv]
    reverse_flow_mwh = float(timeseries["dist_substation_export_mw"].sum() * data.time_step_hours)
    return {
        "schema_version": 1,
        "model_family": "distribution_radial_distflow",
        "mode": mode,
        "periods": len(data.periods),
        "objective_eur": objective_eur,
        "approximations": {
            "phase_model": "balanced_single_phase_equivalent",
            "voltage_model": "linearized_distflow_voltage_drop",
            "thermal_model": "linear_diamond_apparent_power_limit",
            "loss_model": "post_solution_quadratic_loss_diagnostic_not_optimised",
        },
        "hosting_capacity_mw": dict(sorted(hosting_capacity.items())),
        "total_hosting_capacity_mw": float(sum(hosting_capacity.values())),
        "min_voltage_pu": float(timeseries[voltage_columns].min().min()),
        "max_voltage_pu": float(timeseries[voltage_columns].max().max()),
        "max_branch_loading_fraction": (
            float(timeseries[loading_columns].max().max()) if loading_columns else 0.0
        ),
        "losses_mwh": (
            float(timeseries[loss_columns].sum().sum() * data.time_step_hours)
            if loss_columns
            else 0.0
        ),
        "pv_curtailment_mwh": (
            float(timeseries[curtail_columns].sum().sum() * data.time_step_hours)
            if curtail_columns
            else 0.0
        ),
        "substation_import_mwh": float(
            timeseries["dist_substation_import_mw"].sum() * data.time_step_hours
        ),
        "substation_export_mwh": reverse_flow_mwh,
        "reverse_flow_mwh": reverse_flow_mwh,
        "customer_side_battery_throughput_mwh": _battery_throughput(
            data, timeseries, "customer_side"
        ),
        "grid_side_battery_throughput_mwh": _battery_throughput(data, timeseries, "grid_side"),
    }


def _battery_throughput(
    data: _PreparedDistribution,
    timeseries: pd.DataFrame,
    side: BatterySide,
) -> float:
    total = 0.0
    for battery in data.batteries:
        if battery.side != side:
            continue
        total += float(
            (
                timeseries[f"dist_{side}_battery_charge_mw__{battery.id}"]
                + timeseries[f"dist_{side}_battery_discharge_mw__{battery.id}"]
            ).sum()
            * data.time_step_hours
        )
    return total


def _set_bounds(
    registry: _Registry,
    lower: FloatArray,
    upper: FloatArray,
    name: str,
    period: int,
    lb: float,
    ub: float,
    *,
    asset_id: str | None = None,
) -> None:
    index = registry.at(name, period, asset_id=asset_id)
    lower[index] = lb
    upper[index] = ub


def _validate_problem(problem: DistributionFeederProblem) -> None:
    bus_ids = [bus.id for bus in problem.buses]
    if len(set(bus_ids)) != len(bus_ids):
        raise ConfigurationError("Distribution bus ids must be unique")
    if problem.substation_bus_id not in set(bus_ids):
        raise ConfigurationError("substation_bus_id must reference a distribution bus")
    branch_ids = [branch.id for branch in problem.branches]
    if len(set(branch_ids)) != len(branch_ids):
        raise ConfigurationError("Distribution branch ids must be unique")
    child_ids = [branch.to_bus_id for branch in problem.branches]
    if len(set(child_ids)) != len(child_ids):
        raise ConfigurationError(
            "Distribution branches must form a radial tree with one parent per bus"
        )
    if problem.substation_bus_id in set(child_ids):
        raise ConfigurationError("Substation bus cannot have an upstream branch")
    referenced = set(child_ids) | {problem.substation_bus_id}
    if referenced != set(bus_ids):
        missing = sorted(set(bus_ids) - referenced)
        extra = sorted(referenced - set(bus_ids))
        raise ConfigurationError(
            "Distribution branches must connect every bus exactly once; "
            f"missing={missing}, unknown={extra}"
        )
    known_buses = set(bus_ids)
    for branch in problem.branches:
        if branch.from_bus_id not in known_buses or branch.to_bus_id not in known_buses:
            raise ConfigurationError(f"Branch {branch.id} references an unknown bus")
        for name, value in (
            ("resistance_pu", branch.resistance_pu),
            ("reactance_pu", branch.reactance_pu),
            ("rating_mva", branch.rating_mva),
        ):
            if value <= 0.0:
                raise ConfigurationError(f"Branch {branch.id} {name} must be positive")
    for bus in problem.buses:
        if bus.voltage_min_pu <= 0.0 or bus.voltage_max_pu <= bus.voltage_min_pu:
            raise ConfigurationError(f"Bus {bus.id} voltage bounds must be positive and ordered")
    for pv in problem.rooftop_pv:
        if pv.bus_id not in known_buses:
            raise ConfigurationError(f"Rooftop PV {pv.id} references an unknown bus")
        for name, value in (
            ("capacity_mw", pv.capacity_mw),
            ("curtailment_cost_eur_per_mwh", pv.curtailment_cost_eur_per_mwh),
            ("hosting_capacity_max_mw", pv.hosting_capacity_max_mw),
        ):
            if value < 0.0:
                raise ConfigurationError(f"Rooftop PV {pv.id} {name} must be non-negative")
    for battery in problem.batteries:
        if battery.bus_id not in known_buses:
            raise ConfigurationError(f"Battery {battery.id} references an unknown bus")
        if battery.side not in {"customer_side", "grid_side"}:
            raise ConfigurationError(
                f"Battery {battery.id} side must be customer_side or grid_side"
            )
        if battery.power_capacity_mw < 0.0 or battery.energy_capacity_mwh < 0.0:
            raise ConfigurationError(f"Battery {battery.id} capacities must be non-negative")
        if not 0.0 <= battery.initial_soc_mwh <= battery.energy_capacity_mwh:
            raise ConfigurationError(f"Battery {battery.id} initial SOC is outside energy capacity")
        if battery.charge_efficiency <= 0.0 or battery.discharge_efficiency <= 0.0:
            raise ConfigurationError(f"Battery {battery.id} efficiencies must be positive")
    for load in problem.flexible_loads:
        if load.bus_id not in known_buses:
            raise ConfigurationError(f"Flexible load {load.id} references an unknown bus")
        if load.cost_eur_per_mwh < 0.0:
            raise ConfigurationError(f"Flexible load {load.id} cost must be non-negative")
    if not 0.0 <= problem.hosting.max_curtailment_fraction <= 1.0:
        raise ConfigurationError("hosting.max_curtailment_fraction must be between 0 and 1")


def _items(payload: Mapping[str, Any], key: str) -> Sequence[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ConfigurationError(f"{key} must be a list")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ConfigurationError(f"{key}[{index}] must be a mapping")
        result.append(item)
    return result


def _bus(item: Mapping[str, Any], periods: int, index: int) -> DistributionBus:
    return DistributionBus(
        id=_required_id(item, f"buses[{index}]"),
        fixed_load_mw=_profile(
            item.get("fixed_load_mw", 0.0), periods, f"buses[{index}].fixed_load_mw"
        ),
        fixed_reactive_load_mvar=_profile(
            item.get("fixed_reactive_load_mvar", 0.0),
            periods,
            f"buses[{index}].fixed_reactive_load_mvar",
        ),
        voltage_min_pu=float(item.get("voltage_min_pu", 0.95)),
        voltage_max_pu=float(item.get("voltage_max_pu", 1.05)),
    )


def _branch(item: Mapping[str, Any], index: int) -> DistributionBranch:
    path = f"branches[{index}]"
    return DistributionBranch(
        id=_required_id(item, path),
        from_bus_id=str(item.get("from_bus_id", "")),
        to_bus_id=str(item.get("to_bus_id", "")),
        resistance_pu=float(item.get("resistance_pu", 0.0)),
        reactance_pu=float(item.get("reactance_pu", 0.0)),
        rating_mva=float(item.get("rating_mva", 0.0)),
    )


def _pv(item: Mapping[str, Any], periods: int, index: int) -> RooftopPV:
    path = f"rooftop_pv[{index}]"
    return RooftopPV(
        id=_required_id(item, path),
        bus_id=str(item.get("bus_id", "")),
        capacity_mw=float(item.get("capacity_mw", 0.0)),
        availability_profile=_profile(
            item.get("availability_profile", 0.0), periods, f"{path}.availability_profile"
        ),
        curtailment_cost_eur_per_mwh=float(item.get("curtailment_cost_eur_per_mwh", 1_000.0)),
        hosting_capacity_max_mw=float(item.get("hosting_capacity_max_mw", 0.0)),
    )


def _battery(item: Mapping[str, Any], index: int) -> BehindMeterBattery:
    path = f"batteries[{index}]"
    return BehindMeterBattery(
        id=_required_id(item, path),
        bus_id=str(item.get("bus_id", "")),
        side=cast(BatterySide, str(item.get("side", "customer_side"))),
        power_capacity_mw=float(item.get("power_capacity_mw", 0.0)),
        energy_capacity_mwh=float(item.get("energy_capacity_mwh", 0.0)),
        initial_soc_mwh=float(item.get("initial_soc_mwh", 0.0)),
        charge_efficiency=float(item.get("charge_efficiency", 0.95)),
        discharge_efficiency=float(item.get("discharge_efficiency", 0.95)),
        variable_cost_eur_per_mwh=float(item.get("variable_cost_eur_per_mwh", 0.0)),
    )


def _flexible_load(item: Mapping[str, Any], periods: int, index: int) -> ControllableDemand:
    path = f"flexible_loads[{index}]"
    return ControllableDemand(
        id=_required_id(item, path),
        bus_id=str(item.get("bus_id", "")),
        max_reduction_mw=_profile(
            item.get("max_reduction_mw", 0.0), periods, f"{path}.max_reduction_mw"
        ),
        cost_eur_per_mwh=float(item.get("cost_eur_per_mwh", 0.0)),
    )


def _required_id(item: Mapping[str, Any], path: str) -> str:
    value = str(item.get("id", ""))
    if not value:
        raise ConfigurationError(f"{path}.id is required")
    return value


def _positive(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = float(payload.get(key, default))
    if value <= 0.0:
        raise ConfigurationError(f"{key} must be positive")
    return value


def _nonnegative(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = float(payload.get(key, default))
    if value < 0.0:
        raise ConfigurationError(f"{key} must be non-negative")
    return value


def _fraction(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = float(payload.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{key} must be between 0 and 1")
    return value


def _profile(value: npt.ArrayLike, periods: int, path: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(periods, float(array), dtype=np.float64)
    if array.ndim != 1 or len(array) != periods:
        raise ConfigurationError(f"{path} must be a scalar or a list with {periods} values")
    if np.any(~np.isfinite(array)):
        raise ConfigurationError(f"{path} must contain finite numbers")
    return array.astype(np.float64)
