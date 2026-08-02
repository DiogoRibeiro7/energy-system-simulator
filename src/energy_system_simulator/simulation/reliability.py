from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from energy_system_simulator.config import ModelConfig
from energy_system_simulator.exceptions import OptimisationError
from energy_system_simulator.simulation.engine import AvailabilityOverrides, SimulationEngine

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
OutageAssetType = Literal["thermal", "renewable", "storage", "line", "import"]


@dataclass(frozen=True)
class OutageModel:
    """Sequential two-state outage model for one asset."""

    asset_id: str
    asset_type: OutageAssetType
    forced_outage_rate: float
    mean_time_to_repair_hours: float
    group_id: str | None = None


@dataclass(frozen=True)
class CommonCauseOutageGroup:
    """Common-cause outage process applied to grouped assets."""

    id: str
    forced_outage_rate: float
    mean_time_to_repair_hours: float


@dataclass(frozen=True)
class ReliabilityStudyConfig:
    replications: int
    seed: int
    outage_models: tuple[OutageModel, ...]
    common_cause_groups: tuple[CommonCauseOutageGroup, ...] = ()
    confidence_level: float = 0.95
    stop_when_eens_relative_half_width_below: float | None = None
    minimum_replications: int = 1
    parallel_workers: int = 1


@dataclass(frozen=True)
class ReliabilityReplication:
    index: int
    seed: int
    metrics: dict[str, float]
    outage_hours_by_asset: dict[str, float]
    outage_unserved_energy_mwh_by_type: dict[str, float]
    failed: bool
    error: str


@dataclass(frozen=True)
class ReliabilityResult:
    metrics: dict[str, float]
    metric_distribution: dict[str, tuple[float, ...]]
    confidence_intervals: dict[str, dict[str, float]]
    attributed_unserved_energy_mwh_by_type: dict[str, float]
    convergence: dict[str, float | bool]
    replications: tuple[ReliabilityReplication, ...]


class ReliabilityStudy:
    """Run sequential Monte Carlo adequacy simulations around deterministic dispatch."""

    def __init__(self, model_config: ModelConfig, study_config: ReliabilityStudyConfig) -> None:
        self.model_config = model_config
        self.study_config = study_config
        self._validate()

    def run(self) -> ReliabilityResult:
        replications = self._run_replications()
        successful = [replication for replication in replications if not replication.failed]
        if not successful:
            raise OptimisationError("All reliability replications failed")
        metrics = self._aggregate_metrics(successful, attempted=len(replications))
        distribution = self._metric_distribution(successful)
        intervals = self._confidence_intervals(successful)
        attribution = self._aggregate_outage_attribution(successful)
        convergence = self._convergence_summary(successful, len(replications))
        return ReliabilityResult(
            metrics=metrics,
            metric_distribution=distribution,
            confidence_intervals=intervals,
            attributed_unserved_energy_mwh_by_type=attribution,
            convergence=convergence,
            replications=tuple(replications),
        )

    def sample_outage_paths(self, replication_index: int, periods: int) -> dict[str, BoolArray]:
        rng = np.random.default_rng(self.study_config.seed + replication_index)
        dt = self.model_config.simulation.time_step_hours
        group_paths = {
            group.id: _sample_two_state_path(
                periods,
                dt,
                group.forced_outage_rate,
                group.mean_time_to_repair_hours,
                rng,
            )
            for group in self.study_config.common_cause_groups
        }
        paths: dict[str, BoolArray] = {}
        for model in self.study_config.outage_models:
            independent = _sample_two_state_path(
                periods,
                dt,
                model.forced_outage_rate,
                model.mean_time_to_repair_hours,
                rng,
            )
            if model.group_id is not None:
                group_path = group_paths[model.group_id]
                independent = independent | group_path
            paths[model.asset_id] = independent
        return paths

    def _run_replications(self) -> list[ReliabilityReplication]:
        if (
            self.study_config.parallel_workers > 1
            and self.study_config.stop_when_eens_relative_half_width_below is None
        ):
            with ThreadPoolExecutor(max_workers=self.study_config.parallel_workers) as executor:
                return list(
                    executor.map(self._run_replication, range(self.study_config.replications))
                )

        replications: list[ReliabilityReplication] = []
        for index in range(self.study_config.replications):
            replication = self._run_replication(index)
            replications.append(replication)
            if self._converged(replications):
                break
        return replications

    def _run_replication(self, index: int) -> ReliabilityReplication:
        seed = self.study_config.seed + index
        try:
            base = SimulationEngine(self.model_config).run()
            periods = len(base.timeseries)
            paths = self.sample_outage_paths(index, periods)
            overrides = self._availability_overrides(paths, periods)
            result = SimulationEngine(self.model_config, availability_overrides=overrides).run()
            metrics = _replication_metrics(
                result.timeseries,
                self.model_config.simulation.time_step_hours,
            )
            return ReliabilityReplication(
                index=index,
                seed=seed,
                metrics=metrics,
                outage_hours_by_asset={
                    asset_id: float(path.sum() * self.model_config.simulation.time_step_hours)
                    for asset_id, path in paths.items()
                },
                outage_unserved_energy_mwh_by_type=self._attributed_unserved_energy(
                    paths,
                    result.timeseries["total_load_shed_mw"].to_numpy(dtype=np.float64),
                ),
                failed=False,
                error="",
            )
        except Exception as exc:  # pragma: no cover - exercised by integration failures.
            return ReliabilityReplication(
                index=index,
                seed=seed,
                metrics={},
                outage_hours_by_asset={},
                outage_unserved_energy_mwh_by_type={},
                failed=True,
                error=str(exc),
            )

    def _availability_overrides(
        self,
        paths: dict[str, BoolArray],
        periods: int,
    ) -> AvailabilityOverrides:
        by_type: dict[OutageAssetType, dict[str, FloatArray]] = {
            "thermal": {},
            "renewable": {},
            "storage": {},
            "line": {},
        }
        import_factor = np.ones(periods, dtype=np.float64)
        for model in self.study_config.outage_models:
            factor = (~paths[model.asset_id]).astype(np.float64)
            if model.asset_type == "import":
                import_factor *= factor
            else:
                by_type[model.asset_type][model.asset_id] = factor
        return AvailabilityOverrides(
            thermal=by_type["thermal"],
            renewable=by_type["renewable"],
            storage=by_type["storage"],
            lines=by_type["line"],
            imports=import_factor,
        )

    def _attributed_unserved_energy(
        self,
        paths: dict[str, BoolArray],
        shed_mw: FloatArray,
    ) -> dict[str, float]:
        dt = self.model_config.simulation.time_step_hours
        by_type = {
            asset_type: 0.0 for asset_type in ("thermal", "renewable", "storage", "line", "import")
        }
        for period, shed in enumerate(shed_mw):
            if shed <= 0.0:
                continue
            active_types = {
                model.asset_type
                for model in self.study_config.outage_models
                if bool(paths[model.asset_id][period])
            }
            if not active_types:
                continue
            share = shed * dt / len(active_types)
            for asset_type in active_types:
                by_type[asset_type] += float(share)
        return by_type

    def _aggregate_metrics(
        self,
        replications: list[ReliabilityReplication],
        attempted: int,
    ) -> dict[str, float]:
        keys = replications[0].metrics.keys()
        return {
            key: float(np.mean([replication.metrics[key] for replication in replications]))
            for key in keys
        } | {
            "replications": float(len(replications)),
            "failed_replications": float(attempted - len(replications)),
        }

    def _metric_distribution(
        self,
        replications: list[ReliabilityReplication],
    ) -> dict[str, tuple[float, ...]]:
        keys = replications[0].metrics.keys()
        return {
            key: tuple(replication.metrics[key] for replication in replications) for key in keys
        }

    def _confidence_intervals(
        self,
        replications: list[ReliabilityReplication],
    ) -> dict[str, dict[str, float]]:
        keys = replications[0].metrics.keys()
        z = _normal_quantile(self.study_config.confidence_level)
        return {
            key: _confidence_interval(
                [replication.metrics[key] for replication in replications],
                z,
            )
            for key in keys
        }

    def _aggregate_outage_attribution(
        self,
        replications: list[ReliabilityReplication],
    ) -> dict[str, float]:
        asset_types = ("thermal", "renewable", "storage", "line", "import")
        return {
            asset_type: float(
                np.mean(
                    [
                        replication.outage_unserved_energy_mwh_by_type.get(asset_type, 0.0)
                        for replication in replications
                    ]
                )
            )
            for asset_type in asset_types
        }

    def _converged(self, replications: list[ReliabilityReplication]) -> bool:
        threshold = self.study_config.stop_when_eens_relative_half_width_below
        successful = [replication for replication in replications if not replication.failed]
        if threshold is None or len(successful) < self.study_config.minimum_replications:
            return False
        interval = _confidence_interval(
            [replication.metrics["expected_unserved_energy_mwh"] for replication in successful],
            _normal_quantile(self.study_config.confidence_level),
        )
        mean = interval["mean"]
        return mean > 0.0 and interval["half_width"] / mean <= threshold

    def _convergence_summary(
        self,
        replications: list[ReliabilityReplication],
        attempted: int,
    ) -> dict[str, float | bool]:
        threshold = self.study_config.stop_when_eens_relative_half_width_below
        interval = _confidence_interval(
            [replication.metrics["expected_unserved_energy_mwh"] for replication in replications],
            _normal_quantile(self.study_config.confidence_level),
        )
        relative = interval["half_width"] / interval["mean"] if interval["mean"] > 0.0 else np.inf
        return {
            "attempted_replications": float(attempted),
            "successful_replications": float(len(replications)),
            "eens_relative_half_width": float(relative),
            "stopping_threshold": float(threshold) if threshold is not None else np.inf,
            "converged": bool(threshold is not None and relative <= threshold),
        }

    def _validate(self) -> None:
        if self.study_config.replications <= 0:
            raise ValueError("Reliability replications must be positive")
        if self.study_config.parallel_workers <= 0:
            raise ValueError("Reliability parallel_workers must be positive")
        if not 0.0 < self.study_config.confidence_level < 1.0:
            raise ValueError("Reliability confidence_level must be in (0, 1)")
        group_ids = {group.id for group in self.study_config.common_cause_groups}
        for group in self.study_config.common_cause_groups:
            _validate_outage_parameters(group.forced_outage_rate, group.mean_time_to_repair_hours)
        for model in self.study_config.outage_models:
            _validate_outage_parameters(model.forced_outage_rate, model.mean_time_to_repair_hours)
            if model.group_id is not None and model.group_id not in group_ids:
                raise ValueError(f"Outage model {model.asset_id!r} references unknown group")


def _sample_two_state_path(
    periods: int,
    dt_hours: float,
    forced_outage_rate: float,
    mean_time_to_repair_hours: float,
    rng: np.random.Generator,
) -> BoolArray:
    _validate_outage_parameters(forced_outage_rate, mean_time_to_repair_hours)
    if forced_outage_rate == 0.0:
        return np.zeros(periods, dtype=np.bool_)
    if forced_outage_rate == 1.0:
        return np.ones(periods, dtype=np.bool_)
    mean_time_to_failure = (
        mean_time_to_repair_hours * (1.0 - forced_outage_rate) / forced_outage_rate
    )
    fail_probability = min(1.0, dt_hours / mean_time_to_failure)
    repair_probability = min(1.0, dt_hours / mean_time_to_repair_hours)
    state = bool(rng.random() < forced_outage_rate)
    result = np.zeros(periods, dtype=np.bool_)
    for period in range(periods):
        result[period] = state
        if state:
            state = not bool(rng.random() < repair_probability)
        else:
            state = bool(rng.random() < fail_probability)
    return result


def _replication_metrics(frame: pd.DataFrame, dt_hours: float) -> dict[str, float]:
    shed = frame["total_load_shed_mw"].to_numpy(dtype=np.float64)
    scarcity = shed > 0.0
    durations = _event_durations(scarcity, dt_hours)
    horizon_hours = len(shed) * dt_hours
    eue = float(shed.sum() * dt_hours)
    return {
        "loss_of_load_probability": float(np.mean(scarcity)),
        "loss_of_load_expectation_hours_per_year": float(np.mean(scarcity) * 8760.0),
        "expected_unserved_energy_mwh": eue,
        "expected_demand_not_served_mw": float(eue / horizon_hours if horizon_hours else 0.0),
        "scarcity_event_frequency_per_year": float(
            len(durations) * 8760.0 / horizon_hours if horizon_hours else 0.0
        ),
        "scarcity_event_mean_duration_hours": float(np.mean(durations) if durations else 0.0),
    }


def _event_durations(mask: npt.NDArray[np.bool_], dt_hours: float) -> list[float]:
    durations: list[float] = []
    active = 0
    for value in mask:
        if value:
            active += 1
        elif active:
            durations.append(active * dt_hours)
            active = 0
    if active:
        durations.append(active * dt_hours)
    return durations


def _confidence_interval(values: list[float], z: float) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "half_width": 0.0}
    mean = float(np.mean(values))
    if len(values) == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "half_width": 0.0}
    standard_error = float(np.std(values, ddof=1) / sqrt(len(values)))
    half_width = z * standard_error
    return {
        "mean": mean,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "half_width": half_width,
    }


def _normal_quantile(confidence_level: float) -> float:
    known = {
        0.8: 1.2815515655446004,
        0.9: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    return known.get(round(confidence_level, 2), 1.959963984540054)


def _validate_outage_parameters(
    forced_outage_rate: float, mean_time_to_repair_hours: float
) -> None:
    if not 0.0 <= forced_outage_rate <= 1.0:
        raise ValueError("forced_outage_rate must be in [0, 1]")
    if forced_outage_rate not in {0.0, 1.0} and mean_time_to_repair_hours <= 0.0:
        raise ValueError("mean_time_to_repair_hours must be positive")
