from __future__ import annotations

import json
from pathlib import Path

from energy_system_simulator.config import load_config
from energy_system_simulator.reporting import write_outputs
from energy_system_simulator.simulation import SimulationEngine


def test_write_outputs_includes_machine_readable_manifest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "example.yaml"
    config = load_config(config_path)
    result = SimulationEngine(config).run()

    write_outputs(result, tmp_path, config=config, config_path=config_path, create_plots=False)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_version"] == "0.1.0"
    assert len(manifest["input_file_sha256"]) == 64
    assert len(manifest["configuration_sha256"]) == 64
    assert manifest["solver"]["name"] == "scipy.optimize.milp"
    assert manifest["solver"]["status"] == "optimal"
    assert manifest["solver"]["backend_status"] == "optimal"
    assert manifest["solver"]["backend_status_code"] == 0
    assert manifest["formulation"]["integer_variables"] == 1344
    assert (
        manifest["terminal_commitment"]["terminal_commitment_mode"]
        == "forbid_incomplete_transitions"
    )
    assert manifest["resolved_configuration"]["paths"]["input_csv"].endswith(
        "data\\example_hourly.csv"
    ) or manifest["resolved_configuration"]["paths"]["input_csv"].endswith(
        "data/example_hourly.csv"
    )
