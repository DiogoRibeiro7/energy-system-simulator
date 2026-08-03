from __future__ import annotations

from pathlib import Path

from energy_system_simulator.experiments import run_research_experiment

if __name__ == "__main__":
    run_research_experiment(Path(__file__).resolve().parents[1], overwrite=True, create_plots=False)
