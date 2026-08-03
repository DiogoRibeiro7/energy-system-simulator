from __future__ import annotations

from pathlib import Path

from energy_system_simulator.experiments import analyze_research_experiment

if __name__ == "__main__":
    analyze_research_experiment(Path(__file__).resolve().parents[1])
