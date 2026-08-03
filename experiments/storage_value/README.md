# Storage Value Under Renewable Penetration

This study estimates the model-conditioned operating value of one battery design
under low and high renewable penetration assumptions.

Reproduce from a clean checkout after installing dependencies:

```bash
poetry run energy-sim run-experiment --study experiments/storage_value --overwrite --no-plots
```

The command writes scenario result files under `outputs/`, then regenerates
tables, figures, figure metadata, and the research report from the result files.

