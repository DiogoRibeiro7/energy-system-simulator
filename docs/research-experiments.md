# Research Experiments

Research experiments add a stricter reproducibility layer on top of scenario
experiments. A study directory contains the research question, hypotheses, model
and data versions, base configuration, scenario definitions, pre-specified
metrics and comparisons, expected outputs, executable scripts, and limitations.

## Study Contract

Each study has a `study.yaml` file with:

- `research_question` and `hypotheses`
- `model_version`, `data_version`, and deterministic `seed`
- `base_config`, `scenario_file`, and `output_directory`
- pre-specified `metrics` with result columns and units
- pre-specified `comparisons` with baseline, scenario, metric, and paired-seed
  status
- `uncertainty_intervals` and deterministic `sensitivity_ranges`
- figure definitions with captions
- explicit statements for what the model can and cannot identify

The runner refuses studies that omit metrics or comparisons. This keeps the
analysis plan separate from the observed results.

## Commands

Run the complete example experiment:

```bash
poetry run energy-sim run-experiment --study experiments/storage_value --overwrite --no-plots
```

Regenerate tables, figures, captions, and report from existing result files:

```bash
poetry run energy-sim analyze-experiment --study experiments/storage_value
```

Reproduce a run from its manifest:

```bash
poetry run energy-sim reproduce-experiment \
  --manifest experiments/storage_value/outputs/research_manifest.json \
  --overwrite \
  --no-plots
```

The manifest verifies hashes for the study definition, base configuration,
scenario definition, input data, and summary results. If any tracked input has
changed, reproduction stops before rerunning.

## Generated Artifacts

The analysis step writes automated Markdown and LaTeX tables with units in
headers and stable numeric rounding. Figures are created from `outputs/summary.csv`;
`figures/figure_metadata.json` records each figure path, source file, metric, and
caption from the study definition.

The report separates assumptions, model outputs, descriptive comparisons,
interpretation, limitations, and explicit statements of what the model can and
cannot identify. Scenario differences are model-conditioned descriptive results;
the template does not treat them as causal claims.
