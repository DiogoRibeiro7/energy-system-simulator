# Rolling-Horizon Simulation

Rolling-horizon simulation solves long input horizons in bounded optimisation
windows. Each window is solved with the configured look-ahead data, but only the
implementation segment is retained. Terminal asset states from the implemented
segment are transferred into the next window.

## Configuration

Enable rolling horizons with an optional schema v2 section:

```yaml
rolling_horizon:
  enabled: true
  optimisation_window_periods: 168
  implementation_periods: 24
  lookahead_periods: 144
  terminal_treatment: relaxed
  forecast_mode: perfect_foresight
  checkpoint_directory: ../outputs/checkpoints
  resume_from_checkpoint: false
  compare_full_horizon: false
```

- `optimisation_window_periods`: total periods solved in each subproblem.
- `implementation_periods`: periods retained before advancing the horizon.
- `lookahead_periods`: periods available to the optimiser beyond the retained
  implementation segment.
- `terminal_treatment`: `inherit`, `relaxed`, or `enforce`. `relaxed` uses
  carry-forward thermal obligations and free internal storage/hydro terminal
  states; the final window keeps the configured terminal policies.
- `forecast_mode`: records whether inputs represent `perfect_foresight` or
  `forecast_inputs`. The current engine uses the configured input columns
  deterministically in both modes.
- `checkpoint_directory`: destination for deterministic JSON checkpoints.
  When omitted, checkpoints are written under the configured output directory.
- `resume_from_checkpoint`: resumes from the deterministic checkpoint.
- `compare_full_horizon`: runs an additional full-horizon solve and reports
  objective and common-timeseries differences for small fixtures.

## State Transfer

The implementation segment transfers:

- thermal commitment, output, consecutive up/down duration, and residual
  minimum up/down obligations through updated initial states;
- storage state of charge for each storage unit;
- reservoir storage for each hydro unit;
- remaining task energy for deferrable and EV-charging demand.

Timestamps are retained only once, so output coverage is complete and contains
no duplicated periods.

## Horizon Effect

Rolling-horizon results can differ from the full-horizon optimum because each
window only sees its optimisation horizon. Look-ahead mitigates myopic decisions
around storage, hydro, commitment, and flexible demand, but it cannot reproduce
full-horizon foresight unless the window spans the full input data. When the
rolling optimisation window covers every period and the implementation segment
also covers every period, the result is equivalent to a full-horizon solve.
