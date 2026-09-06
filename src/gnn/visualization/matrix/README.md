# Matrix Visualization

Heatmap and statistical visualization for model likelihood, transition,
preference, prior, and PyMDP/POMDP tensor matrices.

## Files

- `visualizer.py` — Matrix heatmap engine with 3-D tensor support. POMDP `B`
  tensors use `(next_state, previous_state, action)` and validate stochasticity
  by summing over the next-state axis for each previous-state/action column.
- `extract.py` (60 lines) — Matrix extraction from parsed models
- `compat.py` (40 lines) — Shared helper exports

## Outputs

- PNG/SVG visualizations for matrices and per-action tensor slices.
- CSV exports for 2-D matrices and every 3-D tensor slice.
- POMDP transition analysis panels for entropy, stochasticity, and dominant
  next-state structure.

## See Also

- [Parent: visualization/README.md](../README.md)
