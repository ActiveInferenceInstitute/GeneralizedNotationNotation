# PyTorch Analysis — Technical Specification

**Version**: 1.6.0

## Input Format

- `simulation_results.json` from PyTorch execution step (searched recursively under `pytorch/` paths or `pytorch_simulation_results.json`)
- Fields: `beliefs`, `actions`, `observations`, `efe_history`, `validation`, `model_name`

## Output Format

- PNG plots: `belief_trajectory.png`, `action_distribution.png`, `efe_history.png`
- JSON summary: `{model}/pytorch_analysis.json` (framework, metrics, plots_generated)

## Processing Requirements

- Handles variable-length simulation runs
- Graceful degradation when matplotlib unavailable
- Numerical stability for near-zero probability values

## Error Handling

- Missing results file → returns empty analysis with warning
- Malformed JSON → logs parse error, continues with available data
