# NumPyro Analysis — Technical Specification

**Version**: 1.6.0

## Input Format

- `simulation_results.json` from NumPyro execution step (searched recursively under `numpyro/` paths or `numpyro_simulation_results.json`)
- Fields: `beliefs`, `actions`, `observations`, `efe_history`, `validation`, `model_name`

## Output Format

- PNG plots: `belief_trajectory.png`, `action_distribution.png`, `efe_history.png`
- JSON summary: `{model}/numpyro_analysis.json` (framework, metrics, plots_generated)

## Processing Requirements

- Handles variable-length simulation runs
- Graceful degradation when matplotlib unavailable (logs warnings, skips plots)
- Numerical stability for near-zero probability values

## Error Handling

- Missing results file → returns empty analysis with warning
- Malformed JSON → logs parse error, continues with available data
