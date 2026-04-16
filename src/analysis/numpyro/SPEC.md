# NumPyro Analysis — Technical Specification

**Version**: 1.6.0

## Input Format

- `simulation_results.json` from NumPyro execution step
- Fields: `belief_history`, `action_history`, `efe_values`, `metadata`

## Output Format

- PNG plots: `belief_trajectory.png`, `action_distribution.png`, `efe_analysis.png`
- JSON summary: `analysis_summary.json`

## Processing Requirements

- Handles variable-length simulation runs
- Graceful degradation when matplotlib unavailable (logs warnings, skips plots)
- Numerical stability for near-zero probability values

## Error Handling

- Missing results file → returns empty analysis with warning
- Malformed JSON → logs parse error, continues with available data
