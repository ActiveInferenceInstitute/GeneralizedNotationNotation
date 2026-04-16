# PyMDP Analysis — Technical Specification

**Version**: 1.6.0

## Input

- `simulation_results.json` from PyMDP execution step
- Belief histories, action sequences, EFE values

## Output

- Belief trajectory heatmaps (PNG)
- Action selection analysis (PNG)
- EFE component decomposition (PNG)
- Free energy landscape (PNG)
- `analysis_summary.json`

## Framework

- PyMDP-specific result format (belief arrays, policy distributions)
- Matplotlib + seaborn visualization

## Analysis Types

- Posterior belief evolution over time
- Policy entropy and decisiveness metrics
- Expected Free Energy decomposition (epistemic vs pragmatic value)

## Error Handling

- Missing results → warning + empty analysis
- Variable-length simulations handled automatically
