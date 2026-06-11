# NumPyro Analysis

Framework-specific analysis module for NumPyro simulation outputs.

## Usage

```python
from analysis.numpyro.analyzer import analyze_numpyro_results

analyze_numpyro_results(results_dir="output/12_execute_output", output_dir="output/16_analysis_output")
```

## Outputs

- Belief trajectory plots (PNG)
- Action distribution histograms (PNG)
- EFE component analysis (PNG)
- Summary statistics (JSON)

## Dependencies

- `numpy`, `matplotlib` (required)
- `numpyro` (optional, for advanced result interpretation)

## See Also

- [Parent: analysis/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
- [SPEC.md](SPEC.md) — Technical specification
