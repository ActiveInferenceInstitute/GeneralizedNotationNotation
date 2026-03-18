# NumPyro Analyzer

`src/analysis/numpyro/` reads `simulation_results.json` emitted by NumPyro simulations and produces a compact analysis JSON plus optional plots.

## Entry point

- `generate_analysis_from_logs(results_dir, output_dir=None, verbose=False) -> List[str]`

## Dependencies

- required: `numpy`
- optional: `matplotlib` (plots)

