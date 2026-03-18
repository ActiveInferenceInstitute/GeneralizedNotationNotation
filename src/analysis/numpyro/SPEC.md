# NumPyro Analysis Backend Specification

## Overview

Consumes NumPyro execution outputs and produces analysis artifacts used by Step 16 and downstream reporting.

## Public API

- `generate_analysis_from_logs(results_dir: Path, output_dir: Optional[Path] = None, verbose: bool = False) -> List[str]`

## Discovery contract

The analyzer searches under `results_dir` for:

- `**/numpyro/**/simulation_results.json`
- `**/numpyro_simulation_results.json`
- `simulation_results.json` at the root as a recovery path

## Outputs

For each discovered results file, writes:

- `output_dir/<model_name>/numpyro_analysis.json`

and optionally plot PNGs when `matplotlib` is available.

