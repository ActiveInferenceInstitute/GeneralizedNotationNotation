# ActiveInference.jl Analysis Module

Framework-specific analysis for ActiveInference.jl execution results.

## Public Surface

- `generate_analysis_from_logs(execution_results_dir, output_dir, verbose=False)`
- Step 16 calls this analyzer for current `activeinference_jl` Step 12 outputs.

## Input Contract

The primary input is:

```text
output/12_execute_output/<model>/activeinference_jl/simulation_data/simulation_results.json
```

The JSON schema is `activeinference_jl_simulation_v1` and includes observations by modality, hidden states by factor, actions by control factor, beliefs by factor, expected free energy, policy posterior, validation, matrix provenance, and runtime metadata.

## Outputs

The analyzer writes plots under:

```text
output/16_analysis_output/activeinference_jl/
```

Generated plots include belief heatmaps, action traces, observation traces, and expected-free-energy traces.

## Verification

```bash
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
