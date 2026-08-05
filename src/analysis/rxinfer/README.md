# RxInfer.jl Analysis Module

Framework-specific analysis for RxInfer.jl execution results.

## Public Surface

- `generate_analysis_from_logs(execution_results_dir, output_dir, verbose=False)`
- Step 16 calls this analyzer for current `rxinfer` Step 12 outputs.

## Input Contract

The primary input is:

```text
output/12_execute_output/<model>/rxinfer/simulation_data/simulation_results.json
```

The JSON schema is `rxinfer_simulation_v1` and includes observations by modality, hidden states by factor, actions by control factor, beliefs by factor, expected free energy, policy posterior, `variational_free_energy`, `vfe_per_iteration`, validation, matrix provenance, and runtime metadata. The `variational_free_energy` and `vfe_per_iteration` fields are **per-iteration VFE traces** (length = INFERENCE_ITERATIONS) — the real convergence diagnostic from RxInfer's variational message passing, NOT per-step constants. Beliefs are **smoothed posteriors** from batch inference, not filtered (online) beliefs.

## Outputs

The analyzer writes plots under:

```text
output/16_analysis_output/rxinfer/
```

Generated plots include belief evolution, belief heatmaps, observation/state traces,
entropy, inference accuracy, action frequencies, belief convergence, belief trace,
free energy, observations, and EFE per action heatmap.

## Per-Exemplar Visualization Set

The RxInfer analyzer produces the full per-exemplar visualization suite from
`rxinfer_simulation_v1` results. The set is rendered as matplotlib PNGs (dpi 300)
under `output/16_analysis_output/rxinfer/` and is complementary to the optional
Julia-native `Plots.jl` PNGs emitted at render/execute time.

| Visualization | File suffix | Description |
|---|---|---|
| `belief_evolution` | `_rxinfer_belief_evolution.png` | Belief probability trajectories per hidden state over time |
| `obs_vs_true` | `_rxinfer_obs_vs_true.png` | Observations vs true hidden states per time step |
| `belief_heatmap` | `_rxinfer_belief_heatmap.png` | 2D heatmap of belief mass over time (state × time) |
| `belief_entropy` | `_rxinfer_belief_entropy.png` | Belief uncertainty (bits) over time with max-entropy reference |
| `accuracy` | `_rxinfer_accuracy.png` | Cumulative inference accuracy from argmax beliefs vs true states |
| `action_frequencies` | `_rxinfer_action_frequencies.png` | Distribution of selected actions across the run |
| `belief_convergence` | `_rxinfer_belief_convergence.png` | Convergence of beliefs toward a stable posterior |
| `belief_trace` | `_rxinfer_belief_trace.png` | Full belief trace / trajectory per time step |
| `free_energy` | `_rxinfer_free_energy.png` | Expected (and variational, when present) free energy dynamics |
| `observations` | `_rxinfer_observations.png` | Observation trace / modality-level observation analysis |

Each plot is generated from the current Step-12 `rxinfer_simulation_v1` payload and
is written per-model. Like the render/execute-time artifacts, these figures are
best-effort: any plot that cannot be produced (e.g. missing matplotlib or absent
input arrays) is logged as a warning and skipped, never treated as a failure. The
set is fully backward-compatible with the `rxinfer_simulation_v1` schema.

## Verification

```bash
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
