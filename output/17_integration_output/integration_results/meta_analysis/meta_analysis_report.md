# Parameter Sweep Meta-Analysis Report

**Models**: 10 | **Frameworks**: 8 | **Total Records**: 80

---

## Framework Summary

| Framework | Runs | Avg Runtime | Min | Max | Avg ms/step |
|---|---|---|---|---|---|
| activeinference_jl | 10 | 13.0s | 9.9s | 14.7s | — |
| bnlearn | 10 | 1.9s | 1.9s | 1.9s | — |
| discopy | 10 | 0.2s | 0.2s | 0.3s | — |
| jax | 10 | 0.8s | 0.7s | 0.9s | 52.80 |
| pymdp | 10 | 3.2s | 2.2s | 4.9s | 186.68 |
| pytorch | 10 | 0.5s | 0.5s | 0.5s | 33.18 |
| rxinfer | 10 | 13.9s | 13.3s | 19.1s | — |

## Simulation Quality Metrics

### Quality-Certainty Correlation

Across **20** models, the Pearson correlation between belief entropy and accuracy is **r = -0.841**.

### Observation Accuracy

| Model | Framework | Accuracy |
|---|---|---|
| T=30 | pymdp | 0.833 |
| actinf_pomdp_agent | rxinfer | 0.933 |
| T=30 | pymdp | 0.900 |
| bnlearn_causal_model | rxinfer | 0.933 |
| T=30 | pymdp | 0.933 |
| deep_planning_horizon | rxinfer | 0.933 |
| T=50 | pymdp | 0.400 |
| hmm_baseline | rxinfer | 0.580 |
| T=40 | pymdp | 1.000 |
| markov_chain | rxinfer | 1.000 |
| T=30 | pymdp | 0.267 |
| multi_armed_bandit | rxinfer | 0.200 |
| T=25 | pymdp | 1.000 |
| simple_mdp | rxinfer | 1.000 |
| T=10 | pymdp | 0.800 |
| time_varying_dynamics | rxinfer | 1.000 |
| T=3 | pymdp | 1.000 |
| tmaze_epistemic | rxinfer | 1.000 |
| T=20 | pymdp | 0.900 |
| two_state_bistable | rxinfer | 0.850 |

### Belief Entropy (Final Window)

| Model | Framework | Mean Entropy (nats) |
|---|---|---|
| actinf_pomdp_agent | activeinference_jl | 0.0000 |
| T=30 | jax | -0.0000 |
| T=30 | pymdp | 0.0000 |
| T=30 | pytorch | -0.0000 |
| actinf_pomdp_agent | rxinfer | 0.0000 |
| bnlearn_causal_model | activeinference_jl | 0.5146 |
| T=30 | jax | 0.3725 |
| T=30 | pymdp | 0.3807 |
| T=30 | pytorch | -0.0000 |
| bnlearn_causal_model | rxinfer | 0.4015 |
| deep_planning_horizon | activeinference_jl | 0.9251 |
| T=30 | jax | 0.3361 |
| T=30 | pymdp | 0.1009 |
| T=30 | pytorch | -0.0000 |
| deep_planning_horizon | rxinfer | 0.2077 |
| T=50 | jax | 0.0000 |
| T=50 | pymdp | 1.1491 |
| T=50 | pytorch | -0.0000 |
| hmm_baseline | rxinfer | 1.0253 |
| T=40 | jax | -0.0000 |
| T=40 | pymdp | 0.0000 |
| T=40 | pytorch | -0.0000 |
| markov_chain | rxinfer | 0.0000 |
| multi_armed_bandit | activeinference_jl | 1.0974 |
| T=30 | jax | 0.4899 |
| T=30 | pymdp | 0.7989 |
| T=30 | pytorch | -0.0000 |
| multi_armed_bandit | rxinfer | 0.8342 |
| simple_mdp | activeinference_jl | 0.0001 |
| T=25 | jax | -0.0000 |
| T=25 | pymdp | 0.0000 |
| T=25 | pytorch | -0.0000 |
| simple_mdp | rxinfer | 0.0000 |
| time_varying_dynamics | activeinference_jl | 0.7231 |
| T=10 | jax | 0.5925 |
| T=10 | pymdp | 0.1287 |
| T=10 | pytorch | -0.0000 |
| time_varying_dynamics | rxinfer | 0.1299 |
| T=3 | jax | -0.0000 |
| T=3 | pymdp | 0.0000 |
| T=3 | pytorch | -0.0000 |
| tmaze_epistemic | rxinfer | 0.0000 |
| two_state_bistable | activeinference_jl | 0.6120 |
| T=20 | jax | 0.5573 |
| T=20 | pymdp | 0.3141 |
| T=20 | pytorch | -0.0000 |
| two_state_bistable | rxinfer | 0.6189 |

## Resource Efficiency & Complexity

Analysis of generated runner complexity and compute throughput.
### Inference Throughput

| Model | Framework | Throughput (Steps/sec) |
|---|---|---|
| T=3 | pymdp | 1.39 |
| T=3 | pytorch | 6.53 |
| T=3 | jax | 4.18 |
| T=10 | pymdp | 3.95 |
| T=10 | pytorch | 19.89 |
| T=10 | jax | 13.86 |
| T=20 | pymdp | 7.06 |
| T=20 | pytorch | 43.30 |
| T=20 | jax | 26.46 |
| T=25 | pymdp | 7.97 |
| T=25 | pytorch | 53.86 |
| T=25 | jax | 31.28 |
| T=30 | pymdp | 9.08 |
| T=30 | pytorch | 60.06 |
| T=30 | jax | 37.08 |
| T=30 | pymdp | 9.11 |
| T=30 | pytorch | 64.11 |
| T=30 | jax | 37.38 |
| T=30 | pymdp | 9.16 |
| T=30 | pytorch | 60.52 |
| T=30 | jax | 36.28 |
| T=30 | pymdp | 9.28 |
| T=30 | pytorch | 64.79 |
| T=30 | jax | 38.60 |
| T=40 | pymdp | 10.64 |
| T=40 | pytorch | 75.51 |
| T=40 | jax | 48.58 |
| T=50 | pymdp | 10.17 |
| T=50 | pytorch | 107.27 |
| T=50 | jax | 52.77 |

### Code Complexity Scaling

| N | activeinference_jl LOC | bnlearn LOC | discopy LOC | jax LOC | numpyro LOC | pymdp LOC | pytorch LOC | rxinfer LOC |
|---|---|---|---|---|---|---|---|---|
| None | 282 | 67 | 173 | 292 | 119 | 262 | 106 | 241 |

> [!NOTE]
> Runner code size scales with state space complexity. PyMDP runners exhibit $O(N^3)$ scaling in generated constant matrices.

## Sweep validation

- Machine-readable report: [sweep_validation.json](sweep_validation.json)
- Issues: **info**=0, **warning**=0, **error**=0


## Aggregate statistics

- Full export: [meta_statistics.json](meta_statistics.json)

### Per-framework runtime (successful runs)

| Framework | Runs OK | Median runtime (s) | Mean (s) |
|---|---|---|---|
| activeinference_jl | 10 | 13.6246 | 13.0118 |
| bnlearn | 10 | 1.8893 | 1.9019 |
| discopy | 10 | 0.2346 | 0.2389 |
| jax | 10 | 0.8008 | 0.7981 |
| numpyro | 0 | — | — |
| pymdp | 10 | 3.2533 | 3.2431 |
| pytorch | 10 | 0.4670 | 0.4810 |
| rxinfer | 10 | 13.2980 | 13.9098 |


## Step 3 serialization footprint

| Format | Total size (MB) | Files OK |
|---|---|---|
| markdown | 0.01 | 3 |
| python | 0.03 | 3 |
| json | 0.03 | 3 |
