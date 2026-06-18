# Parameter Sweep Meta-Analysis Report

**Models**: 1 | **Frameworks**: 8 | **Total Records**: 8

---

## Framework Summary

| Framework | Runs | Avg Runtime | Min | Max | Avg ms/step |
|---|---|---|---|---|---|
| activeinference_jl | 1 | 9.4s | 9.4s | 9.4s | 623.42 |
| discopy | 1 | 0.3s | 0.3s | 0.3s | — |
| jax | 1 | 1.0s | 1.0s | 1.0s | 66.22 |
| numpyro | 1 | 1.1s | 1.1s | 1.1s | 71.81 |
| pymdp | 1 | 2.7s | 2.7s | 2.7s | 179.89 |
| rxinfer | 1 | 6.3s | 6.3s | 6.3s | 423.28 |

## Simulation Quality Metrics

### Quality-Certainty Correlation

Across **3** models, the Pearson correlation between belief entropy and accuracy is **r = 1.000**.

### Observation Accuracy

| Model | Framework | Accuracy |
|---|---|---|
| T=15 | activeinference_jl | 0.400 |
| T=15 | pymdp | 0.800 |
| T=15 | rxinfer | 0.400 |

### Belief Entropy (Final Window)

| Model | Framework | Mean Entropy (nats) |
|---|---|---|
| T=15 | activeinference_jl | -0.0000 |
| T=15 | jax | -0.0000 |
| T=15 | numpyro | -0.0000 |
| T=15 | pymdp | 0.0000 |
| T=15 | rxinfer | -0.0000 |

## Resource Efficiency & Complexity

Analysis of generated runner complexity and compute throughput.
### Inference Throughput

| Model | Framework | Throughput (Steps/sec) |
|---|---|---|
| T=15 | numpyro | 13.93 |
| T=15 | pymdp | 5.56 |
| T=15 | jax | 15.10 |
| T=15 | rxinfer | 2.36 |
| T=15 | activeinference_jl | 1.60 |

### Code Complexity Scaling

| N | activeinference_jl LOC | bnlearn LOC | discopy LOC | jax LOC | numpyro LOC | pymdp LOC | pytorch LOC | rxinfer LOC |
|---|---|---|---|---|---|---|---|---|
| None | 236 | 67 | 167 | 292 | 177 | 1,949 | 164 | 236 |

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
| activeinference_jl | 1 | 9.3513 | 9.3513 |
| bnlearn | 0 | — | — |
| discopy | 1 | 0.2982 | 0.2982 |
| jax | 1 | 0.9933 | 0.9933 |
| numpyro | 1 | 1.0771 | 1.0771 |
| pymdp | 1 | 2.6983 | 2.6983 |
| pytorch | 0 | — | — |
| rxinfer | 1 | 6.3492 | 6.3492 |


## Step 3 serialization footprint

| Format | Total size (MB) | Files OK |
|---|---|---|
| markdown | 0.00 | 1 |
| python | 0.01 | 1 |
| json | 0.02 | 1 |
