# Parameter Sweep Meta-Analysis Report

**Models**: 1 | **Frameworks**: 8 | **Total Records**: 8

---

## Framework Summary

| Framework | Runs | Avg Runtime | Min | Max | Avg ms/step |
|---|---|---|---|---|---|
| activeinference_jl | 1 | 10.4s | 10.4s | 10.4s | 695.26 |
| discopy | 1 | 0.3s | 0.3s | 0.3s | — |
| jax | 1 | 1.0s | 1.0s | 1.0s | 69.38 |
| numpyro | 1 | 1.1s | 1.1s | 1.1s | 71.98 |
| pymdp | 1 | 2.9s | 2.9s | 2.9s | 190.25 |
| rxinfer | 1 | 6.6s | 6.6s | 6.6s | 438.65 |

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
| T=15 | numpyro | 13.89 |
| T=15 | pymdp | 5.26 |
| T=15 | jax | 14.41 |
| T=15 | rxinfer | 2.28 |
| T=15 | activeinference_jl | 1.44 |

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
| activeinference_jl | 1 | 10.4288 | 10.4288 |
| bnlearn | 0 | — | — |
| discopy | 1 | 0.3112 | 0.3112 |
| jax | 1 | 1.0407 | 1.0407 |
| numpyro | 1 | 1.0796 | 1.0796 |
| pymdp | 1 | 2.8537 | 2.8537 |
| pytorch | 0 | — | — |
| rxinfer | 1 | 6.5798 | 6.5798 |


## Step 3 serialization footprint

| Format | Total size (MB) | Files OK |
|---|---|---|
| markdown | 0.00 | 1 |
| python | 0.01 | 1 |
| json | 0.02 | 1 |
