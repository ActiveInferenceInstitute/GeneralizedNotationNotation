# Parameter Sweep Meta-Analysis Report

**Models**: 10 | **Frameworks**: 8 | **Total Records**: 80

---

## Framework Summary

| Framework | Runs | Avg Runtime | Min | Max | Avg ms/step |
|---|---|---|---|---|---|
| activeinference_jl | 10 | 7.8s | 7.8s | 7.9s | 550.28 |
| bnlearn | 10 | 2.1s | 2.0s | 2.2s | — |
| discopy | 10 | 0.2s | 0.2s | 0.2s | — |
| jax | 10 | 0.9s | 0.8s | 1.0s | 63.90 |
| pymdp | 10 | 2.7s | 1.7s | 4.3s | 150.99 |
| pytorch | 10 | 0.5s | 0.5s | 0.6s | 34.81 |
| rxinfer | 10 | 5.4s | 5.4s | 5.5s | 381.39 |

## Simulation Quality Metrics

### Quality-Certainty Correlation

Across **30** models, the Pearson correlation between belief entropy and accuracy is **r = -0.252**.

### Observation Accuracy

| Model | Framework | Accuracy |
|---|---|---|
| T=30 | activeinference_jl | 0.867 |
| T=30 | pymdp | 0.833 |
| T=30 | rxinfer | 0.867 |
| T=30 | activeinference_jl | 0.700 |
| T=30 | pymdp | 0.900 |
| T=30 | rxinfer | 0.700 |
| T=30 | activeinference_jl | 0.733 |
| T=30 | pymdp | 0.933 |
| T=30 | rxinfer | 0.733 |
| T=50 | activeinference_jl | 0.380 |
| T=50 | pymdp | 0.400 |
| T=50 | rxinfer | 0.380 |
| T=40 | activeinference_jl | 0.675 |
| T=40 | pymdp | 1.000 |
| T=40 | rxinfer | 0.675 |
| T=30 | activeinference_jl | 0.233 |
| T=30 | pymdp | 0.267 |
| T=30 | rxinfer | 0.233 |
| T=25 | activeinference_jl | 0.880 |
| T=25 | pymdp | 1.000 |
| T=25 | rxinfer | 0.880 |
| T=10 | activeinference_jl | 0.700 |
| T=10 | pymdp | 0.800 |
| T=10 | rxinfer | 0.700 |
| T=3 | activeinference_jl | 0.000 |
| T=3 | pymdp | 0.000 |
| T=3 | rxinfer | 0.000 |
| T=20 | activeinference_jl | 0.750 |
| T=20 | pymdp | 0.900 |
| T=20 | rxinfer | 0.750 |

### Belief Entropy (Final Window)

| Model | Framework | Mean Entropy (nats) |
|---|---|---|
| T=30 | activeinference_jl | -0.0000 |
| T=30 | jax | -0.0000 |
| T=30 | pymdp | 0.0000 |
| T=30 | pytorch | 0.0826 |
| T=30 | rxinfer | -0.0000 |
| T=30 | activeinference_jl | 0.6515 |
| T=30 | jax | 0.3725 |
| T=30 | pymdp | 0.3807 |
| T=30 | pytorch | 0.0000 |
| T=30 | rxinfer | 0.6515 |
| T=30 | activeinference_jl | 0.5906 |
| T=30 | jax | 0.3361 |
| T=30 | pymdp | 0.1009 |
| T=30 | pytorch | 0.0000 |
| T=30 | rxinfer | 0.5906 |
| T=50 | activeinference_jl | 1.2830 |
| T=50 | jax | 0.8783 |
| T=50 | pymdp | 1.1491 |
| T=50 | pytorch | 0.6513 |
| T=50 | rxinfer | 1.2830 |
| T=40 | activeinference_jl | 1.0171 |
| T=40 | jax | -0.0000 |
| T=40 | pymdp | 0.0000 |
| T=40 | pytorch | 0.0000 |
| T=40 | rxinfer | 1.0171 |
| T=30 | activeinference_jl | 0.9140 |
| T=30 | jax | 0.4899 |
| T=30 | pymdp | 0.7989 |
| T=30 | pytorch | 0.5528 |
| T=30 | rxinfer | 0.9140 |
| T=25 | activeinference_jl | 0.3251 |
| T=25 | jax | -0.0000 |
| T=25 | pymdp | 0.0000 |
| T=25 | pytorch | 0.0000 |
| T=25 | rxinfer | 0.3251 |
| T=10 | activeinference_jl | 0.6736 |
| T=10 | jax | 0.5925 |
| T=10 | pymdp | 0.1287 |
| T=10 | pytorch | 0.0027 |
| T=10 | rxinfer | 0.6736 |
| T=3 | activeinference_jl | -0.0000 |
| T=3 | jax | 0.6931 |
| T=3 | pymdp | 0.0000 |
| T=3 | pytorch | 0.0000 |
| T=3 | rxinfer | -0.0000 |
| T=20 | activeinference_jl | 0.6632 |
| T=20 | jax | 0.5598 |
| T=20 | pymdp | 0.3141 |
| T=20 | pytorch | 0.5968 |
| T=20 | rxinfer | 0.6632 |

## Resource Efficiency & Complexity

Analysis of generated runner complexity and compute throughput.
### Inference Throughput

| Model | Framework | Throughput (Steps/sec) |
|---|---|---|
| T=3 | pymdp | 1.76 |
| T=3 | pytorch | 6.31 |
| T=3 | jax | 2.99 |
| T=3 | rxinfer | 0.55 |
| T=3 | activeinference_jl | 0.38 |
| T=10 | pymdp | 5.23 |
| T=10 | pytorch | 20.53 |
| T=10 | jax | 12.74 |
| T=10 | rxinfer | 1.85 |
| T=10 | activeinference_jl | 1.27 |
| T=20 | pymdp | 9.00 |
| T=20 | pytorch | 37.04 |
| T=20 | jax | 25.44 |
| T=20 | rxinfer | 3.68 |
| T=20 | activeinference_jl | 2.55 |
| T=25 | pymdp | 10.10 |
| T=25 | pytorch | 45.79 |
| T=25 | jax | 29.92 |
| T=25 | rxinfer | 4.61 |
| T=25 | activeinference_jl | 3.19 |
| T=30 | pymdp | 11.30 |
| T=30 | pytorch | 62.43 |
| T=30 | jax | 36.04 |
| T=30 | pymdp | 11.40 |
| T=30 | pytorch | 58.43 |
| T=30 | jax | 37.02 |
| T=30 | pymdp | 10.02 |
| T=30 | pytorch | 50.18 |
| T=30 | jax | 32.77 |
| T=30 | pymdp | 10.62 |
| T=30 | pytorch | 59.83 |
| T=30 | jax | 37.35 |
| T=30 | rxinfer | 5.49 |
| T=30 | activeinference_jl | 3.81 |
| T=30 | rxinfer | 5.52 |
| T=30 | activeinference_jl | 3.84 |
| T=30 | rxinfer | 5.52 |
| T=30 | activeinference_jl | 3.83 |
| T=30 | rxinfer | 5.53 |
| T=30 | activeinference_jl | 3.81 |
| T=40 | pymdp | 11.94 |
| T=40 | pytorch | 79.47 |
| T=40 | jax | 46.14 |
| T=40 | rxinfer | 7.22 |
| T=40 | activeinference_jl | 5.10 |
| T=50 | pymdp | 11.53 |
| T=50 | pytorch | 104.30 |
| T=50 | jax | 51.63 |
| T=50 | rxinfer | 9.23 |
| T=50 | activeinference_jl | 6.38 |

### Code Complexity Scaling

| N | activeinference_jl LOC | bnlearn LOC | discopy LOC | jax LOC | numpyro LOC | pymdp LOC | pytorch LOC | rxinfer LOC |
|---|---|---|---|---|---|---|---|---|
| None | 236 | 67 | 173 | 292 | 121 | 484 | 108 | 236 |

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
| activeinference_jl | 10 | 7.8399 | 7.8459 |
| bnlearn | 10 | 2.0497 | 2.0649 |
| discopy | 10 | 0.2423 | 0.2419 |
| jax | 10 | 0.8340 | 0.8608 |
| numpyro | 0 | — | — |
| pymdp | 10 | 2.6430 | 2.7104 |
| pytorch | 10 | 0.5024 | 0.5125 |
| rxinfer | 10 | 5.4310 | 5.4420 |


## Step 3 serialization footprint

| Format | Total size (MB) | Files OK |
|---|---|---|
| markdown | 0.01 | 3 |
| python | 0.03 | 3 |
| json | 0.03 | 3 |
