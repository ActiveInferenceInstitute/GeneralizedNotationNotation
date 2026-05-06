# Parameter Sweep Meta-Analysis Report

**Models**: 21 | **Frameworks**: 1 | **Total Records**: 21

---

## Sweep Configuration

- **State space sizes (N)**: 2, 4, 8, 16, 32, 64, 128
- **Timestep counts (T)**: 10, 50, 100
- **Grid dimensions**: 7 × 3 = 21 cells
- **Frameworks**: pymdp

## Runtime Comparison

| N | T | pymdp |
|---|---|---|
| 2 | 10 | 4.4s |
| 2 | 50 | 7.8s |
| 2 | 100 | 12.2s |
| 4 | 10 | 3.9s |
| 4 | 50 | 7.7s |
| 4 | 100 | 9.7s |
| 8 | 10 | 5.2s |
| 8 | 50 | 8.4s |
| 8 | 100 | 10.0s |
| 16 | 10 | 6.4s |
| 16 | 50 | 12.5s |
| 16 | 100 | 19.9s |
| 32 | 10 | 7.0s |
| 32 | 50 | 15.3s |
| 32 | 100 | 24.5s |
| 64 | 10 | 7.5s |
| 64 | 50 | 13.1s |
| 64 | 100 | 17.8s |
| 128 | 10 | 16.1s |
| 128 | 50 | 19.3s |
| 128 | 100 | 17.9s |

## Framework Summary

| Framework | Runs | Avg Runtime | Min | Max | Avg ms/step |
|---|---|---|---|---|---|
| pymdp | 21 | 11.7s | 3.9s | 24.5s | 374.86 |

## Simulation Quality Metrics

### Quality-Certainty Correlation

Across **21** models, the Pearson correlation between belief entropy and accuracy is **r = 0.682**.

### Observation Accuracy

| Model | Framework | Accuracy |
|---|---|---|
| N=128, T=10 | pymdp | 0.700 |
| N=128, T=100 | pymdp | 0.890 |
| N=128, T=50 | pymdp | 0.900 |
| N=16, T=10 | pymdp | 0.700 |
| N=16, T=100 | pymdp | 0.890 |
| N=16, T=50 | pymdp | 0.900 |
| N=2, T=10 | pymdp | 0.800 |
| N=2, T=100 | pymdp | 0.960 |
| N=2, T=50 | pymdp | 0.960 |
| N=32, T=10 | pymdp | 0.700 |
| N=32, T=100 | pymdp | 0.890 |
| N=32, T=50 | pymdp | 0.900 |
| N=4, T=10 | pymdp | 0.800 |
| N=4, T=100 | pymdp | 0.910 |
| N=4, T=50 | pymdp | 0.920 |
| N=64, T=10 | pymdp | 0.700 |
| N=64, T=100 | pymdp | 0.890 |
| N=64, T=50 | pymdp | 0.900 |
| N=8, T=10 | pymdp | 0.800 |
| N=8, T=100 | pymdp | 0.910 |
| N=8, T=50 | pymdp | 0.920 |

### Belief Entropy (Final Window)

| Model | Framework | Mean Entropy (nats) |
|---|---|---|
| N=128, T=10 | pymdp | 0.0031 |
| N=128, T=100 | pymdp | 0.4692 |
| N=128, T=50 | pymdp | 0.4692 |
| N=16, T=10 | pymdp | 0.0161 |
| N=16, T=100 | pymdp | 0.3049 |
| N=16, T=50 | pymdp | 0.4012 |
| N=2, T=10 | pymdp | 0.0357 |
| N=2, T=100 | pymdp | 0.0949 |
| N=2, T=50 | pymdp | 0.1542 |
| N=32, T=10 | pymdp | 0.0096 |
| N=32, T=100 | pymdp | 0.3217 |
| N=32, T=50 | pymdp | 0.4258 |
| N=4, T=10 | pymdp | 0.0353 |
| N=4, T=100 | pymdp | 0.1842 |
| N=4, T=50 | pymdp | 0.3330 |
| N=64, T=10 | pymdp | 0.0055 |
| N=64, T=100 | pymdp | 0.4480 |
| N=64, T=50 | pymdp | 0.4480 |
| N=8, T=10 | pymdp | 0.0255 |
| N=8, T=100 | pymdp | 0.1987 |
| N=8, T=50 | pymdp | 0.3720 |

## Scaling Analysis

- **T=10**: Runtime scales as **O(N^0.28)** ($R^2$=0.821, RMSE=0.182) (N=2.0→128.0: 4.4s → 16.1s)
- **T=50**: Runtime scales as **O(N^0.23)** ($R^2$=0.877, RMSE=0.118) (N=2.0→128.0: 7.8s → 19.3s)
- **T=100**: Runtime scales as **O(N^0.17)** ($R^2$=0.489, RMSE=0.239) (N=2.0→128.0: 12.2s → 17.9s)
- **N=2**: Runtime scales as **O(T^0.42)** ($R^2$=0.975, RMSE=0.065) (T=10.0→100.0: 4.4s → 12.2s)
- **N=4**: Runtime scales as **O(T^0.39)** ($R^2$=0.998, RMSE=0.016) (T=10.0→100.0: 3.9s → 9.7s)
- **N=8**: Runtime scales as **O(T^0.28)** ($R^2$=0.999, RMSE=0.008) (T=10.0→100.0: 5.2s → 10.0s)
- **N=16**: Runtime scales as **O(T^0.48)** ($R^2$=0.984, RMSE=0.059) (T=10.0→100.0: 6.4s → 19.9s)
- **N=32**: Runtime scales as **O(T^0.53)** ($R^2$=0.993, RMSE=0.042) (T=10.0→100.0: 7.0s → 24.5s)
- **N=64**: Runtime scales as **O(T^0.37)** ($R^2$=0.997, RMSE=0.020) (T=10.0→100.0: 7.5s → 17.8s)
- **N=128**: Runtime scales as **O(T^0.06)** ($R^2$=0.544, RMSE=0.050) (T=10.0→100.0: 16.1s → 17.9s)

### Theoretical vs. Empirical Complexity

The theoretical complexity for a dense PyMDP Active Inference agent is **O(T × N³)**.
- **N-scaling**: The empirical exponents (α ≈ 0.1–0.3) are significantly lower than the theoretical α=3.0. This indicates that at these scales (N ≤ 128), wall-clock time is dominated by constant **JIT compilation overhead** and JAX framework initialization rather than matrix operation complexity.
- **T-scaling**: The empirical exponents (β ≈ 0.5–0.7) are also lower than the theoretical β=1.0. This suggests that the iterative inference loop benefits from JAX's optimized kernel execution, reducing the per-step cost as T increases.

## Resource Efficiency & Complexity

Analysis of generated runner complexity and compute throughput.
### Inference Throughput

| Model | Framework | Throughput (Steps/sec) |
|---|---|---|
| N=2, T=10 | pymdp | 2.25 |
| N=2, T=50 | pymdp | 6.42 |
| N=2, T=100 | pymdp | 8.23 |
| N=4, T=10 | pymdp | 2.54 |
| N=4, T=50 | pymdp | 6.54 |
| N=4, T=100 | pymdp | 10.34 |
| N=8, T=10 | pymdp | 1.91 |
| N=8, T=50 | pymdp | 5.96 |
| N=8, T=100 | pymdp | 9.99 |
| N=16, T=10 | pymdp | 1.56 |
| N=16, T=50 | pymdp | 4.01 |
| N=16, T=100 | pymdp | 5.03 |
| N=32, T=10 | pymdp | 1.43 |
| N=32, T=50 | pymdp | 3.26 |
| N=32, T=100 | pymdp | 4.08 |
| N=64, T=10 | pymdp | 1.33 |
| N=64, T=50 | pymdp | 3.81 |
| N=64, T=100 | pymdp | 5.63 |
| N=128, T=10 | pymdp | 0.62 |
| N=128, T=50 | pymdp | 2.59 |
| N=128, T=100 | pymdp | 5.60 |

### Code Complexity Scaling

| N | pymdp LOC |
|---|---|
| 2 | 272 |
| 4 | 376 |
| 8 | 992 |
| 16 | 5,200 |
| 32 | 36,272 |
| 64 | 275,056 |
| 128 | 2,147,312 |

> [!NOTE]
> Runner code size scales with state space complexity. PyMDP runners exhibit $O(N^3)$ scaling in generated constant matrices.

## Sweep validation

- Machine-readable report: [sweep_validation.json](sweep_validation.json)
- Issues: **info**=0, **warning**=0, **error**=0

- Expected grid cells (N×T): **21**, records with (N,T): **21**


## Aggregate statistics

- Full export: [meta_statistics.json](meta_statistics.json)

### Per-framework runtime (successful runs)

| Framework | Runs OK | Median runtime (s) | Mean (s) |
|---|---|---|---|
| pymdp | 21 | 10.0066 | 11.7459 |

### Log-log runtime vs N (PyMDP slopes by T)

- **T=10**: slope=0.281, R²=0.821, n=7
- **T=50**: slope=0.227, R²=0.877, n=7
- **T=100**: slope=0.168, R²=0.489, n=7


## Step 3 serialization footprint

| Format | Total size (MB) | Files OK |
|---|---|---|
| markdown | 72.59 | 21 |
| python | 138.01 | 21 |
| json | 161.13 | 21 |


## Visualizations & Data Reports

- **Sweep Data**: Data Report: [sweep_data.csv](visualizations/data/sweep_data.csv)
- **Runtime Heatmap Pymdp**: Image: [runtime_heatmap_pymdp.png](visualizations/pymdp/heatmaps/runtime_heatmap_pymdp.png) | Data Report: [runtime_heatmap_pymdp.csv](visualizations/pymdp/heatmaps/runtime_heatmap_pymdp.csv)
- **Runtime Scaling Curves**: Image: [runtime_scaling_curves.png](visualizations/cross_framework/scaling/runtime_scaling_curves.png) | Data Report: [runtime_scaling_curves.csv](visualizations/cross_framework/scaling/runtime_scaling_curves.csv)
- **Framework Runtime Comparison**: Image: [framework_runtime_comparison.png](visualizations/cross_framework/comparisons/framework_runtime_comparison.png) | Data Report: [framework_runtime_comparison.csv](visualizations/cross_framework/comparisons/framework_runtime_comparison.csv)
- **Time Per Step**: Image: [time_per_step.png](visualizations/cross_framework/comparisons/time_per_step.png) | Data Report: [time_per_step.csv](visualizations/cross_framework/comparisons/time_per_step.csv)
- **Accuracy Comparison**: Image: [accuracy_comparison.png](visualizations/cross_framework/comparisons/accuracy_comparison.png) | Data Report: [accuracy_comparison.csv](visualizations/cross_framework/comparisons/accuracy_comparison.csv)
- **Entropy Vs States**: Image: [entropy_vs_states.png](visualizations/cross_framework/comparisons/entropy_vs_states.png) | Data Report: [entropy_vs_states.csv](visualizations/cross_framework/comparisons/entropy_vs_states.csv)
- **Accuracy Heatmap Pymdp**: Image: [accuracy_heatmap_pymdp.png](visualizations/pymdp/heatmaps/accuracy_heatmap_pymdp.png) | Data Report: [accuracy_heatmap_pymdp.csv](visualizations/pymdp/heatmaps/accuracy_heatmap_pymdp.csv)
- **Entropy Heatmap Pymdp**: Image: [entropy_heatmap_pymdp.png](visualizations/pymdp/heatmaps/entropy_heatmap_pymdp.png) | Data Report: [entropy_heatmap_pymdp.csv](visualizations/pymdp/heatmaps/entropy_heatmap_pymdp.csv)
- **Runtime Surface 3D Pymdp**: [runtime_surface_3d_pymdp.png](visualizations/pymdp/surfaces/runtime_surface_3d_pymdp.png)
- **Compute Efficiency**: [compute_efficiency.png](visualizations/cross_framework/comparisons/compute_efficiency.png)
- **Resource Scaling Loc**: [resource_scaling_loc.png](visualizations/cross_framework/scaling/resource_scaling_loc.png)
- **Accuracy Entropy Correlation**: [accuracy_entropy_correlation.png](visualizations/cross_framework/comparisons/accuracy_entropy_correlation.png)
- **Throughput Vs N**: Image: [throughput_vs_n.png](visualizations/cross_framework/scaling/throughput_vs_n.png) | Data Report: [throughput_vs_n.csv](visualizations/cross_framework/scaling/throughput_vs_n.csv)
- **Runtime Distribution**: [runtime_distribution.png](visualizations/cross_framework/comparisons/runtime_distribution.png)
- **Scaling Exponent Summary**: [scaling_exponent_summary.png](visualizations/cross_framework/scaling/scaling_exponent_summary.png)
- **Code Efficiency**: [code_efficiency.png](visualizations/cross_framework/scaling/code_efficiency.png)
- **Comprehensive Dashboard**: [comprehensive_dashboard.png](visualizations/cross_framework/comprehensive_dashboard.png)
- **Accuracy Vs Timesteps**: [accuracy_vs_timesteps.png](visualizations/cross_framework/comparisons/accuracy_vs_timesteps.png)
- **Gnn Serialization Footprint**: [gnn_serialization_footprint.png](visualizations/data/gnn_serialization_footprint.png)