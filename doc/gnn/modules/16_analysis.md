# Step 16: Analysis — Statistical and Post-Simulation Analysis

## Overview

Performs comprehensive analysis of GNN files and execution results. Includes statistical analysis, complexity metrics, performance benchmarks, and post-simulation visualization with per-framework and cross-framework comparison dashboards.

## Usage

```bash
python src/16_analysis.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/16_analysis.py` (53 lines) |
| Module | `src/analysis/` |
| Processor | `src/analysis/processor.py` (475 lines) |
| Analyzer | `src/analysis/analyzer.py` (1141 lines) |
| Post-simulation | `src/analysis/post_simulation.py` (273 lines) |
| Visualizations | `src/analysis/visualizations.py` (1241 lines) |
| Math utilities | `src/analysis/math_utils.py` |
| Framework extractors | `src/analysis/framework_extractors.py` |

### Per-Framework Analyzers

| Analyzer | Path |
|----------|------|
| PyMDP | `src/analysis/pymdp/analyzer.py` |
| ActiveInference.jl | `src/analysis/activeinference_jl/analyzer.py` |
| DisCoPy | `src/analysis/discopy/analyzer.py` |
| JAX | `src/analysis/jax/analyzer.py` |
| RxInfer | `src/analysis/rxinfer/analyzer.py` |

## Analysis Pipeline

1. **Statistical analysis** — Variable/connection extraction, distributions, correlations
2. **Complexity metrics** — Cyclomatic, cognitive, structural complexity, maintainability index
3. **Performance benchmarks** — Parse time, memory usage, estimated runtime
4. **Post-simulation analysis** — Reads from `output/12_execute_output/`:
   - Per-framework simulation data extraction (beliefs, actions, free energy)
   - Belief evolution plots, action analysis, heatmaps
   - Free energy dynamics, observation analysis
   - Unified cross-framework dashboards and comparison visualizations

## Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `perform_statistical_analysis()` | `analyzer.py` | Structural statistics |
| `calculate_complexity_metrics()` | `analyzer.py` | Code complexity |
| `analyze_framework_outputs()` | `analyzer.py` | Cross-framework comparison |
| `visualize_all_framework_outputs()` | `visualizations.py` | Comprehensive per-framework plots |
| `generate_unified_framework_dashboard()` | `visualizations.py` | Multi-panel comparison |
| `analyze_execution_results()` | `post_simulation.py` | Post-simulation aggregation |

## Active Inference Metrics

| Metric | Function |
|--------|----------|
| Shannon entropy | `compute_shannon_entropy()` |
| KL divergence | `compute_kl_divergence()` |
| Variational free energy | `compute_variational_free_energy()` |
| Expected free energy | `compute_expected_free_energy()` |
| Information gain | `compute_information_gain()` |

## Output

- **Directory**: `output/16_analysis_output/`
  - `analysis_results.json` — Combined analysis data
  - `analysis_summary.md` — Human-readable summary
  - `framework_comparison_report.md` — Cross-framework comparison
  - `pymdp/`, `jax/`, `rxinfer/`, `activeinference_jl/`, `discopy/` — Per-framework visualizations
  - `cross_framework/` — Unified dashboards and comparison plots

## Source

- **Script**: [src/16_analysis.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/16_analysis.py)
