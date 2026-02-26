---
name: gnn-statistical-analysis
description: GNN advanced statistical analysis and result aggregation. Use when performing statistical analysis on simulation results, cross-simulation aggregation, computing information-theoretic metrics, or creating analytical visualizations of pipeline outputs.
---

# GNN Statistical Analysis (Step 16)

## Purpose

Performs advanced statistical analysis on pipeline outputs including simulation results, cross-framework comparison, Active Inference metrics, and comprehensive statistical reporting.

## Key Commands

```bash
# Run analysis
python src/16_analysis.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 16 --verbose
```

## API

```python
from analysis import (
    process_analysis, perform_statistical_analysis,
    calculate_variable_statistics, calculate_connection_statistics,
    calculate_complexity_metrics, calculate_maintainability_index,
    analyze_framework_outputs, generate_framework_comparison_report,
    analyze_simulation_traces, analyze_free_energy,
    compute_shannon_entropy, compute_kl_divergence,
    compute_variational_free_energy, compute_expected_free_energy,
    generate_analysis_summary
)

# Process analysis step (used by pipeline)
process_analysis(target_dir, output_dir, verbose=True)

# Statistical analysis
stats = perform_statistical_analysis(parsed_data)

# Complexity metrics
metrics = calculate_complexity_metrics(model_data)

# Framework comparison
report = generate_framework_comparison_report(results)

# Active Inference metrics
entropy = compute_shannon_entropy(distribution)
kl_div = compute_kl_divergence(p, q)
vfe = compute_variational_free_energy(beliefs, observations)
efe = compute_expected_free_energy(policy, beliefs)
```

## Key Exports

- `process_analysis` — main pipeline processing function
- `perform_statistical_analysis` — core statistical analysis
- `calculate_complexity_metrics`, `calculate_maintainability_index`, `calculate_technical_debt`
- `analyze_framework_outputs`, `generate_framework_comparison_report`
- `compute_shannon_entropy`, `compute_kl_divergence`, `compute_variational_free_energy`, `compute_expected_free_energy`
- `analyze_simulation_traces`, `analyze_free_energy`, `analyze_policy_convergence`
- `generate_analysis_summary` — summary report generation

## Output

- Analysis reports in `output/16_analysis_output/`
- Statistical summaries and aggregations
- Comparative visualizations


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_analysis`
- `get_analysis_results`
- `compute_complexity_metrics`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
