---
name: gnn-intelligent-analysis
description: GNN AI-powered pipeline analysis and executive reports. Use when generating AI-driven executive summaries, performing intelligent pipeline health assessments, or creating comprehensive AI-enhanced analysis of GNN processing results.
---

# GNN Intelligent Analysis (Step 24)

## Purpose

Provides AI-powered analysis of the entire pipeline execution, generating executive summaries, health assessments, intelligent recommendations, and comprehensive AI-enhanced reports.

## Key Commands

```bash
# Run intelligent analysis
python src/24_intelligent_analysis.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline (final step)
python src/main.py --only-steps 24 --verbose
```

## API

```python
from intelligent_analysis import (
    process_intelligent_analysis, IntelligentAnalyzer,
    analyze_pipeline_summary, analyze_individual_steps,
    generate_executive_report, identify_bottlenecks,
    generate_recommendations, calculate_pipeline_health_score,
    classify_failure_severity, detect_performance_patterns,
    generate_optimization_suggestions
)

# Process intelligent analysis step (used by pipeline)
process_intelligent_analysis(target_dir, output_dir, verbose=True)

# Use the IntelligentAnalyzer class
analyzer = IntelligentAnalyzer()

# Analyze pipeline summary
summary = analyze_pipeline_summary(pipeline_data)

# Generate executive report
report = generate_executive_report(analysis_results)

# Health scoring
score = calculate_pipeline_health_score(pipeline_data)

# Identify bottlenecks and generate recommendations
bottlenecks = identify_bottlenecks(step_data)
recs = generate_recommendations(analysis_results)

# Performance pattern detection
patterns = detect_performance_patterns(metrics)
suggestions = generate_optimization_suggestions(patterns)
```

## Key Exports

- `process_intelligent_analysis` — main pipeline processing function
- `IntelligentAnalyzer` / `AnalysisContext` / `StepAnalysis` — analysis classes
- `analyze_pipeline_summary`, `analyze_individual_steps` — analysis functions
- `generate_executive_report` — executive summary generation
- `calculate_pipeline_health_score`, `classify_failure_severity`
- `detect_performance_patterns`, `generate_optimization_suggestions`
- `identify_bottlenecks`, `generate_recommendations`

## Pipeline Position

This is the **final step** (Step 24). It has access to all outputs from Steps 0–23 and provides the capstone analysis.

## Output

- AI analysis reports in `output/24_intelligent_analysis_output/`
- Executive summary documents
- Pipeline health scorecards


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_intelligent_analysis`
- `get_analysis_capabilities`
- `get_intelligent_analysis_module_info`

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
