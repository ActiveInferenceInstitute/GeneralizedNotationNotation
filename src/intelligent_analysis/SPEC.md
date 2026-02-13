# Intelligent Analysis Module Specification

## Overview

AI-powered pipeline analysis with LLM-backed insights, per-step flag detection, health scoring, and executive report generation. Step 24 of the GNN pipeline.

## Components

### Core

- `processor.py` - Pipeline analysis processor (890 lines)
  - `process_intelligent_analysis()` — main entry point
  - `analyze_pipeline_summary()` — extract key metrics
  - `analyze_individual_steps()` — per-step yellow/red flag detection
  - `identify_bottlenecks()` — performance bottleneck identification
  - `extract_failure_context()` — root cause analysis
  - `generate_recommendations()` — actionable optimization suggestions
  - `generate_executive_report()` — comprehensive markdown report
  - `_run_llm_analysis()` — LLM-powered insight generation
  - `_generate_rule_based_summary()` — fallback when LLM unavailable
- `analyzer.py` - Analysis helper classes (476 lines)
  - `IntelligentAnalyzer` — stateful analysis class with health scoring
  - `AnalysisContext` — dataclass for pipeline context
  - `calculate_pipeline_health_score()` — weighted score (0-100)
  - `classify_failure_severity()` — critical/major/minor classification
  - `detect_performance_patterns()` — pattern recognition
  - `generate_optimization_suggestions()` — optimization recommendations
- `mcp.py` - MCP tool integration

### Data Classes

- `StepAnalysis` — per-step analysis record (status, duration, memory, flags)
- `AnalysisContext` — pipeline context with property accessors for steps, failures, performance

## Analysis Types

- Failure root cause analysis
- Performance bottleneck identification
- Per-step yellow/red flag detection
- Health scoring (success rate, warning rate, duration/memory efficiency)
- LLM-powered executive insights (with rule-based fallback)
- Trend and comparative analysis

## Key Exports

```python
from intelligent_analysis import (
    process_intelligent_analysis,
    IntelligentAnalyzer,
    AnalysisContext,
    StepAnalysis,
    calculate_pipeline_health_score,
    generate_executive_report,
)
```
