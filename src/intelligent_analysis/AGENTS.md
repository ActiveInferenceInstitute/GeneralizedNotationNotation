# Intelligent Analysis Module - Agent Scaffolding

## Overview

The intelligent_analysis module (Step 24) provides AI-powered analysis of pipeline execution results. It analyzes failures, identifies performance bottlenecks, performs per-step analysis with yellow/red flag detection, and generates executive reports using the LLM infrastructure.

## Module Structure

```
intelligent_analysis/
├── __init__.py       # Public API exports and module utilities
├── AGENTS.md         # This documentation
├── processor.py      # Core analysis processing logic
└── analyzer.py       # IntelligentAnalyzer class and analysis functions
```

## Key Components

### Processor (`processor.py`)
- `process_intelligent_analysis()` - Main entry point for pipeline step
- `analyze_pipeline_summary()` - Analyze overall pipeline execution
- `analyze_individual_steps()` - Per-step analysis with flag detection
- `generate_executive_report()` - Generate comprehensive reports
- `identify_bottlenecks()` - Find performance issues
- `extract_failure_context()` - Extract failure details
- `generate_recommendations()` - Generate improvement suggestions
- `StepAnalysis` - Data class for step analysis results

### Analyzer (`analyzer.py`)
- `IntelligentAnalyzer` - Main analyzer class with LLM integration
- `AnalysisContext` - Context container for analysis state
- `calculate_pipeline_health_score()` - Compute overall health metrics
- `classify_failure_severity()` - Categorize failure types
- `detect_performance_patterns()` - Identify performance trends
- `generate_optimization_suggestions()` - LLM-powered optimization advice

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Pipeline Analysis | Enabled | Full pipeline execution analysis |
| Failure Root Cause | Enabled | Deep failure analysis and causes |
| Performance Optimization | Enabled | Bottleneck identification |
| LLM-Powered Insights | Enabled | AI-generated analysis and suggestions |
| Executive Reports | Enabled | Markdown, JSON, and HTML reports |
| Per-Step Analysis | Enabled | Individual step health analysis |
| Yellow/Red Flags | Enabled | Warning and error detection per step |
| Rule-Based Fallback | Enabled | Works without LLM infrastructure |
| MCP Integration | Enabled | MCP tool registration available |

## Usage

### Command Line
```bash
# Full intelligent analysis
python src/24_intelligent_analysis.py --verbose

# Skip LLM analysis (rule-based only)
python src/24_intelligent_analysis.py --skip-llm

# Custom bottleneck threshold (seconds)
python src/24_intelligent_analysis.py --bottleneck-threshold 30.0

# Specific LLM model
python src/24_intelligent_analysis.py --analysis-model "gpt-4"
```

### Programmatic
```python
from intelligent_analysis import process_intelligent_analysis

result = process_intelligent_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output"),
    verbose=True,
    skip_llm=False,
    bottleneck_threshold=60.0
)
```

## Output

Results are written to `output/24_intelligent_analysis_output/`:
- `executive_report.md` - Human-readable executive summary
- `executive_report.json` - Machine-readable analysis data
- `executive_report.html` - HTML formatted report
- `step_analysis/` - Per-step analysis files
- `recommendations.json` - Prioritized improvement recommendations

## Analysis Types

1. **Failure Analysis** - Root cause identification for failed steps
2. **Performance Analysis** - Bottleneck detection and timing analysis
3. **Health Scoring** - Pipeline and per-step health metrics
4. **Trend Analysis** - Pattern detection across runs
5. **Optimization Suggestions** - AI-powered improvement recommendations

## Dependencies

- **Required**: pathlib, json, logging
- **Optional**: LLM processor (for AI-powered analysis)
- **Optional**: numpy, pandas (for statistical analysis)

## Data Flow

```
Pipeline Summary (00_pipeline_summary/)
    ↓
Intelligent Analysis
    ↓
├── Step Analysis (per-step flags)
├── Bottleneck Detection
├── Failure Root Cause
└── Executive Report (24_intelligent_analysis_output/)
```

---

**Version**: 2.0.0
**Last Updated**: 2026-01-23
**Status**: Production Ready
