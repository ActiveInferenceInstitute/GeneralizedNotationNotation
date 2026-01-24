# Intelligent Analysis Module

**Step 24** - AI-powered pipeline analysis and executive report generation.

## Overview

The intelligent_analysis module provides comprehensive AI-powered analysis of GNN pipeline execution results. It analyzes pipeline logs, identifies failures and performance bottlenecks, detects warning flags per step, and generates executive reports with actionable recommendations.

## Key Features

- **Pipeline Health Scoring**: Calculates overall pipeline health from execution metrics
- **Failure Root Cause Analysis**: Deep analysis of why steps failed
- **Performance Bottleneck Detection**: Identifies slow steps and optimization opportunities
- **Per-Step Flag Detection**: Yellow (warning) and red (error) flags for each step
- **LLM-Powered Insights**: AI-generated analysis when LLM infrastructure is available
- **Rule-Based Fallback**: Works without LLM using heuristic analysis
- **Executive Reports**: Markdown, JSON, and HTML formatted reports

## Module Structure

```
intelligent_analysis/
├── __init__.py       # Public API exports
├── AGENTS.md         # Agent scaffolding documentation
├── README.md         # This file
├── processor.py      # Core processing logic
└── analyzer.py       # IntelligentAnalyzer class and utilities
```

## Usage

### Command Line

```bash
# Full intelligent analysis
python src/24_intelligent_analysis.py --verbose

# Skip LLM (rule-based only)
python src/24_intelligent_analysis.py --skip-llm

# Custom bottleneck threshold
python src/24_intelligent_analysis.py --bottleneck-threshold 30.0
```

### Programmatic

```python
from intelligent_analysis import process_intelligent_analysis

result = process_intelligent_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output"),
    verbose=True
)
```

## Core Components

### StepAnalysis Dataclass

Represents detailed analysis of a single pipeline step:

```python
@dataclass
class StepAnalysis:
    step_number: int
    script_name: str
    description: str
    status: str
    duration_seconds: float
    memory_mb: float
    exit_code: int
    flags: List[str]
    flag_type: str  # "none", "yellow", "red"
    summary: str
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `process_intelligent_analysis()` | Main entry point |
| `analyze_pipeline_summary()` | Extract insights from pipeline summary |
| `analyze_individual_steps()` | Per-step analysis with flag detection |
| `generate_executive_report()` | Create formatted reports |
| `identify_bottlenecks()` | Find performance issues |
| `generate_recommendations()` | AI-generated improvement suggestions |

### IntelligentAnalyzer Class

The main analyzer class with LLM integration:

```python
from intelligent_analysis import IntelligentAnalyzer

analyzer = IntelligentAnalyzer(llm_enabled=True)
report = analyzer.analyze(pipeline_summary)
```

## Output Structure

```
output/24_intelligent_analysis_output/
├── executive_report.md          # Human-readable report
├── executive_report.json        # Machine-readable data
├── executive_report.html        # HTML formatted report
├── step_analysis/               # Per-step analysis files
│   ├── step_00_template.json
│   ├── step_01_setup.json
│   └── ...
├── recommendations.json         # Prioritized recommendations
└── analysis_summary.json        # Overall summary
```

## Flag Detection

### Yellow Flags (Warnings)
- Step duration > 2x average
- Memory usage > 100MB
- Non-zero warnings in output
- Retry attempts detected

### Red Flags (Errors)
- Step failure (non-zero exit code)
- Timeout (duration > threshold)
- Critical errors in stderr
- Resource exhaustion

## Dependencies

- **Required**: pathlib, json, logging, dataclasses
- **Optional**: LLM processor (for AI-powered analysis)
- **Optional**: numpy, pandas (for statistical analysis)

## Integration

This module reads from:
- `output/00_pipeline_summary/pipeline_execution_summary.json`
- Individual step logs and outputs

This module produces:
- Executive reports in `output/24_intelligent_analysis_output/`

---

**Version**: 2.0.0
**Last Updated**: 2026-01-23
**Status**: Production Ready
