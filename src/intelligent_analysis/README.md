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
- **Rule-Based Recovery**: Works without LLM using heuristic analysis
- **Executive Reports**: Markdown and JSON formatted reports

## Module Structure

```
intelligent_analysis/
├── __init__.py       # Public API exports, tool checks
├── AGENTS.md         # Agent scaffolding documentation
├── README.md         # This file
├── processor.py      # Core processing logic and report generation
├── analyzer.py       # IntelligentAnalyzer class and analysis utilities
├── remediation.py    # ContractViolation fix suggestions (auxiliary)
└── mcp.py            # MCP tool registrations
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
import logging
from intelligent_analysis import process_intelligent_analysis

result = process_intelligent_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output"),
    logger=logging.getLogger("pipeline"),
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
    status: StepStatus        # from pipeline.context
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
| `generate_recommendations()` | Rule-based improvement suggestions |

### IntelligentAnalyzer Class

The main analyzer class with LLM integration:

```python
from intelligent_analysis import IntelligentAnalyzer, AnalysisContext

context = AnalysisContext(summary_data=pipeline_summary)
analyzer = IntelligentAnalyzer(context=context)
results = analyzer.analyze()
```

## Output Structure

```
output/24_intelligent_analysis_output/
├── intelligent_analysis_report.md      # Human-readable executive report
├── analysis_data.json                  # Machine-readable analysis data
└── intelligent_analysis_summary.json   # Compact summary with counts and paths
```

## Flag Detection

### Yellow Flags (Warnings)
- Step duration > 60 s (SLOW) or > 120 s (VERY_SLOW)
- Duration > 3x the pipeline average
- Memory usage > 500 MB (HIGH) or > 1000 MB (CRITICAL)
- Retry attempts or dependency warnings detected

### Red Flags (Errors)
- FAILED status or non-zero exit code

## Dependencies

- **Required**: pathlib, json, logging, asyncio, dataclasses

## Integration

This module reads from:
- `output/00_pipeline_summary/pipeline_execution_summary.json`
- Individual step logs and outputs

This module produces:
- Executive reports in `output/24_intelligent_analysis_output/`

---

**Last Updated**: 2026-09-02
**Status**: Production Ready


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
