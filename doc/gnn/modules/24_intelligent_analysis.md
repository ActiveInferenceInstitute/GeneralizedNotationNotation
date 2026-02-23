# Step 24: Intelligent Analysis — LLM-Powered Pipeline Analysis

## Overview

Performs intelligent analysis of the pipeline execution using available LLM infrastructure. Reads pipeline summaries and execution logs, analyzes failures and performance bottlenecks, and generates an executive report. Supports both LLM-powered and rule-based analysis modes.

## Usage

```bash
# Full analysis with LLM
python src/24_intelligent_analysis.py --target-dir input/gnn_files --output-dir output --verbose

# Rule-based only (no LLM API needed)
python src/24_intelligent_analysis.py --skip-llm --verbose

# Custom bottleneck threshold
python src/24_intelligent_analysis.py --bottleneck-threshold 120.0 --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/24_intelligent_analysis.py` (41 lines) |
| Module | `src/intelligent_analysis/` |
| Processor | `src/intelligent_analysis/processor.py` |
| Module function | `process_intelligent_analysis()` |

Uses direct import — one of the cleanest orchestrators alongside Step 3.

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--analysis-model` | `str` | `None` | Specific LLM model to use |
| `--skip-llm` | `bool` | `False` | Use rule-based analysis only |
| `--bottleneck-threshold` | `float` | `60.0` | Duration threshold (seconds) for bottleneck detection |

## Prerequisites

Depends on Step 23 (Report) for pipeline summary data received as input.

## Output

- **Directory**: `output/24_intelligent_analysis_output/`
- Executive reports, bottleneck analysis, failure categorization, and optimization recommendations

## Source

- **Script**: [src/24_intelligent_analysis.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/24_intelligent_analysis.py)
