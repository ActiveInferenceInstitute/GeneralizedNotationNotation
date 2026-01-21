# Output Directory - Agent Documentation

## Overview

The `output/` directory is a **generated artifacts directory** - not a processing module. It contains the results of pipeline execution.

## Purpose

- Stores all generated artifacts from pipeline steps 0-23
- Contains pipeline execution summary and metadata
- Holds rendered code, visualizations, reports, and analysis results

## Agent Guidelines

### What Agents Should Know

1. **Do Not Modify**: This directory contains machine-generated content
2. **Ephemeral Content**: Contents may be overwritten on each pipeline run
3. **Not Version Controlled**: Directory is in `.gitignore`
4. **Read-Only Access**: Use for retrieving pipeline results only

### Key Files

- `00_pipeline_summary/pipeline_execution_summary.json` - Complete execution metadata
- `N_step_output/` - Output from pipeline step N

### Accessing Results

```python
from pathlib import Path
import json

output_dir = Path("src/output")
summary_path = output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"

if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)
    print(f"Pipeline completed in {summary['total_duration_seconds']}s")
```

## No Processing Functions

Unlike other module directories, `output/` contains no Python code or processing functions. It is purely a data directory.

---

**Status**: Generated Directory (Not a Module)
