# Step 23: Report — Pipeline Report Generation

## Overview

Generates comprehensive reports aggregating results from all prior pipeline steps into readable documents.

## Usage

```bash
python src/23_report.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/23_report.py` (63 lines) |
| Module | `src/report/` |
| Processor | `src/report/processor.py` |
| Module function | `process_report()` |

## Output

- **Directory**: `output/23_report_output/`
- Comprehensive pipeline reports and executive summaries

## Source

- **Script**: [src/23_report.py](../../src/23_report.py)
