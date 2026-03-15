# Step 19: Research — Research Processing

## Overview

Orchestrates research processing for GNN models, integrating with academic literature and research methodologies.

## Usage

```bash
python src/19_research.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/19_research.py` (63 lines) |
| Module | `src/research/` |
| Processor | `src/research/processor.py` |
| Module function | `process_research()` |

## Output

- **Directory**: `output/19_research_output/`
- Research analysis reports and literature integration summaries

## Source

- **Script**: [src/19_research.py](../../src/19_research.py)
