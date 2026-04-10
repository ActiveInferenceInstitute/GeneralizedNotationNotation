# Step 0: Template — Pipeline Pattern Demonstration

## Overview

Demonstrates the thin orchestrator pattern that all pipeline steps follow. Serves as a reference implementation and infrastructure validation tool.

## Usage

```bash
python src/0_template.py --target-dir input/gnn_files --output-dir output --verbose
python src/0_template.py --simulate-error  # Test error handling
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/0_template.py` (64 lines) |
| Module | `src/template/` |
| Processor | `src/template/processor.py` |
| Module function | `process_template_standardized()` |

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--simulate-error` | `bool` | Simulate an error for testing error handling |

## Output

- **Directory**: `output/0_template_output/`
- Template processing results, pattern validation reports, and error handling demonstrations

## Source

- **Script**: [src/0_template.py](../../../src/0_template.py)
