---
name: gnn-template-init
description: GNN pipeline template and initialization system. Use when creating new pipeline steps, bootstrapping project structure, or understanding the thin orchestrator pattern for GNN module development.
---

# GNN Template & Initialization (Step 0)

## Purpose

Initializes the pipeline environment and provides the canonical template for creating new pipeline steps. This is the entry point for the 25-step GNN pipeline.

## Key Commands

```bash
# Run template initialization
python src/0_template.py --target-dir input/gnn_files --output-dir output --verbose

# Use as part of full pipeline
python src/main.py --only-steps 0 --verbose
```

## API

```python
from template import (
    process_template_standardized, process_single_file,
    validate_file, generate_correlation_id,
    safe_template_execution, demonstrate_utility_patterns,
    get_version_info
)

# Process template step (used by pipeline)
result = process_template_standardized(target_dir, output_dir, logger)

# Process a single file
result = process_single_file(file_path, output_dir, logger)

# Validate a file
is_valid = validate_file(file_path)

# Generate correlation ID for tracing
corr_id = generate_correlation_id()
```

## Key Exports

- `process_template_standardized` — main pipeline processing function
- `process_single_file` — process individual file
- `validate_file` — file validation
- `generate_correlation_id` — unique ID for pipeline tracing
- `safe_template_execution` — execution with error handling
- `get_version_info` — module version metadata

## Thin Orchestrator Pattern

```python
# N_module.py — Thin orchestrator (<150 lines)
from utils.pipeline_template import create_standardized_pipeline_script
run_script = create_standardized_pipeline_script("N_module.py", process_func, "Description")
```

- Orchestrators handle ONLY: arg parsing, logging, output dirs, delegation
- ALL domain logic lives in `src/module/processor.py`
- Exit codes: 0=success, 1=error, 2=warnings


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `template.process_file`
- `template.process_directory`
- `template.get_info`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [pipeline_step_template.py](../pipeline_step_template.py) — Canonical template


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
