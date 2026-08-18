---
name: gnn-pipeline-orchestration
description: GNN pipeline orchestration and configuration management. Use when configuring pipeline execution, setting step inclusion/exclusion, managing pipeline state, or customizing the 25-step execution flow.
---

# GNN Pipeline Orchestration

## Purpose

Manages pipeline configuration, step ordering, execution state, and orchestration logic for the 25-step GNN pipeline. Controls which steps run, their order, and how data flows between them.

## Key Commands

```bash
# Full pipeline
python src/main.py --target-dir input/gnn_files --verbose

# Run specific steps only
python src/main.py --only-steps "3,5,11,12" --verbose

# Skip specific steps
python src/main.py --skip-steps "15,16" --verbose

# Dry run (show what would execute)
python src/main.py --dry-run --verbose
```

## API

```python
from pipeline import (
    PipelineConfig,
    StepConfig,
    STEP_METADATA,
    get_pipeline_config,
    set_pipeline_config,
    run_pipeline,
    get_pipeline_status,
    validate_pipeline_config,
    create_pipeline_config,
    execute_pipeline_step,
    execute_pipeline_steps,
    get_output_dir_for_script,
    run_enhanced_health_check,
    EnhancedHealthChecker,
    StepExecutionResult,
)

# Create and run pipeline
config = create_pipeline_config(target_dir="input/", output_dir="output/")
result = run_pipeline(config)

# Execute specific steps
result = execute_pipeline_step(step_number=3, config=config)
results = execute_pipeline_steps([3, 5, 11, 12], config=config)

# Get pipeline status
status = get_pipeline_status()

# Validate configuration
is_valid = validate_pipeline_config(config)

# Health checking
checker = EnhancedHealthChecker()
health = run_enhanced_health_check()
```

## Key Exports

- `PipelineConfig` / `StepConfig` — configuration dataclasses
- `STEP_METADATA` — metadata for all 25 steps
- `run_pipeline` — execute the full pipeline
- `execute_pipeline_step` / `execute_pipeline_steps` — step-level execution
- `create_pipeline_config` / `validate_pipeline_config` — configuration
- `EnhancedHealthChecker` / `run_enhanced_health_check` — health monitoring
- `StepExecutionResult` — step result dataclass

## Configuration Options

| Option | Description |
| -------- | ------------- |
| `--target-dir` | Input directory containing GNN files |
| `--output-dir` | Output directory for all results |
| `--only-steps` | Comma-separated list of steps to run |
| `--skip-steps` | Comma-separated list of steps to skip |
| `--verbose` | Enable detailed logging |
| `--strict` | Fail on warnings (default: continue) |

## Output

- `output/pipeline_execution_summary.json` — Comprehensive execution report
- Per-step output directories (`output/N_step_output/`)


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `get_pipeline_config_info`
- `get_pipeline_status`
- `get_pipeline_steps`
- `validate_pipeline_dependencies`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [../main.py](../main.py) — Main orchestrator script


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
