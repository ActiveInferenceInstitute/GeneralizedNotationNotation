# Pipeline Module Specification

## Overview
Pipeline orchestration, configuration, and execution utilities.

## Components

### Configuration
- `config.py` - Pipeline configuration management
- `config_schema.py` - Configuration schema definitions

### Execution
- `executor.py` - Step execution engine
- `pipeline_step_template.py` - Template for new steps

### Utilities
- `output_utils.py` - Output directory management
- `validation.py` - Pipeline validation

### v3.0.0 Long-Running Orchestration (safe-by-design; no live mutation)
- `durable_streams.py` - `StreamManifest` (file/array, content-checksummed) + `ExecutionTrace`
  integrity & deterministic replay (`validate_stream_manifest`, `trace_integrity`, `replay_trace`)
- `run_session.py` - resumable `RunSession` manifests: atomic `checkpoint`/`load_session`,
  `remaining_units`/`status_report`, path-safe `cancel_safe_cleanup`
- `container_plan.py` - `generate_container_plan` (hardened) + static `security_review` +
  `RollbackDescriptor` + deterministic `compute_plan_hash` (no container/cluster is executed)
- Acceptance: `scripts/run_v3_orchestration_acceptance.py --strict`; reference:
  `doc/pipeline/v3_orchestration.md`

## Key Exports
```python
from pipeline import execute_pipeline_step, get_output_dir_for_script
from pipeline.config import PipelineConfig, get_step_config
```

## Step Naming Convention
Steps follow `N_name.py` pattern (0-24)


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
