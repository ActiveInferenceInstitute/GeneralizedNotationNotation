# Pipeline Module - Agent Scaffolding

## Module Overview

**Purpose**: Pipeline orchestration, configuration management, and execution coordination for the GNN processing system

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Pipeline Infrastructure / Orchestration

**Status**: ✅ Production Ready

**v3.0.0 orchestration (safe-by-design, no live mutation)**: `durable_streams.py` (stream
manifests + replayable execution traces), `run_session.py` (resumable run sessions with atomic
checkpoint/resume + path-safe cleanup), and `container_plan.py` (auditable container plans + static
security review + rollback). These generate/validate **data only** — no container or cluster is
executed. Acceptance: `scripts/run_v3_orchestration_acceptance.py --strict`. Reference:
[`doc/pipeline/v3_orchestration.md`](../../doc/pipeline/v3_orchestration.md).

**Version**: 3.2.0

**Last Updated**: 2026-09-04

---

## Core Functionality

### Primary Responsibilities

1. Pipeline execution orchestration and step coordination
2. Configuration management and validation
3. Step discovery and dependency management
4. Pipeline health monitoring and diagnostics
5. Execution planning and resource estimation
6. Pipeline validation and verification

### Key Capabilities

- Multi-step pipeline orchestration
- Dynamic step discovery and configuration
- Pipeline health monitoring and alerting
- Resource estimation and allocation
- Execution plan generation
- Performance tracking and optimization
- Error recovery and retry mechanisms

---

## API Reference

### Configuration Functions

#### `get_pipeline_config() -> Dict[str, Any]`

**Description**: Get the complete pipeline configuration including step metadata and settings

**Returns**: `Dict[str, Any]` - Pipeline configuration dictionary with:

- `steps`: List of step names
- `output_base_dir`: Base output directory
- `log_level`: Logging level
- `step_configs`: Step-specific configurations

**Example**:

```python
from pipeline import get_pipeline_config

config = get_pipeline_config()
print(f"Steps: {config['steps']}")
```

#### `set_pipeline_config(config: Dict[str, Any]) -> None`

**Description**: Set pipeline configuration

**Parameters**:

- `config` (Dict[str, Any]): Configuration dictionary to set

**Returns**: `None`

#### `get_output_dir_for_script(script_name: str, base_output_dir: Optional[Path] = None) -> Path`

**Description**: Get standardized output directory for a pipeline script

**Parameters**:

- `script_name` (str): Name of the pipeline script (e.g., "3_gnn.py")
- `base_output_dir` (Optional[Path]): Base output directory (default: Path("output"))

**Returns**: `Path` - Output directory path (e.g., "output/3_gnn_output/")

**Example**:

```python
from pipeline import get_output_dir_for_script
from pathlib import Path

output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
```

### Step Validation and DAG Functions

#### `validate_pipeline_step(step_name: str) -> bool`

Validate that a step name is known to the pipeline (present in
`STEP_METADATA`).

**Location**: `src/pipeline/__init__.py`

#### `discover_pipeline_steps() -> list[str]`

Return the ordered list of registered step script stems (derived from
`step_registry.STEP_METADATA_DICT`, the single source of truth).

**Location**: `src/pipeline/__init__.py`

#### `resolve_execution_order(step_dependencies, total_steps=None, skip_steps=None, raise_on_circular=False) -> List[List[int]]`

**Location**: `src/pipeline/dag.py`

Topologically sort **step numbers** into parallel execution tiers (Kahn's
algorithm with tier grouping). `step_dependencies` maps each step number to
the step numbers it depends on (`utils.pipeline_step_dependencies.PIPELINE_STEP_DEPENDENCIES`
is the canonical dependency table). `total_steps` defaults to the registry
step count. Circular dependencies are appended as a final tier unless
`raise_on_circular=True`, which raises `ValueError`.

#### `find_circular_dependencies(step_dependencies, nodes=None) -> Set[int]`

**Location**: `src/pipeline/dag.py` (added 2026-09-04)

Return the set of step numbers bound up in dependency cycles (cycle members
plus any step that transitively depends on one). Powers the
`circular_dependencies` field of the `validate_pipeline_dependencies` MCP
tool.

#### `visualize_dag(tiers, step_names=None) -> str`

Render DAG tiers (output of `resolve_execution_order`) as a human-readable
multi-line string for logging.

> **Note**: The functions `validate_step_prerequisites`, `validate_pipeline_step_sequence`, and `generate_execution_plan` referenced in earlier documentation versions do not exist as standalone functions. Prerequisite checking is handled by `pipeline/pipeline_validator.py` (an E2E runtime tester) and dependency ordering is in `pipeline/dag.py`.

### Execution Planning

**Execution planning** is handled inline by `main.py` using `PipelineContext` and `StepRecord` dataclasses from `pipeline/context.py`, not via a separate `pipeline_planner.py` module.

**Example**:

```python
from pipeline.config import get_output_dir_for_script
from pipeline.dag import resolve_execution_order

output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
```

### Execution Functions

#### `run_pipeline(pipeline_data=None, *, target_dir=None, output_dir=None, steps="all", verbose=False) -> dict`

Execute the pipeline through ``main.py`` and return a compact summary dict
with `success`, `steps_executed`, `errors`, `warnings`, `target_dir`,
`output_dir`, `exit_code`, `duration`, and (when the run wrote a summary)
`summary_file` / `overall_status`. `steps` accepts `"all"`, a single step
name/number, a comma-separated list, or an iterable of those — normalized by
`resolve_step_numbers`.

#### `resolve_step_numbers(steps, pipeline_data=None) -> list[int]`

**Added 2026-09-04.** Normalize step identifiers ("11", "11_render",
"11_render.py", "3,5", iterables of those, or `"all"`/`None`) to a sorted,
de-duplicated list of registered step numbers. Unknown tokens are dropped.

#### `execute_pipeline_step(step_name: str, step_config: dict, pipeline_data: dict) -> StepExecutionResult`

Execute a single numbered pipeline step via ``main.execute_pipeline_step``.

**Returns**: `StepExecutionResult` — dataclass with `step_name`, `success`,
`duration`, `output`, `error`, `warnings`, `remediation`.

#### `get_pipeline_status() -> dict`

Static readiness probe: `{"status": "ready", "timestamp", "steps_available",
"steps_completed"}`. Step availability is derived from the registry; live
progress is reported by the per-run `pipeline_execution_summary.json` instead.

### Health Check Functions

#### `run_enhanced_health_check() -> Dict[str, Any]`

**Description**: Run the pipeline health check (components, dependencies, step discovery, config sanity). Returns a status dict instead of raising.

**Returns**: `Dict[str, Any]` - Health check results with:

- `overall_status` (str): "healthy", "degraded", or "unhealthy"
- `component_status` (Dict[str, str]): Status of each component
- `issues` (List[str]): Detected issues
- `recommendations` (List[str]): Recommended actions

---

## Dependencies

### Required Dependencies

- `pathlib` - Path manipulation
- `typing` - Type hints
- `logging` - Logging functionality

### Internal Dependencies

- `utils.argument_utils` - Argument parsing utilities
- `utils.logging_utils` - Structured (JSON-L + text) logging helpers
- `utils.pipeline_template` - Pipeline template utilities

---

## Configuration

### Environment Variables

None dedicated to this module. Pipeline-level behavior is configured through
`input/config.yaml` (loaded by `src/main.py` via `utils/arg_parsing.py`) and
CLI flags.

### Configuration Files

- `pipeline_config.yaml` - Optional project-local pipeline configuration
  (default lookup path in `pipeline/config.py`)
- `input/config.yaml` - The pipeline's primary configuration file

### Default Settings

Configuration defaults (`timeout: 3600`, `retries: 3`) are set in
`pipeline/config.py` (`PipelineConfig`); step-level defaults live with each
module.

---

## Usage Examples

### Basic Pipeline Configuration

```python
from pipeline.config import get_pipeline_config, get_output_dir_for_script

# Get current configuration
config = get_pipeline_config()
print(f"Output directory: {config['output_dir']}")

# Get output directory for specific step
output_dir = get_output_dir_for_script("3_gnn.py", Path("output"))
print(f"GNN output directory: {output_dir}")
```

### Step Validation

```python
from pipeline import validate_pipeline_step, discover_pipeline_steps

# Verify a step is known (keys are script stems)
assert validate_pipeline_step("3_gnn")

# List all steps
for step in discover_pipeline_steps():
    print(step)
```

### DAG-Based Ordering

```python
from pipeline.dag import resolve_execution_order, find_circular_dependencies
from utils.pipeline_step_dependencies import PIPELINE_STEP_DEPENDENCIES

# Resolve tiers over the canonical dependency table
tiers = resolve_execution_order(dict(PIPELINE_STEP_DEPENDENCIES))
print(f"Resolved tiers: {tiers}")

# Detect cycle-bound steps in an arbitrary graph
assert find_circular_dependencies({0: [1], 1: []}) == set()
```

---

## Output Specification

### Output Products

- `pipeline_execution_summary.json` - Execution summary (written by `main.py` under `00_pipeline_summary/`)
- `pipeline_health_report_<timestamp>.json` - Health monitoring report (written by `utils.pipeline_monitor`)

### Output Directory Structure

```text
output/
├── pipeline_config.yaml
├── pipeline_execution_summary.json
├── pipeline_health_report.json
└── step_execution_reports/
    ├── 0_template_execution.json
    ├── 1_setup_execution.json
    └── ...
```

---

## Performance Characteristics

### Latest Execution

- **Duration**: Variable (depends on pipeline length)
- **Memory**: ~10-50MB for orchestration
- **Status**: ✅ Production Ready

### Expected Performance

- **Orchestration Overhead**: < 5% of total pipeline time
- **Configuration Loading**: < 100ms
- **Step Discovery**: < 500ms
- **Health Monitoring**: < 10ms per check

---

## Error Handling

### Pipeline Errors

1. **Configuration Errors**: Invalid pipeline configuration
2. **Dependency Errors**: Missing step dependencies
3. **Resource Errors**: Insufficient resources for execution
4. **Timeout Errors**: Step execution timeout
5. **Validation Errors**: Invalid step sequence or parameters

### Recovery Strategies

- **Auto-retry**: Automatic retry for transient failures
- **Graceful degradation**: Continue with available steps
- **Resource reallocation**: Adjust resource allocation
- **Configuration repair**: Attempt to fix configuration issues

---

## Integration Points

### Orchestrated By

- **Script**: `main.py` (Main pipeline orchestrator)
- **Function**: Pipeline execution coordination

### Imports From

- `utils.argument_utils` - Argument parsing
- `utils.logging_utils` - Structured logging
- `utils.pipeline_template` - Template utilities

### Imported By

- All pipeline scripts (0_template.py through 24_intelligent_analysis.py)
- `tests.test_pipeline_*` - Pipeline tests
- `mcp.pipeline_tools` - MCP pipeline tools

### Data Flow

```text
Configuration → Step Discovery → Dependency Validation → Execution Planning → Step Execution → Health Monitoring
```

---

## Testing

### Test Files

The pipeline test suite lives in `src/tests/pipeline/` (45+ files). Key
examples:

- `src/tests/pipeline/test_pipeline_orchestration.py` - Orchestration, config, DAG tiers
- `src/tests/pipeline/test_pipeline_integration.py` - Integration tests
- `src/tests/pipeline/test_pipeline_refactor_contracts.py` - Shared-building-block
  contracts (`pipeline._io`, `dag.find_circular_dependencies`,
  `execution.resolve_step_numbers`, `select_model_families`, preflight
  `skip_steps` gate) added 2026-09-04
- `src/tests/pipeline/test_pipeline_performance.py` - Performance tests

### Test Coverage

Measure on demand — no static number is kept in this file:

```bash
uv run --extra dev python -m pytest src/tests/pipeline/ --cov=src/pipeline --cov-report=term-missing
```

### Key Test Scenarios

1. Pipeline configuration validation
2. Step dependency resolution
3. Execution plan generation
4. Health monitoring functionality
5. Error recovery mechanisms

---

## MCP Integration

### Tools Registered

Registered by `register_tools()` in `pipeline/mcp.py`:

- `get_pipeline_steps` - Step metadata and dependencies
- `get_pipeline_status` - Current execution status, recent logs, statistics
- `validate_pipeline_dependencies` - Missing or circular dependency check
- `get_pipeline_config_info` - Detailed configuration information
- `get_v3_orchestration_capabilities`, `run_v3_container_security_review`, `run_v3_orchestration_self_check` - v3.0.0 safe-by-design orchestration tools (data only, no live mutation)
---

## Troubleshooting

### Common Issues

#### Issue 1: Step discovery fails

**Symptom**: Pipeline can't find or load pipeline steps  
**Cause**: Step scripts missing or naming convention mismatch  
**Solution**:

- Verify all step scripts exist in `src/` directory
- Check script naming follows pattern `N_module.py`
- Ensure scripts are executable and have correct imports
- Review step discovery logs

#### Issue 2: Configuration validation errors

**Symptom**: Pipeline fails with configuration errors  
**Cause**: Invalid configuration values or missing required settings  
**Solution**:

- Verify configuration file format is valid
- Check all required configuration keys are present
- Use default configuration if issues persist

---


## Version History

### 2026-09-04 — Composability refactor (fleet worker)

Internal quality pass; every external entry point's behavior is preserved:

- **Shared atomic writes**: `pipeline/_io.py` now owns the mkstemp +
  `os.replace` recipe. `durable_streams`, `run_session.checkpoint`,
  `run_manifest._write_index`, and `hasher.index_run` all delegate (the
  history index write is now atomic too).
- **Shared family selection**: `model_family_acceptance.select_model_families()`
  is the single filter rule; `semantic_fidelity` and `session_acceptance`
  delegate instead of re-implementing it.
- **Cycle detection**: new `dag.find_circular_dependencies()`; the
  `validate_pipeline_dependencies` MCP tool now reports real
  `circular_dependencies` instead of a stubbed empty list.
- **Single version source**: `pipeline/_version.py`; `execution.get_pipeline_info()`
  now reports the package version instead of a stale "1.0.0".
- **Registry-derived counts**: `dag.resolve_execution_order`'s default
  `total_steps` and `execution.get_pipeline_status/get_pipeline_info` derive
  step counts from `step_registry` instead of hardcoding 25.
- **Path constants**: `config.DEFAULT_TARGET_DIR` / `config.DEFAULT_OUTPUT_DIR`
  replace the repeated `"input/gnn_files"` / `"output"` literals
  (`execution`, `context`, package defaults).
- **Status-set dedup**: `execution._SUCCESS_STATUSES` replaces the two inline
  copies of `{"SUCCESS", "SUCCESS_WITH_WARNINGS", "SKIPPED"}`.
- **Comma-separated step lists**: `resolve_step_numbers("3,5")` now works
  (mirrors the `--only-steps` CLI form; previously silently resolved to `[]`).
- **Preflight**: `validate_config` now validates `pipeline.skip_steps` with
  the canonical `read_skip_steps` parser (bad values are a preflight error
  instead of a mid-run failure).
- **Package surface**: `__all__` is fully typed and complete (adds
  `PipelineContext`, `StepRecord`, `StepStatus`, `resolve_step_numbers`,
  `get_module_info`, `validate_pipeline_step`, `discover_pipeline_steps`,
  `DEFAULT_TARGET_DIR`, `DEFAULT_OUTPUT_DIR`).

### Current Version: 3.2.0

**Features**:

- Pipeline orchestration
- Configuration management
- Step discovery and dependency management
- Health monitoring

**Known Issues**:

- None currently

### Roadmap

- Candidate: DAG-based parallel execution for independent step clusters
- Candidate: per-step streaming metrics published via MCP

---

## References

### Related Documentation

- [Pipeline Overview](../../README.md)
- [Utils Module](../utils/AGENTS.md)

### External Resources

- [v3 Orchestration](../../doc/pipeline/v3_orchestration.md)
- [Stage Hardening Review](../../doc/pipeline/pipeline_stage_hardening_review.md)


---

**Last Updated**: 2026-09-04
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 3.2.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API

## Step Registry — `step_registry.py`

The canonical pipeline step registry lives in [`step_registry.py`](step_registry.py). Every other step list in the codebase derives from this single source of truth.

- **Key types:** `StepInfo` (frozen dataclass), `STEPS` (ordered list of 25 entries).
- **Lookup:** `step_for_name("11_render")`, `step_for_stem("11_render")`.
- **Filtering:** `get_core_steps()` → 24 steps, `get_llm_steps()` → 1 step.
- **Adding a new step:** Add one `StepInfo(...)` to `STEPS` — all downstream consumers update automatically.
