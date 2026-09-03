# Utils Module

This module provides core utilities used throughout the GNN pipeline, including unified logging, argument parsing, pipeline orchestration, and common helper functions that ensure consistency across all modules.

## Module Structure

```
src/utils/
├── __init__.py                      # Module initialization and exports
├── AGENTS.md                        # AI agent scaffolding documentation
├── README.md                        # This documentation
├── SPEC.md                          # Module specification
│
├── # Logging & Diagnostics
├── logging_utils.py                 # Facade over structured logging (setup_step_logging, log_step_*)
├── structured_logging.py            # Structured pipeline logging with correlation IDs
├── visual_logging.py                # Visual log formatting
├── diagnostic_logging.py            # Diagnostic logging utilities
│
├── # Configuration & Arguments
├── arg_definitions.py               # Shared STEP_ARGUMENTS/constants
├── arg_parsing.py                   # ArgumentParser (parse_step_arguments, build_step_command_args)
├── argument_utils.py                # Thin re-export module over the arg_* modules
├── path_conversion.py               # validate_and_convert_paths
├── step_config.py                   # StepConfiguration (shared step defaults)
├── config_loader.py                 # YAML/JSON config loading (load_config, get/set_config_value)
│
├── # Pipeline Infrastructure
├── pipeline.py                      # Pipeline utilities (get_output_dir_for_script, ...)
├── pipeline_template.py             # Standardized pipeline script templates
├── pipeline_dependencies.py         # Step dependency management
├── pipeline_monitor.py              # Pipeline health monitoring
├── pipeline_planner.py              # Execution planning
├── pipeline_validator.py            # Validation utilities
├── pipeline_arguments.py            # Pipeline-level argument building
├── pipeline_config_merge.py         # Configuration merging
├── pipeline_step_dependencies.py    # Step dependency declarations
│
├── # Dependency Management
├── dependency_audit.py              # Dependency auditing
├── dependency_installer.py          # Dependency installation
├── dependency_manager.py            # Dependency management
├── dependency_validator.py          # Dependency validation
│
├── # Error Handling & Recovery
├── error_handling.py                # Error handling framework
├── error_recovery.py                # ErrorRecoveryManager, ErrorContext
│
├── # Resource & Performance
├── resource_manager.py              # Resource monitoring (get_current_memory_usage)
├── performance_tracker.py           # PerformanceTracker, track_operation_standalone
├── timeout_manager.py               # Timeout management
├── visualization_optimizer.py       # Visualization optimization
│
├── # Testing & Validation
├── test_utils.py                    # Test utilities
├── script_validator.py              # Script validation
├── validation_schemas.py            # Shared validation schemas
│
├── # Utilities
├── base_processor.py                # Base processor class
├── execution_utils.py               # Execution helpers
├── io_utils.py                      # I/O utilities
├── network_utils.py                 # Network utilities
├── path_utils.py                    # Path utilities
├── safe_eval.py                     # Bounded literal evaluation for untrusted strings
├── system_utils.py                  # System utilities
├── venv_utils.py                    # Virtual environment helpers
├── framework_availability.py        # Optional framework detection
├── jax_stack_validation.py          # JAX/Optax/Flax/pymdp 1.x probe (step 1)
├── matplotlib_setup.py              # Matplotlib backend configuration
│
├── # Specialized
├── mcp.py                           # MCP integration
├── migration_helper.py              # Migration utilities
├── simulation_monitor.py            # Simulation monitoring
├── simulation_utils.py              # Simulation utilities
├── code_metrics.py                  # Code metrics collection
└── logging/                         # Logging subpackage (see logging/README.md)
```

## Core Components

```mermaid
graph TD
    Pipeline[Pipeline Scripts] --> Utils{Utils Module}

    Utils --> Log[Unified Logging]
    Utils --> Args[Arg Parser]
    Utils --> Files[File Utils]
    Utils --> Valid[Validation]
    Utils --> Config[Configuration]
    Utils --> Error[Error Recovery]
    Utils --> Perf[Performance Tracking]

    Log --> StructLog[Structured Logs]
    Args --> Config[Configuration]
    Files --> IOSafe[Safe IO Ops]
    Valid --> Checks[Path/Config Checks]
    Error --> Recovery[Recovery Strategies]
    Perf --> Metrics[Performance Metrics]

    StructLog & Config & IOSafe & Checks & Recovery & Metrics --> Standard[Standardization]
```

### Unified Logging System

`utils/logging_utils.py` is the canonical entry point:

#### `setup_step_logging(step_name: str, verbose: bool = False) -> logging.Logger`
Sets up standardized logging for a pipeline step with correlation-ID tracking.

#### `setup_main_logging(verbose: bool = False) -> logging.Logger`
Sets up logging for the main pipeline orchestrator.

#### `log_step_start(logger, message)` / `log_step_success(logger, message)` / `log_step_error(logger, message)` / `log_step_warning(logger, message)`
Step lifecycle logging helpers. `utils/structured_logging.py` provides richer variants (`log_step_start(logger, step_name, **context)`) with metadata support.

#### `get_performance_summary() -> Dict[str, Any]`
Returns timing/memory metrics recorded by the structured logger.

### Argument Parsing

#### `ArgumentParser` (`utils/arg_parsing.py`, re-exported via `utils/argument_utils.py`)
Standard argument parser with pipeline-wide support.

#### `ArgumentParser.parse_step_arguments(step_name) -> argparse.Namespace`
Parses arguments for a specific pipeline step with recovery support. Standard arguments: `--target-dir`, `--output-dir`, `--verbose`, `--recursive` (plus step-specific definitions from `arg_definitions.STEP_ARGUMENTS`).

#### `build_step_command_args(step_name, args) -> List[str]`
Builds the command-line argument list for invoking a step script.

#### `audit_step_contracts() -> Dict[str, Any]`
Audits for drift between `STEP_ARGUMENTS`, `StepConfiguration`, parser defaults, and command-builder propagation. Exit codes are canonical: `0=success`, `1=error`, `2=success with warnings/skipped`.

### Pipeline Orchestration Utilities

#### `get_output_dir_for_script(script_name: str, base_output_dir: Optional[Path] = None) -> Path`
Gets the standardized per-step output directory (e.g. `"3_gnn.py"` → `output/3_gnn_output/`).

### Configuration

#### `load_config(config_path: Optional[Path] = None) -> GNNPipelineConfig`
Loads pipeline configuration (defaults when no path given).

#### `get_config_value(config, key) -> Any` / `set_config_value(config, key, value) -> Any`
Get/set configuration values with dot-notation keys.

## Usage Examples

### Basic Logging Setup

```python
from utils.logging_utils import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
)

logger = setup_step_logging("my_step", verbose=True)

log_step_start(logger, "Starting my_step")
try:
    results = perform_processing()
    log_step_success(logger, "my_step completed")
except Exception as e:
    log_step_error(logger, f"my_step failed: {e}")
    raise
```

### Argument Parsing

```python
from utils.argument_utils import ArgumentParser

args = ArgumentParser.parse_step_arguments("my_step")
target_dir = args.target_dir
output_dir = args.output_dir
verbose = args.verbose
```

### Pipeline Orchestration

```python
from utils.pipeline import get_output_dir_for_script

output_dir = get_output_dir_for_script("my_script.py", Path("output"))
```

### Memory Monitoring

```python
from utils.resource_manager import get_current_memory_usage

memory_before = get_current_memory_usage()
# ... do some work ...
memory_after = get_current_memory_usage()
print(f"Memory delta: {memory_after - memory_before} MB")
```

## Integration with Pipeline

### Standard Module Pattern

```python
from utils.logging_utils import setup_step_logging, log_step_start, log_step_success, log_step_error
from pipeline import get_output_dir_for_script

logger = setup_step_logging("my_module", verbose=args.verbose)

def process_my_module(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool:
    try:
        log_step_start(logger, "Starting my_module")
        # Core processing logic here
        log_step_success(logger, "my_module completed")
        return True
    except Exception as e:
        log_step_error(logger, f"my_module failed: {e}")
        return False
```

## Error Handling

`utils/error_recovery.py` provides `ErrorRecoveryManager(logger=None)` with `handle_error(context: ErrorContext) -> bool` and helpers such as `format_and_log_error(logger, error, context)`. Errors are described by `ErrorContext` objects (operation, severity, message, error_code, details).

## Testing and Validation

Tests live in `src/tests/utils/` (including `test_utils_core.py` and `test_new_utils.py`).

Run: `uv run --extra dev python -m pytest src/tests/utils/ -v`

## Dependencies

Standard library plus `pyyaml` (config loading) and `psutil` (resource monitoring) — both core pyproject dependencies. No rich/click/pydantic/structlog requirements.

## Troubleshooting

- **No log output**: verify the `verbose` flag on `setup_step_logging` / `setup_main_logging` and that the logger is not filtered by an upstream handler.
- **Argument parsing errors**: run the step with `--help` to see the argument definitions registered in `utils/arg_definitions.py`.
- **Debug logging**: set the logger level to `DEBUG` (`logging.getLogger().setLevel(logging.DEBUG)`) or pass `verbose=True`.

## Summary

The Utils module provides core utilities used throughout the GNN pipeline: unified logging with correlation IDs, standardized argument parsing with step contracts, pipeline orchestration helpers, configuration loading, dependency validation, and error recovery. These utilities form the foundation for consistent pipeline behavior across all 25 steps.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information.

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
