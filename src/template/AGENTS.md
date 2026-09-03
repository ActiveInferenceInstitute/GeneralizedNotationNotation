# Template Module - Agent Scaffolding

## Module Overview

**Purpose**: Pipeline template and initialization system for the GNN processing pipeline

**Pipeline Step**: Step 0: Template initialization (0_template.py)

**Category**: Pipeline Infrastructure / Initialization

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-02

---

## Core Functionality

### Primary Responsibilities
1. Pipeline initialization and template generation
2. Infrastructure demonstration and pattern validation
3. Template processing and customization
4. Pipeline architecture documentation
5. Example generation and testing

### Key Capabilities
- Dynamic pipeline template generation
- Infrastructure pattern demonstration
- Template customization and validation
- Pipeline architecture documentation
- Example and test data generation

---

## API Reference

### Module exports

- `VERSION_INFO` — dict with `version`, `name`, `description`, `author` (included in `__all__`)
- `FEATURES` — capability flags for tooling and discovery

### Public Functions

#### `process_template_standardized(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False, verbose: bool = False, **kwargs) -> bool`
**Description**: Process pipeline template with standardized patterns. This is the main processing function called by the thin orchestrator.

**Parameters**:
- `target_dir` (Path): Target directory for template processing
- `output_dir` (Path): Output directory for results
- `logger` (logging.Logger): Logger instance for logging
- `recursive` (bool): Process subdirectories recursively (default: False)
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional processing options

**Returns**: `bool` - True if template processing succeeded, False otherwise

**Example**:
```python
from template import process_template_standardized
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_template_standardized(
    target_dir=Path("input/"),
    output_dir=Path("output/0_template_output/"),
    logger=logger,
    recursive=True,
    verbose=True,
)
```

#### `process_single_file(input_file: Path, output_dir: Path, options: Dict[str, Any]) -> bool`
**Description**: Process a single file using the template logic. Writes `<stem>/<stem>_processed<ext>` and a `<stem>_report.json` into a per-file subdirectory of `output_dir`.

**Parameters**:
- `input_file` (Path): Path to input file to process
- `output_dir` (Path): Directory to save output files
- `options` (Dict[str, Any]): Processing options dictionary (required positional argument)

**Returns**: `bool` - True if file processing succeeded, False otherwise

#### `validate_file(input_file: Path) -> Dict[str, Any]`
**Description**: Validate a file against template requirements.

**Parameters**:
- `input_file` (Path): Path to file to validate

**Returns**: `Dict[str, Any]` - Validation result dictionary with:
- `status` (str): "ok" when the file exists, is a regular file, and is readable; "error" otherwise
- `error` (str): Error message (present when `status` is "error")
- `file_path` (str): The validated path

#### `safe_template_execution(logger, correlation_id)`
**Description**: Context manager for safe template execution with comprehensive error handling, correlation-aware logging, and resource tracking.

**Parameters**:
- `logger` (logging.Logger): Logger instance for the execution context
- `correlation_id` (str): Correlation ID to tag log entries with

**Usage**:
```python
from template import safe_template_execution, generate_correlation_id

with safe_template_execution(logger, generate_correlation_id()):
    process_template_standardized(target_dir, output_dir, logger)
```

#### `get_version_info() -> Dict[str, str]`
**Description**: Get module version and metadata information.

**Returns**: `Dict[str, str]` - Version information dictionary with:
- `version` (str): Module version string
- `name` (str): Module name
- `description` (str): Module description
- `author` (str): Module author

#### `generate_correlation_id() -> str`
**Description**: Generate a short correlation ID for pipeline tracking and request correlation.

**Returns**: `str` - Correlation ID string (first 8 hex chars of a UUID4)

**Example**:
```python
from template import generate_correlation_id

correlation_id = generate_correlation_id()
# Returns: "550e8400" (first 8 hex chars of a UUID4)
```

#### `demonstrate_utility_patterns(context: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]`
**Description**: Demonstrate utility patterns and capabilities for documentation and testing purposes.

**Parameters**:
- `context` (Dict[str, Any]): Processing context dictionary
- `logger` (logging.Logger): Logger instance for demonstration logging
**Returns**: `Dict[str, Any]` - Demonstration results dictionary with:

- `timestamp`, `correlation_id` (str): Execution timestamp and correlation ID
- `patterns_demonstrated` (List[str]): List of demonstrated patterns
- `infrastructure_status` (Dict[str, Any]): Status of each infrastructure subsystem
- `performance_metrics` (Dict[str, float]): Performance metrics

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `uuid` - Unique ID generation
- `datetime` - Timestamp generation

### Internal Dependencies
- `utils.pipeline_template` - Pipeline template utilities

## Configuration

No module-level configuration file. The step is configured entirely through standard pipeline arguments (`--target-dir`, `--output-dir`, `--recursive`, `--verbose`).

---

## Usage Examples

### Basic Template Processing
```python
from template.processor import process_template_standardized

success = process_template_standardized(
    target_dir="input/", output_dir="output/0_template_output", logger=logger
)
```

### Utility Pattern Demonstration
```python
from template.processor import demonstrate_utility_patterns

results = demonstrate_utility_patterns(context, logger)
print(f"Patterns demonstrated: {len(results['demonstrations'])}")
```

### Correlation ID Generation
```python
from template.processor import generate_correlation_id

correlation_id = generate_correlation_id()
print(f"Generated ID: {correlation_id}")
```

---

## Output Specification

### Output Products
- `template_processing_summary.json` - Template processing results
- Per input file, under `output_dir/<file_stem>/`:
  - `<file_stem>_processed<ext>` - Processed copy of the input file
  - `<file_stem>_report.json` - Per-file processing report

### Output Directory Structure
```
output/0_template_output/
├── template_processing_summary.json
└── <file_stem>/
    ├── <file_stem>_processed<ext>
    └── <file_stem>_report.json
```

---

## Performance Characteristics

### Expected Performance
- **Template Processing**: < 1 second
- **Pattern Demonstration**: 1-2 seconds

---

## Error Handling

### Template Errors
1. **Template Generation**: Template creation failures
2. **Pattern Validation**: Pattern validation errors
3. **File I/O**: File operation failures
4. **Configuration**: Invalid template configuration

### Recovery Strategies
- **Template Regeneration**: Recreate templates from defaults
- **Pattern Simplification**: Use simpler patterns
- **Documentation Recovery**: Generate basic documentation
- **Error Logging**: Comprehensive error reporting

---

## Integration Points

### Orchestrated By
- **Script**: `0_template.py` (Step 0)
- **Function**: `process_template_standardized()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_template_*` - Template tests

### Data Flow
```
Template Input → Processing → Pattern Demonstration → Validation → Documentation → Output
```

---

## Testing

### Test Files
- `src/tests/template/test_template_overall.py` - Module-level tests (imports, outputs, and core behaviors)
- `src/tests/pipeline/test_pipeline_scripts.py` - Orchestrator-level checks that include `0_template.py`

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/test_template*.py \
    --cov=src/template --cov-report=term-missing
```
### Key Test Scenarios
1. Template processing and generation
2. Pattern demonstration and validation
3. Documentation creation
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `template.process_file` - Process a single file (`mcp.py`)
- `template.process_directory` - Process a directory (`mcp.py`)
- `template.get_info` - Return template module metadata (`mcp.py`)

### Registration
Tools are registered by `template.mcp.register_tools(registry)`; there is no `@mcp_tool` decorator in this module.

---

## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
